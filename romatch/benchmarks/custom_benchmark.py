import torch
import numpy as np
import tqdm
from romatch.datasets.custom_dataset import CustomBuilder
from romatch.utils import warp_kpts
from torch.utils.data import ConcatDataset
import romatch

class UmbraBenchmark:
    def __init__(self, val_json_path, h=384, w=512, num_samples=1000) -> None:
        """
        Бенчмарк для оценки на валидационном датасете Umbra.

        Args:
            val_json_path (str): Путь к файлу pairs_val.json.
            h (int): Высота изображений для бенчмарка.
            w (int): Ширина изображений для бенчмарка.
            num_samples (int): Количество пар для оценки.
        """
        custom_data_builder = CustomBuilder(data_root=val_json_path)
        # Убедимся, что image_size соответствует разрешению (h)
        val_scenes = custom_data_builder.build_scenes(image_size=h)
        self.dataset = ConcatDataset(val_scenes)
        
        # Ограничиваем количество семплов, если датасет большой
        self.num_samples = min(num_samples, len(self.dataset))
        
        # Размеры, которые ожидает geometric_dist
        self.w1 = w
        self.h1 = h

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
            mask, x2 = warp_kpts(
                x1.double(),
                depth1.double(),
                depth2.double(),
                T_1to2.double(),
                K1.double(),
                K2.double(),
            )
            # Переводим координаты из [-1, 1] в пиксели
            x2_px = torch.stack(
                (self.w1 * (x2[..., 0] + 1) / 2, self.h1 * (x2[..., 1] + 1) / 2), dim=-1
            )
            prob = mask.float().reshape(b, h1, w1)
            
        x2_hat = dense_matches[..., 2:]
        x2_hat_px = torch.stack(
            (self.w1 * (x2_hat[..., 0] + 1) / 2, self.h1 * (x2_hat[..., 1] + 1) / 2), dim=-1
        )
        
        # Считаем евклидово расстояние в пикселях
        gd = (x2_hat_px - x2_px.reshape(b, h1, w1, 2)).norm(dim=-1)
        
        # Учитываем только валидные точки, где была глубина
        valid_gd = gd[prob == 1]
        if len(valid_gd) == 0: # Если нет валидных точек, возвращаем 0
            return torch.tensor(0.0).to(gd.device), 0.0, 0.0, 0.0, prob

        pck_1 = (valid_gd < 1.0).float().mean()
        pck_3 = (valid_gd < 3.0).float().mean()
        pck_5 = (valid_gd < 5.0).float().mean()
        
        return valid_gd, pck_1, pck_3, pck_5, prob

    def benchmark(self, model, batch_size=8):
        model.train(False)
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0
            
            # Семплер, чтобы каждый раз выбирать случайные пары
            sampler = torch.utils.data.RandomSampler(
                self.dataset, replacement=False, num_samples=self.num_samples
            )
            
            dataloader = torch.utils.data.DataLoader(
                self.dataset, batch_size=batch_size, num_workers=batch_size, sampler=sampler
            )
            
            if len(dataloader) == 0:
                print("Dataloader для бенчмарка пуст, пропускаем оценку.")
                return {}

            for data in tqdm.tqdm(dataloader, desc="Running Umbra Benchmark", disable = romatch.RANK > 0):
                im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
                    data["im_A"].cuda(),
                    data["im_B"].cuda(),
                    data["im_A_depth"].cuda(),
                    data["im_B_depth"].cuda(),
                    data["T_1to2"].cuda(),
                    data["K1"].cuda(),
                    data["K2"].cuda(),
                )
                
                matches, certainty = model.match(im_A, im_B, batched=True)
                
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(
                    depth1, depth2, T_1to2, K1, K2, matches
                )

                if torch.is_tensor(gd) and gd.numel() > 0:
                    gd_tot += gd.mean()

                pck_1_tot += pck_1
                pck_3_tot += pck_3
                pck_5_tot += pck_5

        return {
            "umbra_epe": gd_tot.item() / len(dataloader),
            "umbra_pck_1": pck_1_tot.item() / len(dataloader),
            "umbra_pck_3": pck_3_tot.item() / len(dataloader),
            "umbra_pck_5": pck_5_tot.item() / len(dataloader),
        }
