"""
UmbraDenseBenchmark - бенчмарк для оценки качества сопоставления
на парах изображений Umbra SAR.

Вычисляет метрики:
- EPE (End-Point Error) - средняя геометрическая ошибка
- PCK@1, PCK@3, PCK@5 - процент корректных соответствий
"""

import torch
import tqdm
from romatch.utils import warp_kpts, warp_kpts_planar
import romatch
import os
from romatch.tools.visualize_pipeline import visualize_total

class UmbraDenseBenchmark:
    def __init__(self, scene_info, image_size=640, vis_dir=None, planar_mode=False) -> None:
        """
        Args:
            scene_info: словарь с ключами 'image_paths' и 'pairs'
            image_size: размер изображения для обработки
            vis_dir: директория для сохранения визуализаций
        """
        from romatch.datasets.umbra import UmbraScene

        self.dataset = UmbraScene(scene_info, image_size=image_size, scene_name="umbra_val")
        self.vis_dir = vis_dir
        self.planar_mode = planar_mode
        if self.vis_dir is not None:
            os.makedirs(self.vis_dir, exist_ok=True)

    def geometric_dist(self, depth1, depth2, T_1to2, K1, K2, dense_matches):
        """Вычисляет геометрическое расстояние между предсказанными и истинными соответствиями"""
        b, h1, w1, d = dense_matches.shape
        with torch.no_grad():
            x1 = dense_matches[..., :2].reshape(b, h1 * w1, 2)
            if self.planar_mode:
                mask, x2 = warp_kpts_planar(
                    x1.double(),
                    T_1to2.double()
                )
            else:
                mask, x2 = warp_kpts(
                    x1.double(),
                    depth1.double(),
                    depth2.double(),
                    T_1to2.double(),
                    K1.double(),
                    K2.double(),
                )
            x2 = torch.stack((w1 * (x2[..., 0] + 1) / 2, h1 * (x2[..., 1] + 1) / 2), dim=-1)
            prob = mask.float().reshape(b, h1, w1)
        x2_hat = dense_matches[..., 2:]
        x2_hat = torch.stack((w1 * (x2_hat[..., 0] + 1) / 2, h1 * (x2_hat[..., 1] + 1) / 2), dim=-1)
        gd = (x2_hat - x2.reshape(b, h1, w1, 2)).norm(dim=-1)
        gd = gd[prob == 1]
        pck_1 = (gd < 1.0).float().mean()
        pck_3 = (gd < 3.0).float().mean()
        pck_5 = (gd < 5.0).float().mean()
        return gd, pck_1, pck_3, pck_5, prob

    

    def benchmark(self, model, batch_size=4):
        """
        Запускает бенчмарк на датасете

        Args:
            model: модель для тестирования
            batch_size: размер батча

        Returns:
            dict: словарь с метриками
        """
        model.train(False)
        results = {}
        with torch.no_grad():
            gd_tot = 0.0
            pck_1_tot = 0.0
            pck_3_tot = 0.0
            pck_5_tot = 0.0

            sampler = torch.utils.data.WeightedRandomSampler(
                torch.ones(len(self.dataset)), replacement=False, num_samples=len(self.dataset)
            )
            B = batch_size
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=B, num_workers=4, sampler=sampler)

            for idx, data in tqdm.tqdm(enumerate(dataloader), disable=romatch.RANK > 0, desc="Umbra Benchmark"):
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
                gd, pck_1, pck_3, pck_5, prob = self.geometric_dist(depth1, depth2, T_1to2, K1, K2, matches)

                # Визуализация
                if self.vis_dir is not None and idx < 1:  # визуализируем только первый батч
                    results["visual"] = [
                        visualize_total(
                            im_A=im_A[b],
                            im_B=im_B[b],
                            matches=matches[b],
                            certainty=certainty[b],
                            T_1to2=T_1to2[b],
                            K1=K1[b],
                            K2=K2[b],
                            depth1=depth1[b],
                            depth2=depth2[b],
                            planar_mode=self.planar_mode,
                        )[0]
                        for b in range(min(B, im_A.shape[0]))
                    ]

                gd_tot, pck_1_tot, pck_3_tot, pck_5_tot = (
                    gd_tot + gd.mean(),
                    pck_1_tot + pck_1,
                    pck_3_tot + pck_3,
                    pck_5_tot + pck_5,
                )

        num_batches = len(dataloader)
        results |= {
            "umbra_epe": gd_tot.item() / num_batches,
            "umbra_pck_1": pck_1_tot.item() / num_batches,
            "umbra_pck_3": pck_3_tot.item() / num_batches,
            "umbra_pck_5": pck_5_tot.item() / num_batches,
        }

        # Выводим результаты
        if romatch.RANK == 0:
            print("\n=== Umbra Benchmark Results ===")
            print(f"EPE: {results['umbra_epe']:.3f}")
            print(f"PCK@1px: {results['umbra_pck_1']:.3f}")
            print(f"PCK@3px: {results['umbra_pck_3']:.3f}")
            print(f"PCK@5px: {results['umbra_pck_5']:.3f}")
            if self.vis_dir is not None:
                for idx, fig in enumerate(results["visual"]):
                    fig.savefig(os.path.join(self.vis_dir, f"bench_{idx:02d}.png"))
                print(f"Visualizations saved to: {self.vis_dir}")

        return results
