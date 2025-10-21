from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from romatch.utils.utils import get_gt_warp, get_gt_warp_planar
import wandb
import romatch
import math

class RobustLosses(nn.Module):
    def __init__(
        self,
        ce_weight=0.01,
        local_dist=4.0,
        local_largest_scale=8,
        alpha = 1.,
        c = 1e-3,
        scale_weights = {1:1, 2:1, 4:1, 8:1, 16:1},
        tb_writer: SummaryWriter = None,
        planar_mode = False,
    ):
        """
        Инициализирует модуль RobustLosses для вычисления потерь для модели RoMa.

        Args:
            ce_weight (float): Вес для "уверенности" (certainty) в лоссах. 
                               Определяет баланс между лоссом на классификацию/регрессию 
                               и лоссом на уверенность модели в своих предсказаниях. 
                               Типичные значения: 0.01 - 0.1.
            local_dist (float or dict): Задает порог для End-Point-Error (EPE) в пикселях 
                                        для создания маски "локальных" правильных соответствий. 
                                        Используется для фильтрации грубых выбросов на ранних 
                                        стадиях обучения или на больших масштабах. 
                                        Может быть числом или словарем, где ключи - это масштабы.
            local_largest_scale (int): Максимальный масштаб, на котором применяется `local_dist` 
                                       фильтрация. Например, если 8, то фильтрация будет 
                                       применяться на масштабах >= 8.
            alpha (float or dict): Параметр, контролирующий форму робастной функции потерь 
                                   (general Charbonnier/pseudo-Huber loss).
                                   - alpha = 2.0: L2-подобная потеря.
                                   - alpha = 1.0: Charbonnier-подобная потеря.
                                   - alpha -> 0.0: L1-подобная потеря.
                                   Более низкие значения делают лосс менее чувствительным к выбросам.
                                   Может быть числом или словарем для разных масштабов.
            c (float): Параметр масштаба для робастной функции потерь. Определяет точку 
                       перехода от квадратичного поведения к линейному (для alpha < 2). 
                       Чем меньше `c`, тем раньше лосс становится линейным, т.е. более 
                       робастным к большим ошибкам. Типичное значение: 1e-3 - 1e-5.
            scale_weights (dict): Словарь с весами для каждого масштаба разрешения 
                                  (ключи: 1, 2, 4, 8, ...). Позволяет взвешивать вклад 
                                  каждого масштаба в итоговую функцию потерь.
            tb_writer (SummaryWriter, optional): Экземпляр SummaryWriter для логирования 
                                                 метрик в TensorBoard.
            planar_mode (bool): Если True, вычисляет ground-truth warp, предполагая, что 
                                сцена является плоской. В противном случае использует карты 
                                глубины для вычисления warp.
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.local_dist = local_dist
        self.local_largest_scale = local_largest_scale
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.scale_weights = scale_weights
        self.tb_writer = tb_writer
        self.planar_mode = planar_mode

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty, scale):
        """
        Вычисляет classification loss для Global Matching (GM) модуля.

        Этот лосс состоит из двух частей:
        1. Cross-Entropy loss для классификации, где модель предсказывает ячейку в 
           грубой сетке, которой принадлежит ground-truth соответствие.
        2. Binary Cross-Entropy loss для "уверенности" (certainty), где модель 
           учится предсказывать, существует ли соответствие для данной точки.

        Args:
            x2 (torch.Tensor): Ground truth координаты соответствий (B, H, W, 2).
            prob (torch.Tensor): Маска достоверности ground truth соответствий (B, H, W).
            scale_gm_cls (torch.Tensor): Выход классификатора GM (B, C, H, W).
            gm_certainty (torch.Tensor): Выход уверенности GM (B, 1, H, W).
            scale (int): Текущий масштаб.

        Returns:
            dict: Словарь с вычисленными лоссами (certainty и classification).
        """
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)], indexing='ij')
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2)
            GT = (G[None,:,None,None,:]-x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty.squeeze(1), prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
            
        losses = {
            f"gm_certainty_loss_{scale}": certainty_loss.mean(),
            f"gm_cls_loss_{scale}": cls_loss.mean(),
        }
        if self.tb_writer:
            for k,v in losses.items():
                self.tb_writer.add_scalar("train/" + k, v.item(), romatch.GLOBAL_STEP)
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        
        return losses

    def delta_cls_loss(self, x2, prob, flow_pre_delta, delta_cls, certainty, scale, offset_scale):
        """
        Вычисляет classification loss для Delta refinement модуля.

        Аналогично `gm_cls_loss`, но для модуля уточнений (Delta). Модель предсказывает
        смещение относительно предсказания с более грубого уровня.

        Args:
            x2 (torch.Tensor): Ground truth координаты соответствий.
            prob (torch.Tensor): Маска достоверности ground truth соответствий.
            flow_pre_delta (torch.Tensor): Поле потока с предыдущего, более грубого уровня.
            delta_cls (torch.Tensor): Выход классификатора Delta.
            certainty (torch.Tensor): Выход уверенности Delta.
            scale (int): Текущий масштаб.
            offset_scale (float): Масштабный коэффициент для смещений в сетке классификации.

        Returns:
            dict: Словарь с вычисленными лоссами (certainty и classification).
        """
        with torch.no_grad():
            B, C, H, W = delta_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps = cls_res,device = device) for _ in range(2)])
            G = torch.stack((G[1], G[0]), dim = -1).reshape(C,2) * offset_scale
            GT = (G[None,:,None,None,:] + flow_pre_delta[:,None] - x2[:,None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(delta_cls, GT, reduction  = 'none')[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(certainty.squeeze(1), prob)
        if not torch.any(cls_loss):
            cls_loss = (certainty_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"delta_certainty_loss_{scale}": certainty_loss.mean(),
            f"delta_cls_loss_{scale}": cls_loss.mean(),
        }
        if self.tb_writer:
            for k,v in losses.items():
                self.tb_writer.add_scalar("train/" + k, v.item(), romatch.GLOBAL_STEP)
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        """
        Вычисляет regression loss с использованием робастной функции потерь.

        Лосс состоит из двух частей:
        1. Робастный регрессионный лосс на End-Point-Error (EPE) между предсказанным 
           и ground-truth потоком. Форма лосса контролируется параметрами `alpha` и `c`.
        2. Binary Cross-Entropy loss для "уверенности" (certainty).

        Args:
            x2 (torch.Tensor): Ground truth координаты соответствий.
            prob (torch.Tensor): Маска достоверности ground truth соответствий.
            flow (torch.Tensor): Предсказанное поле потока.
            certainty (torch.Tensor): Предсказанная уверенность.
            scale (int): Текущий масштаб.
            eps (float, optional): Малая константа для численной стабильности. Defaults to 1e-8.
            mode (str, optional): Префикс для имени лосса ('delta' или 'gm'). Defaults to "delta".

        Returns:
            dict: Словарь с вычисленными лоссами (certainty, regression и метрика pck_05).
        """
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        losses = {}
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            losses |= {"pck_05": pck_05}

        ce_loss = F.binary_cross_entropy_with_logits(certainty.squeeze(1), prob)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses |= {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        if self.tb_writer:
            for k,v in losses.items():
                self.tb_writer.add_scalar("train/" + k, v.item(), romatch.GLOBAL_STEP)
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def forward(self, corresps, batch):
        """
        Основной метод для вычисления итоговой функции потерь.

        Проходит по всем масштабам (`scales`), для каждого вычисляет ground truth warp 
        (`gt_warp`), а затем соответствующую функцию потерь (классификация или регрессия) 
        для каждого модуля (Global Matching и Delta refinement). Итоговый лосс является 
        взвешенной суммой лоссов по всем масштабам.

        Args:
            corresps (dict): Словарь с предсказаниями модели для разных масштабов.
            batch (dict): Словарь с входными данными (изображения, глубины, камеры и т.д.).

        Returns:
            torch.Tensor: Итоговое скалярное значение функции потерь.
        """
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        scale_weights = self.scale_weights
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_cls, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("gm_cls"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            if self.planar_mode:
                gt_warp, gt_prob = get_gt_warp_planar(
                    batch["T_1to2"],
                    H=h,
                    W=w,
                )
            else:
                gt_warp, gt_prob = get_gt_warp(                
                    batch["im_A_depth"],
                    batch["im_B_depth"],
                    batch["T_1to2"],
                    batch["K1"],
                    batch["K2"],
                    H=h,
                    W=w,
                )
            x2 = gt_warp.float()
            prob = gt_prob
            
            # Добавляем канальное измерение для совместимости, если его нет
            if prob.dim() == 3:
                prob = prob.unsqueeze(1)

            if self.local_largest_scale >= scale:
                prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
                prob = prob * (
                        prev_epe[:, None]
                        < (2 / 512) * (self.local_dist[scale] * scale))
            
            # TODO: Fix prob shape inconsistency
            prob = prob.squeeze(1)

            if scale_gm_cls is not None:
                gm_cls_losses = self.gm_cls_loss(x2, prob, scale_gm_cls, scale_gm_certainty, scale)
                gm_loss = self.ce_weight * gm_cls_losses[f"gm_certainty_loss_{scale}"] + gm_cls_losses[f"gm_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * gm_loss
            
            if delta_cls is not None:
                delta_cls_losses = self.delta_cls_loss(x2, prob, flow_pre_delta, delta_cls, scale_certainty, scale, offset_scale)
                delta_cls_loss = self.ce_weight * delta_cls_losses[f"delta_certainty_loss_{scale}"] + delta_cls_losses[f"delta_cls_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * delta_cls_loss
            else:
                delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
                reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
                tot_loss = tot_loss + scale_weights[scale] * reg_loss
            prev_epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1).detach()
        return tot_loss
