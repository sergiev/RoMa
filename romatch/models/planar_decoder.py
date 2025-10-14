"""
Оптимизированный декодер для planar scenes (SAR ортофото).

Основные отличия от стандартного RoMa decoder:
1. Упрощенный global matcher (direct regression вместо GP + Transformer)
2. Увеличенные радиусы local correlation для компенсации ошибок
3. Прямая regression flow вместо classification
4. Оптимизирован для homography трансформаций
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from romatch.utils.utils import get_autocast_params


class SimplifiedGlobalMatcher(nn.Module):
    """
    Упрощенный global matcher для planar scenes.
    Вместо GP + Transformer использует correlation + MLP для прямой regression.
    """

    def __init__(
        self,
        feat_dim=512,
        hidden_dim=512,
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.amp_dtype = amp_dtype

        # Feature projection
        self.proj_q = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.proj_k = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Correlation refinement
        self.corr_refine = nn.Sequential(
            nn.Conv2d(256 + 256 + 2, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1),  # flow (2) + certainty (1)
        )

    def compute_global_correlation(self, f1, f2):
        """Вычисляет global correlation map используя dot product."""
        B, C, H, W = f1.shape

        # Normalize features
        f1_norm = F.normalize(f1, dim=1)
        f2_norm = F.normalize(f2, dim=1)

        # Reshape for correlation
        f1_flat = f1_norm.view(B, C, H * W)  # B, C, HW
        f2_flat = f2_norm.view(B, C, H * W)  # B, C, HW

        # Compute correlation: для каждого пикселя в f1 найти best match в f2
        corr = torch.bmm(f1_flat.transpose(1, 2), f2_flat)  # B, HW, HW

        # Soft argmax для получения coordinates
        corr_soft = F.softmax(corr, dim=2)  # B, HW, HW

        # Grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=f1.device), torch.linspace(-1, 1, W, device=f1.device), indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=0).view(2, H * W)  # 2, HW

        # Weighted sum для получения predicted coordinates
        pred_coords = torch.bmm(corr_soft, grid.t().unsqueeze(0).expand(B, -1, -1))  # B, HW, 2
        pred_coords = pred_coords.view(B, H, W, 2).permute(0, 3, 1, 2)  # B, 2, H, W

        # Confidence из max correlation
        confidence = corr.max(dim=2)[0].view(B, 1, H, W)  # B, 1, H, W

        return pred_coords, confidence

    def forward(self, f1, f2):
        """
        Args:
            f1: features from image A (B, C, H, W)
            f2: features from image B (B, C, H, W)
        Returns:
            flow: predicted flow field (B, 2, H, W)
            certainty: certainty map (B, 1, H, W)
        """
        B, C, H, W = f1.shape

        # Project features
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(
            f1.device, enabled=True, dtype=self.amp_dtype
        )
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            q = self.proj_q(f1)  # B, 256, H, W
            k = self.proj_k(f2)  # B, 256, H, W

            # Global correlation
            coarse_flow, coarse_conf = self.compute_global_correlation(q, k)

            # Refine with local features
            combined = torch.cat([q, k, coarse_flow], dim=1)  # B, 256+256+2, H, W
        refined = self.corr_refine(combined.float())  # B, 3, H, W

        flow = refined[:, :2]  # B, 2, H, W
        certainty = refined[:, 2:3]  # B, 1, H, W

        return flow, certainty


class PlanarDecoder(nn.Module):
    """
    Декодер, оптимизированный для planar scenes (SAR ортофото).

    Основные улучшения:
    - Упрощенный global matching на масштабе 16
    - Увеличенные радиусы local correlation (7→11, 3→7, 2→4)
    - Прямая regression вместо classification
    - Меньше параметров, быстрее обучение
    """

    def __init__(
        self,
        global_matcher,
        proj,
        conv_refiner,
        scales=["16", "8", "4", "2", "1"],
        detach=True,
        flow_upsample_mode="bilinear",
        amp_dtype=torch.float16,
    ):
        super().__init__()
        self.global_matcher = global_matcher
        self.proj = proj
        self.conv_refiner = conv_refiner
        self.scales = scales
        self.detach = detach
        self.flow_upsample_mode = flow_upsample_mode
        self.amp_dtype = amp_dtype
        self.refine_init = 4

    def get_placeholder_flow(self, b, h, w, device):
        """Создает identity flow (каждый пиксель сопоставляется сам с собой)."""
        coarse_coords = torch.meshgrid(
            torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device),
            torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device),
            indexing="ij",
        )
        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[None].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        return coarse_coords

    def forward(self, f1, f2, gt_warp=None, gt_prob=None, upsample=False, flow=None, certainty=None, scale_factor=1):
        """
        Args:
            f1: feature pyramid для image A (dict: scale -> tensor)
            f2: feature pyramid для image B (dict: scale -> tensor)
            upsample: если True, начинает с масштаба 8 вместо 16
            flow, certainty: начальные значения для upsample режима
        Returns:
            corresps: dict с предсказаниями на каждом масштабе
        """
        all_scales = self.scales if not upsample else ["8", "4", "2", "1"]
        sizes = {scale: f1[scale].shape[-2:] for scale in f1}
        h, w = sizes[1]
        b = f1[1].shape[0]
        device = f1[1].device
        coarsest_scale = int(all_scales[0])

        corresps = {}

        # Инициализация flow
        if not upsample:
            flow = self.get_placeholder_flow(b, *sizes[coarsest_scale], device)
            certainty = torch.zeros(b, 1, *sizes[coarsest_scale], device=device)
        else:
            flow = F.interpolate(flow, size=sizes[coarsest_scale], mode="bilinear", align_corners=False)
            certainty = F.interpolate(certainty, size=sizes[coarsest_scale], mode="bilinear", align_corners=False)

        # Итерация по масштабам
        for new_scale in all_scales:
            ins = int(new_scale)
            corresps[ins] = {}
            f1_s, f2_s = f1[ins], f2[ins]

            # Feature projection
            if new_scale in self.proj:
                autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(
                    f1_s.device, str(f1_s.device) == "cuda", self.amp_dtype
                )
                with torch.autocast(autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
                    if not autocast_enabled:
                        f1_s, f2_s = f1_s.to(torch.float32), f2_s.to(torch.float32)
                    f1_s, f2_s = self.proj[new_scale](f1_s), self.proj[new_scale](f2_s)

            # Global matching на масштабе 16
            if ins == 16 and not upsample:
                flow, certainty = self.global_matcher(f1_s, f2_s)
                if self.training:
                    corresps[ins].update({"gm_flow": flow, "gm_certainty": certainty})

            # Local refinement
            if new_scale in self.conv_refiner:
                if self.training:
                    corresps[ins].update({"flow_pre_delta": flow})

                delta_flow, delta_certainty = self.conv_refiner[new_scale](
                    f1_s, f2_s, flow, scale_factor=scale_factor, logits=certainty
                )

                if self.training:
                    corresps[ins].update({"delta_flow": delta_flow})

                # Normalize displacement
                displacement = ins * torch.stack(
                    (
                        delta_flow[:, 0].float() / (self.refine_init * w),
                        delta_flow[:, 1].float() / (self.refine_init * h),
                    ),
                    dim=1,
                )

                flow = flow + displacement
                certainty = certainty + delta_certainty

            # Сохранить результаты
            corresps[ins].update({"certainty": certainty, "flow": flow})

            # Upsample для следующего масштаба
            if new_scale != "1":
                flow = F.interpolate(flow, size=sizes[ins // 2], mode=self.flow_upsample_mode)
                certainty = F.interpolate(certainty, size=sizes[ins // 2], mode=self.flow_upsample_mode)

                if self.detach:
                    flow = flow.detach()
                    certainty = certainty.detach()

        return corresps
