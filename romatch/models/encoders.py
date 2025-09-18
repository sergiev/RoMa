from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
from romatch.utils.utils import get_autocast_params


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16, in_chans=3) -> None:
        super().__init__()
        vgg_model = tvm.vgg19_bn(pretrained=pretrained)
        
        if in_chans != 3:
            # Модифицируем первый сверточный слой для приема 1 канала
            first_conv = vgg_model.features[0]
            new_first_conv = nn.Conv2d(in_chans, first_conv.out_channels, 
                                       kernel_size=first_conv.kernel_size,
                                       stride=first_conv.stride, padding=first_conv.padding,
                                       bias=(first_conv.bias is not None))
            
            if pretrained:
                # Усредняем веса по входным каналам, чтобы сохранить общую "энергию"
                new_first_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
                if first_conv.bias is not None:
                    new_first_conv.bias.data = first_conv.bias.data
            
            vgg_model.features[0] = new_first_conv
        
        self.layers = nn.ModuleList(vgg_model.features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs=None, amp=False, use_vgg=False, dinov2_weights=None, amp_dtype=torch.float16, in_chans=1):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")
        from .transformer.dinov2 import vit_large
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
            in_chans=3, # <--- Передаем in_chans в DINO
        )

        dinov2_vitl14 = vit_large(**vit_kwargs).eval()
        dinov2_vitl14.load_state_dict(state_dict=dinov2_weights, strict=False) # strict=False из-за измененного слоя
        
        if in_chans != 3:
            # Модифицируем patch_embed в DINO
            patch_embed = dinov2_vitl14.patch_embed
            new_patch_embed = nn.Conv2d(in_chans, patch_embed.proj.out_channels,
                                        kernel_size=patch_embed.proj.kernel_size,
                                        stride=patch_embed.proj.stride,
                                        padding=patch_embed.proj.padding)
            if dinov2_weights: # Если есть веса
                # Усредняем веса
                new_patch_embed.weight.data = patch_embed.proj.weight.data.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
                if patch_embed.proj.bias is not None:
                    new_patch_embed.bias.data = patch_embed.proj.bias.data
            
            dinov2_vitl14.patch_embed.proj = new_patch_embed
        
        
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        cnn_kwargs['in_chans'] = in_chans # <--- Передаем in_chans в CNN
        
        if not use_vgg:
            self.cnn = ResNet50(**cnn_kwargs)
        else:
            self.cnn = VGG19(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        B,C,H,W = x.shape
        feature_pyramid = self.cnn(x)
        
        if not upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,1024,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid