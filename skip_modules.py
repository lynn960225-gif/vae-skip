import types
import torch
import torch.nn as nn
import torch.nn.functional as F

class LastLevelSkipFusion(nn.Module):
    def __init__(self, decoder_channels=128, cond_channels=3, hidden=64):
        super().__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, decoder_channels, 3, padding=1),
        )
        self.fuse_conv = nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1)
        nn.init.zeros_(self.fuse_conv.weight)
        nn.init.zeros_(self.fuse_conv.bias)

    def forward(self, dec_feat, cond_img):
        cond_resized = F.interpolate(cond_img, size=dec_feat.shape[-2:], mode="bilinear", align_corners=False)
        cond_feat = self.cond_conv(cond_resized)
        return dec_feat + self.fuse_conv(torch.cat([dec_feat, cond_feat], dim=1))


def patch_decoder_with_skip(decoder, skip_fusion):
    def patched_forward(self, sample, latent_embeds=None, cond_img=None):
        sample = self.conv_in(sample)

        # 完全保留原本的gradient checkpointing分支，不破坏显存优化
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            sample = self._gradient_checkpointing_func(self.mid_block, sample, latent_embeds)
            for up_block in self.up_blocks:
                sample = self._gradient_checkpointing_func(up_block, sample, latent_embeds)
        else:
            sample = self.mid_block(sample, latent_embeds)
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # ---- 插入点：up_blocks循环结束、conv_norm_out之前 ----
        if cond_img is not None:
            sample = skip_fusion(sample, cond_img)

        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample

    decoder.forward = types.MethodType(patched_forward, decoder)
    return decoder