import torch
import torch.nn as nn
from models.backbone.swin import SwinTransformer
from mmcv.ops import SyncBatchNorm
from models.decode_heads.uper_head import UPerHead
from models.backbone.resnet import resnet50
from models.modules.LAM import AFModule

class SwinRes_UperNet(nn.Module):
    def __init__(self, num_classes=21, sync_bn=True, freeze_bn=False):
        super(SwinRes_UperNet, self).__init__()

        if sync_bn == True:
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        else:
            norm_cfg = dict(type='BN', requires_grad=True)
        self.in_channels = [128, 256, 512, 1024]
        self.backbone1 = SwinTransformer(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            use_abs_pos_embed=False,
            drop_path_rate=0.3,
            patch_norm=True,
            pretrained='/root/share/pretrain/swin_base_patch4_window7_224_22k.pth'
        )

        self.backbone2 = resnet50(pretrained=True)

        self.AFMs = nn.ModuleList()
        self.lc_convs = nn.ModuleList()
        for in_channel in self.in_channels:
            AFM = AFModule(inplace=in_channel)
            lc_conv = nn.Conv2d(in_channel * 2, in_channel, 3, padding=1)
            self.AFMs.append(AFM)
            self.lc_convs.append(lc_conv)

        self.decoder = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg
        )

        self.freeze_bn = freeze_bn
        

    def forward(self, img):
        x1 = self.backbone1(img)
        x2 = self.backbone2(img)
        c2 = [
            lc_conv(x2[i])
            for i, lc_conv in enumerate(self.lc_convs)
        ]
        e = [
            AFM(x1[i], c2[i])
            for i, AFM in enumerate(self.AFMs)
        ]
        out = self.decoder.forward(e)
        return out


    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, SyncBatchNorm):
    #             m.eval()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.eval()

if __name__ == '__main__':
    model = SwinResNet(num_classes=2, sync_bn=False).cuda()
    a = torch.rand(2, 3, 512, 512).cuda()
    b = model(a)
    print(b.shape)

