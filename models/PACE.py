"""
@File        :PACE.py
@Author      :Xinyu Lu
@EMail       :xinyulu@stu.xmu.edu.cn
"""

import torch.nn.functional as F
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()

        self.Resnet = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size,
                      groups=dim, padding='same'),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

        self.Conv_1x1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
        )
        self.identity = nn.Identity()

    def forward(self, x):
        id = self.identity(x)
        x = self.Resnet(x)
        x = self.Conv_1x1(x)
        x = x + id
        x = F.gelu(x)

        return x


class Pace(nn.Module):
    def __init__(self, dim=1024, depth=4, kernel_size=9, patch_size=16, pool_dim=256, n_classes=3, overlapped=False):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.pool_dim = pool_dim

        stride = patch_size // 2 if overlapped else patch_size
        self.spec_emb = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=patch_size, stride=stride),
            nn.BatchNorm1d(dim),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(p=0.2)
        self.backbone = nn.ModuleList([])

        for _ in range(depth):
            self.backbone.append(
                BasicBlock(dim=dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            # nn.Linear(16*1024, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Linear(dim*(pool_dim//patch_size), n_classes)
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = F.adaptive_avg_pool1d(x, self.pool_dim)
        x = self.spec_emb(x) # [B, dim, pool_dim // patch_size]
        x = self.dropout(x)
        for block in self.backbone:
            x = block(x)
        x = self.head(x) # [B, 1024, 16]

        return x

    def get_args(self):
        print(
            f'dim: {self.dim}, depth: {self.depth}, kernel size: {self.kernel_size}, patch size: {self.patch_size}, pool dim: {self.pool_dim}')


def PACE(n_classes=30, **kwargs):
    return Pace(n_classes=n_classes, **kwargs)
    # return ConvMixer(dim=1024, depth=4, kernel_size=9, patch_size=16, pool_dim=256, n_classes=n_classes)


if __name__ == "__main__":
    import time
    net = PACE(n_classes=30, dim=1024, depth=4, kernel_size=9, patch_size=16, pool_dim=256, overlapped=False)
    inp = torch.randn((1000, 1, 1024))
    start = time.time()
    print(net(inp).shape)
    end = time.time()
    print(end-start)

    from thop import profile
    flops,params = profile(net, inputs=(inp,))
    print(flops/1e6, params/1e6)
    print(net)
