"""Lead-count-agnostic 1D SE-ResNet for multi-label ECG classification.

Same architecture is used for every single-lead model (in_channels=1) and for
the 12-lead ceiling model (in_channels=12). Only the stem's in_channels differs,
so cross-lead differences are attributable to lead content, not architecture.
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x):                       # x: (B, C, T)
        s = x.mean(dim=2)                       # (B, C)
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(2)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1, k=7):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.se = SEBlock(cout)
        self.drop = nn.Dropout(0.1)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout),
            )

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return torch.relu(x + idt)


class SEResNet1D(nn.Module):
    def __init__(self, in_channels=1, n_classes=11, width=(32, 64, 128, 256), blocks=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width[0], 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        layers = []
        cin = width[0]
        for i, (w, nb) in enumerate(zip(width, blocks)):
            for b in range(nb):
                stride = 2 if (b == 0 and i > 0) else 1
                layers.append(ResBlock1D(cin, w, stride=stride))
                cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(cin, n_classes),
        )

    def forward(self, x):                       # x: (B, in_channels, T)
        return self.head(self.body(self.stem(x)))   # logits (B, n_classes)


# ---- additional architectures (resubmission: architecture-invariance check) ----
# Both follow the strongest families of the PTB-XL benchmark (Strodthoff et al
# 2021), which evaluated them at 100 Hz; the input is average-pooled 500 -> 100 Hz
# inside the model so every architecture still receives the identical signal.

class InceptionModule1D(nn.Module):
    def __init__(self, cin, nf=32, ks=40, bottleneck=32):
        super().__init__()
        kss = [ks // (2 ** i) for i in range(3)]
        kss = [k if k % 2 else k - 1 for k in kss]             # odd -> 'same' padding
        self.bottleneck = nn.Conv1d(cin, bottleneck, 1, bias=False) if cin > 1 else nn.Identity()
        cb = bottleneck if cin > 1 else cin
        self.convs = nn.ModuleList([nn.Conv1d(cb, nf, k, padding=k // 2, bias=False) for k in kss])
        self.pool_conv = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                       nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)

    def forward(self, x):
        b = self.bottleneck(x)
        x = torch.cat([c(b) for c in self.convs] + [self.pool_conv(x)], dim=1)
        return torch.relu(self.bn(x))


class InceptionTime1D(nn.Module):
    """InceptionTime (Ismail Fawaz et al 2020): depth 6, residual every 3 modules."""

    def __init__(self, in_channels=1, n_classes=11, nf=32, depth=6, ks=40, pool=5):
        super().__init__()
        self.down = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        self.mods, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        cin, res_in = in_channels, in_channels
        for d in range(depth):
            self.mods.append(InceptionModule1D(cin, nf, ks))
            cin = nf * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_in, cin, 1, bias=False),
                                                    nn.BatchNorm1d(cin)))
                res_in = cin
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(cin, n_classes))

    def forward(self, x):
        x = self.down(x)
        res = x
        for d, m in enumerate(self.mods):
            x = m(x)
            if d % 3 == 2:
                x = torch.relu(x + self.shortcuts[d // 3](res))
                res = x
        return self.head(x)


class XBottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, cin, mid, stride=1, k=5):
        super().__init__()
        cout = mid * self.expansion
        self.body = nn.Sequential(
            nn.Conv1d(cin, mid, 1, bias=False), nn.BatchNorm1d(mid), nn.ReLU(inplace=True),
            nn.Conv1d(mid, mid, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(mid), nn.ReLU(inplace=True),
            nn.Conv1d(mid, cout, 1, bias=False), nn.BatchNorm1d(cout))
        nn.init.zeros_(self.body[-1].weight)                    # zero-init last BN (xresnet)
        idt = []
        if stride != 1:
            idt.append(nn.AvgPool1d(stride, ceil_mode=True))    # xresnet "ResNet-D" shortcut
        if cin != cout:
            idt += [nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout)]
        self.idt = nn.Sequential(*idt)

    def forward(self, x):
        return torch.relu(self.body(x) + self.idt(x))


class XResNet1D(nn.Module):
    """xresnet1d (He et al 2019 bag of tricks, 1D as in Strodthoff et al 2021)."""

    def __init__(self, in_channels=1, n_classes=11, layers=(3, 4, 23, 3), k=5, pool=5):
        super().__init__()
        self.down = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        stem, c = [], in_channels
        for i, co in enumerate((32, 32, 64)):
            stem += [nn.Conv1d(c, co, k, stride=2 if i == 0 else 1, padding=k // 2, bias=False),
                     nn.BatchNorm1d(co), nn.ReLU(inplace=True)]
            c = co
        self.stem = nn.Sequential(*stem, nn.MaxPool1d(3, stride=2, padding=1))
        blocks, cin = [], 64
        for i, (mid, n) in enumerate(zip((64, 128, 256, 512), layers)):
            for b in range(n):
                blocks.append(XBottleneck1D(cin, mid, stride=2 if (b == 0 and i > 0) else 1, k=k))
                cin = mid * XBottleneck1D.expansion
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(cin, n_classes))

    def forward(self, x):
        return self.head(self.body(self.stem(self.down(x))))


ARCHS = {
    "seresnet": SEResNet1D,
    "inceptiontime": InceptionTime1D,
    "xresnet1d50": lambda **kw: XResNet1D(layers=(3, 4, 6, 3), **kw),
    "xresnet1d101": lambda **kw: XResNet1D(layers=(3, 4, 23, 3), **kw),
}


def build_model(arch, in_channels, n_classes):
    return ARCHS[arch](in_channels=in_channels, n_classes=n_classes)
