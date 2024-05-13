import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BaseModel(nn.Module):
    def __init__(self, backbone, normalize=True,
                 mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261],
                 meta_data=None,
                 input_channels=3):
        super(BaseModel, self).__init__()
        mu = torch.tensor(mean)
        std = torch.tensor(std)

        if input_channels == 3:
            mu = mu.view(3, 1, 1)
            std = std.view(3, 1, 1)
        else:
            mu = mu.view(1, 1, 1)
            std = std.view(1, 1, 1)
        self.meta_data = meta_data
        if self.meta_data is None:
            self.meta_data = {}

        self.backbone = backbone

        if device:
            mu = mu.to(device)
            std = std.to(device)
        self.do_norm = normalize
        self.norm = lambda x: (x - mu) / std

    def forward(self, x):
        if self.do_norm:
            x = self.norm(x)
        out = self.backbone(x)
        return out