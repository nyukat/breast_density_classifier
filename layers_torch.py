import torch
import torch.nn as nn
import torch.nn.functional as F


class AllViewsGaussianNoise(nn.Module):
    """Add gaussian noise across all 4 views"""

    def __init__(self, gaussian_noise_std, device):
        super(AllViewsGaussianNoise, self).__init__()
        self.gaussian_noise_std = gaussian_noise_std
        self.device = device

    def forward(self, x):
        if not self.gaussian_noise_std:
            return x

        return {
            "L-CC": self._add_gaussian_noise(x["L-CC"]),
            "L-MLO": self._add_gaussian_noise(x["L-MLO"]),
            "R-CC": self._add_gaussian_noise(x["R-CC"]),
            "R-MLO": self._add_gaussian_noise(x["R-MLO"]),
        }

    def _add_gaussian_noise(self, single_view):
        return single_view + torch.Tensor(*single_view.shape).normal_(std=self.gaussian_noise_std).to(self.device)


class AllViewsConvLayer(nn.Module):
    """Convolutional layers across all 4 views"""

    def __init__(self, in_channels, number_of_filters=32, filter_size=(3, 3), stride=(1, 1)):
        super(AllViewsConvLayer, self).__init__()
        self.cc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )
        self.mlo = nn.Conv2d(
            in_channels=in_channels,
            out_channels=number_of_filters,
            kernel_size=filter_size,
            stride=stride,
        )

    def forward(self, x):
        return {
            "L-CC": F.relu(self.cc(x["L-CC"])),
            "L-MLO": F.relu(self.mlo(x["L-MLO"])),
            "R-CC": F.relu(self.cc(x["R-CC"])),
            "R-MLO": F.relu(self.mlo(x["R-MLO"])),
        }

    @property
    def ops(self):
        return {
            "CC": self.cc,
            "MLO": self.mlo,
        }


class AllViewsMaxPool(nn.Module):
    """Max-pool across all 4 views"""

    def __init__(self):
        super(AllViewsMaxPool, self).__init__()

    def forward(self, x, stride=(2, 2), padding=(0, 0)):
        return {
            "L-CC": F.max_pool2d(x["L-CC"], kernel_size=stride, stride=stride, padding=padding),
            "L-MLO": F.max_pool2d(x["L-MLO"], kernel_size=stride, stride=stride, padding=padding),
            "R-CC": F.max_pool2d(x["R-CC"], kernel_size=stride, stride=stride, padding=padding),
            "R-MLO": F.max_pool2d(x["R-MLO"], kernel_size=stride, stride=stride, padding=padding),
        }


class AllViewsAvgPool(nn.Module):
    """Average-pool across all 4 views"""

    def __init__(self):
        super(AllViewsAvgPool, self).__init__()

    def forward(self, x):
        return {
            "L-CC": self._avg_pool(x["L-CC"]),
            "L-MLO": self._avg_pool(x["L-MLO"]),
            "R-CC": self._avg_pool(x["R-CC"]),
            "R-MLO": self._avg_pool(x["R-MLO"]),
        }

    @staticmethod
    def _avg_pool(single_view):
        n, c, h, w = single_view.size()
        return single_view.view(n, c, -1).mean(-1)


class AllViewsPad(nn.Module):
    """Pad tensor across all 4 views"""

    def __init__(self):
        super(AllViewsPad, self).__init__()

    def forward(self, x, pad):
        return {
            "L-CC": F.pad(x["L-CC"], pad),
            "L-MLO": F.pad(x["L-MLO"], pad),
            "R-CC": F.pad(x["R-CC"], pad),
            "R-MLO": F.pad(x["R-MLO"], pad),
        }

