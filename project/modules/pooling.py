import numpy as np
from numpy.lib.stride_tricks import as_strided

from .module import Module


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

    def forward(self, x):
        B, C, L = x.shape
        oL = (L - self.kernel_size) // self.stride + 1

        strided_x = as_strided(
            x,
            shape=(B, C, oL, self.kernel_size),
            strides=(
                x.strides[0],
                x.strides[1],
                x.strides[2] * self.stride,
                x.strides[2],
            ),
        )

        output = np.max(strided_x, axis=-1)

        if self.is_training:
            maxs = output.repeat(self.stride, axis=2)
            x_window = x[:, :, : oL * self.stride]
            mask = np.equal(x_window, maxs)
            self._input_cache = x
            self._cache["mask"] = mask

        return output

    def backward_delta(self, input, delta):
        mask = self._cache["mask"]
        dA = delta.repeat(self.stride, axis=2)
        dA = np.multiply(dA, mask)
        pad = np.zeros(input.shape)
        pad[:, :, : dA.shape[2]] = dA
        return pad

    def backward_update_gradient(self, input, delta):
        pass


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kh = self.kw = kernel_size
        else:
            self.kh, self.kw = kernel_size

        if stride is None:
            self.stride = (self.kh, self.kw)
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        oH = (H - self.kh) // self.stride[0] + 1
        oW = (W - self.kw) // self.stride[1] + 1

        strided_x = as_strided(
            x,
            shape=(B, C, oH, oW, self.kh, self.kw),
            strides=(
                x.strides[0],
                x.strides[1],
                x.strides[2] * self.stride[0],
                x.strides[3] * self.stride[1],
                x.strides[2],
                x.strides[3],
            ),
        )

        output = np.max(strided_x, axis=(-2, -1))

        if self.is_training:
            maxs = output.repeat(2, axis=2).repeat(2, axis=3)
            x_window = x[:, :, : oH * self.stride[0], : oW * self.stride[1]]
            mask = np.equal(x_window, maxs)
            self._input_cache = x
            self._cache["mask"] = mask

        return output

    def backward_delta(self, input, delta):
        mask = self._cache["mask"]
        dA = delta.repeat(self.stride[0], axis=2).repeat(self.stride[1], axis=3)
        dA = np.multiply(dA, mask)
        pad = np.zeros(input.shape)
        pad[:, :, : dA.shape[2], : dA.shape[3]] = dA
        return pad

    def backward_update_gradient(self, input, delta):
        pass
