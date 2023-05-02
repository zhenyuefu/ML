import numpy as np
from numpy.lib.stride_tricks import as_strided

from .module import Module


class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, x):
        self._input_cache = x
        batch_size, length, chan_in = x.shape
        output_length = (length - self.k_size) // self.stride + 1
        output = np.zeros((batch_size, output_length, chan_in))

        strided_x = as_strided(
            x,
            shape=(batch_size, output_length, self.k_size, chan_in),
            strides=(
                x.strides[0],
                x.strides[1] * self.stride,
                x.strides[1],
                x.strides[2],
            ),
        )
        output = np.max(strided_x, axis=2)

        return output

    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        output_length = delta.shape[1]

        delta_input = np.zeros_like(input)
        strided_input = as_strided(
            input,
            shape=(batch_size, output_length, self.k_size, chan_in),
            strides=(
                input.strides[0],
                input.strides[1] * self.stride,
                input.strides[1],
                input.strides[2],
            ),
        )
        strided_delta_input = as_strided(
            delta_input,
            shape=(batch_size, output_length, self.k_size, chan_in),
            strides=(
                delta_input.strides[0],
                delta_input.strides[1] * self.stride,
                delta_input.strides[1],
                delta_input.strides[2],
            ),
        )

        mask = np.equal(strided_input, np.max(strided_input, axis=2, keepdims=True))
        np.add.at(
            strided_delta_input,
            (slice(None), slice(None), slice(None), slice(None)),
            np.multiply(delta[:, :, np.newaxis, :], mask),
        )

        return delta_input

    def backward_update_gradient(self, input, delta):
        pass
