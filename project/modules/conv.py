import numpy as np
from numpy.lib.stride_tricks import as_strided

from .module import Module


class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters["weights"] = (
            2 * np.random.randn(k_size, chan_in, chan_out) - 1
        ) * 0.01

        if bias:
            self._parameters["bias"] = np.zeros(chan_out)
        else:
            self._parameters["bias"] = None

    def forward(self, x):
        self._input_cache = x
        batch_size, length, _ = x.shape
        output_length = (length - self.k_size) // self.stride + 1
        output = np.zeros((batch_size, output_length, self.chan_out))

        strided_x = as_strided(
            x,
            shape=(batch_size, output_length, self.k_size, self.chan_in),
            strides=(
                x.strides[0],
                x.strides[1] * self.stride,
                x.strides[1],
                x.strides[2],
            ),
        )

        output = np.tensordot(
            strided_x, self._parameters["weights"], axes=([2, 3], [0, 1])
        )
        if self._parameters["bias"] is not None:
            output += self._parameters["bias"]

        return output

    def backward_update_gradient(self, input, delta):
        batch_size, length, _ = input.shape
        output_length = delta.shape[1]

        strided_input = as_strided(
            input,
            shape=(batch_size, output_length, self.k_size, self.chan_in),
            strides=(
                input.strides[0],
                input.strides[1] * self.stride,
                input.strides[1],
                input.strides[2],
            ),
        )

        self._gradient["weights"] = np.tensordot(
            strided_input, delta, axes=([0, 1], [0, 1])
        )

        if self._parameters["bias"] is not None:
            self._gradient["bias"] = np.sum(delta, axis=(0, 1))

    def backward_delta(self, input, delta):
        batch_size, length, _ = input.shape
        output_length = delta.shape[1]

        delta_input = np.zeros_like(input)
        strided_delta_input = as_strided(
            delta_input,
            shape=(batch_size, output_length, self.k_size, self.chan_in),
            strides=(
                delta_input.strides[0],
                delta_input.strides[1] * self.stride,
                delta_input.strides[1],
                delta_input.strides[2],
            ),
        )

        np.add.at(
            strided_delta_input,
            (slice(None), slice(None), slice(None), slice(None)),
            np.tensordot(delta, self._parameters["weights"], axes=([2], [2])),
        )

        return delta_input


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._input_cache = x
        batch_size, length, chan_in = x.shape
        return x.reshape(batch_size, length * chan_in)

    def backward_update_gradient(self, input, delta):
        pass  # No parameters to update

    def backward_delta(self, input, delta):
        batch_size, length, chan_in = input.shape
        return delta.reshape(batch_size, length, chan_in)
