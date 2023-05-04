import numpy as np
from numpy.lib.stride_tricks import as_strided

from .module import Module


class Conv1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride=1, bias=True):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.stride = stride

        self._parameters["weights"] = np.random.normal(
            0, 0.02, (chan_out, chan_in, self.kernel_size)
        )

        if bias:
            self._parameters["bias"] = np.zeros(chan_out)
        else:
            self._parameters["bias"] = None

    def forward(self, input):
        if self.is_training:
            self._input_cache = input
        B, C, iL = input.shape
        oC, iC, kL = self._parameters["weights"].shape

        assert iC == C, "Input channel dimension mismatch"
        oL = (iL - self.kernel_size) // self.stride + 1

        weights = self._parameters["weights"]
        bias = self._parameters["bias"]

        shape = (C, kL, B, oL)
        strides = (iL, 1, C * iL, self.stride)
        strides = input.itemsize * np.array(strides)

        x_stride = np.lib.stride_tricks.as_strided(
            input,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        output = np.einsum("srp, rpbl -> bsl", weights, x_stride)
        if bias is not None:
            output += bias[:, None]

        assert output.shape == (B, oC, oL), "Output shape mismatch"
        return output

    def backward_update_gradient(self, input, delta):
        B, C, iL = input.shape
        oC, iC, kL = self._parameters["weights"].shape
        _, _, oL = delta.shape

        shape = (C, kL, B, oL)
        strides = (iL, 1, C * iL, self.stride)
        strides = input.itemsize * np.array(strides)

        x_stride = np.lib.stride_tricks.as_strided(
            input,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        dw = np.einsum("bsl, rpbl -> srp", delta, x_stride)
        assert dw.shape == self._parameters["weights"].shape, "Gradient shape mismatch"

        self._gradient["weights"] = dw
        if self._parameters["bias"] is not None:
            db = np.sum(delta, axis=(0, -1))
            self._gradient["bias"] = db

    def backward_delta(self, input, delta):
        B, C, iL = input.shape
        oC, iC, kL = self._parameters["weights"].shape
        B, oC, oL = delta.shape

        W = self._parameters["weights"]
        # W在最后一维上翻转180度
        W = np.flip(W, axis=-1)

        shape = (oC, kL, B, iL)
        strides = (oL, 1, oC * oL, self.stride)
        strides = delta.itemsize * np.array(strides)

        delta_stride = np.lib.stride_tricks.as_strided(
            delta,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        # s: oC, r: iC, p: kL
        # s:oC p:kL, b:B, l:iL
        # dx: b:B, r:iC, l:iL
        dx = np.einsum("srp, spbl -> brl", W, delta_stride)
        assert dx.shape == input.shape, "Delta shape mismatch"

        return dx


class Conv2d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride=1, bias=True):
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        if isinstance(kernel_size, int):
            self.kw = self.kh = kernel_size
        else:
            self.kh, self.kw = kernel_size
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        shape = (chan_out, chan_in, self.kh, self.kw)
        fan_in = np.prod(shape[1:])
        std = np.sqrt(2.0 / fan_in)
        self._parameters["weights"] = np.random.normal(0, std, shape)

        if bias:
            self._parameters["bias"] = np.ones(chan_out) * 0.01
        else:
            self._parameters["bias"] = None

    def forward(self, input):
        if self.is_training:
            self._input_cache = input
        B, C, iH, iW = input.shape
        oC, iC, kH, kW = self._parameters["weights"].shape

        assert iC == C, "Input channel dimension mismatch"
        oH = (iH - self.kh) // self.stride[0] + 1
        oW = (iW - self.kw) // self.stride[1] + 1

        weights = self._parameters["weights"]
        bias = self._parameters["bias"]

        # Create view of the input with the kernel shape using as_strided
        shape = (C, kH, kW, B, oH, oW)
        strides = (iH * iW, iW, 1, C * iH * iW, iW * self.stride[0], self.stride[1])
        strides = input.itemsize * np.array(strides)

        x_stride = np.lib.stride_tricks.as_strided(
            input,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        # weights: (s:oC, r:iC, p:kH, q:kW)
        # x_stride: (r:iC, p:kH, q:kW, b:B, h:oH, w:oW)
        output = np.einsum("srpq, rpqbhw -> bshw", weights, x_stride)
        if bias is not None:
            output += bias[:, None, None]

        assert output.shape == (B, oC, oH, oW), "Output shape mismatch"
        return output

    def backward_update_gradient(self, input, delta):
        r"""
        :math: \delta_w = input \star delta
        :math: \delta_b = \sum_{b, h, w} delta_{b, c, h, w}
        """
        B, C, iH, iW = input.shape
        oC, iC, kH, kW = self._parameters["weights"].shape
        _, _, oH, oW = delta.shape

        shape = (C, kH, kW, B, oH, oW)
        strides = (iH * iW, iW, 1, C * iH * iW, iW * self.stride[0], self.stride[1])
        strides = input.itemsize * np.array(strides)

        x_stride = np.lib.stride_tricks.as_strided(
            input,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        dw = np.einsum("bshw, rpqbhw -> srpq", delta, x_stride)
        assert dw.shape == self._parameters["weights"].shape, "Gradient shape mismatch"

        self._gradient["weights"] = dw
        if self._parameters["bias"] is not None:
            db = np.sum(delta, axis=(0, -1, -2))
            self._gradient["bias"] = db

    def backward_delta(self, input, delta):
        r"""
        :math: \delta_{l-1} = \delta \star rot180(W)
        """
        B, C, iH, iW = input.shape
        oC, iC, kH, kW = self._parameters["weights"].shape
        B, oC, oH, oW = delta.shape

        W = self._parameters["weights"]
        W = np.flip(W, axis=(-1, -2))

        shape = (oC, kH, kW, B, iH, iW)
        strides = (oH * oW, oW, 1, oC * oH * oW, oW * self.stride[0], self.stride[1])
        strides = delta.itemsize * np.array(strides)

        delta_stride = np.lib.stride_tricks.as_strided(
            delta,
            shape=shape,
            strides=strides,
            writeable=False,
        )

        # s: oC, r: iC, p: kH, q: kW
        # s:oC p:kH q:kW, b:B, h:iH, w:iW
        # dx: b:B, r:iC, h:iH, w:iW
        dx = np.einsum("srpq, spqbhw -> brhw", W, delta_stride)
        assert dx.shape == input.shape, "Delta shape mismatch"

        return dx
