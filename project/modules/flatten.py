from .module import Module


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.is_training:
            self._input_cache = x
        return x.reshape(x.shape[0], -1)

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)

    def backward_update_gradient(self, input, delta):
        pass
