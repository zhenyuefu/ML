from collections import OrderedDict
from .module import Module


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x):
        for module in self._modules.values():
            x = module.forward(x)
        return x

    def backward_update_gradient(self, input, delta):
        for module in reversed(self._modules.values()):
            module_input = module._input_cache
            module.backward_update_gradient(module_input, delta)
            delta = module.backward_delta(module_input, delta)

    def backward_delta(self, input, delta):
        for module in reversed(self._modules.values()):
            module_input = module._input_cache
            delta = module.backward_delta(module_input, delta)
        return delta

    def update_parameters(self, learning_rate=1e-3):
        for module in self._modules.values():
            module.update_parameters(learning_rate)

    def zero_grad(self):
        for module in self._modules.values():
            module.zero_grad()
