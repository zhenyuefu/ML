from collections import OrderedDict


class Module(object):
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        self._parameters = OrderedDict()
        self._gradient = OrderedDict()
        self._modules = OrderedDict()
        self._input_cache = None
        self._cache = OrderedDict()
        self.is_training = True

    def forward(self, data):
        raise NotImplementedError

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError

    def update_parameters(self, lr=1e-3):
        for param_name, gradient in self._gradient.items():
            param_value = self._parameters[param_name]
            self._parameters[param_name] = param_value - lr * gradient

    def zero_grad(self):
        self._gradient.clear()

    def parameters(self):
        return self._parameters

    def add_module(self, name, module):
        r"""Adds a child module to the current module.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{module} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError('module name can\'t contain "."')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')
        self._modules[name] = module

    def eval(self):
        self.is_training = False
        for module in self._modules.values():
            module.eval()
