from collections import OrderedDict
import numpy as np


class Module(object):
    def __init__(self):
        self._parameters = OrderedDict()
        self._gradient = OrderedDict()

    def forward(self, data):
        raise NotImplementedError

    def backward_update_gradient(self, input, delta):
        raise NotImplementedError

    def backward_delta(self, input, delta):
        raise NotImplementedError

    def update_parameters(self, gradient_step=1e-3):
        for param, grad in zip(self._parameters, self._gradient):
            param -= gradient_step * grad

    def zero_grad(self):
        self._gradient.clear()

    def parameters(self):
        return self._parameters

    def register_parameter(self, name, param):
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name: name of the parameter. The parameter can be accessed
                from this module using the given name
            parameter: parameter to be added to the module.
        """
        if not hasattr(self, 'parameters'):
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        if name in self.parameters:
            raise ValueError(f"Parameter with name '{name}' already exists.")
        self._parameters[name] = param
        setattr(self, name, param)
            
            