class Optimizer:
    def __init__(self, net, loss, learning_rate):
        self.net = net
        self.loss = loss
        self.learning_rate = learning_rate

    def step(self, batch_x, batch_y):
        raise NotImplementedError("Subclasses must implement this method.")
