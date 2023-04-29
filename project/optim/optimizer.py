class Optimizer:
    def __init__(self, net, loss, learning_rate):
        self.net = net
        self.loss = loss
        self.learning_rate = learning_rate

    def step(self, batch_x, batch_y) -> float:
        # Compute network output
        output = self.net.forward(batch_x)

        # Calculate the loss
        loss_value = self.loss.forward(batch_y, output)

        # Perform backpropagation
        delta = self.loss.backward(batch_y, output)
        self.net.backward_update_gradient(batch_x, delta)

        # Update network parameters
        self.net.update_parameters(learning_rate=self.learning_rate)

        return loss_value
