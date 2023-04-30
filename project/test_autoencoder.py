import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from modules import Linear, TanH, Sigmoid, Sequential, BCELoss
from optim.sgd import SGD
from optim.optimizer import Optimizer

# Load USPS dataset
data = pkl.load(open("data/usps.pkl", "rb"))
X_train = np.array(data["X_train"], dtype=float)
X_test = np.array(data["X_test"], dtype=float)


encoder = Sequential(Linear(256, 100), TanH(), Linear(100, 10), TanH())

hidden_weights = encoder._modules["0"]._parameters["weight"]
latent_weights = encoder._modules["2"]._parameters["weight"]

decoder = Sequential(Linear(10, 100), TanH(), Linear(100, 256), Sigmoid())

decoder._modules["0"].set_weight(latent_weights.T)
decoder._modules["2"].set_weight(hidden_weights.T)

autoencoder = Sequential(encoder, decoder)
bce_loss = BCELoss()
optimizer = Optimizer(autoencoder, bce_loss, 0.02)

SGD(X_train, X_train, 256, 1000, optimizer, iterations_per_epoch=200)


def reconstruct_images(encoder, decoder, input_images):
    latent_representations = encoder.forward(input_images)
    reconstructed_images = decoder.forward(latent_representations)
    return reconstructed_images


def display_images(original_images, reconstructed_images, num_images=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_images):
        # Display original image
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(16, 16), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed image
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i].reshape(16, 16), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# Test the autoencoder on the test set
n_samples = 10
test_images = X_test[:n_samples]

reconstructed_images = reconstruct_images(encoder, decoder, test_images)

# Visualize the original and decoded images
display_images(test_images, reconstructed_images)
