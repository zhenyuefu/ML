import numpy as np
from tqdm import tqdm


def SGD(
    X, Y, batch_size, num_iterations, optimizer, shuffle=True, iterations_per_epoch=50
):
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size

    num_epochs = num_iterations // iterations_per_epoch

    for epoch in range(num_epochs):
        if shuffle:
            indices = np.random.permutation(num_samples)
            X = X[indices]
            Y = Y[indices]

        progress_bar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch + 1}")

        for _ in progress_bar:
            iteration_loss = 0.0
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = (i + 1) * batch_size
                batch_x = X[batch_start:batch_end]
                batch_y = Y[batch_start:batch_end]

                batch_loss = optimizer.step(batch_x, batch_y)
                iteration_loss += batch_loss

            iteration_loss /= num_batches

            # Update the progress bar with the current loss
            progress_bar.set_postfix({"Loss": iteration_loss})

        progress_bar.close()
