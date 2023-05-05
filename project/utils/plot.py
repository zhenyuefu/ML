# Plot loss
import matplotlib.pyplot as plt


def plot_loss(
    loss_list, accuracy, batch_size, num_iterations, learning_rate, file_name
):
    plt.plot(loss_list)
    # 在右上角中添加测试集准确率,和训练参数

    # 如果accuracy是None,则不显示
    if accuracy is not None:
        text = "Accuracy: {:.2f}%\n".format(accuracy * 100)
    else:
        text = ""
    text += "Batch size: {}\n".format(batch_size)
    text += "Iterations: {}\n".format(num_iterations)
    text += "Learning rate: {}".format(learning_rate)

    plt.text(
        0.9,
        0.9,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(file_name)
