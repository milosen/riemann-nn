import torch
import torch.nn as nn
import numpy as np
from numpy import pi, ndarray
import matplotlib.pyplot as plt
import torch.optim as optim
from matplotlib import figure

learning_rate = 0.005
num_layers = 10
num_epochs = 1000
N = 500


def plot_liner_classifier(plot: figure, w: ndarray, b: float) -> None:
    """
    @note not sure about this yet
    @param plot: matplotlib figure object
    @param w: weights of linear classifier
    @param b: bias of linear classifier
    """
    xmin, xmax = plot.axes.get_xlim()
    plot.plot([(-w[0, 0]*x - b)/w[0, 1] for x in range(int(xmin), int(xmax))], 'k')


def spiral2d(data_points: int = 200, randomize: bool = False) -> tuple:
    """
    @note idea from https://gist.github.com/45deg
    @param data_points:
    @param randomize:
    @return:
    """
    dims = 2

    def add_noise(r: bool) -> ndarray:
        if r:
            return np.random.randn(data_points, dims)
        else:
            return np.zeros((data_points, dims))

    theta = np.sqrt(np.random.rand(data_points))*2*pi  # np.linspace(0,2*pi,100)

    r_a = 2*theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + add_noise(randomize)

    r_b = -2*theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + add_noise(randomize)

    res_a = np.append(x_a, np.zeros((data_points, 1)), axis=1)
    res_b = np.append(x_b, np.ones((data_points, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    return res, x_a, x_b


class Net(nn.Module):
    def __init__(self, layers, identity_conn=False):
        super(Net, self).__init__()
        self.activations = []
        self.linear = nn.Linear(2, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=2)
        self.identity_connection = identity_conn
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(layers)])

    def forward(self, x, return_activations=False):
        self.activations = []
        # transforms.Normalize(mean=[0, 0], std=[1, 1])
        # ModuleList can act as an iterable, or be indexed using ints
        for i, layer in enumerate(self.layers):
            if self.identity_connection:
                z = layer(x)
                x = x + torch.relu(z) * 1 / num_layers
            else:
                x = layer(x)
                x = torch.relu(x)
            x = self.bn1(x)
            if return_activations:
                self.activations.append(x)
        return torch.sigmoid(self.linear(x))

    def plot_geometry(self, a_numpy, b_numpy):
        print("Hi")
        a = torch.from_numpy(a_numpy).float()
        b = torch.from_numpy(b_numpy).float()

        self.forward(a, return_activations=True)
        act_cls1 = self.activations
        self.forward(b, return_activations=True)
        act_cls2 = self.activations

        fig, plots = plt.subplots(1, num_layers + 1, figsize=[5 * num_layers, 5])
        iter_plots = iter(plots)
        plot = next(iter_plots)
        plot.scatter(a.detach().numpy()[:, 0], a_numpy[:, 1])
        plot.scatter(b.detach().numpy()[:, 0], b_numpy[:, 1])
        plot.set_title('Layer 0')
        for ln, layer_activations in enumerate(zip(act_cls1, act_cls2), start=1):
            cls1, cls2 = map(lambda x: x.detach().numpy(), layer_activations)
            plot = next(iter_plots)
            plot.scatter(cls1[:, 0], cls1[:, 1])
            plot.scatter(cls2[:, 0], cls2[:, 1])
            plot.set_title(f'Layer {ln}')
            if ln == num_layers:
                for p, par in enumerate(self.parameters()):
                    if p > 1:
                        break
                    elif p == 1:
                        b = par.detach().numpy()
                    else:
                        w = par.detach().numpy()
                plot_liner_classifier(plot, w, b)
        plt.show()


def main():
    net = Net(num_layers, identity_conn=True)  # 2 layers

    # create your optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    data, x_a, x_b = spiral2d(N, randomize=False)

    input_data = torch.from_numpy(data[:, :2]).float()
    target = torch.from_numpy(data[:, 2]).float().view(N*2, 1)

    for epoch in range(num_epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input_data)
        loss = criterion(output, target)
        if epoch % (num_epochs / 10) == 0:
            print(f"epoch = {epoch}", end=", ")
            print(f"batch loss = {loss.item()}")
            # net.show_result(x_a, x_b)
        loss.backward()
        optimizer.step()

    net.plot_geometry(x_a, x_b)


if __name__ == "__main__":
    main()
