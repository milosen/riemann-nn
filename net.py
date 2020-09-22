import os
import torch
import torch.nn as nn
import numpy as np
from numpy import pi, ndarray
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.collections import LineCollection
from torch.autograd.functional import jacobian
from matplotlib.patches import Ellipse
from itertools import zip_longest
from sklearn.datasets import make_circles, make_moons


class RiemannianMetric:
    def __init__(self, matrix=None, dim=2):
        self.dims = dim
        if matrix is None:
            self.matrix = np.eye(self.dims)
        else:
            self.matrix = matrix

    def transform_tensor_entry(self, jacobi: ndarray, index: tuple) -> int:
        su = 0
        i1, i2 = index
        for all in range(self.dims):
            for bll in range(self.dims):
                su += jacobi[all][i1] * jacobi[bll][i2] * self.matrix[all][bll]
        return su

    def transform_coordinates(self, jacobi: ndarray) -> object:
        su = 0 * self.matrix
        for al in range(self.dims):
            for bl in range(self.dims):
                indices = (al, bl)
                su[al][bl] = self.transform_tensor_entry(jacobi, indices)
        self.matrix = su
        return self


class CustomLoss(nn.BCEWithLogitsLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.functional.F.binary_cross_entropy_with_logits(input, target, self.weight,
                                                                   pos_weight=self.pos_weight, reduction=self.reduction)


class Layer(nn.Module):
    def __init__(self, act_func, identity_conn=False):
        super(Layer, self).__init__()
        self.identity_connection = identity_conn
        if act_func is None:
            self.act_func = lambda x: x
        else:
            self.act_func = act_func
        self.linear_map = nn.Linear(2, 2)
        self.linear_map2 = nn.Linear(2, 2)

    def forward(self, x):
        if self.identity_connection:
            return x + self.act_func(self.linear_map(x))
        else:
            return self.act_func(self.linear_map(x))


def print_ln(ll, obj='grid'):
    print(f'Plotting {obj} for Layer {ll}')


class Net(nn.Module):
    def __init__(self, layers: int, identity_conn: bool, c2: bool, init_gain: float, activation):
        super(Net, self).__init__()
        self.num_layers = layers
        self.init_gain = init_gain
        self.activations = []
        self.jacobians = []
        self.linear_out = nn.Linear(2, 1)
        self.layers = nn.ModuleList([Layer(activation, identity_conn) for _ in range(layers)])
        if c2:
            self.forward_func = self.forward_c2
        else:
            self.forward_func = self.forward_c1

    @staticmethod
    def get_cmap(n, name='hsv'):
        """
        @param n: number of required colors
        @param name: a standard mpl colormap name
        @return: list of n distinct RGB colors
        """
        return plt.cm.get_cmap(name, n)

    @staticmethod
    def plot_grid(x, y, ax=None, **kwargs):
        ax = ax or plt.gca()
        segs1 = np.stack((x, y), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        ax.add_collection(LineCollection(segs1, **kwargs, zorder=1))
        ax.add_collection(LineCollection(segs2, **kwargs, zorder=1))
        ax.autoscale()

    @staticmethod
    def plot_liner_classifier(plot: figure, w: ndarray, b: ndarray, xmin: float, xmax: float) -> None:
        """
        @param xmin:
        @param xmax:
        @param plot: matplotlib figure object
        @param w: weights of linear classifier
        @param b: bias of linear classifier
        """
        def f(x):
            return (-w[0, 0] * x - b[0]) / w[0, 1]

        plot.plot([xmin, xmax], [f(xmin), f(xmax)], 'k')

    def init_weights(self):
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_normal_(m.weight, gain=self.init_gain)
                m.bias.data.fill_(0)
        self.apply(init)

    def init_forward(self):
        self.activations = []
        self.jacobians = []

    def save_forward(self, coordinates, coord_change_func):
        new_coords = coord_change_func(coordinates)
        self.jacobians.append(jacobian(coord_change_func, coordinates))
        self.activations.append(new_coords)

    def forward_c1(self, x, save_activations=False):
        self.init_forward()
        for i, layer in enumerate(self.layers):
            a = layer(x)
            if save_activations:
                self.save_forward(x, layer)
            x = a
        return self.linear_out(x)

    def forward_c2(self, x, save_activations=False):
        self.init_forward()
        prev_layer = x
        for i, layer in enumerate(self.layers):
            a = layer(x)
            if i != 0:
                # y = 2*x(l) + f(x(l);l) - x(l-1)
                a += x - prev_layer
            if save_activations:
                # not sure about this yet
                self.save_forward(x, layer)
            prev_layer = x
            x = a
        return self.linear_out(x)

    def forward(self, x: torch.Tensor, save_activations: bool = False) -> torch.Tensor:
        return self.forward_func(x, save_activations)

    def plot_points(self, plots: figure, cls_a: ndarray, cls_b: ndarray, plot_classifier: bool = False) -> None:
        # forward pass data points
        iter_plots = iter(plots)
        a = torch.from_numpy(cls_a).float()
        b = torch.from_numpy(cls_b).float()
        self.forward(a, save_activations=True)
        act_cls1 = self.activations
        self.forward(b, save_activations=True)
        act_cls2 = self.activations

        plot = next(iter_plots)
        plot.scatter(cls_a[:, 0], cls_a[:, 1])
        plot.scatter(cls_b[:, 0], cls_b[:, 1])
        for ln, layer_activations in enumerate(zip(act_cls1, act_cls2), start=1):
            print_ln(ln, 'points')
            cls1, cls2 = map(lambda t: t.detach().numpy(), layer_activations)
            plot = next(iter_plots)
            plot.scatter(cls1[:, 0], cls1[:, 1])
            plot.scatter(cls2[:, 0], cls2[:, 1])
            if ln == self.num_layers:
                cls12 = np.append(cls1, cls2)
                xmin = np.min(cls12)
                xmax = np.max(cls12)
                params = self.parameters()
                w = next(params).detach().numpy()
                b = next(params).detach().numpy()
                if plot_classifier:
                    Net.plot_liner_classifier(plot, w, b, xmin, xmax)

    def plot_grids(self, plots: figure, xmin, xmax, grid_dim_x: int, grid_dim_y: int, plot_tensors: bool) -> None:
        # calculate grid points
        iter_plots = iter(plots)
        grid_size = grid_dim_x * grid_dim_y
        grid_x, grid_y = np.meshgrid(np.linspace(xmin, xmax, grid_dim_x), np.linspace(xmin, xmax, grid_dim_y))
        grid_numpy_array = np.array([grid_x.reshape(grid_size), grid_y.reshape(grid_size)]).T
        grid_tensor = torch.from_numpy(grid_numpy_array).float()
        # forward pass grid points
        self.forward(grid_tensor, save_activations=True)
        plot = next(iter_plots)
        print_ln(0)
        Net.plot_grid(grid_x, grid_y, ax=plot, color="lightgrey")
        for e, grid in enumerate(self.activations, start=1):
            plot = next(iter_plots)
            print_ln(e)
            grid = grid.detach().numpy()
            xx = grid.T.reshape(2, grid_dim_x, grid_dim_y)[0]
            yy = grid.T.reshape(2, grid_dim_x, grid_dim_y)[1]
            Net.plot_grid(xx, yy, ax=plot, color="lightgrey")
        if plot_tensors:
            self.plot_tensors(plots, grid_numpy_array, grid_tensor)

    def plot_tensors(self, plots: figure, grid_numpy_array: ndarray, grid_tensor: torch.Tensor) -> None:
        # every point gets it's own color for the metric tensor plot
        cmap = Net.get_cmap(len(grid_numpy_array))
        print("Plotting tensor glyphs...")
        for e, grid_point in enumerate(grid_numpy_array):
            point = grid_tensor[e]
            self.forward(point, save_activations=True)
            g = RiemannianMetric()
            g_numpy = g.matrix
            iter_jacobi = reversed(self.jacobians)
            for en, layer in enumerate(zip_longest(reversed(self.activations), reversed(plots),
                                                   fillvalue=torch.from_numpy(grid_point))):
                point, plot = layer
                x, y = point.detach().numpy()
                if en != 0:
                    jacobi = next(iter_jacobi)
                    g_numpy = g.transform_coordinates(jacobi.detach().numpy()).matrix
                eig_vals, eig_vecs = np.linalg.eig(g_numpy)
                eig_vals = np.sqrt(eig_vals)
                indices = np.argsort(eig_vals)
                angle = np.arccos(eig_vecs[indices[1]][0] / np.linalg.norm(eig_vecs[indices[1]]))
                width, height = eig_vals[indices[1]], eig_vals[indices[0]]
                plot.add_artist(Ellipse((x, y), width, height, angle * 360 / (2 * pi),
                                        zorder=3, facecolor=cmap(e), edgecolor='k', lw=0.5))

    def plot_geometry(self, a_numpy, b_numpy, grid_dim_x=15, grid_dim_y=15, plot_rows=1, plot_cols=11,
                      plot_grids=False, plot_points=True, plot_tensors=False, plot_classifier=False, save_img='test'):
        """
        @param a_numpy:
        @param b_numpy:
        @param grid_dim_x:
        @param grid_dim_y:
        @param plot_rows:
        @param plot_cols:
        @param plot_grids:
        @param plot_points:
        @param plot_tensors:
        @param plot_classifier:
        @return:
        """
        # prepare plots
        fig, plots = plt.subplots(plot_rows, plot_cols, figsize=[5 * plot_cols, 5 * plot_rows])
        if plot_rows > 1:
            plots = plots.flatten()
        xmin = min(np.min(a_numpy), np.min(b_numpy))
        xmax = max(np.max(a_numpy), np.max(b_numpy))

        if plot_grids:
            self.plot_grids(plots, xmin, xmax, grid_dim_x, grid_dim_y, plot_tensors)

        if plot_points:
            self.plot_points(plots, a_numpy, b_numpy, plot_classifier)

        for e, plot in enumerate(plots):
            plot.set_title(f'Layer {e}')

        if not os.path.exists('img_out'):
            os.mkdir('img_out')
        i = 0
        while os.path.exists(f'img_out/{save_img}-{i}.png'):
            i += 1
        plt.savefig(f'img_out/{save_img}-{i}')
        plt.show()


def spiral2d(data_points: int, n_noise: float, spiral_length_rad: float) -> tuple:
    """
    @note idea from https://gist.github.com/45deg
    @param data_points:
    @param n_noise:
    @param spiral_length_rad:
    @return:
    """
    dims = 2
    points_per_cls = int(data_points/2)

    theta = np.sqrt(np.random.rand(points_per_cls))*spiral_length_rad

    r_a = 2*theta + pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + n_noise * np.random.randn(points_per_cls, dims)

    r_b = -2*theta - pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + n_noise * np.random.randn(points_per_cls, dims)

    res_a = np.append(x_a, np.zeros((points_per_cls, 1)), axis=1)
    res_b = np.append(x_b, np.ones((points_per_cls, 1)), axis=1)

    res = np.append(res_a, res_b, axis=0)
    np.random.shuffle(res)

    return res, x_a, x_b


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    correct = 0
    incorrect = 0
    total = 0
    n = output.shape[0]
    output = output.view(n).detach().numpy()
    target = target.view(n).detach().numpy()
    for z in zip(output, target):
        o, t = z
        correct_positive = o > 0 and t == 1
        correct_negative = o < 0 and t == 0
        if correct_positive or correct_negative:
            correct += 1
        else:
            incorrect += 1
        total += 1
    return correct / total


def data(n_samples: int, n_noise: float, n_shuffle: bool, data_type='circles'):
    if data_type == 'moons':
        return make_moons(n_samples=n_samples, noise=n_noise, shuffle=n_shuffle)
    if data_type != 'circles':
        print("Warning: Only 'spiral', 'moons' or 'circles' are admissible data manifolds. Default will be used.")
    return make_circles(n_samples=n_samples, noise=n_noise, shuffle=n_shuffle)
