"""Backend supported: pytorch"""
import deepxde as dde
import numpy as np
from sys import exit
from deepxde.backend import torch
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config
import argparse

dde.config.disable_xla_jit()
dde.config.set_random_seed(9999)
dde.config.set_default_float("float32")

# torch.cuda.set_device(0)


class MPFNN(dde.nn.PFNN):
    def __init__(self, layer_sizes, second_layer_sizes, activation, kernel_initializer):
        super(MPFNN, self).__init__(layer_sizes, activation, kernel_initializer)
        self.first_layer_sizes = layer_sizes
        self.second_layer_sizes = second_layer_sizes
        self.activation = activations.get(activation)

        # Fully connected network
        self.firstFNN = dde.nn.PFNN(
            self.first_layer_sizes, self.activation, kernel_initializer
        )
        self.secondFNN = dde.nn.PFNN(
            self.second_layer_sizes, self.activation, kernel_initializer
        )

    def forward(self, inputs):
        x = inputs

        if self._input_transform is not None:
            x = self._input_transform(x)

        x_firstFNN = self.firstFNN(x)
        x_secondFNN = self.secondFNN(x)

        x = torch.cat((x_firstFNN, x_secondFNN), dim=1)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        return x


def main(refData, network):
    data = np.load(refData, allow_pickle="TRUE")
    nodal_coor, disp, strain = (
        data.item()["nodal_coor"],
        data.item()["nodal_disp"],
        data.item()["nodal_strain"],
    )

    ux_mean, ux_std = np.mean(disp[:, 0]), np.std(disp[:, 0])
    uy_mean, uy_std = np.mean(disp[:, 1]), np.std(disp[:, 1])

    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    losses = [
        dde.PointSetBC(nodal_coor[:, :2], strain[:, :3], component=[7, 8, 9]),
    ]

    def pde(x, f):
        Fxx, Fyy, Fxy, Fyx = f[:, 3:4], f[:, 4:5], f[:, 5:6], f[:, 6:7]

        detF = Fxx * Fyy - Fxy * Fyx

        invFxx = Fyy / detF
        invFyy = Fxx / detF
        invFxy = -Fxy / detF
        invFyx = -Fyx / detF

        E = f[:, 2:3]
        nu = 0.3
        lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        # compressible 1st PK Stress(incompressible 1st PK stress: P = -pF^(-T)+muF)
        # Derivation based on strain energy function from
        # Javier Bonet, Richard D. Wood. Nonlinear continuum mechanics for
        # finite element analysis. Cambridge University Press, 1997.

        lnF = torch.log(detF)
        Pxx = mu * Fxx + (lmbd * lnF - mu) * invFxx
        Pxy = mu * Fxy + (lmbd * lnF - mu) * invFyx
        Pyx = mu * Fyx + (lmbd * lnF - mu) * invFxy
        Pyy = mu * Fyy + (lmbd * lnF - mu) * invFyy

        NPxx, NPyy, NPxy, NPyx = f[:, 10:11], f[:, 11:12], f[:, 12:13], f[:, 13:14]

        Sxx_x = dde.grad.jacobian(Pxx, x, i=0, j=0)
        Syy_y = dde.grad.jacobian(Pyy, x, i=0, j=1)
        Syx_y = dde.grad.jacobian(Pyx, x, i=0, j=1)
        Sxy_x = dde.grad.jacobian(Pxy, x, i=0, j=0)

        momentum_x = Sxx_x + Syx_y
        momentum_y = Sxy_x + Syy_y

        if "1" in network or "2" in network:
            stress_xx = (Pxx - NPxx) / E
            stress_yy = (Pyy - NPyy) / E
            stress_xy = (Pxy - NPxy) / E
            stress_yx = (Pyx - NPyx) / E

            pde_losses = [
                momentum_x,
                momentum_y,
                stress_xx,
                stress_yy,
                stress_xy,
                stress_yx,
            ]

        else:
            pde_losses = [momentum_x, momentum_y]

        return pde_losses

    data = dde.data.PDE(geom, pde, losses, anchors=nodal_coor[:, :2])

    def feature_transform(x):
        y = x[:, 1:2]
        x = x[:, 0:1]
        pi = np.pi

        return torch.concat(
            (
                x,
                torch.sin(pi * x),
                torch.sin(pi * 2 * x),
                torch.sin(pi * 3 * x),
                torch.sin(pi * 4 * x),
                torch.sin(pi * 5 * x),
                y,
                torch.sin(pi * y),
                torch.sin(pi * 2 * y),
                torch.sin(pi * 3 * y),
                torch.sin(pi * 4 * y),
                torch.sin(pi * 5 * y),
            ),
            axis=1,
        )

    def output_transform(x, y):
        Nux, Nuy, NE = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        Pxx, Pyy, Pxy, Pyx = y[:, 3:4], y[:, 4:5], y[:, 5:6], y[:, 6:7]

        Nux = Nux * ux_std + ux_mean
        Nuy = Nuy * uy_std + uy_mean

        if "A" in network or "C" in network:
            # enforce displacement boundary conditions
            Nux = x[:, 0:1] * Nux - 0.2
            Nux = (1 - x[:, 0:1]) * Nux + x[:, 0:1] * 0.2

            Nuy = x[:, 1:2] * Nuy - 0.2
            Nuy = (1 - x[:, 1:2]) * Nuy + x[:, 1:2] * 0.2

        NE = 1 + 4 * torch.sigmoid(NE)

        duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
        duydy = dde.grad.jacobian(Nuy, x, i=0, j=1)
        duxdy = dde.grad.jacobian(Nux, x, i=0, j=1)
        duydx = dde.grad.jacobian(Nuy, x, i=0, j=0)

        Fxx = duxdx + 1.0
        Fxy = duxdy
        Fyx = duydx
        Fyy = duydy + 1.0

        Exx = 0.5 * (Fxx**2 + Fxy**2 - 1)
        Eyy = 0.5 * (Fyx**2 + Fyy**2 - 1)
        Exy = 0.5 * (Fxx * Fyx + Fxy * Fyy)

        return torch.concat(
            [Nux, Nuy, NE, Fxx, Fyy, Fxy, Fyx, Exx, Eyy, Exy, Pxx, Pyy, Pxy, Pyx],
            axis=1,
        )

    # Standard PINN
    if "A" in network or "B" in network:
        num_input_var = 2
    # Fourier-feature PINN
    elif "C" in network or "D" in network:
        num_input_var = 12

    if "1" in network:
        net = MPFNN(
            [num_input_var, 50, 50, [25] * 2, [25] * 2, [25] * 2, 2],
            [num_input_var] + [75] * 5 + [5],
            "swish",
            "Glorot normal",
        )
    elif "2" in network:
        net = MPFNN(
            [num_input_var] + [75] * 5 + [2],
            [num_input_var] + [75] * 5 + [5],
            "swish",
            "Glorot normal",
        )
    elif "3" in network:
        net = dde.nn.PFNN(
            [num_input_var, 50, 50, [25] * 3, [25] * 3, [25] * 3, 3],
            "swish",
            "Glorot normal",
        )
    elif "4" in network:
        net = dde.nn.PFNN(
            [num_input_var, [25] * 3, [25] * 3, [25] * 3, [25] * 3, [25] * 3, 3],
            "swish",
            "Glorot normal",
        )
    elif "5" in network:
        net = dde.nn.FNN([num_input_var] + [75] * 5 + [3], "swish", "Glorot normal")

    if "C" in network or "D" in network:
        net.apply_feature_transform(feature_transform)

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    iteration = 500000

    if "1" in network or "2" in network:
        model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 1, 1, 1e2])
    else:
        model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1e2])

    losshistory, train_state = model.train(epochs=iteration)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)

    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    output = model.predict(X_star)

    data = {"Coor": X_star, "PINN output": output}
    np.save("PINN_Prediction.npy", data)


def plot_data(refData):
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)

    ground_truth = np.load(refData, allow_pickle="TRUE")
    gt_coor, gt_elasticity_map = (
        ground_truth.item()["nodal_coor"],
        ground_truth.item()["property_map"],
    )

    PINN = np.load("PINN_Prediction.npy", allow_pickle=True)
    pinn_coor, pinn_elasticity_map = (
        PINN.item()["Coor"],
        PINN.item()["PINN output"][:, 2],
    )

    print(pinn_coor.shape, pinn_elasticity_map.shape)

    gt_elasticity_map = griddata(
        gt_coor[:, 0:2], gt_elasticity_map, (X, Y), method="cubic"
    )
    pinn_elasticity_map = griddata(
        pinn_coor, pinn_elasticity_map.reshape(-1), (X, Y), method="cubic"
    )

    print(
        "Young's modulus L2 relative error (%): ",
        np.linalg.norm(gt_elasticity_map - pinn_elasticity_map)
        / np.linalg.norm(gt_elasticity_map)
        * 100,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.75))

    ax = axes[0]
    im0 = ax.imshow(
        gt_elasticity_map,
        interpolation="nearest",
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="equal",
        cmap="gray",
    )
    plt.colorbar(
        im0,
        ax=ax,
        ticks=[
            np.min(gt_elasticity_map),
            (np.max(gt_elasticity_map) + np.min(gt_elasticity_map)) / 2,
            np.max(gt_elasticity_map),
        ],
        format="%.2f",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"$E$")

    ax = axes[1]
    im1 = ax.imshow(
        pinn_elasticity_map,
        interpolation="nearest",
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="equal",
        cmap="gray",
    )
    plt.colorbar(
        im1,
        ax=ax,
        ticks=[
            np.min(pinn_elasticity_map),
            (np.max(pinn_elasticity_map) + np.min(pinn_elasticity_map)) / 2,
            np.max(pinn_elasticity_map),
        ],
        format="%.2f",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"$E^*$")

    ax = axes[2]
    abs_error = np.abs(gt_elasticity_map - pinn_elasticity_map)
    max_error = np.max(abs_error)
    min_error = np.min(abs_error)
    im2 = ax.imshow(
        abs_error,
        interpolation="nearest",
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="equal",
        cmap="jet",
    )
    plt.colorbar(
        im2,
        ax=ax,
        ticks=[min_error, (max_error + min_error) / 2, max_error],
        format="%.1e",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"|$E$ - $E^*$|")

    plt.tight_layout()
    plt.savefig("compare_E.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/GRF_equi_disp0.4_neo.npy")
    parser.add_argument("--network", type=str, default="2B")
    args = parser.parse_args()

    if int(args.network[0]) > 5 or args.network[1].capitalize() > "D":
        print("Network architecture invalid.")
        exit()

    if "neo" not in args.data:
        print("Reference data are not characterized by Neo-Hookean model.")
        exit()

    main(args.data, args.network)
    plot_data(args.data)
