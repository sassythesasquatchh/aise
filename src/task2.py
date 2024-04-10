import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import time
import torch.nn as nn
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

from os.path import join
import pandas as pd

import datetime
import json

import optuna

PATH_TO_DATA = r"C:\Users\patri\OneDrive\Patricks OneDrive Share\Documents\ETHZ\Fruhling_23\AISE\data\Task2\DataSolution.txt"
LOG_PATH = r"C:\Users\patri\OneDrive\Patricks OneDrive Share\Documents\ETHZ\Fruhling_23\AISE\logs\task_2"


class NeuralNet(nn.Module):

    def __init__(
        self, input_dimension, output_dimension, n_hidden_layers, neurons, retrain_seed
    ):
        super(NeuralNet, self).__init__()
        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = nn.Tanh()
        self.output_activation = nn.ReLU()

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.retrain_seed = retrain_seed
        # Random Seed for weight initialization
        self.init_xavier()

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_activation(self.output_layer(x))

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain("tanh")
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)


# Initial condition to solve the heat equation u0(x)=-sin(pi x)
def initial_condition(x):
    return torch.ones(x.shape[0], 1)


# Exact solution for the heat equation ut = u_xx with the IC above
def exact_solution(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]

    u = -torch.exp(-np.pi**2 * t) * torch.sin(np.pi * x)
    return u


def exact_conductivity(inputs):
    t = inputs[:, 0]
    x = inputs[:, 1]
    k = torch.sin(np.pi * x) + 1.1

    return k


def source(inputs):
    s = -np.pi**2 * exact_solution(inputs) * (1 - exact_conductivity(inputs))
    return s


def fluid_flow(t):
    zero = torch.zeros(t.shape).to(t.device)
    return (
        torch.heaviside(t, zero)
        - torch.heaviside(t - 1.0, zero)
        - torch.heaviside(t - 2.0, zero)
        + torch.heaviside(t - 3.0, zero)
        + torch.heaviside(t - 4.0, zero)
        - torch.heaviside(t - 5.0, zero)
        - torch.heaviside(t - 6.0, zero)
        + torch.heaviside(t - 7.0, zero)
    )


class Pinns:
    def __init__(
        self, n_int_, n_sb_, n_tb_, u_lambda_, n_neurons_, n_hidden_layers_, rois_
    ):
        self.n_int = n_int_
        self.n_sb = n_sb_
        self.n_tb = n_tb_
        self.n_roi = n_int_ // 4

        # Extrema of the solution domain (t,x) in [0,8] x [0,1]
        self.domain_extrema = torch.tensor(
            [[0, 8], [0, 1]]  # Time dimension
        )  # Space dimension

        if rois_ == "none":
            self.rois = []
        else:
            self.rois = [
                torch.tensor([[0, 1.5], [0, 0.2]]),
                torch.tensor([[3.7, 5.5], [0, 0.2]]),
                torch.tensor([[0, 8], [0.25, 0.7]]),
                torch.tensor([[0, 0.2], [0.0, 1.0]]),
                torch.tensor([[1.9, 4.1], [0.5, 1.0]]),
                torch.tensor([[5.9, 8], [0.5, 1.0]]),
                torch.tensor([[3.8, 4.2], [0.0, 0.05]]),
            ]
        if rois_ == "more":
            for i in range(1, 8):
                self.rois.append(torch.tensor([[i - 0.2, i + 0.2], [0.0, 1.0]]))

        # Number of space dimensions
        self.space_dimensions = 1

        # Parameter to balance role of data and PDE
        self.lambda_u = u_lambda_

        # PDE Coefficients
        self.T_hot = 4
        self.T_cold = 1
        self.alpha_f = 0.005
        self.h_f = 5

        # Model hyperparameters
        tf_params = {
            "n_hidden_layers": 6,
            "n_neurons": 25,
            "hidden_layer_activation": "tanh",
            "output_activation": "relu",
        }
        self.tf_params = tf_params

        ts_params = {
            "n_hidden_layers": 4,
            "n_neurons": 20,
            "hidden_layer_activation": "tanh",
            "output_activation": "relu",
        }
        self.ts_params = ts_params

        # set seed for reproducibility
        self.seed = 42

        # Filepath to the measurement data
        self.measurement_data_filepath = PATH_TO_DATA

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # FF Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=n_hidden_layers_,
            neurons=n_neurons_,
            retrain_seed=self.seed,
        ).to(self.device)

        # FF Dense NN to approximate the conductivity we wish to infer
        self.approximate_coefficient = NeuralNet(
            input_dimension=self.domain_extrema.shape[0],
            output_dimension=1,
            n_hidden_layers=n_hidden_layers_,
            neurons=n_neurons_,
            retrain_seed=self.seed,
        ).to(self.device)

        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(
            dimension=self.domain_extrema.shape[0]
        )

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = (
            self.assemble_datasets()
        )

        # number of sensors to record temperature
        self.n_sensor = 50

        self.metadata_file = join(LOG_PATH, "metadata.json")

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return (
            tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0])
            + self.domain_extrema[:, 0]
        )

    def convert_range(self, tens, domain):
        assert tens.shape[1] == self.domain_extrema.shape[0]
        return tens * (domain[:, 1] - domain[:, 0]) + domain[:, 0]

    ################################################################################################
    # Function returning the input-output tensor required to assemble the training set S_tb corresponding to the temporal boundary
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[0, 0]
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 0] = torch.full(input_tb[:, 0].shape, t0)
        output_tb = initial_condition(input_tb[:, 1]).reshape(-1, 1)

        return input_tb, output_tb

    # Function returning the input-output tensor required to assemble the training set S_sb corresponding to the spatial boundary
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[1, 0]
        xL = self.domain_extrema[1, 1]

        x0_input_list = []
        xL_input_list = []

        x0_output_list = []
        xL_output_list = []

        n_sb_per_domain = self.n_sb // 8

        for i in range(8):
            domain = torch.tensor(
                [[i, i + 1], [0, 1]]
            )  # Time dimension  # Space dimension

            input_sb_section = self.convert_range(
                self.soboleng.draw(n_sb_per_domain), domain
            )
            input_sb_section_0 = torch.clone(input_sb_section)
            input_sb_section_0[:, 1] = torch.full(input_sb_section_0[:, 1].shape, x0)

            input_sb_section_L = torch.clone(input_sb_section)
            input_sb_section_L[:, 1] = torch.full(input_sb_section_L[:, 1].shape, xL)

            x0_input_list.append(input_sb_section_0)
            xL_input_list.append(input_sb_section_L)

            if i % 4 == 0:  # charging
                output_sb_section_0 = self.T_hot * torch.ones(
                    (input_sb_section_0.shape[0], 1)
                )
                output_sb_section_L = torch.zeros((input_sb_section_0.shape[0], 1))

            elif i % 4 == 2:  # discharging
                output_sb_section_0 = torch.zeros((input_sb_section_0.shape[0], 1))
                output_sb_section_L = self.T_cold * torch.ones(
                    (input_sb_section_0.shape[0], 1)
                )

            elif i % 2 == 1:  # idle
                output_sb_section_0 = torch.zeros((input_sb_section_0.shape[0], 1))
                output_sb_section_L = torch.zeros((input_sb_section_0.shape[0], 1))

            x0_output_list.append(output_sb_section_0)
            xL_output_list.append(output_sb_section_L)

        # input_sb = self.convert(self.soboleng.draw(self.n_sb))

        # input_sb_0 = torch.clone(input_sb)
        # input_sb_0[:, 1] = torch.full(input_sb_0[:, 1].shape, x0)

        # input_sb_L = torch.clone(input_sb)
        # input_sb_L[:, 1] = torch.full(input_sb_L[:, 1].shape, xL)

        # output_sb_0 = torch.zeros((input_sb.shape[0], 1))
        # output_sb_L = torch.zeros((input_sb.shape[0], 1))

        return torch.cat(x0_input_list + xL_input_list, 0), torch.cat(
            x0_output_list + xL_output_list, 0
        )

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        inputs = [input_int]
        outputs = [output_int]
        for roi in self.rois:
            input_int_roi = self.convert_range(self.soboleng.draw(self.n_roi), roi)
            output_int_roi = torch.zeros((input_int_roi.shape[0], 1))
            inputs.append(input_int_roi)
            outputs.append(output_int_roi)
        return torch.cat(inputs, 0), torch.cat(outputs, 0)

    def get_measurement_data(self):
        # torch.random.manual_seed(42)
        # # take measurments every 0.001 sec on a set of randomly placed (in space) sensors
        # t = torch.linspace(0, self.domain_extrema[0, 1], 25)
        # x = torch.linspace(self.domain_extrema[1, 0], self.domain_extrema[1, 1], self.n_sensor)

        # # x = torch.rand(self.n_sensor)
        # # x = x * (self.domain_extrema[1, 1] - self.domain_extrema[1, 0]) +  self.domain_extrema[1, 0]

        # input_meas = torch.cartesian_prod(t, x)

        # output_meas = exact_solution(input_meas).reshape(-1,1)
        # noise = 0.01*torch.randn_like(output_meas)
        # output_meas = output_meas + noise

        measurements = torch.tensor(
            pd.read_csv(self.measurement_data_filepath, header=0).values,
            dtype=torch.float32,
        )
        input_meas = measurements[:, 0:2]
        output_meas = measurements[:, 2].reshape(-1, 1)

        return input_meas.to(self.device), output_meas.to(self.device)

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb, output_sb = self.add_spatial_boundary_points()  # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()  # S_int

        training_set_sb = DataLoader(
            torch.utils.data.TensorDataset(input_sb, output_sb),
            batch_size=2 * self.space_dimensions * self.n_sb,
            shuffle=False,
        )
        training_set_tb = DataLoader(
            torch.utils.data.TensorDataset(input_tb, output_tb),
            batch_size=self.n_tb,
            shuffle=False,
        )
        training_set_int = DataLoader(
            torch.utils.data.TensorDataset(input_int, output_int),
            batch_size=self.n_int + len(self.rois) * self.n_roi,
            shuffle=False,
        )

        return training_set_sb, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb)
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_sb):
        input_sb.requires_grad = True
        u_pred_sb = self.approximate_solution(input_sb)

        grad_u_x = torch.autograd.grad(u_pred_sb.sum(), input_sb, create_graph=True)[0][
            :, 1
        ]

        return u_pred_sb, grad_u_x

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_int):
        input_int.requires_grad = True
        T_f = self.approximate_solution(input_int).reshape(
            -1,
        )
        T_s = self.approximate_coefficient(input_int).reshape(
            -1,
        )

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_u = torch.autograd.grad(T_f.sum(), input_int, create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int, create_graph=True)[
            0
        ][:, 1]

        # s = source(input_int)

        with torch.no_grad():
            U_f = fluid_flow(input_int[:, 0])

        residual = (
            grad_u_t
            + U_f * grad_u_x
            - self.alpha_f * grad_u_xx
            + self.h_f * (T_f - T_s)
        )

        return residual.reshape(
            -1,
        )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(
        self,
        inp_train_sb,
        u_train_sb,
        inp_train_tb,
        u_train_tb,
        inp_train_int,
        verbose=True,
    ):

        u_pred_sb, grad_u_sb_x = self.apply_boundary_conditions(inp_train_sb)
        u_pred_tb = self.apply_initial_condition(inp_train_tb)

        inp_train_meas, u_train_meas = self.get_measurement_data()
        u_pred_meas = self.approximate_solution(inp_train_meas)

        assert u_pred_sb.shape[1] == u_train_sb.shape[1]
        assert u_pred_tb.shape[1] == u_train_tb.shape[1]
        assert u_pred_meas.shape[1] == u_train_meas.shape[1]

        r_int = self.compute_pde_residual(inp_train_int)
        # r_sb = u_train_sb - u_pred_sb
        r_tb = u_train_tb - u_pred_tb
        r_meas = u_train_meas - u_pred_meas

        r_sb = grad_u_sb_x.clone()
        section_length = self.n_sb // 8
        for i in [0, 4]:  # dirichlet conditions for charging phase at x0
            r_sb[i * section_length : (i + 1) * section_length] = (
                u_pred_sb[i * section_length : (i + 1) * section_length]
                - u_train_sb[i * section_length : (i + 1) * section_length]
            ).reshape(
                -1,
            )

        for i in [10, 14]:  # dirichlet conditions for discharging phase at xL
            r_sb[i * section_length : (i + 1) * section_length] = (
                u_pred_sb[i * section_length : (i + 1) * section_length]
                - u_train_sb[i * section_length : (i + 1) * section_length]
            ).reshape(
                -1,
            )

        loss_sb = torch.mean(abs(r_sb) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int = torch.mean(abs(r_int) ** 2)
        loss_meas = torch.mean(abs(r_meas) ** 2)

        loss_u = loss_sb + loss_tb + loss_meas

        loss = torch.log10(self.lambda_u * loss_u + loss_int)
        if verbose:
            print(
                "Total loss: ",
                round(loss.item(), 4),
                "| PDE Loss: ",
                round(torch.log10(loss_int).item(), 4),
                "| Function Loss: ",
                round(torch.log10(loss_u).item(), 4),
            )

        return loss

    def save(self, loss_history):
        filename_tf = join(
            "models",
            "task_3",
            datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "_tf.pt",
        )
        filename_ts = join(
            "models",
            "task_3",
            datetime.datetime.now().strftime("%m-%d %H:%M:%S") + "_ts.pt",
        )

        salient_info = {}
        salient_info["n_hidden_layers"] = self.n_hidden_layers
        salient_info["n_neurons"] = self.n_neurons
        salient_info["final_loss"] = loss_history[-1]
        salient_info["min_loss"] = min(loss_history)
        salient_info["tf_model_path"] = filename_tf
        salient_info["ts_model_path"] = filename_ts
        salient_info["seed"] = self.seed
        salient_info["tf_model_hyperparams"] = self.tf_params
        salient_info["ts_model_hyperparams"] = self.ts_params
        salient_info["lambda"] = self.lambda_u
        salient_info["rois"] = [roi.numpy().tolist() for roi in self.rois]
        salient_info["n_int"] = self.n_int
        salient_info["n_sb"] = self.sb
        salient_info["n_tb"] = self.tb
        salient_info["n_roi"] = self.roi

        torch.save(
            self.approximate_solution.state_dict(), join("models", "task_2", "tf.pt")
        )
        torch.save(
            self.approximate_coefficient.state_dict(), join("models", "task_2", "ts.pt")
        )

        with open(self.metadata_file, "a") as f:
            json.dump(salient_info, f)


def fit(model: Pinns, num_epochs, optimizer, verbose=True):
    history = list()
    device = model.device
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print(
                "################################ ",
                epoch,
                " ################################",
            )

        for j, (
            (inp_train_sb, u_train_sb),
            (inp_train_tb, u_train_tb),
            (inp_train_int, u_train_int),
        ) in enumerate(
            zip(model.training_set_sb, model.training_set_tb, model.training_set_int)
        ):

            def closure():
                optimizer.zero_grad()
                loss = model.compute_loss(
                    inp_train_sb.to(device),
                    u_train_sb.to(device),
                    inp_train_tb.to(device),
                    u_train_tb.to(device),
                    inp_train_int.to(device),
                    verbose=verbose,
                )
                loss.backward()

                history.append(loss.item())
                return loss

            optimizer.step(closure=closure)

    print("Final Loss: ", history[-1])

    return history


def objective(trial):

    # parameters:
    # number of points
    # lambda
    # number of neurons
    # number of hidden layers
    # rois (more or less)

    num_points = trial.suggest_int("num_points", 100, 1000, log=True)
    u_lambda = trial.suggest_float("u_lambda", 0.1, 1000, log=True)
    n_neurons = trial.suggest_int("n_neurons", 20, 40, log=True)
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 2, 6)
    rois = trial.suggest_categorical("rois", ["more", "less", "none"])

    n_int = num_points
    n_sb = num_points // 4
    n_tb = num_points // 4

    pinn = Pinns(n_int, n_sb, n_tb, u_lambda, n_neurons, n_hidden_layers, rois)

    optimizer_choice = trial.suggest_categorical("optimizer", ["Adam", "LBFGS"])
    if optimizer_choice == "Adam":
        optimizer = optim.Adam(
            list(pinn.approximate_solution.parameters())
            + list(pinn.approximate_coefficient.parameters()),
            lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        )
        n_epochs = 20000
    else:
        optimizer = optim.LBFGS(
            list(pinn.approximate_solution.parameters())
            + list(pinn.approximate_coefficient.parameters()),
            lr=float(0.5),
            max_iter=50000,
            max_eval=50000,
            history_size=150,
            line_search_fn="strong_wolfe",
            tolerance_change=1.0 * np.finfo(float).eps,
        )
        n_epochs = 5

    hist = fit(pinn, num_epochs=n_epochs, optimizer=optimizer, verbose=True)

    # trial.report(val_loss, step=epoch)
    # if trial.should_prune():
    #     raise optuna.TrialPruned()

    return min(hist)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    logged_data = trial.params
    logged_data["loss"] = trial.value

    with open(join(LOG_PATH, "task2.json"), "w") as f:
        json.dump(trial.params, f)
