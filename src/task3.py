import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore
import torch.optim as optim  # type: ignore
from os.path import join
import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna
import math

PATH_TO_DATA = r"C:\Users\patri\OneDrive\Patricks OneDrive Share\Documents\ETHZ\Fruhling_23\AISE\data\Task3\housing.csv"
LOG_PATH = r"C:\Users\patri\OneDrive\Patricks OneDrive Share\Documents\ETHZ\Fruhling_23\AISE\logs\task_3"


def process_data(reduced_set=False):
    full_df = pd.read_csv(PATH_TO_DATA)

    numerical_features = list(full_df.columns)
    numerical_features.remove("ocean_proximity")
    numerical_features.remove("median_house_value")

    max_house_age = full_df["housing_median_age"].max()
    full_df["age_clipped"] = full_df["housing_median_age"] == max_house_age

    full_df["median_house_value_log"] = np.log1p(full_df["median_house_value"])

    skewed_features = [
        "households",
        "median_income",
        "population",
        "total_bedrooms",
        "total_rooms",
    ]
    log_numerical_features = []
    for f in skewed_features:
        full_df[f + "_log"] = np.log1p(full_df[f])
        log_numerical_features.append(f + "_log")

    lin = LinearRegression()

    # we will train our model based on all numerical non-target features with not NaN total_bedrooms
    appropriate_columns = full_df.drop(
        [
            "median_house_value",
            "median_house_value_log",
            "ocean_proximity",
            "total_bedrooms_log",
        ],
        axis=1,
    )
    train_data = appropriate_columns[~pd.isnull(full_df).any(axis=1)]
    lin.fit(train_data.drop(["total_bedrooms"], axis=1), train_data["total_bedrooms"])
    full_df["total_bedrooms_is_nan"] = pd.isnull(full_df).any(axis=1).astype(int)
    full_df["total_bedrooms"].loc[pd.isnull(full_df).any(axis=1)] = lin.predict(
        full_df.drop(
            [
                "median_house_value",
                "median_house_value_log",
                "total_bedrooms",
                "total_bedrooms_log",
                "ocean_proximity",
                "total_bedrooms_is_nan",
            ],
            axis=1,
        )[pd.isnull(full_df).any(axis=1)]
    )
    full_df["total_bedrooms_log"] = np.log1p(full_df["total_bedrooms"])
    full_df = full_df.dropna()

    ocean_proximity_dummies = pd.get_dummies(
        full_df["ocean_proximity"],
        drop_first=True,
    )
    dummies_names = list(ocean_proximity_dummies.columns)
    full_df = pd.concat([full_df, ocean_proximity_dummies[: full_df.shape[0]]], axis=1)

    full_df = full_df.drop(["ocean_proximity"], axis=1)

    sf_coord = [-122.4194, 37.7749]
    la_coord = [-118.2437, 34.0522]

    full_df["distance_to_SF"] = np.sqrt(
        (full_df["longitude"] - sf_coord[0]) ** 2
        + (full_df["latitude"] - sf_coord[1]) ** 2
    )

    full_df["distance_to_LA"] = np.sqrt(
        (full_df["longitude"] - la_coord[0]) ** 2
        + (full_df["latitude"] - la_coord[1]) ** 2
    )

    if reduced_set:
        features_to_scale = log_numerical_features + [
            "distance_to_SF",
            "distance_to_LA",
        ]
    else:
        features_to_scale = (
            numerical_features
            + log_numerical_features
            + ["distance_to_SF", "distance_to_LA"]
        )

    scaler = StandardScaler()

    scaled_features = pd.DataFrame(
        scaler.fit_transform(full_df[features_to_scale]),
        columns=features_to_scale,
        index=full_df.index,
    )

    X = pd.concat(
        [full_df[dummies_names + ["age_clipped"]], scaled_features],
        axis=1,
        ignore_index=True,
    )
    y = full_df["median_house_value"]

    X = X.to_numpy(dtype=np.float32)
    y = y.to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).view(-1, 1).float()

    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).view(-1, 1).float()

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    validation_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=64, shuffle=True
    )

    return train_dataloader, validation_dataloader


class NeuralNet(nn.Module):

    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        retrain_seed,
        output_activation=None,
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
        # self.activation = nn.Tanh()
        # self.activation = nn.LeakyReLU()
        self.activation = nn.ReLU()
        # self.output_activation=nn.ReLU()
        if output_activation is None:
            self.output_activation = nn.Identity()

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


def objective(trial):

    reduced_set = trial.suggest_categorical("reduced_set", [True, False])
    train_dataloader, validation_dataloader = process_data(reduced_set)

    input_dim = 12 if reduced_set else 20

    n_hidden_layers = trial.suggest_int("n_hidden_layers", 6, 10)
    neurons = int(trial.suggest_float("neurons", 20, 512, log=True))
    retrain_seed = 13

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    net = NeuralNet(input_dim, 1, n_hidden_layers, neurons, retrain_seed)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    early_stopping_rounds = 5
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(2000):
        net.train()
        train_loss = 0.0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataloader.dataset)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in validation_dataloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
            val_loss /= len(validation_dataloader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= early_stopping_rounds:
                print(f"Early stopping at epoch {epoch+1}")
                break

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


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
    logged_data["rsme"] = math.sqrt(trial.value)

    with open(os.path.join(LOG_PATH, "task3_study2.json"), "w") as f:
        json.dump(trial.params, f)
