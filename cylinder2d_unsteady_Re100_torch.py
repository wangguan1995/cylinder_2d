# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import csv
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset


def load_csv_file(
    file_path: str,
    keys,
    alias_dict={},
    delimeter: str = ",",
    encoding: str = "utf-8",
):
    # read all data from csv file
    with open(file_path, "r", encoding=encoding) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimeter)
        raw_data = collections.defaultdict(list)
        for _, line_dict in enumerate(reader):
            for key, value in line_dict.items():
                raw_data[key].append(value)

    # convert to numpy array
    data_dict = {}
    for key in keys:
        fetch_key = alias_dict[key] if key in alias_dict else key
        if fetch_key not in raw_data:
            raise KeyError(f"fetch_key({fetch_key}) do not exist in raw_data.")
        data_dict[key] = np.asarray(raw_data[fetch_key], float).reshape([-1, 1])

    return data_dict


def convert_to_array(dict, keys):
    return np.concatenate([dict[key] for key in keys], axis=-1)


def convert_to_dict(array, keys):
    split_array = np.split(array, len(keys), axis=-1)
    return {key: split_array[i] for i, key in enumerate(keys)}


def combine_array_with_time(x, t):
    nx = len(x)
    tx = []
    for ti in t:
        tx.append(np.hstack((np.full([nx, 1], float(ti), dtype=float), x)))
    tx = np.vstack(tx)
    return tx


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        h_nD = 50
        self.main = nn.Sequential(
            nn.Linear(3, h_nD),
            nn.Tanh(),
            nn.Linear(h_nD, h_nD),
            nn.Tanh(),
            nn.Linear(h_nD, h_nD),
            nn.Tanh(),
            nn.Linear(h_nD, h_nD),
            nn.Tanh(),
            nn.Linear(h_nD, 3),
        )

    # This function defines the forward rule of
    # output respect to input.
    def forward(self, x):
        output = self.main(x)
        return output


def jacobian(x, y):
    return torch.autograd.grad(
        x, y, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True
    )[0]


def hessian(x, y):
    return jacobian(jacobian(x, y), y)


def loss_pde(model_net, x, y, t, nu, rho):
    device = torch.device("cuda")
    x = x.to(device)
    y = y.to(device)
    t = t.to(device)

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    net_in = torch.cat((x, y, t), 1)
    out = model_net(net_in)
    u = out[:, 0]
    v = out[:, 1]
    P = out[:, 2]
    u = u.view(len(u), -1)
    v = v.view(len(v), -1)
    P = P.view(len(P), -1)

    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True
    )[0]
    v_t = torch.autograd.grad(
        v, t, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True
    )[0]

    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True
    )[0]
    u_y = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]
    P_x = torch.autograd.grad(
        P, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True
    )[0]
    loss_1 = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) + 1 / rho * P_x

    v_x = torch.autograd.grad(
        v, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]
    v_xx = torch.autograd.grad(
        v_x, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]

    v_y = torch.autograd.grad(
        v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]

    v_yy = torch.autograd.grad(
        v_y, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]
    P_y = torch.autograd.grad(
        P, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True
    )[0]

    loss_2 = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) + 1 / rho * P_y
    loss_3 = u_x + v_y

    # MSE LOSS
    loss_f = nn.MSELoss()

    loss = (
        loss_f(loss_1, torch.zeros_like(loss_1))
        + loss_f(loss_2, torch.zeros_like(loss_2))
        + loss_f(loss_3, torch.zeros_like(loss_3))
    )

    return loss


def loss_bc(model_net, input, label, name, weight):
    device = torch.device("cuda")
    input = input[name]
    label = label[name]

    x = torch.FloatTensor(input["x"]).to(device)
    y = torch.FloatTensor(input["y"]).to(device)
    t = torch.FloatTensor(input["t"]).to(device)
    u_label = torch.FloatTensor(label["u"]).to(device) if "u" in label.keys() else None
    v_label = torch.FloatTensor(label["v"]).to(device) if "v" in label.keys() else None
    p_label = torch.FloatTensor(label["p"]).to(device) if "p" in label.keys() else None

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    net_in = torch.cat((x, y, t), 1)
    out = model_net(net_in)
    u = out[:, 0]
    v = out[:, 1]
    p = out[:, 2]
    loss_f = nn.MSELoss()
    loss_u = (u - u_label) if u_label is not None else 0
    loss_v = (v - v_label) if v_label is not None else 0
    loss_p = (p - p_label) if p_label is not None else 0
    loss_1 = weight * (loss_u + loss_v + loss_p)
    loss = loss_f(loss_1, torch.zeros_like(loss_1))
    return loss


class MyIterableDataset(IterableDataset):
    def __init__(self, input):
        device = torch.device("cpu")
        self.input = {
            key: torch.FloatTensor(val).to(device) for key, val in input.items()
        }

    def __iter__(self):
        yield self.input


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        while True:
            dataloader_iter = iter(self.dataloader)
            for batch in dataloader_iter:
                yield batch

    def __len__(self):
        return len(self.dataloader)


def readbc(
    file_path: str,
    input_keys,
    label_keys,
    alias_dict=None,
    weight_dict=None,
    timestamps=None,
):

    # read raw data from file
    raw_data = load_csv_file(
        file_path,
        input_keys + label_keys,
        alias_dict,
    )
    # filter raw data by given timestamps if specified
    if timestamps is not None:
        if "t" in raw_data:
            # filter data according to given timestamps
            raw_time_array = raw_data["t"]
            mask = []
            for ti in timestamps:
                mask.append(np.nonzero(np.isclose(raw_time_array, ti).flatten())[0])
            raw_data = convert_to_array(raw_data, input_keys + label_keys)
            mask = np.concatenate(mask, 0)
            raw_data = raw_data[mask]
            raw_data = convert_to_dict(raw_data, input_keys + label_keys)
        else:
            # repeat data according to given timestamps
            raw_data = convert_to_array(raw_data, input_keys + label_keys)
            raw_data = combine_array_with_time(raw_data, timestamps)
            input_keys = ("t",) + tuple(input_keys)
            raw_data = convert_to_dict(raw_data, input_keys + label_keys)

    # fetch input data
    input = {key: value for key, value in raw_data.items() if key in input_keys}
    # fetch label data
    label = {key: value for key, value in raw_data.items() if key in label_keys}

    # prepare weights
    weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
    if weight_dict is not None:
        for key, value in weight_dict.items():
            if isinstance(value, (int, float)):
                weight[key] = np.full_like(next(iter(label.values())), value)
            elif callable(value):
                func = value
                weight[key] = func(input)
                if isinstance(weight[key], (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), weight[key])
            else:
                raise NotImplementedError(f"type of {type(value)} is invalid yet.")
    return input, label, weight


def parse_args():
    parser = argparse.ArgumentParser("paddlescience running script")
    parser.add_argument("-e", "--epochs", type=int, help="training epochs")
    parser.add_argument("-o", "--output_dir", type=str, help="output directory")
    parser.add_argument(
        "--to_static",
        action="store_true",
        help="whether enable to_static for forward computation",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    seed = 42
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = parse_args()

    # set output directory
    OUTPUT_DIR = (
        "./output_cylinder2d_unsteady_torch" if not args.output_dir else args.output_dir
    )
    os.chdir("/workspace/wangguan/Pdsc_debug/examples/cylinder/2d_unsteady/")

    logger = logging.getLogger("cylinder2d_unsteady")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 输出到文件
    file_handler = logging.FileHandler(
        "./cylinder2d_unsteady_torch_train.log", mode="a", encoding="utf-8"
    )

    # 输出到控制台
    stream_handler = logging.StreamHandler()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # set model
    device = torch.device("cuda")
    model_net = MLP().to(device)

    # set timestamps
    TIME_START, TIME_END = 1, 50
    NUM_TIMESTAMPS = 50
    TRAIN_NUM_TIMESTAMPS = 30

    train_timestamps = np.linspace(
        TIME_START, TIME_END, NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")
    train_timestamps = np.random.choice(train_timestamps, TRAIN_NUM_TIMESTAMPS)
    train_timestamps.sort()
    t0 = np.array([TIME_START], dtype="float32")

    val_timestamps = np.linspace(
        TIME_START, TIME_END, NUM_TIMESTAMPS, endpoint=True
    ).astype("float32")

    # set time-geometry
    input_pde = load_csv_file(
        "./datasets/domain_train.csv",
        ("x", "y"),
        alias_dict={"x": "Points:0", "y": "Points:1"},
    )
    num_t = TRAIN_NUM_TIMESTAMPS + 1
    input_pde["t"] = np.repeat(
        np.concatenate((t0, train_timestamps), axis=0).reshape(num_t, 1),
        input_pde["x"].shape[0],
        axis=0,
    )
    input_pde["x"] = np.tile(input_pde["x"], (num_t, 1))
    input_pde["y"] = np.tile(input_pde["y"], (num_t, 1))

    # set dataloader config
    ITERS_PER_EPOCH = 1

    # pde/bc/sup constraint use t1~tn, initial constraint use t0
    NPOINT_PDE, NTIME_PDE = 9420, len(train_timestamps)
    NPOINT_INLET_CYLINDER = 161
    NPOINT_OUTLET = 81
    ALIAS_DICT = {"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"}

    # set constraint

    # set training hyper-parameters
    EPOCHS = 40000 if not args.epochs else args.epochs
    EVAL_FREQ = 400

    # set optimizer
    optimizer = optim.Adam(
        model_net.parameters(), lr=0.001, betas=(0.9, 0.99), eps=10**-15
    )

    def construct_dataloader(input, batchsize):
        device = torch.device("cuda")
        dataset = MyIterableDataset(input)
        # dataloader = DataLoader(dataset, batch_size=batchsize, num_workers = 0, drop_last = False)
        return InfiniteDataLoader(dataset)

    pde_data_iter = iter(construct_dataloader(input_pde, NPOINT_PDE * NTIME_PDE))
    CHKPT_DIR = OUTPUT_DIR + "/checkpoint"
    os.makedirs(CHKPT_DIR, exist_ok=True)

    input_dict, label_dict = {}, {}
    input_dict["in"], label_dict["in"], _ = readbc(
        **{
            "file_path": "./datasets/domain_inlet_cylinder.csv",
            "input_keys": ("x", "y"),
            "label_keys": ("u", "v"),
            "alias_dict": ALIAS_DICT,
            "weight_dict": {"u": 10, "v": 10},
            "timestamps": train_timestamps,
        }
    )

    input_dict["out"], label_dict["out"], _ = readbc(
        **{
            "file_path": "./datasets/domain_outlet.csv",
            "input_keys": ("x", "y"),
            "label_keys": ("p",),
            "alias_dict": ALIAS_DICT,
            "timestamps": train_timestamps,
        }
    )

    input_dict["ic"], label_dict["ic"], _ = readbc(
        **{
            "file_path": "./datasets/initial/ic0.1.csv",
            "input_keys": ("x", "y"),
            "label_keys": ("u", "v", "p"),
            "alias_dict": ALIAS_DICT,
            "weight_dict": {"u": 10, "v": 10, "p": 10},
            "timestamps": t0,
        }
    )

    input_dict["sup"], label_dict["sup"], _ = readbc(
        **{
            "file_path": "./datasets/probe/probe1_50.csv",
            "input_keys": ("t", "x", "y"),
            "label_keys": ("u", "v"),
            "alias_dict": ALIAS_DICT,
            "weight_dict": {"u": 10, "v": 10},
            "timestamps": train_timestamps,
        }
    )

    import time

    tic = float(time.time())
    time_point = [0 for i in range(5)]
    batch_cost = 0
    batch_cost_sum = 0
    batch_tic = time.perf_counter()
    ips = []
    for epoch in range(EPOCHS):
        input_pde = next(pde_data_iter)
        # initialize solver
        model_net.zero_grad()
        loss = loss_pde(
            model_net, input_pde["x"], input_pde["y"], input_pde["t"], 0.2, 1
        )
        loss += loss_bc(model_net, input_dict, label_dict, "in", 10)
        loss += loss_bc(model_net, input_dict, label_dict, "out", 1)
        loss += loss_bc(model_net, input_dict, label_dict, "ic", 10)
        loss += loss_bc(model_net, input_dict, label_dict, "sup", 10)
        batch_size = input_pde["x"].size()[0]
        for key, val in input_dict.items():
            batch_size += val["x"].shape[0]
        loss.backward()
        optimizer.step()
        batch_cost += time.perf_counter() - batch_tic
        batch_cost_sum += batch_cost
        batch_cost_avg = batch_cost_sum / (epoch + 1)
        ips.append(batch_size / batch_cost_avg)
        ips_msg = f" ips: {batch_size / batch_cost_avg:.5f} samples/s"
        logger.info(f"[Train][Epoch {epoch+1}/{EPOCHS}] Loss: {loss.item()}, {ips_msg}")
        if epoch % 100 == 0:
            torch.save(
                model_net.state_dict(),
                CHKPT_DIR + "/netParams_" + "_epoch_" + str(epoch) + ".pt",
            )
        batch_tic = time.perf_counter()

    toc = float(time.time())
    elapseTime = toc - tic
    ips_average = sum(ips) / len(ips)
    logger.info(
        f"elapse time in serial = {elapseTime}, average ips is : {ips_average:.5f} samples/s"
    )
