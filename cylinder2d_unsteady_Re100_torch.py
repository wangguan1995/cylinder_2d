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
import time
from pyevtk import hl
from typing import Dict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset
import torch.nn.init as init

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
        data_dict[key] = np.asarray(raw_data[fetch_key], "float32").reshape([-1, 1])

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
        tx.append(np.hstack((np.full([nx, 1], float(ti), dtype='float32'), x)))
    tx = np.vstack(tx)
    return tx


class MLP(torch.nn.Module):
    def __init__(self, w_ini, b_ini):
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
            nn.Linear(h_nD, h_nD),
            nn.Tanh(),
            nn.Linear(h_nD, 3),
        )
        for i in range(5):
            self.main[2 * i].weight = nn.Parameter(torch.from_numpy(w_ini[i].T))
            self.main[2 * i].bias = nn.Parameter(torch.from_numpy(b_ini[i]))
        self.main[-1].weight = nn.Parameter(torch.from_numpy(w_ini[-1].T))
        self.main[-1].bias = nn.Parameter(torch.from_numpy(b_ini[-1]))


    # This function defines the forward rule of
    # output respect to input.
    def forward(self, x):
        output = self.main(x)
        return output

def loss_pde(model_net, x, y, t, nu, rho):
    def jacobian(y_, x_):
        return torch.autograd.grad(y_, x_, create_graph=True, grad_outputs=torch.ones_like(y_))[0]

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    net_in = torch.cat((t, x, y), 1)
    out = model_net(net_in)
    u, v, p = torch.split(out, [1, 1, 1], dim=-1)

    u_t = jacobian(u, t)
    v_t = jacobian(v, t)

    u_x = jacobian(u, x)
    u_xx = jacobian(u_x, x)
    u_y = jacobian(u, y)
    u_yy = jacobian(u_y, y)
    p_x = jacobian(p, x)
    loss_1 = u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) + 1 / rho * p_x

    v_x = jacobian(v, x)
    v_xx = jacobian(v_x, x)

    v_y = jacobian(v, y)

    v_yy = jacobian(v_y, y)
    p_y = jacobian(p, y)

    loss_2 = v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) + 1 / rho * p_y
    loss_3 = u_x + v_y

    loss = (
        nn.functional.mse_loss(loss_1, torch.zeros_like(loss_1))
        + nn.functional.mse_loss(loss_2, torch.zeros_like(loss_2))
        + nn.functional.mse_loss(loss_3, torch.zeros_like(loss_3))
    )

    return loss


def loss_bc(model_net, input, label, name, weight):
    # device = torch.device("cuda")
    input = input[name]
    label = label[name]

    x = input["x"]
    y = input["y"]
    t = input["t"]

    u_label = label.get("u", None)
    v_label = label.get("v", None)
    p_label = label.get("p", None)

    x.requires_grad = True
    y.requires_grad = True
    t.requires_grad = True

    net_in = torch.cat((t, x, y), 1)
    out = model_net(net_in)
    u, v, p = torch.split(out, [1, 1, 1], dim=-1)

    loss_u = (u - u_label) if u_label is not None else 0
    loss_v = (v - v_label) if v_label is not None else 0
    loss_p = (p - p_label) if p_label is not None else 0
    loss_u = nn.functional.mse_loss(loss_u, torch.zeros_like(loss_u)) if u_label is not None else 0
    loss_v = nn.functional.mse_loss(loss_v, torch.zeros_like(loss_v)) if v_label is not None else 0
    loss_p = nn.functional.mse_loss(loss_p, torch.zeros_like(loss_p)) if p_label is not None else 0

    return weight * (loss_u + loss_v + loss_p)


class MyIterableDataset(IterableDataset):
    def __init__(self, input):
        # device = torch.device("cpu")
        self.input = {
            k: torch.from_numpy(v) for k, v in input.items()
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
    for k, v in input.items():
        input[k] = torch.from_numpy(v).float().cuda()
    
    for k, v in label.items():
        label[k] = torch.from_numpy(v).float().cuda()
    
    for k, v in weight.items():
        weight[k] = torch.from_numpy(v).float().cuda()
    
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


def _save_vtu_from_array(filename, coord, value, value_keys, num_timestamps=1):
    """Save data to '*.vtu' file(s).

    Args:
        filename (str): Output filename.
        coord (np.ndarray): Coordinate points with shape of [N, 2] or [N, 3].
        value (np.ndarray): Value of each coord points with shape of [N, M].
        value_keys (Tuple[str, ...]): Names of each dimension of value, such as ("u", "v").
        num_timestamps (int, optional): Number of timestamp over coord and value.
            Defaults to 1.
    """
    if not isinstance(coord, np.ndarray):
        raise ValueError(f"type of coord({type(coord)}) should be ndarray.")
    if value is not None and not isinstance(value, np.ndarray):
        raise ValueError(f"type of value({type(value)}) should be ndarray.")
    if value is not None and len(coord) != len(value):
        raise ValueError(
            f"coord length({len(coord)}) should be equal to value length({len(value)})"
        )
    if len(coord) % num_timestamps != 0:
        raise ValueError(
            f"coord length({len(coord)}) should be an integer multiple of "
            f"num_timestamps({num_timestamps})"
        )
    if coord.shape[1] not in [2, 3]:
        raise ValueError(f"ndim of coord({coord.shape[1]}) should be 2 or 3.")

    # discard extension name
    if filename.endswith(".vtu"):
        filename = filename[:-4]
    npoint = len(coord)
    coord_ndim = coord.shape[1]

    if value is None:
        value = np.ones([npoint, 1], dtype=coord.dtype)
        value_keys = ["dummy_key"]

    data_ndim = value.shape[1]
    nx = npoint // num_timestamps
    for t in range(num_timestamps):
        # NOTE: each array in data_vtu should be 1-dim, i.e. [N, 1] will occur error.
        if coord_ndim == 2:
            axis_x = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 0])
            axis_y = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 1])
            axis_z = np.zeros([nx], dtype='float32')
        elif coord_ndim == 3:
            axis_x = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 0])
            axis_y = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 1])
            axis_z = np.ascontiguousarray(coord[t * nx : (t + 1) * nx, 2])

        data_vtu = {}
        for j in range(data_ndim):
            data_vtu[value_keys[j]] = np.ascontiguousarray(
                value[t * nx : (t + 1) * nx, j]
            )

        if num_timestamps > 1:
            hl.pointsToVTK(f"{filename}_t-{t}", axis_x, axis_y, axis_z, data=data_vtu)
        else:
            hl.pointsToVTK(filename, axis_x, axis_y, axis_z, data=data_vtu)

    if num_timestamps > 1:
        logger.info(
            f"Visualization results are saved to {filename}_t-0.vtu ~ {filename}_t-{num_timestamps - 1}.vtu"
        )
    else:
        logger.info(f"Visualization result is saved to {filename}.vtu")


def save_vtu_from_dict(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
    num_timestamps: int = 1,
):
    """Save dict data to '*.vtu' file.

    Args:
        filename (str): Output filename.
        data_dict (Dict[str, np.ndarray]): Data in dict.
        coord_keys (Tuple[str, ...]): Tuple of coord key. such as ("x", "y").
        value_keys (Tuple[str, ...]): Tuple of value key. such as ("u", "v").
        num_timestamps (int, optional): Number of timestamp in data_dict. Defaults to 1.
    """
    if len(coord_keys) not in [2, 3, 4]:
        raise ValueError(f"ndim of coord ({len(coord_keys)}) should be 2, 3 or 4")

    coord = [data_dict[k] for k in coord_keys if k != "t"]
    value = [data_dict[k] for k in value_keys] if value_keys else None

    coord = np.concatenate(coord, axis=1)

    if value is not None:
        value = np.concatenate(value, axis=1)

    _save_vtu_from_array(filename, coord, value, value_keys, num_timestamps)


def random_points(geom, n, random, time_stamps, criteria=None):
    nt = 30
    t = time_stamps
    nx = int(np.ceil(n / nt))

    _size, _ntry, _nsuc = 0, 0, 0
    x = np.empty(
        shape=(nx, 2), dtype=np.float32
    )
    while _size < nx:
        index = np.random.choice(len(next(iter(geom.values()))), size=nx, replace=False)
        geom_array = np.concatenate([geom[key] for key in geom.keys()], axis=-1)
        _x = geom_array[index]
        if criteria is not None:
            # fix arg 't' to None in criteria there
            criteria_mask = criteria(
                None, *np.split(_x, 2, axis=1)
            ).flatten()
            _x = _x[criteria_mask]
        if len(_x) > nx - _size:
            _x = _x[: nx - _size]
        x[_size : _size + len(_x)] = _x

        _size += len(_x)
        _ntry += 1
        if len(_x) > 0:
            _nsuc += 1

        if _ntry >= 1000 and _nsuc == 0:
            raise ValueError(
                "Sample interior points failed, "
                "please check correctness of geometry and given creteria."
            )

    tx = []
    for ti in t:
        tx.append(
            np.hstack(
                (np.full([nx, 1], ti, dtype='float32'), x)
            )
        )
    tx = np.vstack(tx)
    if len(tx) > n:
        tx = tx[:n]
    return tx

def sample_interior(geom, n, time_stamps, random="pseudo", criteria=None, evenly=False):
    """Sample random points in the geometry and return those meet criteria."""
    x = np.empty(shape=(n, 3), dtype=np.float32)
    _size, _ntry, _nsuc = 0, 0, 0
    while _size < n:
        points = random_points(geom, n, random, time_stamps, criteria)

        if len(points) > n - _size:
            points = points[: n - _size]
        x[_size : _size + len(points)] = points

        _size += len(points)
        _ntry += 1
        if len(points) > 0:
            _nsuc += 1

        if _ntry >= 1000 and _nsuc == 0:
            raise ValueError(
                "Sample interior points failed, "
                "please check correctness of geometry and given creteria."
            )

    # if sdf_func added, return x_dict and sdf_dict, else, only return the x_dict
    sdf_dict = {}
    dim_keys = ['t', 'x', 'y']
    split_array = np.split(x, len(dim_keys), axis=-1)
    x_dict = {key: split_array[i] for i, key in enumerate(dim_keys)}
    return {**x_dict, **sdf_dict}


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
    net_params = (np.load("./net_params.npy",allow_pickle=True)).item()

    model_net = MLP(net_params['w'], net_params['b']).to(device)

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
    time_rect = load_csv_file(
        "./datasets/domain_train.csv",
        ("x", "y"),
        alias_dict={"x": "Points:0", "y": "Points:1"},
    )
    NPOINT_PDE, NTIME_PDE = 9420, len(train_timestamps)
    num_t = TRAIN_NUM_TIMESTAMPS + 1
    input_pde = sample_interior(time_rect, NPOINT_PDE * NTIME_PDE * 1, time_stamps=train_timestamps)
    domain_eval = load_csv_file(
        "./datasets/domain_eval.csv",
        ("t", "x", "y"),
        )

    # set dataloader config
    ITERS_PER_EPOCH = 1

    # pde/bc/sup constraint use t1~tn, initial constraint use t0
    NPOINT_PDE, NTIME_PDE = 9420, len(train_timestamps)
    NPOINT_INLET_CYLINDER = 161
    NPOINT_OUTLET = 81
    ALIAS_DICT = {"x": "Points:0", "y": "Points:1", "u": "U:0", "v": "U:1"}

    # set training hyper-parameters
    EPOCHS = 40000 if not args.epochs else args.epochs
    EVAL_FREQ = 400

    # set optimizer
    optimizer = optim.Adam(
        model_net.parameters(), lr=0.001
    )

    def construct_dataloader(input, batchsize):
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

    tic = float(time.time())
    time_point = [0 for i in range(5)]
    batch_cost = 0
    batch_cost_sum = 0
    ips = []
    batch_cost_list, ips_list = [], []
    for epoch in range(EPOCHS):
        batch_tic = time.perf_counter()
        input_pde = next(pde_data_iter)
        for k, v in input_pde.items():
            input_pde[k] = v.to(device)

        optimizer.zero_grad()
        loss_dict = {}
        loss_dict["pde"] = loss_pde(
            model_net, input_pde["x"], input_pde["y"], input_pde["t"], 0.2, 1
        )
        loss_dict["in"] = loss_bc(model_net, input_dict, label_dict, "in", 10)
        loss_dict["out"] = loss_bc(model_net, input_dict, label_dict, "out", 1)
        loss_dict["ic"] = loss_bc(model_net, input_dict, label_dict, "ic", 10)
        loss_dict["sup"] = loss_bc(model_net, input_dict, label_dict, "sup", 10)

        batch_size = input_pde["x"].size()[0]
        for key, val in input_dict.items():
            batch_size += val["x"].shape[0]
        loss = sum([val for _, val in loss_dict.items()])
        loss_dict_float = {key : val.item() for key, val in loss_dict.items()}
        loss.backward()
        optimizer.step()
        batch_cost = time.perf_counter() - batch_tic
        print(f"[Train][Epoch {epoch+1}/{EPOCHS}] batch cost : {batch_cost}")
        batch_cost_list.append(batch_cost)
        batch_cost_avg = sum(batch_cost_list) / (epoch + 1)
        ips_list.append(batch_size / batch_cost_avg)
        # ips_msg = f" ips: {batch_size / batch_cost_avg:.5f} samples/s"
        # logger.info(f"[Train][Epoch {epoch+1}/{EPOCHS}] Loss: {loss.item()}, {ips_msg}")
        if epoch % 100 == 0:
            torch.save(
                model_net.state_dict(),
                CHKPT_DIR + "/netParams_" + "_epoch_" + str(epoch) + ".pt",
            )
        batch_tic = time.perf_counter()

    if len(ips_list) < 10:
        print(f"epochs number {len(ips_list)} should be bigger than 10.")
    else:
        import copy
        ips_list_cp = copy.deepcopy(ips_list)
        skip_step_1 = 2
        skip_step_2 = max(int(len(ips_list)*0.05), 5)
        del ips_list[:skip_step_1]
        del ips_list[:skip_step_2]
        del ips_list[-skip_step_2:]
        print(f"average ips: {sum(ips_list) / (EPOCHS - skip_step_1 - skip_step_2*2)} samples/second")
        print(ips_list)

    toc = float(time.time())
    elapseTime = toc - tic
    ips_average = sum(ips) / len(ips)
    logger.info(
        f"elapse time in serial = {elapseTime}, average ips is : {ips_average:.5f} samples/s"
    )