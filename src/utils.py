import argparse
import inspect
import json
import math
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time
import pdb
from collections import defaultdict
from multiprocessing import Process
from random import randint
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional
import scipy
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.path import Path
from matplotlib.pyplot import MultipleLocator
from torch import Tensor

import utils_cython, structs

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import get_centerlines_most_aligned_with_trajectory, is_overlapping_lane_seq, remove_overlapping_lane_seq

am = ArgoverseMap()

line_colors = ['#375397', '#F05F78', '#80CBE5', '#ABCB51', '#C8B0B0',"#FFFF00"] # dark blue, red, light blue, green, brown. yellow

_False = False
if _False:
    import utils_cython


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("-e", "--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true')
    parser.add_argument("--data_dir",
                        default='train/data/',
                        type=str)
    parser.add_argument("--data_dir_for_val",
                        default='/home/s0001563/Eye4AI/2022eye4aimotionprediction/argoverse/val/data/',
                        type=str)
    parser.add_argument("--output_dir", default="/home/s0001563/Eye4AI/2022eye4aimotionprediction/DenseTNT/models.densetnt.1", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--temp_file_dir", default=None, type=str)
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=16.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--hidden_size",
                        default=128,
                        type=int)
    parser.add_argument("--hidden_dropout_prob",
                        default=0.1,
                        type=float)
    parser.add_argument("--sub_graph_depth",
                        default=3,
                        type=int)
    parser.add_argument("--global_graph_depth",
                        default=1,
                        type=int)
    parser.add_argument("--debug",
                        action='store_true')
    parser.add_argument("--initializer_range",
                        default=0.02,
                        type=float)
    parser.add_argument("--sub_graph_batch_size",
                        default=8000,
                        type=int)
    parser.add_argument("-d", "--distributed_training",
                        nargs='?',
                        default=8,
                        const=4,
                        type=int)
    parser.add_argument("--cuda_visible_device_num",
                        default=None,
                        type=int)
    parser.add_argument("--use_map",
                        action='store_true')
    parser.add_argument("--reuse_temp_file",
                        action='store_true')
    parser.add_argument("--old_version",
                        action='store_true')
    parser.add_argument("--max_distance",
                        default=50.0,
                        type=float)
    parser.add_argument("--no_sub_graph",
                        action='store_true')
    parser.add_argument("--no_agents",
                        action='store_true')
    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-ep", "--eval_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-tp", "--train_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("--not_use_api",
                        action='store_true')
    parser.add_argument("--core_num",
                        default=16,
                        type=int)
    parser.add_argument("--visualize",
                        action='store_true')
    parser.add_argument("--train_extra",
                        action='store_true')
    parser.add_argument("--use_centerline",
                        action='store_true')
    parser.add_argument("--autoregression",
                        nargs='?',
                        default=None,
                        const=2,
                        type=int)
    parser.add_argument("--lstm",
                        action='store_true')
    parser.add_argument("--add_prefix",
                        default=None)
    parser.add_argument("--attention_decay",
                        action='store_true')
    parser.add_argument("--placeholder",
                        default=0.0,
                        type=float)
    parser.add_argument("--multi",
                        nargs='?',
                        default=None,
                        const=6,
                        type=int)
    parser.add_argument("--method_span",
                        nargs='*',
                        default=[0, 1],
                        type=int)
    parser.add_argument("--nms_threshold",
                        default=None,
                        type=float)
    parser.add_argument("--stage_one_K", type=int)
    parser.add_argument("--master_port", default='12355')
    parser.add_argument("--gpu_split",
                        nargs='?',
                        default=0,
                        const=2,
                        type=int)
    parser.add_argument("--waymo",
                        action='store_true')
    parser.add_argument("--argoverse",
                        action='store_true')
    parser.add_argument("--nuscenes",
                        action='store_true')
    parser.add_argument("--future_frame_num",
                        default=30,
                        type=int)
    parser.add_argument("--future_test_frame_num",
                        default=16,
                        type=int)
    parser.add_argument("--single_agent",
                        action='store_true',
                        default=True)
    parser.add_argument("--agent_type",
                        default=None,
                        type=str)
    parser.add_argument("--inter_agent_types",
                        default=None,
                        nargs=2,
                        type=str)
    parser.add_argument("--mode_num",
                        default=12,
                        type=int)

class Args:
    data_dir = None
    data_kind = None
    debug = None
    train_batch_size = None
    seed = None
    eval_batch_size = None
    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    do_eval = None
    hidden_size = None
    sub_graph_depth = None
    global_graph_depth = None
    train_batch_size = None
    num_train_epochs = None
    initializer_range = None
    sub_graph_batch_size = None
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    old_version = None
    model_recover_path = None
    do_train = None
    max_distance = None
    no_sub_graph = None
    other_params: Dict = None
    eval_params = None
    train_params = None
    no_agents = None
    not_use_api = None
    core_num = None
    visualize = None
    train_extra = None
    hidden_dropout_prob = None
    use_centerline = None
    autoregression = None
    lstm = None
    add_prefix = None
    attention_decay = None
    do_test = None
    placeholder = None
    multi = None
    method_span = None
    waymo = None
    argoverse = None
    nuscenes = None
    single_agent = None
    agent_type = None
    future_frame_num = None
    no_cuda = None
    mode_num = None
    nms_threshold = None
    inter_agent_types = None


args: Args = None

logger = None


def init(args_: Args, logger_):
    global args, logger
    args = args_
    logger = logger_

    if not args.do_eval and not args.debug and os.path.exists(args.output_dir):
        print('{} {} exists'.format(get_color_text('Warning!'), args.output_dir))
        input()

    if args.do_eval:
        assert os.path.exists(args.output_dir)
        assert os.path.exists(args.data_dir_for_val)
    else:
        assert os.path.exists(args.data_dir)

    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.temp_file_dir is None:
        args.temp_file_dir = os.path.join(args.output_dir, 'temp_file')
    else:
        args.reuse_temp_file = True
        args.temp_file_dir = os.path.join(args.temp_file_dir, 'temp_file')

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.temp_file_dir, exist_ok=True)
    if not args.do_eval and not args.debug:
        src_dir = os.path.join(args.output_dir, 'src')
        if os.path.exists(src_dir):
            subprocess.check_output('rm -r {}'.format(src_dir), shell=True, encoding='utf-8')
        os.makedirs(src_dir, exist_ok=False)
        for each in os.listdir('src'):
            is_dir = '-r' if os.path.isdir(os.path.join('src', each)) else ''
            subprocess.check_output(f'cp {is_dir} {os.path.join("src", each)} {src_dir}', shell=True, encoding='utf-8')
        with open(os.path.join(src_dir, 'cmd'), 'w') as file:
            file.write(' '.join(sys.argv))
    args.model_save_dir = os.path.join(args.output_dir, 'model_save')
    os.makedirs(args.model_save_dir, exist_ok=True)

    def init_args_do_eval():
        if args.argoverse:
            args.data_dir = args.data_dir_for_val if not args.do_test else 'test_obs/data/'
        if args.model_recover_path is None:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save', 'model.16.bin')
        elif len(args.model_recover_path) <= 2:
            args.model_recover_path = os.path.join(args.output_dir, 'model_save',
                                                   'model.{}.bin'.format(args.model_recover_path))
        args.do_train = False

        if 'set_predict' in args.other_params:
            # TODO
            pass

        if len(args.method_span) != 2:
            args.method_span = [args.method_span[0], args.method_span[0] + 1]

        if args.mode_num != 6:
            add_eval_param(f'mode_num={args.mode_num}')

    def init_args_do_train():
        # if 'interactive' in args.other_params:
        #     args.data_dir = 'tf_example/validation_interactive/'
        pass

    if args.do_eval:
        init_args_do_eval()
    else:
        init_args_do_train()

    print(dict(sorted(vars(args_).items())))
    # print(json.dumps(vars(args_), indent=4))
    args_dict = vars(args)
    print()
    logger.info("***** args *****")
    for each in ['output_dir', 'other_params']:
        if each in args_dict:
            temp = args_dict[each]
            if each == 'other_params':
                temp = [param if args.other_params[param] is True else (param, args.other_params[param]) for param in
                        args.other_params]
            print("\033[31m" + each + "\033[0m", temp)
    logging(vars(args_), type='args', is_json=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.temp_file_dir, time_begin), exist_ok=True)

    if isinstance(args.data_dir, str):
        args.data_dir = [args.data_dir]

    assert args.do_train or args.do_eval


def add_eval_param(param):
    if param not in args.eval_params:
        args.eval_params.append(param)


def get_name(name='', append_time=False):
    if name.endswith(time_begin):
        return name
    prefix = 'test.' if args.do_test else 'eval.' if args.do_eval and not args.do_train else ''
    prefix = 'debug.' + prefix if args.debug else prefix
    prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


eps = 1e-5

origin_point = None
origin_angle = None


def get_pad_vector(li):
    """
    Pad vector to length of args.hidden_size
    """
    assert len(li) <= args.hidden_size
    li.extend([0] * (args.hidden_size - len(li)))
    return li


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def batch_list_to_batch_tensors_old(batch):
    batch_tensors = []
    for x in zip(*batch):
        batch_tensors.append(x)
    return batch_tensors


def round_value(v):
    return round(v / 100)


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


def get_dis_point2point(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))


def get_angle(x, y):
    return math.atan2(y, x)


def get_sub_matrix(traj, object_type, x=0, y=0, angle=None):
    res = []
    for i in range(0, len(traj), 2):
        if i > 0:
            vector = [traj[i - 2] - x, traj[i - 1] - y, traj[i] - x, traj[i + 1] - y]
            if angle is not None:
                vector[0], vector[1] = rotate(vector[0], vector[1], angle)
                vector[2], vector[3] = rotate(vector[2], vector[3], angle)
            res.append(vector)
    return res


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def rotate_(x, y, cos, sin):
    res_x = x * cos - y * sin
    res_y = x * sin + y * cos
    return res_x, res_y


index_file = 0

file2pred = {}


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)


files_written = {}


def logging(*inputs, prob=1.0, type='1', is_json=False, affi=True, sep=' ', to_screen=False, append_time=False, as_pickle=False):
    """
    Print args into log file in a convenient style.
    """
    if to_screen:
        print(*inputs, sep=sep)
    if not random.random() <= prob or not hasattr(args, 'log_dir'):
        return

    file = os.path.join(args.log_dir, get_name(type, append_time))
    if as_pickle:
        with open(file, 'wb') as pickle_file:
            assert len(inputs) == 1
            pickle.dump(*inputs, pickle_file)
        return
    if file not in files_written:
        with open(file, "w", encoding='utf-8') as fout:
            files_written[file] = 1
    inputs = list(inputs)
    the_tensor = None
    for i, each in enumerate(inputs):
        if isinstance(each, torch.Tensor):
            # torch.Tensor(a), a must be Float tensor
            if each.is_cuda:
                each = each.cpu()
            inputs[i] = each.data.numpy()
            the_tensor = inputs[i]
    np.set_printoptions(threshold=np.inf)

    with open(file, "a", encoding='utf-8') as fout:
        if is_json:
            for each in inputs:
                print(json.dumps(each, indent=4), file=fout)
        elif affi:
            print(*tuple(inputs), file=fout, sep=sep)
            if the_tensor is not None:
                print(json.dumps(the_tensor.tolist()), file=fout)
            print(file=fout)
        else:
            print(*tuple(inputs), file=fout, sep=sep)
            print(file=fout)


mpl.use('Agg')


def larger(a, b):
    return a > b + eps


def equal(a, b):
    return True if abs(a - b) < eps else False


def get_valid_lens(matrix: np.ndarray):
    valid_lens = []
    for i in range(matrix.shape[0]):
        ok = False
        for j in range(2, matrix.shape[1], 2):
            if equal(matrix[i][j], 0) and equal(matrix[i][j + 1], 0):
                ok = True
                valid_lens.append(j)
                break

        assert ok
    return valid_lens


visualize_num = 0


def rot(verts, rad):
    rad = -rad
    verts = np.array(verts)
    rotMat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    transVerts = verts.dot(rotMat)
    return transVerts


class CustomMarker(Path):
    def __init__(self, icon, az):
        import svgpath2mpl
        # if icon == "icon":
        #     verts = iconMat
        # svg = """<svg t="1624195118046" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="19465" xmlns:xlink="http://www.w3.org/1999/xlink" width="700" height="700"><defs><style type="text/css"></style></defs><path d="M812.875093 411.578027l-0.003413 0.01536-43.562667-11.671894V203.436373c0-102.367573-112.216747-185.35424-250.63936-185.35424s-250.641067 82.986667-250.641066 185.35424l-0.360107 10.238294v187.89376l-41.89696 11.226453-0.006827-0.01536c-26.519893 7.120213-44.946773 24.33536-41.166506 38.469973l47.930026-12.84096 0.003414 0.013654 35.136853-9.413974v484.061867l0.360107 7.022933c0 48.899413 112.218453 88.546987 250.641066 88.546987s250.63936-39.645867 250.63936-88.546987V427.36128l36.800854 9.86112 0.00512-0.01536 47.92832 12.84096c3.77856-14.134613-14.64832-31.34976-41.168214-38.469973zM658.152107 87.01952c13.34272-9.344 37.459627 2.075307 53.86752 25.506133s18.889387 49.998507 5.543253 59.342507c-13.34272 9.347413-37.46304-2.0736-53.86752-25.50272-16.406187-23.432533-18.891093-50.00192-5.543253-59.34592z m65.14176 231.66976l-42.922667 82.507093c-88.410453-28.182187-231.00416-29.134507-323.060053-2.84672l-41.96352-79.786666c92.73856-87.42912 315.33056-87.386453 407.94624 0.126293zM325.08416 111.418027c16.406187-23.430827 40.521387-34.850133 53.865813-25.506134 13.346133 9.344 10.862933 35.91168-5.543253 59.342507-16.402773 23.430827-40.521387 34.850133-53.865813 25.504427-13.34784-9.340587-10.86464-35.909973 5.543253-59.3408zM307.2 348.16c28.352853 17.481387 41.51808 150.084267 38.674773 276.48H307.2V348.16z m0 501.76V648.533333h37.94432c-3.43552 88.183467-14.849707 169.470293-34.530987 201.386667h-3.413333z m15.423147 21.143893l47.071573-118.454613 0.116053-0.114347c32.37888 32.37888 262.442667 30.34112 295.401814-2.618026l1.11104 0.269653 49.6896 117.439147c-43.892053 43.88864-350.266027 46.600533-393.39008 3.478186zM737.08032 846.506667h-3.413333c-19.679573-31.916373-31.095467-113.2032-34.52928-201.386667h37.942613v201.386667z m0-225.28h-38.673067c-2.843307-126.395733 10.320213-258.998613 38.673067-276.48v276.48z" fill="#1296db" p-id="19466"></path></svg>"""
        svg = "M812.875093 411.578027l-0.003413 0.01536-43.562667-11.671894V203.436373c0-102.367573-112.216747-185.35424-250.63936-185.35424s-250.641067 82.986667-250.641066 185.35424l-0.360107 10.238294v187.89376l-41.89696 11.226453-0.006827-0.01536c-26.519893 7.120213-44.946773 24.33536-41.166506 38.469973l47.930026-12.84096 0.003414 0.013654 35.136853-9.413974v484.061867l0.360107 7.022933c0 48.899413 112.218453 88.546987 250.641066 88.546987s250.63936-39.645867 250.63936-88.546987V427.36128l36.800854 9.86112 0.00512-0.01536 47.92832 12.84096c3.77856-14.134613-14.64832-31.34976-41.168214-38.469973zM658.152107 87.01952c13.34272-9.344 37.459627 2.075307 53.86752 25.506133s18.889387 49.998507 5.543253 59.342507c-13.34272 9.347413-37.46304-2.0736-53.86752-25.50272-16.406187-23.432533-18.891093-50.00192-5.543253-59.34592z m65.14176 231.66976l-42.922667 82.507093c-88.410453-28.182187-231.00416-29.134507-323.060053-2.84672l-41.96352-79.786666c92.73856-87.42912 315.33056-87.386453 407.94624 0.126293zM325.08416 111.418027c16.406187-23.430827 40.521387-34.850133 53.865813-25.506134 13.346133 9.344 10.862933 35.91168-5.543253 59.342507-16.402773 23.430827-40.521387 34.850133-53.865813 25.504427-13.34784-9.340587-10.86464-35.909973 5.543253-59.3408zM307.2 348.16c28.352853 17.481387 41.51808 150.084267 38.674773 276.48H307.2V348.16z m0 501.76V648.533333h37.94432c-3.43552 88.183467-14.849707 169.470293-34.530987 201.386667h-3.413333z m15.423147 21.143893l47.071573-118.454613 0.116053-0.114347c32.37888 32.37888 262.442667 30.34112 295.401814-2.618026l1.11104 0.269653 49.6896 117.439147c-43.892053 43.88864-350.266027 46.600533-393.39008 3.478186zM737.08032 846.506667h-3.413333c-19.679573-31.916373-31.095467-113.2032-34.52928-201.386667h37.942613v201.386667z m0-225.28h-38.673067c-2.843307-126.395733 10.320213-258.998613 38.673067-276.48v276.48z"
        # import xml.etree.ElementTree as etree
        # from six import StringIO
        # tree = etree.parse(StringIO(svg))
        # root = tree.getroot()
        az = az + math.radians(180)
        verts = svgpath2mpl.parse_path(svg).vertices
        verts[:, 0] -= (867 - 180) / 2 + 180
        verts[:, 1] -= (1008 - 18) / 2 + 18
        vertices = rot(verts, az)
        super().__init__(vertices, codes=svgpath2mpl.parse_path(svg).codes)


def clustering(mapping, goals_2D, scores: np.ndarray, future_frame_num, loss=None, labels: np.ndarray = None,
                       labels_is_valid=None, predict: np.ndarray = None): 
    predict = predict.reshape([args.mode_num, future_frame_num, 2])  

    labels = [labels[i] for i in range(future_frame_num) if labels_is_valid[i]]
    labels = np.array(labels)   
    
    lanes = [] 
    lanes_dir = []
    agent_lanes_dir = []
    distances_list = []
    closest_point_per_lane_list = []
    confidences = [] 

    # Compute probabilities
    goals = [[each[-1,0], each[-1,1]] for each in predict]
    def do_kdtree(combined_x_y_arrays,points):
        mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
        dist, indexes = mytree.query(points)
        return indexes
    goals2goals2D = [np.array(np.floor(g)) for g in goals]
    #find the scores for these goals
    score_indexes = do_kdtree(goals_2D,goals2goals2D).tolist() # log_probabilities
    goals_probs = scipy.special.softmax(scores[score_indexes]) # probabilities of the 12 predicted goals (sum up to 1) 
    # order predict by probabilities
    goals_probs_ids = np.argsort(goals_probs) 
    predict_ordered = [predict[i] for i in goals_probs_ids[::-1]]
    goals = np.array([goals[i] for i in goals_probs_ids[::-1]])
    goals_probs_ordered = [goals_probs[i] for i in goals_probs_ids[::-1]]
    
    clusters = [] # goal clusters
    cluster_lanes = [] # lanes of each cluster      
    agent_dir  = []
    dict_lanes = {} # dict of lanes and their 2D points

    for m, each in enumerate(predict_ordered):            
        # Compute trajectory direction
        agent_vector_dir = each[-2] - each[-4]
        agent_dir.append(np.arctan2(agent_vector_dir[1],agent_vector_dir[0]))
        # Transform point to original coordinate
        to_origin_coordinate(each[-1:], mapping['element_in_batch'])
        # Find nearest centerline to the end point for subsequent clustering 
        lane_id, conf, lines, distances = am.get_nearest_centerline((each[-1]), visualize=False, name=None ,city_name=mapping["city_name"]) 
        ids_list = []
        lane_dir = []
        agent_lane_angle_list = []
        closest_point_per_lane = [] 
        probability = []
        for i, line in enumerate(lines): 
            # Compute lane direction
            closest_waypt_indxs = np.linalg.norm(line - each[-1], axis=1).argsort()[:2]
            closest_point_per_lane.append(line[closest_waypt_indxs[0]])
            prev_waypoint_id = closest_waypt_indxs.min()
            next_waypoint_id = closest_waypt_indxs.max()
            prev_waypoint = line[prev_waypoint_id]
            next_waypoint = line[next_waypoint_id]
            lane_dir_vector = next_waypoint - prev_waypoint 
            lane_dir.append(lane_dir_vector)
            # Compute angle between agent and lane
            agent_lane_angle = abs( agent_dir[-1] - np.arctan2(lane_dir_vector[1],lane_dir_vector[0]) )
            if agent_lane_angle < np.pi / 4: 
                agent_lane_angle_list.append(agent_lane_angle)
                ids_list.append(i) 
            dict_lanes[lane_id[i]] = line 
        if len(ids_list) > 0: 
            lane_id, conf, lines, lane_dir, distances, closest_point_per_lane = [lane_id[i] for i in ids_list], [conf[i] for i in ids_list], [lines[i] for i in ids_list], \
                                            [lane_dir[i] for i in ids_list], [distances[i] for i in ids_list], [closest_point_per_lane[i] for i in ids_list]
    
            conf = list(scipy.special.softmax(conf))
            # Soft clustering into intention-modes with lanes
            # If goals share at least one lane, then they are in the same cluster 
            lanes_m = []
            if len(lane_id) == 1:
                lanes_m.append(lane_id)
                probability = [1.0]
            else:
            # check if lanes should be in different clusters (don't merge or are successors)
                for j in range(1, len(lane_id)): 
                    min_dist = min(np.linalg.norm(lines[0][-1] - lines[j][-1]),np.linalg.norm(lines[0][-1] - lines[j][0]), np.linalg.norm(lines[0][0] - lines[j][-1]))
                    if min_dist > 2.8: 
                        if len(lanes_m) == 0:
                            lanes_m.append(lane_id[:j])
                            lanes_m.append(lane_id[j:j+1])  
                            probability.append(conf[0])
                            probability.append(conf[j])
                        else:
                            # check if the lane is in the same cluster as the previous one
                            included = False
                            for k in range(1,j):
                                min_dist = min(np.linalg.norm(lines[k][-1] - lines[j][-1]),np.linalg.norm(lines[k][-1] - lines[j][0]), np.linalg.norm(lines[k][0] - lines[j][-1])) 
                                if min_dist < 2.8:  
                                    for kidx, lanem in enumerate(lanes_m):
                                        if lane_id[k] in lanem and lane_id[j] not in lanem:   
                                            lanes_m[kidx].append(lane_id[j])
                                            probability.append(conf[j])
                                            included=True
                                            break
                                    break
                            if not included: 
                                lanes_m.append(lane_id[j:j+1])
                                probability.append(conf[j])
                    else:
                        # lane j belongs to the same cluster as lane 0
                        if len(lanes_m) == 0:
                            lanes_m.append(lane_id[:j+1])  
                            probability.append(conf[0])
                            probability.append(conf[j])
                            
                        else:
                            lanes_m[0].append(lane_id[j])
                            probability.append(conf[j])   
            if m == 0:
                clusters = [set([m]) for i in range(len(lanes_m))] 
                cluster_lanes = [set(lm) for lm in lanes_m] 
            else:
                for l_group in lanes_m: 
                    clusterized = False
                    # If any lane in l_group is in cluster_lanes, then they are in the same cluster  
                    for c_n, lane_cn in enumerate(cluster_lanes):
                        # compute minimum last distance between lanes in l_group and lanes in cluster_lanes
                        min_dist = np.min([min(np.linalg.norm(dict_lanes[l][-1] - dict_lanes[l_c][-1]), np.linalg.norm(dict_lanes[l][-1] - dict_lanes[l_c][0]), np.linalg.norm(dict_lanes[l][0] - dict_lanes[l_c][-1])) for l in l_group for l_c in list(lane_cn)])
                        if any(l_group[i] in lane_cn for i in range(len(l_group))) or min_dist < 2.8:   
                            clusters[c_n].update([m])
                            cluster_lanes[c_n].update(l_group)
                            clusterized = True
                            break 
                    if clusterized == False:
                        clusters.append(set([m]))
                        cluster_lanes.append(set(l_group))
        else:
            print("No lanes found with agent_lane_angle < pi/4")
            lane_id = []
        
        lanes.append(lane_id) 
        lanes_dir.append(lane_dir)
        agent_lanes_dir.append(agent_lane_angle_list)
        distances_list.append(distances)
        closest_point_per_lane_list.append(closest_point_per_lane)
        confidences.append(probability) # sum(y_k) = 1)   

    # Hard clustering - choose the highest probable cluster for each mode
    hard_clusters = [[] for i in range(len(clusters))]
    cluster_probs = [0] * len(clusters) 
    max_conf_idx = [(np.array(conf)).argmax() if len(conf)>1 else 0 for conf in confidences] 
    
    for m in range(len(max_conf_idx)):
        for i, lanem in enumerate(lanes[m]):
            for j in range(len(clusters)):
                if lanem in cluster_lanes[j]:
                    cluster_probs[j] += goals_probs_ordered[m]*confidences[m][i] 
                    if i == max_conf_idx[m]:
                        hard_clusters[j].append(m)
    cluster_probs = scipy.special.softmax(cluster_probs) 
    #cluster_std = [np.std(cluster_goals[j], 0) for j in range(len(clusters))]
    cluster_avg = [np.mean(goals[c], axis=0) for c in hard_clusters]   

    # Return the goal id with the highest probabiliy in each cluster 
    cluster_ids = [goals[hard_clusters[i][0]] for i in range(len(hard_clusters)) if len(hard_clusters[i])>0]
    return cluster_ids, agent_dir[cluster_ids] 


def visualize_goals_2D(mapping, goals_2D, scores: np.ndarray, future_frame_num, loss=None, labels: np.ndarray = None,
                       labels_is_valid=None, predict: np.ndarray = None):
    print('in visualize_goals_2D', mapping['file_name'])
    print('speed', mapping.get('seep', None))

    assert predict is not None
    add_end = True
    show_prediction = True
    show_intention = True
    modes_viz = 12     
    predict = predict.reshape([args.mode_num, future_frame_num, 2])
    assert labels.shape == (future_frame_num, 2)
    os.makedirs(os.path.join(args.log_dir, 'visualize_' + time_begin), exist_ok=True)

    if 'eval_time' in mapping:
        assert labels.shape[0] == labels_is_valid.shape[0] == future_frame_num
        eval_time = mapping['eval_time']
        labels = labels[:eval_time]
        predict = predict[:, :eval_time, :]
        labels_is_valid = labels_is_valid[:eval_time]
        future_frame_num = eval_time

    if labels_is_valid is not None:
        assert labels.shape[0] == labels_is_valid.shape[0]
        labels = [labels[i] for i in range(future_frame_num) if labels_is_valid[i]]
        labels = np.array(labels)

    if 'time_offset' in mapping:
        time_offset = mapping['time_offset']
    else:
        time_offset = None

    assert labels is not None
    labels = labels.reshape([-1])
    marker_size = 100
  
    # target_agent_color, target_agent_edge_color = '#0d79e7', '#bcd6ed' # blue
    target_agent_color, target_agent_edge_color = '#4bad34', '#c5dfb3' #green 
    plt.cla()
    sns.set()
    fig = plt.figure(0, figsize=(50.0,50.0)) 

    plt.xlim(-40, 40)
    plt.ylim(-30, 50)

    cmap = plt.cm.get_cmap('Reds')
    vmin = np.log(0.0001)
    vmin = np.log(0.00001)
    scores = np.clip(scores.copy(), a_min=vmin, a_max=np.inf)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=np.max(scores)))
    
    trajs = mapping['trajs']
    if args.argoverse:
        name = os.path.split(mapping['file_name'])[1].split('.')[0]  

    linewidth = 7
    for lane in mapping['vis_lanes']:
        lane = lane[:, :2]
        assert lane.shape == (len(lane), 2), lane.shape
        plt.plot(lane[:, 0], lane[:, 1], linestyle="-", color="black", marker=None, 
                 alpha=0.5,
                 linewidth=2.5,
                 zorder=0)
        # plt.fill(lane[:, 0], lane[:, 1], linestyle="-", color='#a5a5a5',
        #          linewidth=2,
        #          zorder=0)

    def draw_his_trajs():
        # Draw history of agents 
        for i, traj in enumerate(trajs):
            assert isinstance(traj, np.ndarray)
            assert traj.ndim == 2 and traj.shape[1] == 2, traj.shape
            if i == 0:
                # Focal agent
                traj = np.array(traj).reshape([-1])
                t = np.zeros(len(traj) + 2)
                t[:len((traj))] = traj
                t[-2] = labels[0]
                t[-1] = labels[1]

                plt.plot(t[0::2], t[1::2], linestyle="-", color=target_agent_color, marker=None,
                         alpha=1,
                         linewidth=linewidth+1,
                         zorder=100)
                # if 'vis_video' in args.other_params:
                ax.plot(0.0, 0.0, marker=CustomMarker("icon", 0), c=target_agent_color,
                         markersize=marker_size , markeredgecolor=target_agent_edge_color, markeredgewidth=0.5, label = 'Focal Agent')
            elif i == 1:
                    # AV
                    color = '#ff0000'
                    width = 3 
                    plt.plot(traj[:, 0], traj[:, 1], linestyle="-", color=color, marker=None,
                                alpha=1,
                                linewidth=width,
                                zorder=0)
                    # compute yaw
                    yaw = np.arctan2(traj[-2, 1] - traj[0, 1], traj[-2, 0] - traj[0, 0]) - math.pi/2 
                    if name == '3638':
                        yaw = math.pi/2
                    # Compute yaw using direction of the closest lane
                    # Transform point to original coordinate
                    # Find the closest lane
                    """to_origin_coordinate(traj[-1:], mapping['element_in_batch'])
                    lane_id, conf, lines, distances = am.get_nearest_centerline((traj[-1]), visualize=False, city_name=mapping["city_name"]) 
                    to_relative_coordinate(lines[conf.argmax()],  mapping['cent_x'],mapping['cent_y'],mapping['angle'])
                    lane_dir_vector = am.get_lane_direction(lines[conf.argmax()], (traj[-2]), mapping["city_name"])
                    # compute yaw
                    yaw = np.arctan2(lane_dir_vector[1],lane_dir_vector[0]) """
                    ax.plot(traj[-2, 0], traj[-2, 1], marker=CustomMarker("icon", yaw), c=color,
                        markersize=marker_size, markeredgecolor=color, markeredgewidth=0.5, label = 'AV') 
            else:
                if len(traj) >= 2: 
                    color = "darkblue" 
                    label = None 
                    if i == 2:
                        label = 'Other Agents'
                    plt.plot(traj[:, 0], traj[:, 1], linestyle="-", color=color, marker=None,
                                alpha=1,
                                linewidth=linewidth,
                                zorder=0, label=label)
                    
                    circle=plt.Circle((traj[-1, 0],
                                traj[-1, 1]),
                                0.3,
                                facecolor=color,
                                edgecolor=color,
                                lw=2,
                                zorder=3)
                    ax.add_patch(circle)
          
    ax = plt.gca() # get current axes instance
    draw_his_trajs()
    if add_end:  
        ax.plot(labels[-2], labels[-1], markersize=50, color=target_agent_color, marker="*",
                    markeredgecolor='black')

    function1 = plt.plot(labels[0::2], labels[1::2], linestyle="-", color=target_agent_color, linewidth=linewidth,
                            zorder=100, label='Ground truth trajectory') 
    
    name = os.path.join(args.log_dir, 'visualize_' + time_begin,
                                get_name("visualize" + ("" if name == "" else "_" + name) + "_no-pred.png"))


    if False: #goals_2D is not None:
        goals_2D = np.array(goals_2D)
        marker_size = 120
        plt.scatter(goals_2D[:, 0], goals_2D[:, 1], c=scores, cmap=cmap, norm=sm.norm, s=marker_size, alpha=0.5, marker=',')
    
    if show_prediction:
        lanes = [] 
        lanes_dir = []
        agent_lanes_dir = []
        distances_list = []
        closest_point_per_lane_list = []
        confidences = [] 

        # Compute probabilities
        goals = [[each[-1,0], each[-1,1]] for each in predict]
        def do_kdtree(combined_x_y_arrays,points):
            mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
            dist, indexes = mytree.query(points)
            return indexes
        goals2goals2D = [np.array(np.floor(g)) for g in goals]
        #find the scores for these goals
        score_indexes = do_kdtree(goals_2D,goals2goals2D).tolist() # log_probabilities
        goals_probs = scipy.special.softmax(scores[score_indexes]) # probabilities of the 12 predicted goals (sum up to 1) 
        # order predict by probabilities
        goals_probs_ids = np.argsort(goals_probs) 
        predict_ordered = [predict[i] for i in goals_probs_ids[::-1]]
        goals = np.array([goals[i] for i in goals_probs_ids[::-1]])
        goals_probs_ordered = [goals_probs[i] for i in goals_probs_ids[::-1]]
        
        clusters = [] # goal clusters
        cluster_lanes = [] # lanes of each cluster     
        cmap_cool = plt.get_cmap('cool')
        sm_cool = plt.cm.ScalarMappable(cmap=cmap_cool , norm=plt.Normalize(vmin=0, vmax=1)) 
        dict_lanes = {} # dict of lanes and their 2D points
        
        if show_intention:  
            for m, each in enumerate(predict_ordered[:modes_viz]):            
                # Compute trajectory direction
                agent_vector_dir = each[-2] - each[-4]
                agent_dir = np.arctan2(agent_vector_dir[1],agent_vector_dir[0])

                # Transform point to original coordinate
                to_origin_coordinate(each[-1:], mapping['element_in_batch'])

                # Find nearest centerline to the end point for subsequent clustering
                name_fig_lanes = os.path.join(args.log_dir, 'visualize_' + time_begin, f"lanes_mode{m}_"+ name.split('.')[0] + ".png")
                lane_id, conf, lines, distances = am.get_nearest_centerline((each[-1]), visualize=False, name=name_fig_lanes ,city_name=mapping["city_name"]) 
                to_relative_coordinate(each[-1:], mapping['cent_x'],mapping['cent_y'],mapping['angle'])  
                ids_list = []
                lane_dir = []
                agent_lane_angle_list = []
                closest_point_per_lane = [] 
                probability = []
                for i, line in enumerate(lines): 
                    # Convert to relative coorinates to be in the same frame as the trajectory
                    to_relative_coordinate(line, mapping['cent_x'],mapping['cent_y'],mapping['angle'])  
                    # Compute lane direction
                    # compute norms to waypoints
                    closest_waypt_indxs = np.linalg.norm(line - each[-1], axis=1).argsort()[:2]

                    closest_point_per_lane.append(line[closest_waypt_indxs[0]])
                    # Compute lane direction (as a vector)
                    prev_waypoint_id = closest_waypt_indxs.min()
                    next_waypoint_id = closest_waypt_indxs.max()
                    prev_waypoint = line[prev_waypoint_id]
                    next_waypoint = line[next_waypoint_id]
                    lane_dir_vector = next_waypoint - prev_waypoint 
                    lane_dir.append(lane_dir_vector)
                    #Visualize line directions
                    if True:  
                        # plt.plot(line[:, 0], line[:, 1], color=line_colors[m%6], 
                        #        linewidth=linewidth+1, zorder=0.5 ) # plot the centerline 
                        plt.scatter(
                            line[-1][0],
                            line[-1][1],
                            200,
                            marker=".",
                            color="b",
                        )
                        final_dir = line[-1] - line[-2]
                        dx = final_dir[0] * 5
                        dy = final_dir[1] * 5
                        plt.arrow(
                            line[-1][0],
                            line[-1][1],
                            dx,
                            dy,
                            color="r",
                            width=0.2,
                            length_includes_head=True,                    
                            zorder=2,
                        )
                        # ax.annotate(lane_id[i], (line[-1-i][0], line[-1-i][1])) 
                        #for j in range(centerline_length):
                        #    plt.scatter(line[j, 0], line[j, 1], j / 5.0, marker=".", color="k") 
                    # Compute angle between agent and lane
                    agent_lane_angle = abs( agent_dir - np.arctan2(lane_dir_vector[1],lane_dir_vector[0]) )
                    #np.arccos(np.dot(agent_dir_vector, lane_dir_vectors[i]) / (np.linalg.norm(agent_dir_vector) * np.linalg.norm(lane_dir_vectors[i])))
                    if agent_lane_angle < np.pi / 4: 
                        agent_lane_angle_list.append(agent_lane_angle)
                        ids_list.append(i) 
                    dict_lanes[lane_id[i]] = line 
                if len(ids_list) > 0: 
                    lane_id, conf, lines, lane_dir, distances, closest_point_per_lane = [lane_id[i] for i in ids_list], [conf[i] for i in ids_list], [lines[i] for i in ids_list], \
                                                    [lane_dir[i] for i in ids_list], [distances[i] for i in ids_list], [closest_point_per_lane[i] for i in ids_list]
            
                    #conf = list(scipy.special.softmax(conf))

                    # Soft clustering into intention-modes with lanes
                    # If goals share at least one lane, then they are in the same cluster
                    # compute distances among all the closest_point_per_lane themselves 
                    lanes_m = []
                    if len(lane_id) == 1:
                        lanes_m.append(lane_id)
                        probability = [1.0]
                    else:
                    # check if lanes should be in different clusters  (distance between lanes > 2.8m)
                        for j in range(1, len(lane_id)):
                            # if distance in closest_point_per_lane is bigger than 2.8m and the final distance between
                            # lanes is bigger than 2.8m (they don't merge), then lanes are in different clusters 
                            # check if lanes merge, or are successors. 
                            min_dist = min(np.linalg.norm(lines[0][-1] - lines[j][-1]),np.linalg.norm(lines[0][-1] - lines[j][0]), np.linalg.norm(lines[0][0] - lines[j][-1]))
                            if min_dist > 2.8: #np.linalg.norm(closest_point_per_lane[0] - closest_point_per_lane[j]) > 2.8 and 
                                if len(lanes_m) == 0:
                                    lanes_m.append(lane_id[:j])
                                    lanes_m.append(lane_id[j:j+1])  
                                    probability.append(conf[0])
                                    probability.append(conf[j])
                                else:
                                    # check if the lane is in the same cluster as the previous one
                                    included = False
                                    for k in range(1,j):
                                        min_dist = min(np.linalg.norm(lines[k][-1] - lines[j][-1]),np.linalg.norm(lines[k][-1] - lines[j][0]), np.linalg.norm(lines[k][0] - lines[j][-1]))
                                        # if distance in closest_point_per_lane is smaller than 2.8m and the minimum distance between
                                        # lanes is smaller than 2.8m, then lanes are in the same cluster
                                        if min_dist < 2.8: # np.linalg.norm(closest_point_per_lane[j] - closest_point_per_lane[k]) < 2.8 and
                                            for kidx, lanem in enumerate(lanes_m):
                                                if lane_id[k] in lanem and lane_id[j] not in lanem:   
                                                    lanes_m[kidx].append(lane_id[j])
                                                    probability.append(conf[j])
                                                    included=True
                                                    break
                                            break
                                    if not included: #and len(range(1,j))!= 0:
                                        lanes_m.append(lane_id[j:j+1])
                                        probability.append(conf[j])
                            else:
                                # lane j belongs to the same cluster as lane 0
                                if len(lanes_m) == 0:
                                    lanes_m.append(lane_id[:j+1])  
                                    probability.append(conf[0])
                                    probability.append(conf[j])
                                    
                                else:
                                    lanes_m[0].append(lane_id[j])
                                    probability.append(conf[j])   
                    if m == 0:
                        clusters = [set([m]) for i in range(len(lanes_m))] 
                        cluster_lanes = [set(lm) for lm in lanes_m] 
                    else:
                        for l_group in lanes_m: 
                            clusterized = False
                            # If any lane in l_group is in cluster_lanes, then they are in the same cluster  
                            for c_n, lane_cn in enumerate(cluster_lanes):
                                # compute minimum last distance between lanes in l_group and lanes in cluster_lanes
                                min_dist = np.min([min(np.linalg.norm(dict_lanes[l][-1] - dict_lanes[l_c][-1]), np.linalg.norm(dict_lanes[l][-1] - dict_lanes[l_c][0]), np.linalg.norm(dict_lanes[l][0] - dict_lanes[l_c][-1])) for l in l_group for l_c in list(lane_cn)])
                                if any(l_group[i] in lane_cn for i in range(len(l_group))) or min_dist < 2.8:   
                                    clusters[c_n].update([m])
                                    cluster_lanes[c_n].update(l_group)
                                    clusterized = True
                                    break 
                            if clusterized == False:
                                clusters.append(set([m]))
                                cluster_lanes.append(set(l_group))

                    # Plot lines in relative coordinates
                    #for line in lines: 
                    #    plt.plot(line[:, 0], line[:, 1], color=line_colors[m%6], linewidth=linewidth+1, zorder=0.5, label='mode centerline') # plot the centerline
                else:
                    print("No lanes found with agent_lane_angle < pi/4")
                    lane_id = []
                
                lanes.append(lane_id) 
                lanes_dir.append(lane_dir)
                agent_lanes_dir.append(agent_lane_angle_list)
                distances_list.append(distances)
                closest_point_per_lane_list.append(closest_point_per_lane)
                confidences.append(probability) # softmax to make it probability of belonging to each cluster   

            # Save figure 1 - no prediction  
            # legend = ax.legend(loc='upper right', fontsize=60)
            # handles, labels_ax = ax.get_legend_handles_labels()  
            # legend._init_legend_box(handles, labels_ax)
            # legend._set_loc(legend._loc) 
            plt.legend(loc='upper right', fontsize=50, frameon=False)
            plt.savefig(name, bbox_inches='tight') 

            # Save figure 2 - goal set prediction
            for m, each in enumerate(predict_ordered[:modes_viz]): 
                color = "darkorange" 
                label_goals = None
                label = None 
                if m == 0:
                    label_goals = "Predicted goal set"
                    label = "Predicted trajectory"
                function2 = plt.plot(each[:, 0], each[:, 1], linestyle="-", color=color,
                                    marker=None, linewidth=linewidth*1.5, zorder=1, label=label)
                if add_end:
                    ax.plot(each[-1, 0], each[-1, 1], markersize=50, color="darkorange", marker="*",
                                markeredgecolor='black', label = label_goals)
                    # ax.annotate(str(m), (each[-1, 0], each[-1, 1]), color="darkorange") 
            # handles.append(function2[0])
            # labels_ax.append(function2[0].get_label())
            # legend._init_legend_box(handles, labels_ax)
            # legend._set_loc(legend._loc)  
            name = name.replace('no-pred','no-int') 
            plt.legend(loc='upper right', fontsize=50, frameon=False)
            plt.savefig(name, bbox_inches='tight')

            # Create figure 3 - intention prediction  
            if add_end: 
                for line in ax.lines[-30:]:
                    line.set_marker(None)
            for m, each in enumerate(predict_ordered[:modes_viz]): 
                label = None
                if m == 0:
                    label = "Predicted trajectory"
                color = cmap_cool(goals_probs_ordered[m]*5) 
                function2 = plt.plot(each[:, 0], each[:, 1], linestyle="-", color=color,
                                marker=None, linewidth=linewidth*1.5, zorder=1, label=label)

            # Hard clustering - choose the highest probable cluster for each mode
            hard_clusters = [[] for i in range(len(clusters))]
            cluster_probs = [0] * len(clusters) 
            max_conf_idx = [(np.array(conf)).argmax() if len(conf)>1 else 0 for conf in confidences] 
            
            for m in range(len(max_conf_idx)):
                for i, lanem in enumerate(lanes[m]):
                    for j in range(len(clusters)):
                        if lanem in cluster_lanes[j]:
                            cluster_probs[j] += goals_probs_ordered[m]*confidences[m][i] 
                            if i == max_conf_idx[m]:
                                hard_clusters[j].append(m)
                
            """ count = 0
            for m in range(len(max_conf_idx)):
                for j in range(len(clusters)):
                    if lanes[m][max_conf_idx[m]] in cluster_lanes[j]:
                        # compute the cluster probability taking into account the soft assignment 
                        if count == max_conf_idx[m]:
                            hard_clusters[j].append(m)
                            cluster_probs[j] += goals_probs_ordered[m]*confidences[m][max_conf_idx[m]][0]
                            # cluster_goals[j] += goals[m]*confidences[m][count][0]
                            count=0
                            break 
                        count+=1
            del count  """
            cluster_probs = scipy.special.softmax(cluster_probs) 
            #cluster_std = [np.std(cluster_goals[j], 0) for j in range(len(clusters))]
            cluster_avg = [np.mean(goals[c], axis=0) for c in hard_clusters]   

            cmap_bupu = plt.get_cmap('BuPu', )
            sm_bupu = plt.cm.ScalarMappable(cmap=cmap_bupu , norm=plt.Normalize(vmin=0, vmax=1)) 
            
            # Plot the cluster end points 
            if modes_viz > 1:
                for i,c in enumerate(cluster_avg):
                    label = None
                    if i == 0:
                        label = "Cluster center"
                    try:
                        sns.kdeplot(x=goals[list(hard_clusters[i]),0], y=goals[list(hard_clusters[i]),1], norm=sm_cool.norm, weights=np.array(goals_probs_ordered)[list(hard_clusters[i])]*20,
                                        shade=True, thresh=0.06, hue=cluster_probs[i], palette=cmap_cool, hue_norm= plt.Normalize(vmin=0, vmax=1), zorder=0.5, alpha=0.7, bw_adjust=.7)
                    except:
                        pass
                    function3 = plt.plot(c[0], c[1], markersize=50, color=cmap_cool(cluster_probs[i]), marker="o", 
                                markeredgecolor='black', zorder=200, label=label)  #line_colors[clusters[i][0]%6]
                    

                    # Color final stars with their probability color
                    if False:
                        for n_g, goal in enumerate(predict[list(hard_clusters[i]),-1]):
                            plt.plot(goal[0], goal[1], markersize=60, color=cmap_bupu(5*goals_probs_ordered[hard_clusters[i][n_g]]), marker="*",
                                    markeredgecolor='black', linestyle='') 
                function2 = function2 + function3


            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.4)
            #cbar=plt.colorbar(sm, cax=cax)
            #cbar_bupu = plt.colorbar(sm_bupu, cax=divider.append_axes("right", size="2%", pad=0.6))
            #cbar_bupu.set_label('Probability of each goal', rotation=270,size=50,weight='bold')
            #cbar.set_label('Probability scores', rotation=270,size=50,weight='bold')
            cbar_cool = plt.colorbar(sm_cool, cax=cax)
            cbar_cool.ax.get_yaxis().labelpad = 20
            cbar_cool.ax.set_ylabel('Probability of each intention', rotation=270,size=60,weight='bold')
            # cbar_cool.set_label('Probability of each intention', rotation=270,size=60,weight='bold')
            
            
            legend = ax.legend(loc='upper right', fontsize=60)
            handles, labels_ax = ax.get_legend_handles_labels()
            handles.pop(-3)
            labels_ax.pop(-3)
            handles.pop(-3)
            labels_ax.pop(-3)
            legend._legend_box = None
            legend._init_legend_box(handles, labels_ax)
            legend._set_loc(legend._loc)
            name = name.replace('no-int','intention') 
            plt.legend(loc='upper right', fontsize=50, frameon=False)
            plt.savefig(name, bbox_inches='tight') 
    plt.close()
    global visualize_num
    visualize_num += 1
    if visualize_num > 200 and 'vis_video' not in args.other_params and 'vis_all' not in args.other_params:
        print('press any key to continue')
        input()


def load_model(model, state_dict, prefix=''):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix)

    if logger is None:
        return

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, json.dumps(missing_keys, indent=4)))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, json.dumps(unexpected_keys, indent=4)))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


traj_last = None


def batch_init(mapping):
    global traj_last, origin_point, origin_angle
    batch_size = len(mapping)

    global origin_point, origin_angle
    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']

    def load_file2pred():
        global file2pred
        if len(file2pred) == 0:
            with open(args.other_params['set_predict_file2pred'], 'rb') as pickle_file:
                file2pred = pickle.load(pickle_file)


second_span = False

li_vector_num = None


def turn_traj(traj: np.ndarray, object_type='AGENT'):
    vectors = []
    traj = traj.reshape([-1, 2])
    for i, point in enumerate(traj):
        x, y = point[0], point[1]
        if i > 0:
            vector = [point_pre[0], point_pre[1], x, y, i * 0.1, object_type == 'AV',
                      object_type == 'AGENT', object_type == 'OTHERS', 0, i]
            vectors.append(get_pad_vector(vector))
        point_pre = point
    return vectors


def merge_tensors(tensors: List[torch.Tensor], device, hidden_size=None) -> Tuple[Tensor, List[int]]:
    """
    merge a list of tensors into a tensor
    """
    lengths = []
    hidden_size = args.hidden_size if hidden_size is None else hidden_size
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def de_merge_tensors(tensor: Tensor, lengths):
    return [tensor[i, :lengths[i]] for i in range(len(lengths))]


def gather_tensors(tensor: torch.Tensor, indices: List[list]):
    lengths = [len(each) for each in indices]
    assert tensor.shape[0] == len(indices)
    for each in indices:
        each.extend([0] * (tensor.shape[1] - len(each)))
    index = torch.tensor(indices, device=tensor.device)
    index = index.unsqueeze(2).expand(tensor.shape)
    tensor = torch.gather(tensor, 1, index)
    for i, length in enumerate(lengths):
        tensor[i, length:, :].fill_(0)
    # index = torch.zeros(tensor.shape, device=tensor.device, dtype=torch.int)
    return tensor, lengths


def get_closest_polygon(pred: np.ndarray, new_polygons) -> np.ndarray:
    dis = np.inf
    closest_polygon = None

    def get_dis(pred: np.ndarray, polygon):
        dis = 0
        pred = pred.reshape([30, 2])
        for point in pred:
            dis += np.min(np.abs(polygon[:, 0] - point[0]) + np.abs(polygon[:, 1] - point[1]))
        return dis

    for idx_polygon, polygon in enumerate(new_polygons):
        temp = get_dis(pred, polygon)
        if temp < dis:
            dis = temp
            closest_polygon = polygon
    return closest_polygon


NMS_LIST = [2.0, 1.7, 1.4, 2.3, 2.6] + [2.9, 3.2, 3.5, 3.8, 4.1] + [2.7, 2.8, 3.0, 3.1]

NMS_START = 6

DYNAMIC_NMS_START = 30

DYNAMIC_NMS_LIST = [3.2, 3.8, 4.8, 5.4, 6.0] + [6.6, 7.2, 7.8, 8.4, 0.0] + \
                   [2.0, 2.6, 1.5, 0.1]


def select_goals_by_NMS(mapping: Dict, goals_2D: np.ndarray, scores: np.ndarray, threshold, speed, gt_goal=None, mode_num=6):
    argsort = np.argsort(-scores)
    goals_2D = goals_2D[argsort]
    scores = scores[argsort]

    add_eval_param(f'DY_NMS={threshold}')

    speed_scale_factor = utils_cython.speed_scale_factor(speed)
    threshold = threshold * speed_scale_factor

    pred_goals = []
    pred_probs = []

    def in_predict(pred_goals, point, threshold):
        return np.min(get_dis_point_2_points(point, pred_goals)) < threshold

    for i in range(len(goals_2D)):
        if len(pred_goals) > 0 and in_predict(np.array(pred_goals), goals_2D[i], threshold):
            continue
        else:
            pred_goals.append(goals_2D[i])
            pred_probs.append(scores[i])
            if len(pred_goals) == mode_num:
                break

    while len(pred_goals) < mode_num:
        i = np.random.randint(0, len(goals_2D))
        pred_goals.append(goals_2D[i])
        pred_probs.append(scores[i])

    pred_goals = np.array(pred_goals)
    pred_probs = np.array(pred_probs)

    FDE = np.inf
    if gt_goal is not None:
        for each in pred_goals:
            FDE = min(FDE, get_dis_point2point(each, gt_goal))

    mapping['pred_goals'] = pred_goals
    mapping['pred_probs'] = pred_probs


def select_goal_pairs_by_NMS(mapping: Dict, mapping_oppo: Dict, goals_4D: np.ndarray, scores_4D: np.ndarray, threshold, speed, speed_oppo,
                             mode_num=6):
    argsort = np.argsort(-scores_4D)

    goals_4D = goals_4D[argsort]
    scores_4D = scores_4D[argsort]

    def in_predict(pred_goal_pairs, goal_pair, thresholds):
        # pred_goal_pairs [..., 2, 2]
        return np.min(get_dis_point_2_points(goal_pair[0], pred_goal_pairs[:, 0, :])) < thresholds[0] \
               and np.min(get_dis_point_2_points(goal_pair[1], pred_goal_pairs[:, 1, :])) < thresholds[1]

    add_eval_param(f'DY_NMS={threshold}')

    thresholds = (threshold * utils_cython.speed_scale_factor(speed), threshold * utils_cython.speed_scale_factor(speed_oppo))

    pred_goal_pairs = []
    pred_probs = []

    for i in range(len(goals_4D)):
        if len(pred_goal_pairs) > 0 and in_predict(np.array(pred_goal_pairs), goals_4D[i].reshape((2, 2)), thresholds):
            continue
        else:
            pred_goal_pairs.append(goals_4D[i].reshape((2, 2)))
            pred_probs.append(scores_4D[i])
            if len(pred_goal_pairs) == mode_num:
                break

    while len(pred_goal_pairs) < mode_num:
        i = np.random.randint(0, len(pred_goal_pairs))
        pred_goal_pairs.append(goals_4D[i].reshape((2, 2)))
        pred_probs.append(scores_4D[i])

    pred_goal_pairs = np.array(pred_goal_pairs)
    pred_probs = np.array(pred_probs)

    mapping['pred_goals'] = pred_goal_pairs[:, 0, :]
    mapping['pred_probs'] = pred_probs
    mapping_oppo['pred_goals'] = pred_goal_pairs[:, 1, :]
    mapping_oppo['pred_probs'] = pred_probs


def get_FDE(points: np.ndarray, scores: np.ndarray, mapping, gt_goal=None, method=0, idx_in_batch=0, mode_num=6):
    points = points.copy()
    scores = scores.copy()
    polygons = mapping['polygons']

    li = sorted([(point, score) for (point, score) in zip(points, scores)], key=lambda x: x[1], reverse=True)
    points = np.array([each[0] for each in li])
    if 'scale' in mapping:
        scale = mapping['scale']
        points *= 1.0 / scale

    scores = np.array([each[1] for each in li])

    def get_hash(point):
        return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

    if True:
        scores = np.exp(scores)

        def get_scaled_scores(scores, sum=1.0):
            sum_cur = np.sum(scores)
            scores = scores / sum_cur * sum
            return scores

        if method == 1:
            idx = np.searchsorted(-scores, -0.001, side='right')
            scores, points = scores[:idx], points[:idx]

            def fn(a):
                return a + a ** 1.2

            scaled_scores = scores.copy()
            for i in range(len(scores)):
                scaled_scores[i] = fn(scaled_scores[i])
            scaled_scores = get_scaled_scores(scaled_scores, np.sum(scores))
            # for a, b in zip(scores, scaled_scores):
            #     print(a, b)
            scores = scaled_scores

        elif method == 2:
            idx = np.searchsorted(-scores, -0.001, side='right')
            scores, points = scores[:idx], points[:idx]

            def fn(a):
                return a + a ** 0.9

            scaled_scores = scores.copy()
            for i in range(len(scores)):
                scaled_scores[i] = fn(scaled_scores[i])
            scaled_scores = get_scaled_scores(scaled_scores, np.sum(scores))
            # for a, b in zip(scores, scaled_scores):
            #     print(a, b)
            scores = scaled_scores
        elif method == 3:
            idx = np.searchsorted(-scores, -0.001, side='right')
            scores, points = scores[:idx], points[:idx]

            def fn(a):
                return a + a ** 1.1

            scaled_scores = scores.copy()
            for i in range(len(scores)):
                scaled_scores[i] = fn(scaled_scores[i])
            scaled_scores = get_scaled_scores(scaled_scores, np.sum(scores))
            # for a, b in zip(scores, scaled_scores):
            #     print(a, b)
            scores = scaled_scores
        elif method == 4:
            idx = np.searchsorted(-scores, -0.001, side='right')
            scores, points = scores[:idx], points[:idx]

            def fn(a):
                return a + a ** 0.8

            scaled_scores = scores.copy()
            for i in range(len(scores)):
                scaled_scores[i] = fn(scaled_scores[i])
            scaled_scores = get_scaled_scores(scaled_scores, np.sum(scores))
            # for a, b in zip(scores, scaled_scores):
            #     print(a, b)
            scores = scaled_scores
        elif method == 5:
            idx = np.searchsorted(-scores, -0.001, side='right')
            scores, points = scores[:idx], points[:idx]

            def fn(a):
                return a + a ** 0.7

            scaled_scores = scores.copy()
            for i in range(len(scores)):
                scaled_scores[i] = fn(scaled_scores[i])
            scaled_scores = get_scaled_scores(scaled_scores, np.sum(scores))
            # print()
            # for a, b in zip(scores, scaled_scores):
            #     print(str(a)[:6], str(b)[:6])
            scores = scaled_scores
        elif NMS_START <= method < NMS_START + len(NMS_LIST):
            threshold = NMS_LIST[method - NMS_START]
            add_eval_param(f'NMS={threshold}')
            # print('threshold', threshold)
            predict = []
            ans_point_scores = []

            def in_predict(predict, point, threshold):
                return np.min(get_dis_point_2_points(point, predict)) < threshold

            for i in range(len(points)):
                if len(predict) > 0 and in_predict(np.array(predict), points[i], threshold):
                    continue
                else:
                    predict.append(points[i])
                    ans_point_scores.append(scores[i])
                    if len(predict) == mode_num:
                        break
            while len(predict) < mode_num:
                i = np.random.randint(0, len(points))
                predict.append(points[i])
                ans_point_scores.append(scores[i])

            idx_in_batch_2_ans_points[idx_in_batch] = np.array(predict)
            idx_in_batch_2_ans_point_scores[idx_in_batch] = np.array(ans_point_scores)
            FDE = np.inf
            if gt_goal is not None:
                for each in predict:
                    FDE = min(FDE, get_dis_point2point(each, gt_goal))
            method2FDEs[method].append(FDE)
        elif DYNAMIC_NMS_START <= method < DYNAMIC_NMS_START + len(DYNAMIC_NMS_LIST):
            threshold = DYNAMIC_NMS_LIST[method - DYNAMIC_NMS_START]
            add_eval_param(f'DY_NMS={threshold}')
            speed_scale_factor = utils_cython.speed_scale_factor(mapping['speed'])
            threshold = threshold * speed_scale_factor

            # print('threshold', threshold)
            predict = []
            ans_point_scores = []

            def in_predict(predict, point, threshold):
                return np.min(get_dis_point_2_points(point, predict)) < threshold

            for i in range(len(points)):
                if len(predict) > 0 and in_predict(np.array(predict), points[i], threshold):
                    continue
                else:
                    predict.append(points[i])
                    ans_point_scores.append(scores[i])
                    if len(predict) == mode_num:
                        break
            while len(predict) < mode_num:
                i = np.random.randint(0, len(points))
                predict.append(points[i])
                ans_point_scores.append(scores[i])

            idx_in_batch_2_ans_points[idx_in_batch] = np.array(predict)
            idx_in_batch_2_ans_point_scores[idx_in_batch] = np.array(ans_point_scores)
            FDE = np.inf
            if gt_goal is not None:
                for each in predict:
                    FDE = min(FDE, get_dis_point2point(each, gt_goal))
            method2FDEs[method].append(FDE)
            pass
        else:
            assert False

        if method < 6:
            # Note 'method > 0' in train.py
            with open(os.path.join(args.temp_file_dir, time_begin, "cpp_input" + str(idx_in_batch)), "w") as fout:
                print(len(points), file=fout)
                for point, score in zip(points, scores):
                    print(point[0], point[1], score, file=fout)


def get_subdivide_points(polygon, include_self=False, threshold=1.0, include_beside=False, return_unit_vectors=False):
    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    # Compute average distance between points
    average_dis = 0
    for i, point in enumerate(polygon):
        if i > 0:
            average_dis += get_dis(point, point_pre)
        point_pre = point
    average_dis /= len(polygon) - 1

    points = []
    if return_unit_vectors:
        assert not include_self and not include_beside
        unit_vectors = []
    divide_num = 1
    while average_dis / divide_num > threshold:
        divide_num += 1
    for i, point in enumerate(polygon):
        if i > 0:
            for k in range(1, divide_num):
                def get_kth_point(point_a, point_b, ratio):
                    return (point_a[0] * (1 - ratio) + point_b[0] * ratio,
                            point_a[1] * (1 - ratio) + point_b[1] * ratio)

                points.append(get_kth_point(point_pre, point, k / divide_num))
                if return_unit_vectors:
                    unit_vectors.append(get_unit_vector(point_pre, point))
        if include_self or include_beside:
            points.append(point)
        point_pre = point
    if include_beside:
        points_ = []
        for i, point in enumerate(points):
            if i > 0:
                der_x = point[0] - point_pre[0]
                der_y = point[1] - point_pre[1]
                scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
                der_x *= scale
                der_y *= scale
                der_x, der_y = rotate(der_x, der_y, math.pi / 2)
                for k in range(-2, 3):
                    if k != 0:
                        points_.append((point[0] + k * der_x, point[1] + k * der_y))
                        if i == 1:
                            points_.append((point_pre[0] + k * der_x, point_pre[1] + k * der_y))
            point_pre = point
        points.extend(points_)
    if return_unit_vectors:
        return points, unit_vectors
    return points
    # return points if not return_unit_vectors else points, unit_vectors


def get_one_subdivide_polygon(polygon):
    new_polygon = []
    for i, point in enumerate(polygon):
        if i > 0:
            new_polygon.append((polygon[i - 1] + polygon[i]) / 2)
        new_polygon.append(point)
    return new_polygon


def get_subdivide_polygons(polygon, threshold=2.0):
    if len(polygon) == 1:
        polygon = [polygon[0], polygon[0]]
    elif len(polygon) % 2 == 1:
        polygon = list(polygon)
        polygon = polygon[:len(polygon) // 2] + polygon[-(len(polygon) // 2):]
    assert_(len(polygon) >= 2)

    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def get_average_dis(polygon):
        average_dis = 0
        for i, point in enumerate(polygon):
            if i > 0:
                average_dis += get_dis(point, point_pre)
            point_pre = point
        average_dis /= len(polygon) - 1
        return average_dis

    average_dis = get_average_dis(polygon)

    if average_dis > threshold:
        length = len(polygon)
        point_a = polygon[length // 2 - 1]
        point_b = polygon[length // 2]
        point_mid = (point_a + point_b) / 2
        polygon_a = polygon[:length // 2]
        polygon_a = get_one_subdivide_polygon(polygon_a)
        polygon_a = polygon_a + [point_mid]
        polygon_b = polygon[length // 2:]
        polygon_b = get_one_subdivide_polygon(polygon_b)
        polygon_b = [point_mid] + polygon_b
        assert_(len(polygon) == len(polygon_a))
        # print('polygon', np.array(polygon), 'polygon_a',np.array(polygon_a), average_dis, get_average_dis(polygon_a))
        return get_subdivide_polygons(polygon_a) + get_subdivide_polygons(polygon_b)
    else:
        return [polygon]


method2FDEs = defaultdict(list)


def get_neighbour_points(points, topk_ids=None, mapping=None, neighbour_dis=2):
    # grid = np.zeros([300, 300], dtype=int)
    grid = {}
    for fake_idx, point in enumerate(points):
        x, y = round(float(point[0])), round(float(point[1]))

        # not compatible argo
        for i in range(-neighbour_dis, neighbour_dis + 1):
            for j in range(-neighbour_dis, neighbour_dis + 1):
                grid[(x + i, y + j)] = 1
    points = list(grid.keys())
    return points


def get_neighbour_points_new(points, neighbour_dis=2, density=1.0):
    grid = {}

    for fake_idx, point in enumerate(points):
        x, y = round(float(point[0])), round(float(point[1]))
        if -100 <= x <= 100 and -100 <= y <= 100:
            i = x - neighbour_dis
            while i < x + neighbour_dis + eps:
                j = y - neighbour_dis
                while j < y + neighbour_dis + eps:
                    grid[(i, j)] = True
                    j += density
                i += density
    points = list(grid.keys())
    points = get_points_remove_repeated(points, density)
    return points


def get_neighbour_points_for_lanes(polygons):
    points = []
    for polygon in polygons:
        points.extend(polygon)
    return get_neighbour_points(points)


def calc_bitmap(bitmap, polygon):
    for point_idx, point in enumerate(polygon):
        if point_idx > 0:
            walk_bitmap(bitmap, point_pre, point, calc_bitmap=True)
        point_pre = point
    pass


def walk_bitmap(bitmap, point_a, point_b, calc_bitmap=False, check_bitmap=False):
    point_a = (round(float(point_a[0])) + 150, round(float(point_a[1])) + 150)
    point_b = (round(float(point_b[0])) + 150, round(float(point_b[1])) + 150)
    xs = [0, 0, 1, -1]
    ys = [1, -1, 0, 0]
    while True:
        if 0 <= point_a[0] < 300 and 0 <= point_a[1] < 300:
            if calc_bitmap:
                bitmap[point_a[0]][point_a[1]] = 1
            if check_bitmap:
                if bitmap[point_a[0]][point_a[1]]:
                    return True
        if point_a == point_b:
            break
        min_dis = np.inf
        arg_min = None
        for tx, ty in zip(xs, ys):
            x, y = point_a[0] + tx, point_a[1] + ty
            dis = np.sqrt((x - point_b[0]) ** 2 + (y - point_b[1]) ** 2)
            if dis < min_dis:
                min_dis = dis
                arg_min = (x, y)
        point_a = arg_min
    return False


def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)


idx_in_batch_2_ans_points = {}
idx_in_batch_2_ans_point_scores = {}


def run_process_todo(queue, queue_res, speed=None, eval_time=None):
    id = np.random.randint(5)
    print('in run_process_todo', get_time(), id)


def run_process(queue, queue_res, args):
    id = np.random.randint(5)
    utils_cython.args = args
    objective = 'MR'
    if 'MRminFDE' in args.other_params:
        objective = 'MRminFDE'
    opti_time = float(args.other_params.get('opti_time', 10000.0))

    li = []
    while True:
        # print('a', round(time.time() - start_time, 2))
        value = queue.get()
        if value is None:
            break
        idx_in_batch, file_name, (goals_2D, scores), kwargs = value
        scores = np.exp(scores)
        if file_name == 'test_obs/data/33670.csv':
            print('aaa', len(scores), np.sum(scores), scores, goals_2D)

        if 'MRminFDE' in args.other_params:
            assert 'cnt_sample' in args.other_params
            MRratio = float(args.other_params['MRminFDE']) if args.other_params['MRminFDE'] is not True else 1.0

        start_time = time.time()

        if 'cnt_sample' in args.other_params:
            num_step = 1000
            kwargs.update(dict(
                num_step=num_step,
                cnt_sample=args.other_params['cnt_sample'],
                MRratio=MRratio,
            ))
            assert args.other_params['cnt_sample'] > 1

        results = utils_cython.get_optimal_targets(goals_2D, scores, file_name, objective, opti_time, kwargs=kwargs)

        li.append(round(time.time() - start_time, 2))

        expectation, ans_points, pred_probs = results
        queue_res.put((idx_in_batch, expectation, ans_points, pred_probs))
    pass

    print('out run_process', get_time(), id)


def select_goals_by_optimization(batch_gt_points, mapping, close=False):
    this = select_goals_by_optimization
    if not hasattr(this, 'processes'):
        # if end:
        #     return
        queue = multiprocessing.Queue(args.core_num)
        queue_res = multiprocessing.Queue()
        processes = [
            Process(target=run_process, args=(queue, queue_res, args,))
            for _ in range(args.core_num)]
        for each in processes:
            each.start()
        this.processes = processes
        this.queue = queue
        this.queue_res = queue_res

    queue = this.queue
    queue_res = this.queue_res

    if close:
        for i in range(args.core_num):
            queue.put(None)
        for each in select_goals_by_optimization.processes:
            each.join()
        return

    start_time = time.time()
    batch_size, future_frame_num, _ = batch_gt_points.shape

    batch_file_name = get_from_mapping(mapping, 'file_name')

    assert args.core_num >= 2

    run_times = 8
    for _ in range(run_times):
        for i in range(batch_size):
            kwargs = {}
            pass

            queue.put((i, batch_file_name[i], mapping[i]['goals_2D_scores'], kwargs))

    while not queue.empty():
        pass

    expectations = np.ones(batch_size) * 10000.0
    batch_ans_points = np.zeros([batch_size, args.mode_num, 2])
    batch_pred_probs = np.zeros([batch_size, args.mode_num])
    for _ in range(run_times * batch_size):
        i, expectation, ans_points, pred_probs = queue_res.get()
        if expectation < expectations[i]:
            expectations[i] = expectation
            batch_ans_points[i] = ans_points
            batch_pred_probs[i] = pred_probs

    # print('here', round(time.time() - start_time, 2))

    for i in range(batch_size):
        FDE = np.inf
        if not args.do_test:
            FDE = np.min(get_dis_point_2_points(batch_gt_points[i][-1], batch_ans_points[i]))
        method2FDEs[0].append(FDE)

        ans_points = batch_ans_points[i].copy()
        if args.argoverse:
            to_origin_coordinate(ans_points, i)

    return batch_ans_points, batch_pred_probs


def to_origin_coordinate(points, idx_in_batch, scale=None):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                    point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])
        if scale is not None:
            point[0] *= scale
            point[1] *= scale


def to_relative_coordinate(points, x, y, angle):
    for point in points:
        point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)


def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


time_begin = get_time()


def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename, sys._getframe().f_back.f_lineno)
    assert satisfied


def get_miss_rate(li_FDE, dis=2.0):
    return np.sum(np.array(li_FDE) > dis) / len(li_FDE) if len(li_FDE) > 0 else None


def ids_to_matrix(ids_list: List[List[int]], size, device):
    tensor = torch.zeros([len(ids_list), size], device=device)
    for idx, each_list in enumerate(ids_list):
        if len(each_list) == 0:
            continue
        tensor[idx].scatter_(0, torch.tensor(each_list, device=device), 1.0)
    return tensor


def get_max_hidden(hidden_states: Tensor, pooling_mask: Tensor):
    num_query = pooling_mask.shape[0]
    num_key = pooling_mask.shape[1]
    assert num_key == hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    pooling_mask = (1.0 - pooling_mask) * -10000.0
    hidden_states = hidden_states.unsqueeze(0).expand([num_query, num_key, hidden_size])
    hidden_states = hidden_states + pooling_mask.unsqueeze(2)
    return torch.max(hidden_states, dim=1)[0]


return_values = None


def model_return(*inputs):
    if args.distributed_training:
        global return_values
        return_values = inputs[1:]
        return inputs[0]
    else:
        return inputs


def get_color_text(text, color='red'):
    if color == 'red':
        return "\033[31m" + text + "\033[0m"
    else:
        assert False


other_errors_dict = defaultdict(list)


def other_errors_put(error_type, error):
    other_errors_dict[error_type].append(error)


def other_errors_to_string():
    res = {}
    for each, value in other_errors_dict.items():
        res[each] = np.mean(value)
    return str(res)


def get_points_remove_repeated(points, threshold=1.0):
    grid = {}

    def get_hash_point(point):
        return round(point[0] / threshold), round(point[1] / threshold)

    def get_de_hash_point(point):
        return float(point[0] * threshold), float(point[1] * threshold)

    for each in points:
        grid[get_hash_point(each)] = True
    return [get_de_hash_point(each) for each in list(grid.keys())]


def get_dis_point_2_points(point, points):
    assert points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


def get_dis_point_2_polygons(point, polygons):
    dis = np.zeros(len(polygons))
    for i, each in enumerate(polygons):
        dis[i] = np.min(np.sqrt(np.square(each[:, 0] - point[0]) + np.square(each[:, 1] - point[1])))
    # dis = np.square(polygons[:, :, 0] - point[0]) + np.square(polygons[:, :, 1] - point[1])
    # dis = np.min(dis, axis=-1)
    # dis = np.sqrt(dis)
    return dis


_zip = zip


def zip(*inputs):
    for each in inputs:
        assert len(each) == len(inputs[0])
    return _zip(*inputs)


def zip_enum(*inputs):
    for each in inputs:
        assert len(each) == len(inputs[0])
    return zip(range(len(inputs[0])), *inputs)


def point_in_points(point, points):
    points = np.array(points)
    if points.ndim != 2:
        return False
    dis = get_dis_point_2_points(point, points)
    return np.min(dis) < 1.0 + eps


def get_pseudo_label(predicts, labels, self_cost=None, kwargs={}):
    if self_cost is None:
        self_cost = np.zeros(len(predicts))
    if isinstance(labels, list):
        cost_list = []
        pseudo_label_list = []
        for each in labels:
            pseudo_label, cost, _ = \
                utils_cython.get_pseudo_label(predicts.astype(np.float32), each.astype(np.float32), self_cost.astype(np.float32), kwargs)
            pseudo_label_list.append(pseudo_label)
            cost_list.append(cost)

        argmin = np.argmin(np.array(cost_list))
        return pseudo_label_list[argmin], cost_list[argmin], None
    else:
        return utils_cython.get_pseudo_label(predicts.astype(np.float32), labels.astype(np.float32), self_cost.astype(np.float32), kwargs)


def get_file_name_int(file_name):
    return int(os.path.split(file_name)[1][:-4])


def assign(a, b, n=2):
    if n == 2:
        a[0], a[1] = b[0], b[1]
    else:
        assert False


def my_print(*args):
    print(*args)


i_epoch = None


def get_from_mapping(mapping: List[Dict], key=None):
    if key is None:
        line_context = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        key = line_context.split('=')[0].strip()
    return [each[key] for each in mapping]


ap_list = None


def metric_values_to_string(metric_values, metric_names, metric=None, index=None, append=False):
    if metric_values == None:
        print('metric_values is None')
        return
    lines = []
    for i, m in enumerate(
            ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
        if metric is None or metric == m:
            for j, n in enumerate(metric_names):
                if index is None or index == j:
                    if append and metric_values[i][j] > 0.0:
                        ap_list.append(float(metric_values[i][j]))
                    lines.append('{}/{}: {}'.format(m, n, metric_values[i][j]))
    return '\n'.join(lines)


def pool_forward(rank, queue, result_queue, run):
    while True:
        file = queue.get()
        if file is None:
            break
        result = run(*file)
        result_queue.put(result)


class Pool:
    def __init__(self, core_num, files, run):
        self.core_num = core_num
        self.queue = multiprocessing.Queue(core_num)
        self.result_queue = multiprocessing.Queue(core_num)
        self.processes = [multiprocessing.Process(target=pool_forward, args=(rank, self.queue, self.result_queue, run,)) for rank in
                          range(self.core_num)]
        self.files = files
        for each in self.processes:
            each.start()
        for file in files:
            assert file is not None
            self.queue.put(file)

    def join(self):
        results = []
        for i in range(len(self.files)):
            results.append(self.result_queue.get())

        while not self.queue.empty():
            pass

        for i in range(self.core_num):
            self.queue.put(None)

        for each in self.processes:
            each.join()

        return results


motion_metrics = None
metric_names = None

trajectory_type_2_motion_metrics = {}


def get_trajectory_upsample(inputs: np.ndarray, future_frame_num, future_test_frame_num):
    stride = future_frame_num // future_test_frame_num
    shape_prefix = list(inputs.shape[:-2])
    assert len(shape_prefix) > 0
    inputs = inputs.reshape(-1, future_test_frame_num, future_frame_num)
    outputs = np.zeros(len(inputs), future_frame_num, 2)
    outputs[:, stride - 1::stride, :] = inputs
    outputs = outputs.reshape(*shape_prefix, future_frame_num, 2)
    return outputs


def get_eval_identifier():
    eval_identifier = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15 and '=' in each:
            each = each.split('=')[0]
        if len(each) > 15:
            each = 'long'
        eval_identifier += '.' + str(each)
    eval_identifier = get_name(eval_identifier, append_time=True)
    return eval_identifier


def get_wait5_rank(rank):
    rank = rank + 1
    return rank // 2


# def shape_equal(shape, shape_):
#     if len(shape) != len(shape_):
#         return False

class Normalizer:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.origin = rotate(0.0 - x, 0.0 - y, yaw)

    def __call__(self, points, reverse=False):
        points = np.array(points)
        if points.shape == (2,):
            points.shape = (1, 2)
        assert len(points.shape) <= 3
        if len(points.shape) == 3:
            for each in points:
                each[:] = self.__call__(each, reverse)
        else:
            assert len(points.shape) == 2
            for point in points:
                if reverse:
                    point[0], point[1] = rotate(point[0] - self.origin[0],
                                                point[1] - self.origin[1], -self.yaw)
                else:
                    point[0], point[1] = rotate(point[0] - self.x,
                                                point[1] - self.y, self.yaw)

        return points


def satisfy_one_of(conds, other_params):
    for each in conds:
        if each in other_params:
            return True
    return False


def get_static_var(obj, name, default=None, path=None):
    if not hasattr(obj, name):
        if default is not None:
            value = default
        elif path is not None:
            value = structs.load(path)
        else:
            assert False
        setattr(obj, name, value)
    return getattr(obj, name)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
