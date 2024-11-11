"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import numpy as np
import torch
import os
import torch.nn.functional as F
from inspect import currentframe, getframeinfo
from termcolor import colored

def print_indebugmode(message):
    previous_frame = currentframe().f_back
    (
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = getframeinfo(previous_frame)
    print (colored("[*** 4Debug] ", 'red') + 
        filename + ':' + str(line_number) + ' - ', message)
    
def count_parameters(model):
    valid_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    invalid_n = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return valid_n, invalid_n + valid_n

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def check_dict_k_v(x, x_name):
    print ("\n checking ", x_name)
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            print ("[xx] {} : {}".format(k, v.shape))
        elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
            print ("[xx] {} : lists of # {} tensors, like v[0] = {}".format(
                k, len(v), v[0].shape)
                )
        elif isinstance(v, float):
            print ("[xx] {} : float {}".format(k, v))
        else:
            print ("[xx] {} : other types {}".format(k, v))

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    lines = [l for l in lines if not l.startswith('#')]
    return lines

def write_to_file(txt_fn, files_list):
    with open(txt_fn, 'w') as fw:
        for l in files_list:
            fw.write(l + "\n")
    print("Done! Saved {} names to {}".format(len(files_list), txt_fn))


def kitti_colormap(disparity, maxval=-1):
	"""
	A utility function to reproduce KITTI fake colormap
	Arguments:
	- disparity: numpy float32 array of dimension HxW
	- maxval: maximum disparity value for normalization (if equal to -1, the maximum value in disparity will be used)

	Returns a numpy uint8 array of shape HxWx3.
	"""
	if maxval < 0:
		maxval = np.max(disparity)
                #print ('maxval = %f' % maxval)

	colormap = np.asarray([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],[0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
	weights = np.asarray([8.771929824561404,5.405405405405405,8.771929824561404,5.747126436781609,8.771929824561404,5.405405405405405,8.771929824561404,0])
	cumsum = np.asarray([0,0.114,0.299,0.413,0.587,0.701,0.8859999999999999,0.9999999999999999])

	colored_disp = np.zeros([disparity.shape[0], disparity.shape[1], 3])
	values = np.expand_dims(np.minimum(np.maximum(disparity/maxval, 0.), 1.), -1)
	bins = np.repeat(np.repeat(np.expand_dims(np.expand_dims(cumsum,axis=0),axis=0), disparity.shape[1], axis=1), disparity.shape[0], axis=0)
	diffs = np.where((np.repeat(values, 8, axis=-1) - bins) > 0, -1000, (np.repeat(values, 8, axis=-1) - bins))
	index = np.argmax(diffs, axis=-1)-1

	w = 1-(values[:,:,0]-cumsum[index])*np.asarray(weights)[index]


	colored_disp[:,:,2] = (w*colormap[index][:,:,0] + (1.-w)*colormap[index+1][:,:,0])
	colored_disp[:,:,1] = (w*colormap[index][:,:,1] + (1.-w)*colormap[index+1][:,:,1])
	colored_disp[:,:,0] = (w*colormap[index][:,:,2] + (1.-w)*colormap[index+1][:,:,2])

	return (colored_disp*np.expand_dims((disparity>0),-1)*255).astype(np.uint8)


def monodepth_colormap(
    input_tensor: torch.Tensor, 
    normalize: bool = True, 
    is_channel_first: bool = True
) -> np.ndarray:
    """Applies a colormap to the input tensor and optionally normalizes and transposes it.

    Args:
        input_tensor: A tensor or numpy array to which the colormap will be applied.
        normalize: If True, the input values will be normalized to the range [0, 1].
        is_channel_first: If True, the output will be transposed to match PyTorch's channel-first format.

    Returns:
        A numpy array with the colormap applied, with optional normalization and transposition.
    """
    import matplotlib.pyplot as plt
    _DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.detach().cpu().numpy()

    color = input_tensor
    if normalize:
        max_value = float(color.max())
        min_value = float(color.min())
        range_value = max_value - min_value if max_value != min_value else 1e5
        color = (color - min_value) / range_value

    if color.ndim == 4:
        color = color.transpose([0, 2, 3, 1]) #[B,C,H,W]-->[B,H,W,C]
        color = _DEPTH_COLORMAP(color)
        color = color[:, :, :, 0, :3]
        if is_channel_first:
            color = color.transpose(0, 3, 1, 2) # back to [B,C,H,W]
    elif color.ndim == 3:
        color = _DEPTH_COLORMAP(color)
        color = color[:, :, :, :3]
        if is_channel_first:
            color = color.transpose(0, 3, 1, 2)
    elif color.ndim == 2:
        color = _DEPTH_COLORMAP(color)
        color = color[..., :3]
        if is_channel_first:
            color = color.transpose(2, 0, 1) # [C,H,W]

    return color


def change_sec_to_hm_str(t):
    """Convert time in seconds to a string
    e.g. 10239 sec -> '02h50m39s'
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    h = t // 60
    return f"{h:02d}h{m:02d}m{s:02d}s"

def check_nan_inf(inp, name):
    assert not torch.isnan(inp).any(), \
        "Found Nan in input {}, shape = {}, val = {}".format(
            name, inp.shape, inp)
    assert not torch.isinf(inp).any(), \
        "Found Inf in input {}, shape = {}, val = {}".format(
            name, inp.shape, inp)


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))

@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

def is_empty(x: torch.Tensor) -> bool:
    return x.numel() == 0

# Code was adopted from DeepVideoMVS (https://github.com/ardaduz/deep-video-mvs)
# MIT license.
# GEOMETRIC UTILS
def pose_distance(reference_pose, measurement_pose):
    """
    :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose (not extrinsic matrix!)
    :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world pose (not extrinsic matrix!)
    :return combined_measure: float, combined pose distance measure
    :return R_measure: float, rotation distance measure
    :return t_measure: float, translation distance measure
    """
    rel_pose = np.dot(np.linalg.inv(reference_pose), measurement_pose)
    R = rel_pose[:3, :3]
    t = rel_pose[:3, 3]
    R_measure = np.sqrt(2 * (1 - min(3.0, np.matrix.trace(R)) / 3))
    t_measure = np.linalg.norm(t)
    combined_measure = np.sqrt(t_measure ** 2 + R_measure ** 2)
    return combined_measure, R_measure, t_measure

