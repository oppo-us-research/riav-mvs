"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import cv2
import numpy as np
from datetime import datetime
import os
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
from PIL import Image  # using pillow-simd for increased speed
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_errors(gt_input, pred_input):
    """Computation of error metrics between predicted 
       and ground truth depths
    """
    gt = gt_input.copy().astype(np.float32)
    pred = pred_input.copy().astype(np.float32)
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def save_metrics_to_csv_file(mean_errors, csv_file, model_name,
    dir_you_specifiy, bad_x_thred_list, 
    mean_median_aligned_errors = None, 
    your_note = None):

    """ save as csv file, Excel file format """
    timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    print ("write ", csv_file, "\n")
    # assume the result dir is in this format:
    # "./results/mvs-pytorch-vkt2-val-D192-epo20/depth-epo-001"
    # we want to extract the last few dirs: i.e., "mvs-pytorch-vkt2-val-D192-epo20/depth-epo-001"
    #tmp_pos = self.args.outdir.find("/results")
    #tmp_dir = self.args.outdir[tmp_pos+1+len("/results"):]
    tmp_dir = dir_you_specifiy
    epo_info_pos = tmp_dir.find('depth-epo-')
    epo_info = tmp_dir[epo_info_pos:epo_info_pos+len('depth-epo-0xx')]
    #print ("[???] tmp_dir = ", tmp_dir)
    messg = timeStamp + ",method={},resultDir,{}".format(model_name, tmp_dir) + \
        (",{},{:.6f}"*9).format(
        "abs", mean_errors[0],
        "abs_inv", mean_errors[1],
        "abs_rel", mean_errors[2],
        "sq_rel", mean_errors[3],
        "rmse", mean_errors[4],
        "rmse_log", mean_errors[5],
        "a1(%)", mean_errors[6]*100.0,
        "a2(%)", mean_errors[7]*100.0,
        "a3(%)", mean_errors[8]*100.0
        )

    num_bad_x = len(bad_x_thred_list)
    for i in range(num_bad_x):
        messg += (",bad-{:.1f}(%),{:.6f}").format(bad_x_thred_list[i], 100.0*mean_errors[9+i] )
    
    messg += ",all for log," + \
        ("{:.6f},"*6).format(* mean_errors[0:6].tolist()) + \
        "{:.3f},{:.3f},{:.3f},".format(mean_errors[6]*100.0, mean_errors[7]*100.0, mean_errors[8]*100.0) + \
        ("{:.3f},"*3).format( *(100.0*mean_errors[9:9+num_bad_x]) )
    
    if mean_median_aligned_errors is not None:
        messg += ",all for median scaled," + \
            ("{:.6f},"*6).format(* mean_median_aligned_errors[0:6].tolist()) + \
            "{:.3f},{:.3f},{:.3f},".format(*mean_median_aligned_errors[6:9]*100.0) + \
            ("{:.3f},"*3).format( *(100.0*mean_median_aligned_errors[9:9+num_bad_x]) )
    
    messg += ",{}".format(epo_info)

    if your_note is not None and your_note != '':
        messg += ",{}\n".format(your_note)
    else:
        messg += "\n"

    with open( csv_file, 'w') as fwrite:
        fwrite.write(messg)


def compute_errors_v2(gt_input, pred_input, min_depth, max_depth, bad_x_thred = [1.0, 2.0, 3.0]):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    valid = (gt_input >= min_depth) & (gt_input <= max_depth)
    gt = gt_input.copy()[valid].astype(np.float32)
    pred = pred_input.copy()[valid].astype(np.float32)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # newly added metric
    diff = gt - pred
    abs_diff = np.abs(diff)
    abs_error = np.mean(abs_diff)
    abs_inverse_error = np.mean(np.abs(1.0 / gt - 1.0 / pred))

    # newly added metric
    bad_x_errs = []
    for bad_thre in bad_x_thred:
        bad_x_err = (abs_diff >= bad_thre).mean()
        bad_x_errs.append(bad_x_err)


    rmse = diff ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(abs_diff / gt)

    sq_rel = np.mean((diff ** 2) / gt)

    res = [abs_error, abs_inverse_error, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3] + bad_x_errs
    return res

# torch version
@torch.no_grad()
def compute_errors_batch(
    img_global_id, # used to find image path;
    groundtruth, #[N,1,H,W]
    prediction, #[N,1,H,W]
    min_depth, max_depth,
    bad_x_thred = [1.0, 2.0, 3.0],
    is_median_scaling = False
    ):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    assert prediction.size(1) == 1 and groundtruth.size(1) == 1
    assert prediction.size() == groundtruth.size() 
    batch = groundtruth.size(0)

    bad_x_num = len(bad_x_thred)
    errors = []
    meds = []
    for b_idx in range(batch):
        #reshapes a tensor t with one element to a scalar;
        img_id = torch.reshape(img_global_id[b_idx], []) 
        gt = groundtruth.detach().clone()[b_idx, 0] #[H, W]
        pred = prediction.detach().clone()[b_idx, 0] # [H, W]
        valid_mask = (gt >= min_depth) & (gt <= max_depth)
        valid_count = valid_mask.float().sum()
        # only evluate the valid region
        pred = pred[valid_mask]
        gt = gt[valid_mask]

        if is_median_scaling:
            med_gt = torch.median(gt)
            med_pred =  torch.median(pred)
            ratio = med_gt / med_pred
            meds.append(torch.stack([ratio, med_gt, med_pred], dim=-1))
            pred = pred *ratio
        
        thresh, _ = torch.max(torch.stack([gt/pred, pred/gt], dim=0), dim=0)
        a1 = (thresh < 1.25     ).float().sum() / valid_count
        a2 = (thresh < 1.25 ** 2).float().sum() / valid_count
        a3 = (thresh < 1.25 ** 3).float().sum() / valid_count

        diff = gt - pred
        abs_diff = diff.abs()
        abs_error = abs_diff.sum() / valid_count
        abs_inverse_error = (1.0 / gt - 1.0 / pred).abs().sum() / valid_count

        rmse = torch.sqrt( (diff ** 2).sum()/ valid_count )

        rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
        rmse_log = torch.sqrt(rmse_log.sum()/ valid_count)

        abs_rel = (abs_diff / gt).sum() / valid_count

        sq_rel = ((diff ** 2) / gt).sum() / valid_count
        
        batch_err = [
            abs_error, abs_inverse_error, abs_rel, sq_rel, rmse, rmse_log, 
            a1, a2, a3]

        # newly added metric
        for bad_thre in bad_x_thred:
            bad_x_err = (abs_diff >= bad_thre).float().sum()/ valid_count
            batch_err.append(bad_x_err)
        
        batch_err.append(img_id)
        batch_err = torch.stack(batch_err, dim=-1) #[13], 12 metrics + 1 img-id;
        #if b_idx == 0:
        #    print (f"[???] batch {b_idx}: img_id: {img_id}, batch_err size : {batch_err.shape}")
        errors.append(batch_err)
    
    #errors = torch.stack(errors, dim=0).mean(dim=0) # along batch dim
    errors = torch.stack(errors, dim=0) # along batch dim
    #print (f"[???] res size : {errors.shape}")
    if is_median_scaling:
        meds = torch.stack(meds, dim=0)
        #print (f"[???] meds :  {meds.shape}")
        
        return errors, meds
    else:
        return errors

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

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


def manydepth_colormap(depth, maxval=-1, is_return_np = False):
    toplot = depth.squeeze()
    _vmin=toplot.min()
    if maxval < 0:
        _vmax=np.percentile(toplot, 95)
    else:
        _vmax = maxval
    #print ("{} : vmin = {}, vmax = {}".format(plot_name, _vmin, _vmax))
    normalizer = mpl.colors.Normalize(vmin = _vmin, vmax = _vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
    if is_return_np:
        return colormapped_im
    else:
        im = pil.fromarray(colormapped_im)
        return im

def error_colormap(depth_err, maxval=-1, is_return_np = False, cmap = 'jet'):
    toplot = depth_err.squeeze()
    _vmin=toplot.min()
    if maxval < 0:
        _vmax=np.percentile(toplot, 95)
    else:
        _vmax = maxval
    #print ("{} : vmin = {}, vmax = {}".format(plot_name, _vmin, _vmax))
    normalizer = mpl.colors.Normalize(vmin = _vmin, vmax = _vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap= cmap)
    colormapped_im = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)
    if is_return_np:
        return colormapped_im
    else:
        im = pil.fromarray(colormapped_im)
        return im