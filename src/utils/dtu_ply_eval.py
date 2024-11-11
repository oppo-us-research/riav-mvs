#----------------------------
# Modified from DTUeval-python (https://github.com/jzhangbs/DTUeval-python/blob/master/eval.py)
# MIT license
#----------------------------

import os
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
#import argparse
from path import Path
from datetime import datetime
import time
import cv2
import json


class DTU_ply_eval(object):
    def __init__(self, args, ply_eval_name=None):
        super(DTU_ply_eval, self).__init__()
        self.args = args
        
        # Can tune those parameters 
        self.args.downsample_density = 0.2
        self.args.patch_size = 60
        self.args.max_dist = 20
        self.args.visualize_threshold = 10
        self.args.mode = 'pcd' # or 'mesh'

        self.dtu_test_sets_list = [
                1, 4, 9, 10, 11, 12, 13, 15, 
                23, 24, 29, 
                32, 33, 34, 48, 49, 62, 75, 77, 
                110, 114, 118
            ]
        if ply_eval_name is None:
            self.result_dir = os.path.join(args.outdir, "ply_eval")
        else:
            self.result_dir = os.path.join(args.outdir, ply_eval_name)
        print (f"Saving ply_eval to {self.result_dir}")
        self.njobs = args.num_workers
        self.method = args.method
        print ("self.njobs = ", self.njobs)
        print ("dtu ply dir: ", self.result_dir)
        self.vis_out_dir = self.result_dir
        os.makedirs(self.result_dir, exist_ok=True)


    def sample_single_tri(self, input_):
        n1, n2, v1, v2, tri_vert = input_
        c = np.mgrid[:n1+1, :n2+1]
        c += 0.5
        c[0] /= max(n1, 1e-7)
        c[1] /= max(n2, 1e-7)
        c = np.transpose(c, (1,2,0))
        k = c[c.sum(axis=-1) < 1]  # m2
        q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
        return q

    def write_vis_pcd(self, file, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(file, pcd)

    def eval_per_scan(self, file_name, scan):
        thresh = self.args.downsample_density
        assert file_name.endswith(f"{scan:03}.ply"), "name scan mismatch"
        if self.args.mode == 'mesh':
            pbar = tqdm(total=9)
            pbar.set_description('read data mesh')
            data_mesh = o3d.io.read_triangle_mesh(file_name)

            vertices = np.asarray(data_mesh.vertices)
            triangles = np.asarray(data_mesh.triangles)
            tri_vert = vertices[triangles]

            pbar.update(1)
            pbar.set_description('sample pcd from mesh')
            v1 = tri_vert[:,1] - tri_vert[:,0]
            v2 = tri_vert[:,2] - tri_vert[:,0]
            l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
            l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
            area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
            non_zero_area = (area2 > 0)[:,0]
            l1, l2, area2, v1, v2, tri_vert = [
                arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
            ]
            thr = thresh * np.sqrt(l1 * l2 / area2)
            n1 = np.floor(l1 / thr)
            n2 = np.floor(l2 / thr)

            with mp.Pool() as mp_pool:
                new_pts = mp_pool.map(
                    self.sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

            new_pts = np.concatenate(new_pts, axis=0)
            data_pcd = np.concatenate([vertices, new_pts], axis=0)
        
        elif self.args.mode == 'pcd':
            pbar = tqdm(total=8)
            pbar.set_description('read data pcd')
            data_pcd_o3d = o3d.io.read_point_cloud(file_name)
            data_pcd = np.asarray(data_pcd_o3d.points)

        pbar.update(1)
        pbar.set_description('random shuffle pcd index')
        shuffle_rng = np.random.default_rng()
        shuffle_rng.shuffle(data_pcd, axis=0)

        pbar.update(1)
        pbar.set_description('downsample pcd')
        nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=self.njobs)
        nn_engine.fit(data_pcd)
        rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
        mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
        for curr, idxs in enumerate(rnn_idxs):
            if mask[curr]:
                mask[idxs] = 0
                mask[curr] = 1
        data_down = data_pcd[mask]

        pbar.update(1)
        pbar.set_description('masking data pcd')
        obs_mask_file = loadmat(f'{self.args.ply_data_path}/ObsMask/ObsMask{scan}_10.mat')
        ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
        BB = BB.astype(np.float32)

        patch = self.args.patch_size
        inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
        data_in = data_down[inbound]

        data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
        grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
        data_grid_in = data_grid[grid_inbound]
        in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
        data_in_obs = data_in[grid_inbound][in_obs]

        pbar.update(1)
        pbar.set_description('read STL pcd')
        stl_pcd = o3d.io.read_point_cloud(f'{self.args.ply_data_path}/Points/stl/stl{scan:03}_total.ply')
        stl = np.asarray(stl_pcd.points)

        pbar.update(1)
        pbar.set_description('compute data2stl')
        nn_engine.fit(stl)
        dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
        max_dist = self.args.max_dist
        mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

        pbar.update(1)
        pbar.set_description('compute stl2data')
        ground_plane = loadmat(f'{self.args.ply_data_path}/ObsMask/Plane{scan}.mat')['P']

        stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
        above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
        stl_above = stl[above]

        nn_engine.fit(data_in)
        dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
        mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

        pbar.update(1)
        pbar.set_description('visualize error')
        vis_dist = self.args.visualize_threshold
        R = np.array([[1,0,0]], dtype=np.float64)
        G = np.array([[0,1,0]], dtype=np.float64)
        B = np.array([[0,0,1]], dtype=np.float64)
        W = np.array([[1,1,1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
        data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
        self.write_vis_pcd(f'{self.vis_out_dir}/vis_{scan:03}_d2s.ply', data_down, data_color)
        stl_color = np.tile(B, (stl.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
        stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
        self.write_vis_pcd(f'{self.vis_out_dir}/vis_{scan:03}_s2d.ply', stl, stl_color)

        pbar.update(1)
        pbar.set_description('done')
        pbar.close()
        over_all = (mean_d2s + mean_s2d) / 2
        print(f"Acc.(mm)={mean_d2s:.4f}, Comp.(mm)={mean_s2d:.4f}, Overall(mm)={over_all:.4f}")
        #print(mean_d2s, mean_s2d, over_all)
        return mean_d2s, mean_s2d, over_all
    
    def save_metrics_to_csv_file(self, mean_errors, csv_file):
        """ save as csv file, Excel file format """
        timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        tmp_dir = self.result_dir
        messg = timeStamp + ",method={},resultDir,{}".format(self.method, tmp_dir)
        messg += ",Acc.(mm),{:.4f},Comp.(mm),{:.4f},Overall(mm),{:.4f}".format(*mean_errors)
        with open( csv_file, 'w') as fwrite:
            fwrite.write(messg + "\n")
        print ("Done! Write ", csv_file, "\n")
    
    def __call__(self, scans_to_do = None):
        errors = []
        if scans_to_do is None:
            scans_to_do = self.dtu_test_sets_list
        num_scans = len(scans_to_do)
        res_dict = {
            'method': self.method,
            'dataset': self.args.dataset,
            'input_ply_dir': self.args.ply_data_path,
            'timeStamp': datetime.now().strftime('%Y-%m-%d_%H:%M:%S'),
            "metric_info": f"accuracy (mm) / completeness (mm) / Overrall (mm). But for DTU, lower is better",
            "metrics": {}
            }
        
        dst_csv_file = os.path.join(self.args.csv_dir, f'mvs-ply-eval-perscan-{self.args.machine_name}.csv')
        for idx, scan in enumerate(scans_to_do):
            #file_name = Path(self.args.outdir)/ f"{scan:03d}.ply"
            file_name = Path(self.result_dir)/ f"{scan:03d}.ply"
            print ("processing {}/{}: scan = {}".format(idx+1, num_scans, file_name))
            # err: accuracy(mm), completeness(mm), overall(mm)
            err = self.eval_per_scan(file_name = file_name, scan = scan)
            errors.append(err)
            res_dict["metrics"][scan] = list(err)

            # save each scan error, just in case the code crashes; 
            csv_file = os.path.join(self.result_dir, f"err_{scan:03d}.csv")
            with open( csv_file, 'w') as fwrite:
                fwrite.write(f"{self.result_dir},scan{scan:03d},{err[0]:.5f},{err[1]:.5f},{err[2]:.5f}\n")
            os.system(f'cat {csv_file} >> {dst_csv_file}')

        errors = np.array(errors)
        
        
        mean_errors = np.nanmean(errors, axis=0)
        message = "mean_errors: acc. / comp. / overall = & {:.4f} & {:.4f} & {:.4f}".format(*mean_errors)
        print (message)
        res_dict["metrics_avg"] = message
        json_file = os.path.join(self.result_dir, 'mvs-ply-eval.json')
        with open(json_file, 'w') as f:
            json.dump(res_dict, f, indent=2)
            print (f"Just saved mvs eval metric to {json_file}")
        
        csv_file = os.path.join(self.result_dir, "mvs-ply-eval.csv")
        self.save_metrics_to_csv_file(mean_errors, csv_file)
        
        dst_csv_file = os.path.join(self.args.csv_dir, f'mvs-ply-eval-{self.args.machine_name}.csv')
        os.system(f'cat {csv_file} >> {dst_csv_file}')
        print (f"cat {csv_file} to {dst_csv_file}")


