import re
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


class DTUDatasetFinetuneNeuS(Dataset):
    def __init__(self, confs, mode):
        super(DTUDatasetFinetuneNeuS, self).__init__()

        self.mode = mode
        self.data_dir = confs['data_dir']
        self.interval_scale = confs.get_float('interval_scale')
        self.num_interval = confs.get_int('num_interval')
        self.img_hw = confs['img_hw']
        self.n_rays = confs.get_int('n_rays')
        self.factor = confs.get_float('factor')
        self.num_views = 3

        self.scene = confs.get_string('scene')
        self.ref_view = confs.get_int('ref_view')
        self.val_res_level = confs.get_int('val_res_level', default=1)
        
        self.pairs = self.get_pairs()
        self.all_views = [self.ref_view] + list(self.pairs[self.ref_view])[:(self.num_views-1)]
        
        print("self.all_views:", self.all_views)
        
        camera_dict = np.load(os.path.join(self.data_dir, 'neus_data/data_DTU/dtu_{}/cameras_sphere.npz'.format(self.scene)))
        self.world_mats_np = [camera_dict['world_mat_%d' % vid].astype(np.float32) for vid in self.all_views]
        self.scale_mats_np = [camera_dict['scale_mat_%d' % vid].astype(np.float32) for vid in self.all_views]
        
        # for i in range(len(self.scale_mats_np)):
        #     self.scale_mats_np[i][:3, :3] *= 1.75
        
        self.intrs = []
        self.c2ws = []
        self.near_fars = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrs.append(torch.from_numpy(intrinsics).float())
            self.c2ws.append(torch.from_numpy(pose).float())
            
            camera_o = pose[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2)).astype(np.float32)
            near = dist - 1
            far = dist + 1
            self.near_fars.append([0.95 * near, 1.05 * far])
            
        self.intrs = torch.stack(self.intrs)
        self.c2ws = torch.stack(self.c2ws)
        self.near_fars = torch.from_numpy(np.stack(self.near_fars).astype(np.float32))
        self.scale_mat = torch.from_numpy(self.scale_mats_np[0].astype(np.float32))
        self.scale_factor = 1.0 / self.scale_mat[0, 0]
        
        self.images_lis = [os.path.join(self.data_dir, 'neus_data/data_DTU/dtu_{}/image/{:0>6}.png'.format(self.scene, vid)) for vid in self.all_views]
        self.masks_lis = [os.path.join(self.data_dir, 'neus_data/data_DTU/dtu_{}/mask/{:0>3}.png'.format(self.scene, vid)) for vid in self.all_views]
        self.images = [np.array(Image.open(im_name), dtype=np.float32) / 256.0 for im_name in self.images_lis]
        self.images = np.stack([cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST) for img in self.images])
        self.images = torch.from_numpy(self.images.astype(np.float32))
        self.masks = [np.array(Image.open(im_name), dtype=np.float32) for im_name in self.masks_lis]
        self.masks = np.stack([(cv2.resize(mask, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST) > 10).astype(np.float32) for mask in self.masks])
        self.masks = torch.from_numpy(self.masks.astype(np.float32))
        
        self.pseudo_depth_lis = [os.path.join(self.data_dir, 'PseudoMVSScore/dtu_exp/{}/filtered_avg_depth/{:0>8}.pfm'.format(self.scene, vid)) for vid in self.all_views]
        self.pseudo_depths = [np.array(read_pfm(filename)[0], dtype=np.float32) for filename in self.pseudo_depth_lis]
        self.pseudo_depths = np.stack([cv2.resize(depth, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST) for depth in self.pseudo_depths])
        self.pseudo_depths = torch.from_numpy(self.pseudo_depths.astype(np.float32)) * self.scale_factor
        
        pcd = PlyData.read(os.path.join(self.data_dir, "PseudoMVSDepth/mvsnet{:0>3}_l3.ply".format(int(self.scene[4:]))))
        px = pcd['vertex']['x']
        py = pcd['vertex']['y']
        pz = pcd['vertex']['z']
        pxyz = np.stack([px, py, pz], axis=1)
        pxyz = torch.from_numpy(pxyz)
        self.pseudo_pts = (pxyz - self.scale_mat[:3, 3][None]) / self.scale_mat[0, 0]

    def get_pairs(self):
        
        pair_file = "Cameras/pair.txt"
        print("Using existing pair file...")
        with open(os.path.join(self.data_dir, pair_file)) as f:
            num_viewpoint = int(f.readline())
            pairs = [[]] * num_viewpoint
            # viewpoints (49)
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                pairs[ref_view] = np.array(src_views[:10])
        pairs = np.array(pairs)

        return pairs
    
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sqrt(torch.sum(rays_d**2, dim=-1, keepdim=True))
        b = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
    
    def get_all_images(self):
        near, far = self.near_fars[0].reshape(1, 2).split(split_size=1, dim=1)
        outputs = {
            "imgs": self.images.permute(0, 3, 1, 2),
            "c2ws": self.c2ws,
            "intrs":self.intrs,
            "near": near,
            "far": far,
            "near_fars": self.near_fars
        }
        return outputs
    
    def get_random_rays(self, vid):
        vid = vid.item()
        
        pixels_x = torch.randint(low=0, high=self.img_hw[1], size=[self.n_rays])
        pixels_y = torch.randint(low=0, high=self.img_hw[0], size=[self.n_rays])
        
        color = self.images[vid][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        mask = self.masks[vid][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(self.intrs[vid].inverse()[None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(self.c2ws[vid, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = self.c2ws[vid, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = self.near_fars[vid].reshape(1, 2).split(split_size=1, dim=1)
        # near, far = self.near_far_from_sphere(rays_o, rays_d)
        
        pseudo_depth = self.pseudo_depths[vid][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        
        random_idx = torch.randint(low=0, high=self.pseudo_pts.shape[0], size=[2048])
        pseudo_pts = self.pseudo_pts[random_idx]
        
        view_ids = [vid] + list(range(self.num_views))[:vid] + list(range(self.num_views))[vid+1:]
        intrs = self.intrs[view_ids]
        c2ws = self.c2ws[view_ids]
        imgs = self.images[view_ids].permute(0, 3, 1, 2)
        
        outputs = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "intrs": intrs,
            "c2ws": c2ws,
            "view_ids": view_ids,
            "imgs": imgs,
            "pseudo_pts": pseudo_pts,
            "pseudo_depth": pseudo_depth
        }
        
        return outputs
    
    def get_rays_at(self, vid):
        tx = torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1] // self.val_res_level)
        ty = torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0] // self.val_res_level)
        pixels_y, pixels_x = torch.meshgrid(ty, tx)
        pixels_x, pixels_y = pixels_x.reshape(-1), pixels_y.reshape(-1)
        
        color = self.images[vid][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        mask = self.masks[vid][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(self.intrs[vid].inverse()[None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(self.c2ws[vid, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = self.c2ws[vid, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = self.near_fars[vid].reshape(1, 2).split(split_size=1, dim=1)
        # near, far = self.near_far_from_sphere(rays_o, rays_d)
        
        view_ids = [vid] + list(range(self.num_views))[:vid] + list(range(self.num_views))[vid+1:]
        intrs = self.intrs[view_ids]
        c2ws = self.c2ws[view_ids]
        imgs = self.images[view_ids].permute(0, 3, 1, 2)
        masks = self.masks[view_ids]
        bound_min=torch.tensor([-1, -1, -1], dtype=torch.float32)
        bound_max=torch.tensor([1, 1, 1], dtype=torch.float32)
        hw = torch.Tensor([self.img_hw[0]//self.val_res_level, self.img_hw[1]//self.val_res_level]).int()
        
        outputs = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "intrs": intrs,
            "c2ws": c2ws,
            "view_ids": view_ids,
            "scale_mat": self.scale_mat,
            "scene": self.scene,
            "imgs": imgs,
            "masks": masks,
            "bound_min": bound_min,
            "bound_max": bound_max,
            "hw": hw
        }
        
        return outputs