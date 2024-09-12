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
import lmdb


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


def load_and_decompress(stream):
    import pickle
    import lz4.frame as lz4f

    stream = lz4f.decompress(stream)
    obj = pickle.loads(stream)
    return obj


class DTUDataset(Dataset):
    def __init__(self, confs, mode):
        super(DTUDataset, self).__init__()

        self.mode = mode
        self.data_dir = confs['data_dir']
        self.num_src_view = confs.get_int('num_src_view')
        self.interval_scale = confs.get_float('interval_scale')
        self.num_interval = confs.get_int('num_interval')
        self.img_hw = confs['img_hw']
        self.n_rays = confs.get_int('n_rays', 0)
        self.factor = confs.get_float('factor')
        self.total_views = 49

        self.split = confs.get_string("split", default=None)
        self.scene = confs.get_list('scene', default=None)
        self.light_idx = confs.get_list('light_idx', default=None)
        self.ref_view = confs.get_list('ref_view', default=None)
        if mode == "val":
            self.val_res_level = confs.get_int('val_res_level', default=1)
        
        if self.scene is None:
            if self.split is not None:
                with open(self.split) as f:
                    scans = f.readlines()
                    self.scene = [line.rstrip() for line in scans]
            else:
                raise ValueError("There are no scenes!")

        # self.intrs, self.w2cs, self.near_fars = self.read_cam_info()
        self.pairs = self.get_pairs()
        self.metas = self.build_list()
        
        # self.pxyz_oris = {}
        # for scan in self.scene:
        #     pcd = PlyData.read(os.path.join(self.data_dir, "PseudoMVSDepth2/mvsnet{:0>3}_l3.ply".format(int(scan[4:]))))
        #     px = pcd['vertex']['x']
        #     py = pcd['vertex']['y']
        #     pz = pcd['vertex']['z']
        #     pxyz_ori = np.stack([px, py, pz], axis=1)
        #     self.pxyz_oris[scan] = torch.from_numpy(pxyz_ori).contiguous()
        
        # self.env = lmdb.open(os.path.join(self.data_dir, "PseudoMVSDepth/lmdb"), readonly=True, lock=False)
        # self.txn = self.env.begin()
        # # self.db_keys = load_and_decompress(self.txn.get(("__point_keys_list__").encode()))

    def get_pairs(self, num_select=10):
        
        pair_file = "Cameras/pair.txt"
        if os.path.exists(os.path.join(self.data_dir, pair_file)):
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
        else:
            print("Calculating pair...")
            w2cs = np.stack(self.w2cs, axis=0)
            c2ws = np.linalg.inv(w2cs)
            dists = np.linalg.norm(c2ws[:, None, :3, 3] - c2ws[None, :, :3, 3], axis=-1)
            eyes = np.eye(dists.shape[0])
            dists[eyes>0] = 1e3
            sorted_vids = np.argsort(dists, axis=1)
            pairs = sorted_vids[:, :num_select]

        return pairs

    def build_list(self):
        metas = []
        
        light_idxs = range(7)
        if self.light_idx is not None:
            light_idxs = self.light_idx

        # scans
        for scan in self.scene:
            # num_viewpoint = len(os.listdir(os.path.join(self.data_dir, 'Rectified/{}_train/'.format(scan)))) // 7
            num_viewpoint = self.total_views

            all_ref_views = [i for i in range(num_viewpoint)] if self.ref_view is None else self.ref_view

            for ref_view in all_ref_views:
                # pairs = list(self.pairs[ref_view])
                # src_views = pairs[:min(self.num_src_view, len(pairs))]
                
                # light conditions 0-6
                for light_idx in light_idxs:
                    metas.append((scan, light_idx, ref_view))
                        
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def read_cam(self, filename):

        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # intrinsics[:2] *= 4
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        depth_max = depth_min + depth_interval * self.num_interval

        intrinsics_[0] *= self.img_hw[1] / 1600
        intrinsics_[1] *= self.img_hw[0] / 1200
        
        return intrinsics_, extrinsics, [depth_min, depth_max]

    def get_scale_mat(self, img_hw, intrs, w2cs, near_fars, factor=0.8):
        bnds = np.zeros((3, 2))
        bnds[:, 0] = np.inf
        bnds[:, 1] = -np.inf
        im_h, im_w = img_hw

        for intr, w2c, near_far in zip(intrs, w2cs, near_fars):
            min_depth, max_depth = near_far

            view_frust_pts = np.stack([
                (np.array([0, 0, im_w, im_w, 0, 0, im_w, im_w]) - intr[0, 2]) * np.array(
                    [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]) / intr[0, 0],
                (np.array([0, im_h, 0, im_h, 0, im_h, 0, im_h]) - intr[1, 2]) * np.array(
                    [min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth]) / intr[1, 1],
                np.array([min_depth, min_depth, min_depth, min_depth, max_depth, max_depth, max_depth, max_depth])
            ])
            view_frust_pts = view_frust_pts.astype(np.float32)
            view_frust_pts = np.linalg.inv(w2c) @ np.concatenate([view_frust_pts, np.ones_like(view_frust_pts[:1])], axis=0)
            view_frust_pts = view_frust_pts[:3]

            bnds[:, 0] = np.minimum(bnds[:, 0], view_frust_pts.min(axis=1))
            bnds[:, 1] = np.maximum(bnds[:, 1], view_frust_pts.max(axis=1))
        
        center = np.array(((bnds[0, 1] + bnds[0, 0]) / 2, (bnds[1, 1] + bnds[1, 0]) / 2,
                           (bnds[2, 1] + bnds[2, 0]) / 2)).astype(np.float32)

        lengths = bnds[:, 1] - bnds[:, 0]

        max_length = lengths.max(axis=0)
        radius = max_length / 2

        radius = radius * factor

        scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
        scale_mat[:3, 3] = center

        return scale_mat, 1. / radius
    
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sqrt(torch.sum(rays_d**2, dim=-1, keepdim=True))
        b = torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def read_img(self, filename):
        # 1200, 1600
        img = np.array(Image.open(filename), dtype=np.float32)
        # # 600 800
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # img = img[44:556, 80:720]  

        img = cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        return img
    
    def read_numpy(self, filename):
        # 1200, 1600
        img = np.load(filename).astype(np.float32)
        # # 600 800
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # img = img[44:556, 80:720]  

        img = cv2.resize(img, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)
        return img
    
    def read_depth(self, filename):
        # 1200, 1600
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # # 600 800
        # depth = cv2.resize(depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        # # 512, 640
        # depth = depth[44:556, 80:720]  

        depth = cv2.resize(depth, self.img_hw[::-1], interpolation=cv2.INTER_NEAREST)

        return depth

    def __getitem__(self, idx):
        scan, light_idx, ref_view = self.metas[idx]
        pairs = list(self.pairs[ref_view])
        if self.mode == "train":
            # src_views = random.sample(pairs[:6], min(self.num_src_view, len(pairs)))
            src_views = pairs[:min(self.num_src_view, len(pairs))]
        else:
            src_views = pairs[:min(self.num_src_view, len(pairs))]
        view_ids = [ref_view] + src_views
        
        imgs = []
        intrs = []
        w2cs = []
        near_fars = []
        masks = []
        # depths = []
        # pseudo_mvs_depths = []
        
        src_idx = np.random.randint(1, len(view_ids))
        
        for i, vid in enumerate(view_ids):
            if vid > 48:
                img_filename = os.path.join(self.data_dir, 'Rectified_raw/{}/rect_{:0>3}_{}_r7000.png'.format(scan, vid + 1, light_idx))
            else:
                img_filename = os.path.join(self.data_dir, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
                
            depth_filename = os.path.join(self.data_dir, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            
            pseudo_mvs_depth_filename = os.path.join(self.data_dir, 'Pseudo_depths/{}/{:0>8}.pfm'.format(scan, vid))
            
            mask_filename = os.path.join(self.data_dir, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            
            cam_file = os.path.join(self.data_dir, 'Cameras/{:0>8}_cam.txt').format(vid)
            
            img = self.read_img(img_filename) / 256.0
            intr, w2c, near_far = self.read_cam(cam_file)
            mask = self.read_img(mask_filename)
            mask = (mask > 10).astype(np.float32)

            near_fars.append(near_far)
            masks.append(mask)
            imgs.append(img)
            intrs.append(intr)
            w2cs.append(w2c)
            
            if i==0:
                ref_depth = self.read_depth(depth_filename)
                ref_pseudo_mvs_depth = self.read_depth(pseudo_mvs_depth_filename)
                
            if i==src_idx:
                src_depth = self.read_depth(depth_filename)
                src_pseudo_mvs_depth = self.read_depth(pseudo_mvs_depth_filename)

        w2c_ref = w2cs[0]
        w2c_ref_inv = np.linalg.inv(w2c_ref)
        new_w2cs = []
        for w2c in w2cs:
            new_w2cs.append(w2c @ w2c_ref_inv)
        w2cs = new_w2cs

        scale_mat, scale_factor = self.get_scale_mat(self.img_hw, intrs, w2cs, near_fars, factor=self.factor)
        
        c2ws = []
        new_near_fars = []
        new_intrs = []
        # new_depths = []
        # new_pseudo_mvs_depths = []
        for intr, w2c in zip(intrs, w2cs):
            P = intr @ w2c @ scale_mat
            P = P[:3, :4]
            new_intr, c2w = load_K_Rt_from_P(None, P)
            c2ws.append(c2w)
            new_intrs.append(new_intr)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2)).astype(np.float32)
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
            # new_depths.append(scale_factor * depth)
            # new_pseudo_mvs_depths.append(scale_factor * pseudo_mvs_depth)
            
        ref_depth = ref_depth * scale_factor
        ref_pseudo_mvs_depth = ref_pseudo_mvs_depth * scale_factor
        src_pseudo_mvs_depth = src_pseudo_mvs_depth * scale_factor
        src_depth = src_depth * scale_factor
        ref_depth = torch.from_numpy(ref_depth.astype(np.float32))
        ref_pseudo_mvs_depth = torch.from_numpy(ref_pseudo_mvs_depth.astype(np.float32))
        src_pseudo_mvs_depth = torch.from_numpy(src_pseudo_mvs_depth.astype(np.float32))
        src_depth = torch.from_numpy(src_depth.astype(np.float32))

        imgs = torch.from_numpy(np.stack(imgs).astype(np.float32))
        intrs = torch.from_numpy(np.stack(new_intrs).astype(np.float32))
        c2ws = torch.from_numpy(np.stack(c2ws).astype(np.float32))
        near_fars = torch.from_numpy(np.stack(new_near_fars).astype(np.float32))
        masks = torch.from_numpy(np.stack(masks).astype(np.float32))
        # depths = torch.from_numpy(np.stack(new_depths).astype(np.float32))
        # pseudo_mvs_depths = torch.from_numpy(np.stack(new_pseudo_mvs_depths).astype(np.float32))
        
        outputs = {
            "imgs": imgs.permute(0, 3, 1, 2).contiguous(),
            "intrs": intrs,
            "c2ws": c2ws,
            "scale_mat": torch.from_numpy(w2c_ref_inv @ scale_mat),
            "view_ids": torch.from_numpy(np.array(view_ids)).long()
        }
        
        ys, xs = torch.meshgrid(torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0]),
                                torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1]))  # pytorch's meshgrid has indexing='ij'
        pixel_all = torch.stack([xs, ys], dim=-1)  # H, W, 2

        if self.mode == "train":
            assert self.n_rays>0, "No sampling rays!"
            
            ref_n_rays = self.n_rays
            
            p_valid = pixel_all[masks[0] > 0.5]  # [num, 2]
            pixels_x_i = torch.randint(low=0, high=self.img_hw[1], size=[ref_n_rays // 4])
            pixels_y_i = torch.randint(low=0, high=self.img_hw[0], size=[ref_n_rays // 4])
            random_idx = torch.randint(low=0, high=p_valid.shape[0], size=[ref_n_rays - ref_n_rays // 4])
            p_select = p_valid[random_idx]
            pixels_x = p_select[:, 0]
            pixels_y = p_select[:, 1]

            pixels_x = torch.cat([pixels_x, pixels_x_i], dim=0)
            pixels_y = torch.cat([pixels_y, pixels_y_i], dim=0)

        else:
            bound_min=torch.tensor([-1, -1, -1], dtype=torch.float32)
            bound_max=torch.tensor([1, 1, 1], dtype=torch.float32)
            outputs.update({"bound_min": bound_min, "bound_max": bound_max, "scene": scan})
            outputs["file_name"] = scan+"_view"+str(ref_view)+"_light"+str(light_idx)
            outputs["hw"] = torch.Tensor([self.img_hw[0]//self.val_res_level, self.img_hw[1]//self.val_res_level]).int()
            outputs["masks"] = masks

            tx = torch.linspace(0, self.img_hw[1] - 1, self.img_hw[1] // self.val_res_level)
            ty = torch.linspace(0, self.img_hw[0] - 1, self.img_hw[0] // self.val_res_level)
            pixels_y, pixels_x = torch.meshgrid(ty, tx)
            pixels_x, pixels_y = pixels_x.reshape(-1), pixels_y.reshape(-1)
        
        pseudo_depth = ref_pseudo_mvs_depth[(pixels_y.long(), pixels_x.long())]   # n_rays
        color = imgs[0][(pixels_y.long(), pixels_x.long())]    # n_rays, 3
        depth = ref_depth[(pixels_y.long(), pixels_x.long())]   # n_rays
        mask = masks[0][(pixels_y.long(), pixels_x.long())]   # n_rays
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # n_rays, 3
        p = torch.matmul(intrs.inverse()[0, None, :3, :3], p[:, :, None]).squeeze() # n_rays, 3
        rays_d = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # n_rays, 3
        rays_d = torch.matmul(c2ws[0, None, :3, :3], rays_d[:, :, None]).squeeze()  # n_rays, 3
        rays_o = c2ws[0, None, :3, 3].expand(rays_d.shape) # n_rays, 3
        near, far = near_fars[0].reshape(1, 2).split(split_size=1, dim=1)
        
        pcd = PlyData.read(os.path.join(self.data_dir, "Pseudo_points/mvsnet{:0>3}_l3.ply".format(int(scan[4:]))))
        px = pcd['vertex']['x']
        py = pcd['vertex']['y']
        pz = pcd['vertex']['z']
        pxyz_ori = np.stack([px, py, pz], axis=1)
        
        random_idx = np.random.randint(low=0, high=pxyz_ori.shape[0], size=[2048])
        pxyz = pxyz_ori[random_idx]
        pxyz = np.matmul(w2c_ref, np.concatenate([pxyz, np.ones_like(pxyz[..., :1])], axis=1)[..., None])[:, :3, 0]
        pseudo_pts = (pxyz - scale_mat[:3, 3][None]) / scale_mat[0, 0]
        pseudo_pts = torch.from_numpy(pseudo_pts)
        
        outputs.update({
            "pixels_x": pixels_x,
            "pixels_y": pixels_y,
            "near_fars": near_fars,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "near": near,
            "far": far,
            "color": color,
            "depth": depth,
            "pseudo_depth": pseudo_depth,
            "mask": mask,
            "mask_ref": masks[0],
            "depth_ref": ref_depth,
            "pseudo_pts": pseudo_pts,
            "pseudo_depth_ref": ref_pseudo_mvs_depth,
            "pseudo_depth_src": src_pseudo_mvs_depth,
            "src_idx": src_idx,
            "mask_src": masks[src_idx],
            "depth_src": src_depth,
        })

        return outputs
    
    def __len__(self):
        return len(self.metas)