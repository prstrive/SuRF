import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F


class Volume(nn.Module):
    def __init__(self, confs):
        super(Volume, self).__init__()
        
        self.base_volume_dim = confs.get_list("base_volume_dim")
        self.bounding = np.array(confs.get_list("bounding", default=[[-1, 1], [-1, 1], [-1, 1]]))
        self.origin = self.bounding[:, 0]
        
        self.agg_mlp = nn.Sequential(
            nn.Linear(4, 8),
            nn.ELU(inplace=True),
            nn.Linear(8, 1)
        )
        
    def init_coords(self):
        self.volume_dim = np.array(self.base_volume_dim)
        self.voxel_size = (self.bounding[:, 1] - self.bounding[:, 0]) / (self.volume_dim - 1)
        
        with torch.no_grad():
            # Create voxel grid
            grid_range = [torch.arange(0, self.volume_dim[axis]) for axis in range(3)]
            grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
            grid = grid.float()  # 3 dx dy dz
            coords = grid.view(3, -1)
            
        up_coords = coords.permute(1, 0).contiguous()
        return up_coords
    
    def up_sample(self, pre_coords, pre_feat, num=8):
        
        self.volume_dim *= 2
        self.voxel_size = (self.bounding[:, 1] - self.bounding[:, 0]) / (self.volume_dim - 1)
        
        with torch.no_grad():
            pre_coords *= 2
            pos_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += 1

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 3)

        return up_coords, up_feat
    
    def back_proj_multiscale(self, feats, coords, intrs, c2ws, stage_idx):
        """
        feats: list coarse2fine (n, c, h, w)
        coords: (n_pts, 3)
        intrs: (n, 4, 4)
        c2ws: (n, 4, 4)
        """
        
        n_views, c, h, w = feats[-1].shape
        
        world_pts = coords * torch.tensor(self.voxel_size).type_as(coords).unsqueeze(0) + torch.tensor(self.origin).type_as(coords).unsqueeze(0)
        world_pts = world_pts.unsqueeze(0).permute(0, 2, 1).contiguous()   # 1, 3, n_pts
        world_pts = torch.cat([world_pts, torch.ones_like(world_pts[:, :1])], dim=1)
        
        with torch.no_grad():
            cam_pts = torch.matmul(torch.inverse(c2ws), world_pts)
            img_pts = torch.matmul(intrs, cam_pts)[:, :3]
            xy = img_pts[:, :2] / img_pts[:, 2:]
            
            norm_x = xy[:, 0] / ((w - 1) / 2) - 1
            norm_y = xy[:, 1] / ((h - 1) / 2) - 1
            
            grid = torch.stack([norm_x, norm_y], dim=-1)    # nv, n_pts, 2
            
            mask = (norm_x.abs() <= 1) & (norm_y.abs() <= 1) & (img_pts[:, 2] > 0)  # nv, n_pts
            mask = mask.unsqueeze(1)    # nv, 1, npts
        
        warp_feat = 0
        for feat in feats[stage_idx:]:
            warp_f = F.grid_sample(feat, grid.unsqueeze(1), padding_mode='zeros', align_corners=True) # nv, c, 1, npts
            warp_feat += warp_f.reshape(n_views, c, -1)
        
        mask_sum = mask.sum(dim=0)
        warp_feat = warp_feat.permute(0, 2, 1).contiguous()
        x = self.agg_mlp(warp_feat)
        x = x.masked_fill(mask.permute(0, 2, 1).contiguous() == 0, -1e9)
        w = F.softmax(x, dim=0)
        
        mean = (warp_feat * w).sum(dim=0)
        var = ((warp_feat * w)**2).sum(dim=0).sub_((warp_feat * w).sum(dim=0).pow_(2))
        feat_vol = torch.cat([mean, var], dim=1)
        mask_vol = (mask_sum > 1).squeeze(0)
            
        return feat_vol, mask_vol
    
    def sparse2dense(self, feats, coords, pre_volume):
        """ 
        pre_volume: (1, c, h, w, d)
        """
        npts, c = feats.shape
        # if pre_volume is None:
        dense_volume = torch.full([1, self.volume_dim[0], self.volume_dim[1], self.volume_dim[2], c], float(0), device=feats.device)
        # else:
        #     dense_volume = F.interpolate(pre_volume, scale_factor=2, mode="trilinear").permute(0, 2, 3, 4, 1)
        if pre_volume is not None:
            pre_density_volume = F.interpolate(pre_volume, scale_factor=2, mode="trilinear").permute(0, 2, 3, 4, 1)
            dense_volume[..., :1] = pre_density_volume
            
        mask_volume = torch.full([1, self.volume_dim[0], self.volume_dim[1], self.volume_dim[2], 1], float(0), device=feats.device)
            
        locs = coords.to(torch.int64)
        dense_volume[:, locs[:, 0], locs[:, 1], locs[:, 2]] = feats
        mask_volume[:, locs[:, 0], locs[:, 1], locs[:, 2]] = torch.ones(feats[..., :1].shape).type_as(mask_volume)
        
        dense_volume = dense_volume.permute(0, 4, 1, 2, 3)
        mask_volume = mask_volume.permute(0, 4, 1, 2, 3)
        
        return dense_volume, mask_volume
    
    def get_index(self, coords):
        
        index_table = torch.full([self.volume_dim[0], self.volume_dim[1], self.volume_dim[2]], float(-1), device=coords.device).to(torch.int64)
        
        feat_idx = torch.arange(coords.shape[0]).to(torch.int64).to(coords.device)
        
        locs = coords.to(torch.int64)
        index_table[locs[:, 0], locs[:, 1], locs[:, 2]] = feat_idx
        
        return index_table
    
    def depth_filtering(self, depths, coords, feats, intrs, c2ws, depth_range):
        
        depths = torch.stack(depths, dim=0).unsqueeze(1)    # nv, 1, h, w
        
        nv, _, h, w = depths.shape
        
        world_pts = coords * torch.tensor(self.voxel_size).type_as(coords).unsqueeze(0) + torch.tensor(self.origin).type_as(coords).unsqueeze(0)
        world_pts = world_pts.unsqueeze(0).permute(0, 2, 1).contiguous()   # 1, 3, n_pts
        world_pts = torch.cat([world_pts, torch.ones_like(world_pts[:, :1])], dim=1)
        
        with torch.no_grad():
            cam_pts = torch.matmul(torch.inverse(c2ws), world_pts)
            img_pts = torch.matmul(intrs, cam_pts)[:, :3]
            coord_depths = img_pts[:, 2:] # nv, n_pts
            
            xy = img_pts[:, :2] / img_pts[:, 2:]
            
            norm_x = xy[:, 0] / ((w - 1) / 2) - 1
            norm_y = xy[:, 1] / ((h - 1) / 2) - 1
            
            grid = torch.stack([norm_x, norm_y], dim=-1)    # nv, n_pts, 2
            
            mask = (norm_x.abs() <= 1) & (norm_y.abs() <= 1) & (img_pts[:, 2] > 0)  # nv, n_pts
            mask = mask.unsqueeze(1)   # nv, 1, npts
            
            warp_depths = F.grid_sample(depths, grid.unsqueeze(1), padding_mode='zeros', align_corners=True) # nv, 1, 1, npts
            warp_depths = warp_depths.reshape(nv, 1, -1)
            
            valid_mask = ((warp_depths - coord_depths).abs() < depth_range) & mask
            valid_mask = (valid_mask.sum(0) > 1).squeeze(0)
        
        valid_coords = coords[valid_mask]
        valid_feats = feats[valid_mask]
        
        return valid_coords, valid_feats
    
    def depth_filtering_geocheck(self, depths, coords, feats, intrs, c2ws, depth_range):
        
        depths = torch.stack(depths, dim=0).unsqueeze(1)    # nv, 1, h, w
        
        nv, _, h, w = depths.shape
        
        with torch.no_grad():
            y, x = torch.meshgrid([torch.arange(0, h, dtype=torch.float32, device=depths.device), 
                                torch.arange(0, w, dtype=torch.float32, device=depths.device)])
            y, x = y.contiguous().view(-1), x.contiguous().view(-1)
            ref_xy = torch.stack([x, y])    # 2, hw
            xyz = torch.stack((x, y, torch.ones_like(x)))[None] * depths.reshape(nv, 1, -1)  # [nv, 3, hw]
            ref_cam_pts = torch.matmul(torch.inverse(intrs)[:, :3, :3], xyz)    # nv, 3, hw
            ref_cam_pts_homo = torch.cat([ref_cam_pts, torch.ones_like(ref_cam_pts[:, :1])], dim=1)
            world_pts = torch.matmul(c2ws, ref_cam_pts_homo) # nv, 4, hw
            src_cam_pts = torch.matmul(torch.inverse(c2ws)[:, None], world_pts[None])[:, :, :3]    # nv, nv, 3, hw
            src_xyz = torch.matmul(intrs[:, None, :3, :3], src_cam_pts)
            src_xy = src_xyz[:, :, :2] / (src_xyz[:, :, 2:] + 1e-8)
            norm_x = src_xy[:, :, 0] / ((w - 1) / 2) - 1
            norm_y = src_xy[:, :, 1] / ((h - 1) / 2) - 1
            grid = torch.stack([norm_x, norm_y], -1)   # nv, nv, hw, 2
            warp_depths = F.grid_sample(depths, grid, padding_mode='zeros', align_corners=True) # nv, 1, nv, hw
            xyz_src = torch.cat([src_xy, torch.ones_like(src_xy[:, :, :1])], dim=2) * warp_depths.permute(0, 2, 1, 3).contiguous()
            xyz_src = torch.matmul(torch.inverse(intrs)[:, None, :3, :3], xyz_src)    # nv, nv, 3, hw
            proj_world_pts = torch.matmul(c2ws, torch.cat([xyz_src, torch.ones_like(xyz_src[:, :, :1])], dim=2))  # nv, nv, 4, hw
            proj_ref_cam_pts = torch.matmul(torch.inverse(c2ws), proj_world_pts).permute(1, 0, 2, 3)[:, :, :3]  # nv, nv, 3, hw
            depth_proj = proj_ref_cam_pts[:, :, 2].reshape(nv, nv, h, w)
            proj_ref_xyz = torch.matmul(intrs[:, None, :3, :3], proj_ref_cam_pts)
            proj_ref_xy = proj_ref_xyz[:, :, :2] / (proj_ref_xyz[:, :, 2:] + 1e-8)
            depth_diffs = (depths - depth_proj).abs() / depths
            depth_masks = (depth_diffs < 0.3).detach().float()
            coord_diffs = ((ref_xy[None, None] - proj_ref_xy)**2).sum(dim=2).sqrt().reshape(nv, nv, h, w)
            coord_masks = (coord_diffs < 5).detach().float()
            geomasks = (depth_masks * coord_masks).sum(dim=1, keepdim=True) > 1 #np.ceil(nv * 0.6)
            # print("geomasks:", geomasks.float().mean().item())
        
        if geomasks.float().mean() > 0.01:
            # print("geomask...")
            depths = depths * geomasks
        
        world_pts = coords * torch.tensor(self.voxel_size).type_as(coords).unsqueeze(0) + torch.tensor(self.origin).type_as(coords).unsqueeze(0)
        world_pts = world_pts.unsqueeze(0).permute(0, 2, 1).contiguous()   # 1, 3, n_pts
        world_pts = torch.cat([world_pts, torch.ones_like(world_pts[:, :1])], dim=1)
        
        with torch.no_grad():
            cam_pts = torch.matmul(torch.inverse(c2ws), world_pts)
            img_pts = torch.matmul(intrs, cam_pts)[:, :3]
            coord_depths = img_pts[:, 2:] # nv, n_pts
            
            xy = img_pts[:, :2] / img_pts[:, 2:]
            
            norm_x = xy[:, 0] / ((w - 1) / 2) - 1
            norm_y = xy[:, 1] / ((h - 1) / 2) - 1
            
            grid = torch.stack([norm_x, norm_y], dim=-1)    # nv, n_pts, 2
            
            mask = (norm_x.abs() <= 1) & (norm_y.abs() <= 1) & (img_pts[:, 2] > 0)  # nv, n_pts
            mask = mask.unsqueeze(1)   # nv, 1, npts
            
        warp_depths = F.grid_sample(depths, grid.unsqueeze(1), padding_mode='zeros', align_corners=True) # nv, 1, 1, npts
        warp_depths = warp_depths.reshape(nv, 1, -1)
        
        valid_mask = ((warp_depths - coord_depths).abs() < depth_range) & mask & (warp_depths > 0)
        valid_mask = (valid_mask.sum(0) > 1).squeeze(0)
        
        valid_coords = coords[valid_mask]
        valid_feats = feats[valid_mask]
        
        return valid_coords, valid_feats