import torch
import torch.nn as nn 
import torch.nn.functional as F

from .projector import lookup_volume


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    
    # weights = weights + 1e-5  # prevent nans
    # pdf = weights / torch.sum(weights, -1, keepdim=True)
    # cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    
    cdf = torch.cumsum(weights, axis=1) / (weights.sum(axis=1)[:,None] + 1e-6)
    
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).type_as(weights)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).type_as(weights)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class MatchingField(nn.Module):
    def __init__(self, confs):
        super(MatchingField, self).__init__()
        
        self.n_samples_depths = confs.get_list("n_samples_depths")
        self.n_importance_depths = confs.get_list("n_importance_depths")
        self.up_sample_steps = confs.get_list("up_sample_steps")
        self.depth_res_levels = confs.get_list("depth_res_levels")
        # self.z_val_ranges = confs.get_list("z_val_ranges")
        
    def density_render(self, rays_o, rays_d, near, far, c2w, density_volume, stage_idx, perturb=False):
        
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        batch_size = len(rays_o)
        
        # sample_dist = (near + 2.0 - far)
        # # sample_dist = (far - near) / self.n_samples_depths[stage_idx]   # Assuming the region of interest is a unit sphere
        # z_vals = torch.linspace(0.0, 1.0, self.n_samples_depths[stage_idx]).type_as(near)
        # z_vals = near + (far - near) * z_vals[None, :]
        
        # # perturb = self.perturb
        # if perturb:
        #     t_rand = (torch.rand([batch_size, 1]) - 0.5).type_as(z_vals)
        #     z_vals = z_vals + t_rand * (far - near) / self.n_samples_depths[stage_idx]
            
        num_stages = near.shape[-1]
        all_z_vals = []
        for i in range(num_stages):
            
            near_stage = near[..., [i]]
            far_stage = far[..., [i]]
            z_vals = torch.linspace(0.0, 1.0, self.n_samples_depths[stage_idx]).type_as(near)
            z_vals = near_stage + (far_stage - near_stage) * z_vals[None, :]

            if perturb:
                t_rand = (torch.rand([batch_size, 1]) - 0.5).type_as(z_vals)
                z_vals = z_vals + t_rand * (far_stage - near_stage) / self.n_samples_depths[stage_idx]
            
            all_z_vals.append(z_vals)
            
        z_vals = torch.cat(all_z_vals, dim=-1)
        z_vals, _ = torch.sort(z_vals, dim=-1)
        
        # with torch.no_grad():
        #     n_importance_i = self.n_importance_depths[stage_idx] // self.up_sample_steps[stage_idx]
        #     for i in range(self.up_sample_steps[stage_idx]):
        #         pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        #         pts = pts.reshape(-1, 3)
                
        #         # density = lookup_volume(pts, density_volume, sample_mode='bilinear')
        #         # density = F.softplus(density).reshape(batch_size, -1)
        #         # dists = z_vals[:, 1:] - z_vals[:, :-1]
        #         # dists = torch.cat([dists, sample_dist], -1)
        #         # # LOG SPACE
        #         # free_energy = dists * density
        #         # shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        #         # alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        #         # transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        #         # weights = alpha * transmittance # probability of the ray hits something here
                
        #         # alpha = lookup_volume(pts, density_volume, sample_mode='bilinear')
        #         # weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).type_as(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1]
                
        #         density = lookup_volume(pts, density_volume, sample_mode='bilinear')
        #         density = density.reshape(batch_size, -1)
        #         weights = F.softmax(density, dim=-1)
                
        #         z_samples = sample_pdf(z_vals, weights, n_importance_i, det=True).detach()
        #         z_vals = torch.cat([z_vals, z_samples], dim=1)
        #         z_vals, _ = torch.sort(z_vals, dim=-1)
        
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)
        
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, -1)
        outside_sphere = (pts_norm > 1.0).float().detach()
        
        # density = lookup_volume(pts, density_volume, sample_mode='bilinear')
        # density = F.softplus(density).reshape(batch_size, -1)
        # dists = z_vals[:, 1:] - z_vals[:, :-1]
        # dists = torch.cat([dists, sample_dist], -1)
        # # LOG SPACE
        # free_energy = dists * density
        # shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        # alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        # transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        # weights = alpha * transmittance # probability of the ray hits something here
        
        # alpha = lookup_volume(pts, density_volume, sample_mode='bilinear')
        # weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).type_as(alpha), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        
        # pts_mask = lookup_volume(pts, mask_volume, sample_mode='nearest')
        # pts_mask_bool = (pts_mask > 0).view(-1)
        # if torch.sum(pts_mask_bool.float()) < 1:
        #     raise ValueError("No valid pseudo pts!")
        # density = torch.ones_like(pts[:, :1]) * 1e-9
        # density[pts_mask_bool] = lookup_volume(pts[pts_mask_bool], density_volume, sample_mode='bilinear')
        density = lookup_volume(pts, density_volume, sample_mode='bilinear')
        density = density.reshape(batch_size, -1)
        weights = F.softmax(density, dim=-1)
        
        cam_rays_d = torch.matmul(torch.inverse(c2w[None, :3, :3]), rays_d[:, :, None]).squeeze()
        render_z_val = (z_vals * weights).sum(dim=1) #/ (weights.sum(dim=1) + 1e-10)
        # render_z_val = (z_vals * weights).sum(dim=1) + (1 - weights.sum(dim=1)) * (near + 2.0).squeeze(1)
        render_depth = render_z_val * cam_rays_d[:, 2] # (nr, ) z_val * cos = depth
        
        # z_val_near = render_z_val - self.z_val_ranges[stage_idx] / 2
        # z_val_far = render_z_val + self.z_val_ranges[stage_idx] / 2
        # depth_min = z_val_near * cam_rays_d[:, 2] 
        # depth_max = z_val_far * cam_rays_d[:, 2] 
        
        norm_w = weights / weights.sum(dim=1, keepdim=True)
        
        # z_val_range = ((z_vals - render_z_val.unsqueeze(-1))**2 * norm_w).sum(dim=-1) ** 0.5 * 3
        # depth_range = ((z_vals * cam_rays_d[:, 2:3] - render_depth.unsqueeze(-1))**2 * norm_w).sum(dim=-1) ** 0.5 * 3
        
        occ_reg = density[:, :6].mean() + (density * outside_sphere).sum() / (outside_sphere.sum() + 1e-10)
        
        # occ_reg = alpha[:, :12]
        
        return render_depth, occ_reg #, z_val_near, z_val_far, depth_min, depth_max
        
    def forward(self, ipts, density_volume, stage_idx, range_ratios, pre_depths=None, perturb=False):
        near_fars = ipts["near_fars"] # (nv, 2)
        c2ws = ipts["c2ws"]   # (nv, 4, 4)
        intrs = ipts["intrs"]   # (nv, 4, 4)
        src_idx = ipts["src_idx"] if "src_idx" in ipts else 0
        
        imgs = ipts["imgs"]
        img_h, img_w = imgs.shape[-2:]
        
        h, w = img_h // self.depth_res_levels[stage_idx], img_w // self.depth_res_levels[stage_idx]
        
        tx = torch.linspace(0, img_w - 1, w)
        ty = torch.linspace(0, img_h - 1, h)
        pixels_y, pixels_x = torch.meshgrid(ty, tx)
        pixels_x, pixels_y = pixels_x.reshape(-1), pixels_y.reshape(-1)
        pixels = torch.stack([pixels_x, pixels_y], -1).type_as(intrs)
        
        nv = intrs.shape[0]
        
        render_depths = []
        # depth_ranges = []
        occ_regs = []
        
        for i in range(nv):
            pixel_all = torch.cat([pixels, torch.ones_like(pixels[:, :1])], dim=-1).float()  # n_rays, 3
            cam_pixel_all = torch.matmul(intrs.inverse()[i, None, :3, :3], pixel_all[:, :, None]).squeeze() # n_rays, 3
            rays_d_all = cam_pixel_all / torch.linalg.norm(cam_pixel_all, ord=2, dim=-1, keepdim=True)    # n_rays, 3
            rays_d_all = torch.matmul(c2ws[i, None, :3, :3], rays_d_all[:, :, None]).squeeze()  # n_rays, 3
            rays_o_all = c2ws[i, None, :3, 3].expand(rays_d_all.shape) # n_rays, 3
            near_ori, far_ori = near_fars[i].reshape(1, 2).split(split_size=1, dim=1)
            if pre_depths is not None:                
                pre_depth = pre_depths[i].detach()[(pixels_y.long(), pixels_x.long())]
                cam_rays_d = torch.matmul(torch.inverse(c2ws[i, None, :3, :3]), rays_d_all[:, :, None]).squeeze()
                pre_z_val = pre_depth.reshape(-1) / cam_rays_d[:, 2]
                stage_range = (far_ori - near_ori).squeeze() * range_ratios[stage_idx]
                near = (pre_z_val  - stage_range / 2).unsqueeze(1)
                far = (pre_z_val  + stage_range / 2).unsqueeze(1)
                near = torch.where(far > far_ori, near - (far - far_ori), near)
                far = torch.where(near < near_ori, far + (near_ori - near), far)
                near = torch.clamp(near, near_ori.squeeze(), far_ori.squeeze())
                far = torch.clamp(far, near_ori.squeeze(), far_ori.squeeze())
                
                pre_stage_range = (far_ori - near_ori).squeeze() * range_ratios[stage_idx-1]
                pre_near = (pre_z_val  - pre_stage_range / 2).unsqueeze(1)
                pre_far = (pre_z_val  + pre_stage_range / 2).unsqueeze(1)
                pre_near = torch.where(pre_far > far_ori, pre_near - (pre_far - far_ori), pre_near)
                pre_far = torch.where(pre_near < near_ori, pre_far + (near_ori - pre_near), pre_far)
                pre_near = torch.clamp(pre_near, near_ori.squeeze(), far_ori.squeeze())
                pre_far = torch.clamp(pre_far, near_ori.squeeze(), far_ori.squeeze())
                
                near = torch.cat([near, pre_near], dim=1)
                far = torch.cat([far, pre_far], dim=1)
            else:
                near = near_ori.repeat(rays_o_all.shape[0], 1)
                far = far_ori.repeat(rays_o_all.shape[0], 1)
            
            if (i==0) or (i==src_idx):
                render_depth, occ_reg = self.density_render(rays_o_all, rays_d_all, near, far, c2ws[i], density_volume, stage_idx, perturb)
            else:
                with torch.no_grad():
                    render_depth, occ_reg = self.density_render(rays_o_all, rays_d_all, near, far, c2ws[i], density_volume, stage_idx, False)
                    
            render_depth = render_depth.reshape(h, w)
            
            render_depth = F.interpolate(render_depth.unsqueeze(0).unsqueeze(0), size=(img_h, img_w), mode="bilinear").squeeze(0).squeeze(0)
            render_depths.append(render_depth)
            occ_regs.append(occ_reg)
            
        return render_depths, occ_regs