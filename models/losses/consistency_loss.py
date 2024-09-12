import torch
import torch.nn.functional as F


def compute_consistency_loss(ref_depth, src_depth, intrs, c2ws, src_idx, mask_ref, mask_src):
    ref_intr = intrs[0]
    src_intr = intrs[src_idx]
    ref_c2w = c2ws[0]
    src_c2w = c2ws[src_idx]
    
    height, width = ref_depth.shape
    
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=ref_depth.device),
                            torch.arange(0, width, dtype=torch.float32, device=ref_depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    ref_xy = torch.stack([x, y])    # 2, hw
    xyz = torch.stack((x, y, torch.ones_like(x))) * ref_depth.reshape(1, -1)  # [3, H*W]
    
    ref_cam_pts = torch.matmul(torch.inverse(ref_intr)[:3, :3], xyz)    # 3, hw
    ref_cam_pts_homo = torch.cat([ref_cam_pts, torch.ones_like(ref_cam_pts[:1])])
    world_pts = torch.matmul(ref_c2w, ref_cam_pts_homo) # 4, hw
    src_cam_pts = torch.matmul(torch.inverse(src_c2w), world_pts)[:3]    # 3, hw
    src_xyz = torch.matmul(src_intr[:3, :3], src_cam_pts)
    src_xy = src_xyz[:2] / (src_xyz[2:] + 1e-8)
    
    norm_x = src_xy[0] / ((width - 1) / 2) - 1
    norm_y = src_xy[1] / ((height - 1) / 2) - 1
    
    grid = torch.stack([norm_x, norm_y], -1).unsqueeze(0).unsqueeze(0)   # 1, 1, hw, 2
    
    warp_depth = F.grid_sample(src_depth.unsqueeze(0).unsqueeze(0), grid, padding_mode='zeros', align_corners=True)
    
    xyz_src = torch.cat([src_xy, torch.ones_like(src_xy[:1])]) * warp_depth.reshape(1, -1)
    xyz_src = torch.matmul(torch.inverse(src_intr)[:3, :3], xyz_src)    # 3, hw
    proj_world_pts = torch.matmul(src_c2w, torch.cat([xyz_src, torch.ones_like(xyz_src[:1])]))  # 4, hw
    proj_ref_cam_pts = torch.matmul(torch.inverse(ref_c2w), proj_world_pts)[:3]
    depth_proj = proj_ref_cam_pts[2].reshape(height, width)
    
    proj_ref_xyz = torch.matmul(ref_intr[:3, :3], proj_ref_cam_pts)
    proj_ref_xy = proj_ref_xyz[:2] / (proj_ref_xyz[2:] + 1e-8)
    
    depth_diff = (ref_depth - depth_proj).abs() / ref_depth
    depth_mask = (depth_diff < 0.01).detach().float() * mask_ref
    depth_consistency_loss = (depth_diff * depth_mask).sum() / (depth_mask.sum() + 1e-8)
    
    # ref_xy[0] = ref_xy[0] / width
    # ref_xy[1] = ref_xy[1] / height
    # proj_ref_xy[0] = proj_ref_xy[0] / width
    # proj_ref_xy[1] = proj_ref_xy[1] / height
    coord_diff = (ref_xy - proj_ref_xy).abs().mean(0)
    coord_mask = (coord_diff < 1).detach().float() * mask_ref.reshape(-1)
    coord_consistency_loss = (coord_diff * coord_mask).sum() / (coord_mask.sum() + 1e-8)
    
    # print("coord_mask:", coord_mask.sum().item(), depth_mask.sum().item())
    
    consistency_loss = depth_consistency_loss + coord_consistency_loss * 0.1
    
    return consistency_loss