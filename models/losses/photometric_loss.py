import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        x = self.refl(x)
        y = self.refl(y)
        mask = self.refl(mask)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output
    
    
def compute_smooth_loss(depth, img, mask):
    mask_x = (mask[..., :-1] + mask[..., 1:]) / 2
    mask_y = (mask[..., :-1, :] + mask[..., 1:, :]) / 2

    grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
    grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_x *= torch.exp(-grad_img_x) * mask_x
    grad_y *= torch.exp(-grad_img_y) * mask_y

    smooth_loss = grad_x.mean() + grad_y.mean()

    return smooth_loss


def compute_ptloss(depth, imgs, mask_ref, intrs, c2ws, ref_idx=0, topk=2):
    """ 
    depth: h, w
    imgs: nv, 3, h, w
    mask_ref: h, w
    intrs: nv, 4 ,4
    c2ws, nv, 4, 4
    """
    
    ref_img = imgs[ref_idx:ref_idx+1]
    src_imgs = torch.cat([imgs[:ref_idx], imgs[ref_idx+1:]])
    ref_intr = intrs[ref_idx]
    src_intrs = torch.cat([intrs[:ref_idx], intrs[ref_idx+1:]])
    ref_c2w = c2ws[ref_idx]
    src_c2ws = torch.cat([c2ws[:ref_idx], c2ws[ref_idx+1:]])
    
    n_srcs, _, height, width = src_imgs.shape
    
    depth = depth.unsqueeze(0).unsqueeze(0)
    mask_ref = mask_ref.unsqueeze(0).unsqueeze(0)
    
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_imgs.device),
                            torch.arange(0, width, dtype=torch.float32, device=src_imgs.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x))) * depth.reshape(1, -1)  # [3, H*W]
    
    ref_cam_pts = torch.matmul(torch.inverse(ref_intr)[:3, :3], xyz)    # 3, hw
    ref_cam_pts_homo = torch.cat([ref_cam_pts, torch.ones_like(ref_cam_pts[:1])])
    world_pts = torch.matmul(ref_c2w, ref_cam_pts_homo) # 4, hw
    src_cam_pts = torch.matmul(torch.inverse(src_c2ws), world_pts[None])[:, :3]    # nsrc, 3, hw
    src_xyz = torch.matmul(src_intrs[:, :3, :3], src_cam_pts)
    src_xy = src_xyz[:, :2] / (src_xyz[:, 2:] + 1e-8)
    
    norm_x = src_xy[:, 0] / ((width - 1) / 2) - 1
    norm_y = src_xy[:, 1] / ((height - 1) / 2) - 1
    
    mask = (norm_x.abs() <= 1) & (norm_y.abs() <= 1) & (src_xyz[:, 2] > 0)
    mask = mask.reshape(n_srcs, 1, height, width)
    
    grid = torch.stack([norm_x, norm_y], -1).unsqueeze(1)   # nsrc, 1, hw, 2
    
    warp_imgs = F.grid_sample(src_imgs, grid, padding_mode='zeros', align_corners=True)
    warp_imgs = warp_imgs.reshape(n_srcs, -1, height, width)
    
    ssim_loss = SSIM()(warp_imgs, ref_img, (mask & (mask_ref>0.5)).float()).mean(dim=1, keepdim=True)    # (nsrc, 1, h, w)
    ssim_loss, _ = torch.topk(ssim_loss, topk, dim=0, largest=False)
    ssim_loss = (ssim_loss * mask_ref).sum() / (mask_ref.sum() + 1e-8)
    
    l1_loss = F.smooth_l1_loss(warp_imgs, ref_img, reduction="none").mean(dim=1, keepdim=True)    # (nsrc, 1, h, w)
    l1_loss, _ = torch.topk(l1_loss, topk, dim=0, largest=False)
    l1_loss = (l1_loss * mask_ref).sum() / (mask_ref.sum() + 1e-8)
    
    ref_dy = ref_img[:, :, :-1, :] - ref_img[:, :, 1:, :]
    mask_ref_y = mask_ref[:, :, :-1, :] * mask_ref[:, :, 1:, :]
    ref_dx = ref_img[:, :, :, :-1] - ref_img[:, :, :, 1:]
    mask_ref_x = mask_ref[:, :, :, :-1] * mask_ref[:, :, :, 1:]
    warped_dy = warp_imgs[:, :, :-1, :] - warp_imgs[:, :, 1:, :]
    warped_dx = warp_imgs[:, :, :, :-1] - warp_imgs[:, :, :, 1:]
    grad_loss_x = F.smooth_l1_loss(warped_dx, ref_dx, reduction='none').mean(dim=1, keepdim=True)
    grad_loss_x, _ = torch.topk(grad_loss_x, topk, dim=0, largest=False)
    grad_loss_x = (grad_loss_x * mask_ref_x).sum() / (mask_ref_x.sum() + 1e-8)
    grad_loss_y = F.smooth_l1_loss(warped_dy, ref_dy, reduction='none').mean(dim=1, keepdim=True)
    grad_loss_y, _ = torch.topk(grad_loss_y, topk, dim=0, largest=False)
    grad_loss_y = (grad_loss_y * mask_ref_y).sum() / (mask_ref_y.sum() + 1e-8)
    grad_loss = grad_loss_x + grad_loss_y
    
    # smooth_loss = compute_smooth_loss(depth, ref_img, mask_ref)
    
    photo_loss =  l1_loss + grad_loss + ssim_loss #+ 0.006 * smooth_loss
    
    return photo_loss
    
    