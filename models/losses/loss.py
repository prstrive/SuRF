import torch
import torch.nn as nn
import torch.nn.functional as F

from .ncc import compute_LNCC2, compute_LNCC2_grid
from .photometric_loss import compute_ptloss
from .consistency_loss import compute_consistency_loss


class Loss(nn.Module):
    def __init__(self, confs):
        super(Loss, self).__init__()
        
        self.color_weight = confs.get_float('color_weight')
        self.sparse_scale_factor = confs.get_float('sparse_scale_factor')
        self.sparse_weight = confs.get_float('sparse_weight')
        self.igr_weight = confs.get_float("igr_weight")
        self.mfc_weight = confs.get_float("mfc_weight")
        self.smooth_weight = confs.get_float("smooth_weight")
        self.depth_weight = confs.get_float('depth_weight')
        self.ptloss_weight = confs.get_float('ptloss_weight')
        self.pseudo_auxi_depth_weight = confs.get_float('pseudo_auxi_depth_weight')
        self.pseudo_sdf_weight = confs.get_float('pseudo_sdf_weight')
        self.stage_weights = confs.get_list('stage_weights')
        self.pseudo_depth_weight = confs.get_float('pseudo_depth_weight')
        
    def forward(self, preds, targets, step=None, mode="train"):
        valid_mask = preds['valid_mask']
        if "mask" in targets:
            valid_mask = valid_mask * targets["mask"].reshape(-1, 1)
        
        color_loss = F.l1_loss(preds["color_fine"], targets["color"], reduction='none')
        color_loss = (color_loss * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-5)
        
        eikonal_loss = preds['gradient_error'].mean() #+ preds["auxi_gradient_error"].mean()
        
        annel_weight = min(1.0, step / 2)
        
        sparse_loss = torch.exp(-torch.abs(preds["sparse_sdf"]) * self.sparse_scale_factor).mean() * annel_weight
        
        smooth_loss = preds["smooth_error"].mean()
                
        ncc = compute_LNCC2(preds["ref_gray_val"], preds["sampled_gray_val"])
        ncc_mask = valid_mask * preds["mid_inside_sphere"]
        mfc_loss = 0.5 * ((ncc * ncc_mask).sum(dim=0) / (ncc_mask.sum(dim=0) + 1e-8)).squeeze(-1)
        
        photo_loss = 0.0
        pseudo_auxi_depth_loss = 0.0
        auxi_depth_loss = 0.0
        auxi_depth_loss0 = 0.0
        src_auxi_depth_loss = 0.0
        src_auxi_depth_loss0 = 0.0
        if mode == "train":
            for i in range(len(self.stage_weights)):
                ref_photo_loss = compute_ptloss(preds[f"depth_stage{i}"], targets["imgs"], targets["mask_ref"], targets["intrs"], targets["c2ws"])
                src_photo_loss = compute_ptloss(preds[f"depth_src_stage{i}"], targets["imgs"], targets["mask_src"], targets["intrs"], targets["c2ws"], ref_idx=targets["src_idx"], topk=1)
                photo_loss += (ref_photo_loss + src_photo_loss) * self.stage_weights[i]
                
                pa_depth_loss = ((preds[f"depth_stage{i}"] - targets["pseudo_depth_ref"]).abs() * (targets["pseudo_depth_ref"] > 0).float()).sum() / ((targets["pseudo_depth_ref"] > 0).float().sum() + 1e-8)
                src_pa_depth_loss = ((preds[f"depth_src_stage{i}"] - targets["pseudo_depth_src"]).abs() * (targets["pseudo_depth_src"] > 0).float()).sum() / ((targets["pseudo_depth_src"] > 0).float().sum() + 1e-8)
                pseudo_auxi_depth_loss += (pa_depth_loss + src_pa_depth_loss) * self.stage_weights[i]
            
            # consistency_loss = compute_consistency_loss(preds["auxi_depth"], preds["src_auxi_depth"], targets["intrs"], targets["c2ws"], targets["src_idx"], targets["mask_ref"], targets["mask_src"])
                
            auxi_depth_loss = ((preds[f"depth_stage{len(self.stage_weights)-1}"] - targets["depth_ref"]).abs() * targets["mask_ref"]).sum() / (targets["mask_ref"].sum() + 1e-8)
            src_auxi_depth_loss = ((preds[f"depth_src_stage{len(self.stage_weights)-1}"] - targets["depth_src"]).abs() * targets["mask_src"]).sum() / (targets["mask_src"].sum() + 1e-8)
            
            auxi_depth_loss0 = ((preds["depth_stage0"] - targets["depth_ref"]).abs() * targets["mask_ref"]).sum() / (targets["mask_ref"].sum() + 1e-8)
            src_auxi_depth_loss0 = ((preds["depth_src_stage0"] - targets["depth_src"]).abs() * targets["mask_src"]).sum() / (targets["mask_src"].sum() + 1e-8)
        
        pseudo_sdf_loss = 0.0
        if "pseudo_sdf" in preds:
            pseudo_sdf_loss = torch.abs(preds["pseudo_sdf"]).mean()
        
        pseudo_depth_loss = 0.0
        if "pseudo_depth" in targets:
            pseudo_depth_loss = ((preds["render_depth"] - targets["pseudo_depth"]).abs() * (targets["pseudo_depth"] > 0).float()).sum() / ((targets["pseudo_depth"] > 0).float().sum() + 1e-8)
        
        depth_loss = 0.0
        if "depth" in targets:
            depth_loss = ((preds["render_depth"] - targets["depth"]).abs() * (targets["depth"]>0).float()).sum() / ((targets["depth"]>0).float().sum() + 1e-8)
        
        loss = color_loss * self.color_weight\
                + eikonal_loss * self.igr_weight \
                + sparse_loss * self.sparse_weight \
                + mfc_loss * self.mfc_weight \
                + smooth_loss * self.smooth_weight \
                + depth_loss * self.depth_weight \
                + photo_loss * self.ptloss_weight \
                + pseudo_auxi_depth_loss * self.pseudo_auxi_depth_weight \
                + pseudo_sdf_loss * self.pseudo_sdf_weight \
                + pseudo_depth_loss * self.pseudo_depth_weight \
        
        loss_outs = {
            "loss": loss,
            "color_loss": color_loss,
            "eikonal_loss": eikonal_loss,
            "sparse_loss": sparse_loss,
            "mfc_loss": mfc_loss,
            "smooth_loss": smooth_loss,
            "depth_loss": depth_loss,
            "photo_loss": photo_loss,
            "auxi_depth_loss": auxi_depth_loss,
            "pseudo_auxi_depth_loss": pseudo_auxi_depth_loss,
            "src_auxi_depth_loss": src_auxi_depth_loss,
            "pseudo_sdf_loss": pseudo_sdf_loss,
            "auxi_depth_loss0": auxi_depth_loss0,
            "src_auxi_depth_loss0": src_auxi_depth_loss0,
            "pseudo_depth_loss": pseudo_depth_loss,
        }
        
        return loss_outs