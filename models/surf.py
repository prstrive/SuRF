import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

from .modules.feature_network import FeatureNetwork
from .modules.volume import Volume
from .modules.implicit_surface import ImplicitSurface
from .modules.matching_field import MatchingField

from .modules.reg_network import SparseCostRegNetList
from torchsparse.tensor import SparseTensor


class SuRF(nn.Module):
    def __init__(self, confs):
        super(SuRF, self).__init__()
        
        self.has_vol = confs.get_bool("has_vol", default=False)
        
        self.range_ratios = confs.get_list("range_ratios")
        self.num_stage = len(self.range_ratios)
        
        if not self.has_vol:
            self.feature_network = FeatureNetwork(confs["feature_network"])
            self.volume = Volume(confs["volume"])
            self.reg_network = SparseCostRegNetList(confs["reg_network"])
            self.matching_field = MatchingField(confs["matching_field"])
            
            self.match_feature_network = FeatureNetwork(confs["feature_network"])
            for param in self.match_feature_network.parameters():
                param.requires_grad = False
                
        self.implicit_surface = ImplicitSurface(confs["implicit_surface"])
        
    def get_optim_params(self, lr_conf):
        mlp_params_to_train = list(self.implicit_surface.parameters())
        grad_vars = [{'params': mlp_params_to_train, 'lr': lr_conf["mlp_lr"]}]
        if not self.has_vol:
            feat_params_to_train = list(self.feature_network.parameters()) + list(self.reg_network.parameters()) + list(self.volume.parameters())
            grad_vars.append({'params': feat_params_to_train, 'lr': lr_conf["feat_lr"]})
        else:
            for v_lr, volume_param in zip(lr_conf["vol_lr"], self.volumes):
                grad_vars.append({'params': volume_param, 'lr': v_lr})
        return grad_vars
    
    def load_params_vol(self, path, device):
        ckpt = torch.load(path)
        model = ckpt["model"]
        self.volumes = model["volumes"].to(device)
        self.mask_volmes = model["mask_volmes"].to(device)
        self.features = model["features"].to(device)
        self.implicit_surface.load_state_dict(model["implicit_surface"])
        self.has_vol = True
        
    def get_params_vol(self):
        params = {
            "volumes": self.volumes,
            "mask_volmes": self.mask_volmes,
            "features": self.features,
            "implicit_surface": self.implicit_surface.state_dict()
        }
        return params
    
    def init_volumes(self, ipts):
        imgs = ipts["imgs"]  # (nv, 3, h, w)
        
        with torch.no_grad():
            features = self.feature_network(imgs)   # coarse to fine
            outputs, volumes, sparse_idxes, mask_volmes, matching_volume = self.build_volumes(ipts, features, False)
            
        self.volumes = nn.ParameterList([nn.Parameter(volume.detach(), requires_grad=True) for volume in volumes])
        self.sparse_idxes = nn.ParameterList([nn.Parameter(sparse_idx.detach(), requires_grad=False) for sparse_idx in sparse_idxes])
        self.mask_volmes = nn.ParameterList([nn.Parameter(volume.detach(), requires_grad=False) for volume in mask_volmes])
        
        self.matching_volume = nn.Parameter(matching_volume.detach(), requires_grad=False)
        self.features = [feat.detach() for feat in features]
        self.has_vol = True
    
    def build_volumes(self, ipts, features, perturb):
        
        intrs = ipts["intrs"]   # (nv, 4, 4)
        c2ws = ipts["c2ws"] # (nv, 4, 4)
        base_range = (ipts["far"] - ipts["near"]).squeeze()
        
        volumes_all = []
        sparse_idx_all = []
        mask_volumes_all = []
        depths = None
        matching_volume = None
        
        outputs = {}
        for s in range(self.num_stage):
            stage_range = base_range * self.range_ratios[s]
            
            if s == 0:
                up_coords = self.volume.init_coords().type_as(intrs)  # (num_pts, 4), b, x, y, z
            else:
                up_coords, up_feats = self.volume.up_sample(up_coords, feats)
                up_coords, up_feats = self.volume.depth_filtering(depths, up_coords, up_feats, intrs, c2ws, stage_range)
            
            feats, frustum_mask = self.volume.back_proj_multiscale(features, up_coords, intrs, c2ws, s)
            
            feats = feats[frustum_mask]
            up_coords = up_coords[frustum_mask]
            
            if s > 0:
                up_feats = up_feats[frustum_mask]
                feats = torch.cat([feats, up_feats], dim=1)

            # torchsparse V2.1 move batch to first dim
            r_coords = torch.cat([torch.zeros_like(up_coords[:, :1]), up_coords], dim=1)
            sparse_feats = SparseTensor(feats=feats, coords=r_coords.to(torch.int32))  # - directly use sparse tensor to avoid point2voxel operations

            out_feats, feats = self.reg_network(sparse_feats, s)
            
            matching_volume, mask_volume = self.volume.sparse2dense(out_feats[:, :1], up_coords, matching_volume)
            
            feat_volume = out_feats[:, 1:]
            sparse_idx = self.volume.get_index(up_coords)
            
            volumes_all.append(feat_volume)
            sparse_idx_all.append(sparse_idx.detach())
            mask_volumes_all.append(mask_volume.detach())
            
            depths, occ_regs = self.matching_field(ipts, matching_volume, s, self.range_ratios, depths, perturb=perturb)
            
            outputs[f"depth_stage{s}"] = depths[0]
            outputs[f"depth_src_stage{s}"] = depths[ipts["src_idx"]] if "src_idx" in ipts else depths[0]
        
        return outputs, volumes_all, sparse_idx_all, mask_volumes_all, matching_volume
        
    def forward(self, mode, ipts, cos_anneal_ratio=1.0, step=None):
        imgs = ipts["imgs"]  # (nv, 3, h, w)
        
        outputs = {}
        if not self.has_vol:
            features = self.feature_network(imgs)   # coarse to fine
            mf_outputs, volumes_all, sparse_idx_all, mask_volumes_all, matching_volume = self.build_volumes(ipts, features, perturb=(mode=="train"))
            outputs.update(mf_outputs)
            if (step is not None) and (step%2==0):
                print("load image feature ckpt")
                last_state = self.feature_network.state_dict()
                self.match_feature_network.load_state_dict(last_state, strict=True)
                for param in self.match_feature_network.parameters():
                    param.requires_grad = False
            with torch.no_grad():
                match_features = self.match_feature_network(imgs)
        else:
            view_ids = ipts["view_ids"]  # list of int
            volumes_all = self.volumes
            sparse_idx_all = self.sparse_idxes
            mask_volumes_all = self.mask_volmes
            
            matching_volume = self.matching_volume
            features = [feat[view_ids] for feat in self.features]
            match_features = [feat[view_ids] for feat in self.features]

        surface_outputs = self.implicit_surface(mode, ipts, matching_volume, volumes_all[::-1], sparse_idx_all[::-1], mask_volumes_all[::-1], features[::-1], match_features[::-1], cos_anneal_ratio, step)
            
        outputs.update(surface_outputs)
        
        return outputs