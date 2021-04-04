import functools
import torch
from torch import nn
from torch.nn import functional as F


class SemanticConsistencyLoss(nn.Module):
    def __init__(self, content_layers=None):
        super(SemanticConsistencyLoss, self).__init__()
        if content_layers is not None:
            self.feat_content_layers = content_layers
        else:
            self.feat_content_layers = {'relu3_2': 1.0}

    def _l1_loss(self, o, t):
        return torch.mean(torch.abs(o - t))

    def forward(self, out, target):
        content_loss_lst = [self.feat_content_layers[layer] * self._l1_loss(out[layer], target[layer])
                            for layer in self.feat_content_layers]
        content_loss = functools.reduce(lambda x, y: x + y, content_loss_lst)
        return content_loss


class IDMRFLoss(nn.Module):
    def __init__(self):
        super(IDMRFLoss, self).__init__()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, feature_maps):
        return feature_maps / torch.sum(feature_maps, dim=1, keepdim=True)

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        return cdist / (torch.min(cdist, dim=1, keepdim=True)[0] + epsilon)

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def _mrf_loss(self, o, t):
        o_feats = o - torch.mean(t, 1, keepdim=True)
        t_feats = t - torch.mean(t, 1, keepdim=True)
        o_normalized = o_feats / torch.norm(o_feats, p=2, dim=1, keepdim=True)
        t_normalized = t_feats / torch.norm(t_feats, p=2, dim=1, keepdim=True)

        cosine_dist_l = []
        b_size = t.size(0)

        for i in range(b_size):
            t_feat_i = t_normalized[i:i + 1, :, :, :]
            o_feat_i = o_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(t_feat_i)

            cosine_dist_i = F.conv2d(o_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)

        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, out, target):
        style_loss_list = [self.feat_style_layers[layer] * self._mrf_loss(out[layer], target[layer])
                           for layer in self.feat_style_layers]

        content_loss_list = [self.feat_content_layers[layer] * self._mrf_loss(out[layer], target[layer])
                             for layer in self.feat_content_layers]

        return functools.reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style + \
            functools.reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content
