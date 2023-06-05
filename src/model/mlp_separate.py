# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.util as util

class MLP(nn.Module):
    def __init__(self, D=8, W=256, d_in=3, d_in_views=3, d_latent=56, skips=[4], combine_type="average", **kwargs):
        """
        :param D: the depth of the mlp
        :param W: the number of the channels of the mlp
        :param d_in: channel(xyz) + channel(views)
        :param d_in_views: channel(views)
        :param d_latents: channel(image features)
        :param skips:List, skip connect

        """
        super(MLP, self).__init__()
        self.D = D  # 网络深度
        self.W = W  # 每层设置的通道数
        self.input_ch_xyz = d_in-d_in_views  # xyz的通道数
        self.input_ch_views = d_in_views  # direct通道数
        self.input_ch_features = d_latent  # 提取的图片特征的通道数
        self.skips = skips  # 加入输入的位置
        self.combine_type = combine_type
        # 生成D层全连接网络，在skip3+1层加入input_pts
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch_xyz + self.input_ch_features, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch_xyz + self.input_ch_features, W) for i in
             range(D - 1)])

        # Implementation according to the official code release
        # 对view处理的网络层
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_views + W, W // 2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        # 输出特征 alpha和rgb的最后层
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)

    def forward(self, x,combine_inner_dims=(1,), **kwargs):
        # 将xyz与view分开
        input_pts, input_views, input_features = \
            torch.split(x, [self.input_ch_xyz, self.input_ch_views, self.input_ch_features], dim=-1)
        h = torch.cat([input_pts, input_features], dim=-1)
        # 全连接网络
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, input_features, h], -1)
        # 网络输出计算
        mean1 = util.combine_interleaved(h, combine_inner_dims, self.combine_type)  # average
        alpha = self.alpha_linear(mean1)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        mean2 = util.combine_interleaved(h, combine_inner_dims, self.combine_type)  # average
        rgb = self.rgb_linear(mean2)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs

    @classmethod
    def from_conf(cls, conf, d_in=3, d_in_views=3, d_latnets=56, **kwargs):
        # PyHocon construction
        return cls(
            D=8,
            W=256,
            d_in=d_in,
            d_in_views=d_in_views,
            d_latents=d_latnets,
            skips=[4],
            combine_type="average",
            **kwargs
        )
def get_mlp_model(conf,):
    input_feature_ch = conf.UNet.base_channels * 7
    if conf.code.include_input:
        input_ch = conf.code.xyz_freqs * 2 * 3 + 3
        input_ch_views = conf.code.view_freqs * 2 * 3 + 3
    else:
        input_ch = conf.code.xyz_freqs * 2 * 3
        input_ch_views = conf.code.view_freqs * 2 * 3
    mlp_coarse = MLP(conf.mlp_coarse.D, conf.mlp_coarse.W, input_ch, input_ch_views,
                     input_feature_ch, conf.mlp_coarse.skips, conf.use_img_code)
    mlp_fine = MLP(conf.mlp_fine.D, conf.mlp_fine.W, input_ch, input_ch_views,
                   input_feature_ch, conf.mlp_fine.skips,conf.use_img_code)
    return mlp_coarse, mlp_fine


if __name__ == '__main__':
    mlp = MLP(input_ch=63, input_ch_views=27, input_feature_ch=8, skips=[4])
    test_data = torch.ones(5, 98)
    outputs = mlp(test_data)
    print(outputs.shape)
