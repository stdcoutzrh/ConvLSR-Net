import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import math

config = {
    "tiny":[64, 128, 256, 512],
    "small": [96, 192, 384, 768],
    "base": [128, 256, 512, 1024]
}


# MSC-FFN
class MultiScaleConvMlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super(MultiScaleConvMlp,self).__init__()

        self.dwconv1 = nn.Conv2d(in_features, in_features//2, 1,1)
        self.dwconv2 = nn.Conv2d(in_features, in_features//4, kernel_size = 3, stride = 1, padding = 1)
        self.dwconv3 = nn.Conv2d(in_features, in_features//4, kernel_size = 7, stride = 1, padding = 3)

        self.act = act_layer()

        self.fc1 = nn.Conv2d(in_features, in_features*4, kernel_size = 1, stride = 1, padding = 0)
        self.fc2 = nn.Conv2d(in_features*4, in_features, kernel_size = 1, stride = 1, padding = 0)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
 
        x = blc2bchw(x,H,W)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x3 = self.dwconv3(x)

        x = torch.cat([x1,x2,x3],dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = bchw2blc(x,H,W)
        return x


def blc2bchw(x,h,w):
    b,l,c = x.shape
    assert l==h*w, "in blc to bchw, h*w != l."
    return x.view(b,h,w,c).permute(0,3,1,2).contiguous()


def bchw2blc(x,h,w):
    b,c,_,_ = x.shape
    return x.permute(0,2,3,1).view(b,-1,c).contiguous()


def window_partition(x, window_size):
    """
        x: (B, H, W, C) Returns:windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C).contiguous()
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)   Returns:x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).contiguous()
    return x


class WindowMSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                attn_drop=0., proj_drop=0.,use_relative_pe=False):

        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.use_relative_pe = use_relative_pe
        if self.use_relative_pe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [num_windows*B, N, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_relative_pe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LSRFormer(nn.Module):
    def __init__(self,dim,heads):
        super(LSRFormer,self).__init__()

        self.dim = dim
        self.channel_ratio = 2
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(dim,dim//self.channel_ratio,2,2,0,groups=dim//8),
            nn.BatchNorm2d(dim//self.channel_ratio))

        self.h_conv = nn.Conv2d(dim//self.channel_ratio,dim//self.channel_ratio,2,2,0)
        self.w_conv = nn.Conv2d(dim//self.channel_ratio,dim//self.channel_ratio,2,2,0)

        self.global_attn = WindowMSA(dim//self.channel_ratio,(4,4),heads)
        self.local_attn = WindowMSA(dim//self.channel_ratio,(4,4),heads,use_relative_pe=True)

        self.conv_out = nn.Conv2d(dim//self.channel_ratio,dim,3,1,1,groups=dim//8)

        self.mlp = MultiScaleConvMlp(in_features=dim//self.channel_ratio, act_layer=nn.GELU, drop=0.1)
        self.norm =nn.LayerNorm(dim//self.channel_ratio)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_index(self,real_h):
        index = []
        windows = real_h // 4
        for i in range(windows):
            if i==0:
                index.append(4-1)
            elif i==windows-1:
                index.append(real_h-4)
            else:
                index.append(i*4)
                index.append(i*4+3)
        return index
  
    def forward(self,x):
        # input x: b c h w 
        # step1:reduce dim of x
        x_reduction = self.conv_reduce(x) # w,h /2 ; c /4 
        x_reduction = x_reduction.permute(0,2,3,1).contiguous() # bchw->bhwc  

        # add :pad feature maps to multiples of window size 4
        H,W = x_reduction.shape[1],x_reduction.shape[2]
        pad_r = int(((4 - W % 4) % 4)/2)
        pad_b= pad_l = pad_t = pad_r
        if pad_r>0:
            x_reduction = F.pad(x_reduction, (0, 0, pad_l, pad_r, pad_t, pad_b),mode='reflect')

        # get index of window border
        border_index = torch.Tensor(self.get_index(x_reduction.shape[2])).int().to(x_reduction.device)

        # long range attn
        x_h = torch.index_select(x_reduction ,1, border_index).permute(0,3,1,2).contiguous()  # [1, 16, 62, 128]
        x_h = self.h_conv(x_h).permute(0,2,3,1).contiguous()    # b c h w -> b h w c  [1, 31, 64, 16]
        b_,h_,w_,c_ = x_h.shape
        x_h = window_partition(x_h,[1,w_]).view(-1,1*w_,c_).contiguous()    # [31, 64, 16]

        x_w = torch.index_select(x_reduction,2,border_index).permute(0,3,2,1).contiguous()   # [1, 16, 62, 128]
        x_w = self.w_conv(x_w).permute(0,2,3,1).contiguous()    # b c h w -> b h w c  [1, 31, 64, 16]
        x_w = window_partition(x_w,[1,w_]).view(-1,1*w_,c_).contiguous()    # [31, 64, 16]
        
        x_total = torch.cat([x_h,x_w],dim=0)
        
        x_h,x_w = torch.chunk(self.global_attn(x_total),2,0)
        x_h,x_w = x_h.contiguous(),x_w.contiguous()
        x_h,x_w = window_reverse(x_h,[1,w_],h_,w_).permute(0,3,1,2).contiguous(),window_reverse(x_w,[1,w_],h_,w_).permute(0,3,2,1).contiguous()
        x_h,x_w = F.interpolate(x_h,scale_factor=2,mode='bilinear', align_corners=True),F.interpolate(x_w, scale_factor=2,mode='bilinear', align_corners=True)
        x_h,x_w = x_h.permute(0,2,3,1).contiguous(),x_w.permute(0,2,3,1).contiguous() # [1, 16, 62, 128] [1, 16, 128, 62]

        x_reduction.index_add_(1,border_index,x_h)
        x_reduction.index_add_(2,border_index,x_w)  # bhwc
        # long range attn end

        # short range attn   
        local_windows = window_partition(x_reduction,[4,4]).view(-1,16,x_reduction.shape[3]).contiguous()
        local_windows = self.local_attn(local_windows)

        # bhwc
        local_windows = window_reverse(local_windows,[4,4],x_reduction.shape[1],x_reduction.shape[2]).contiguous()   #torch.Size([1, 128, 128, 16])
        
        # add:
        if pad_r > 0:
            # remove pad
            x_reduction = x_reduction[:, pad_t:H+pad_t, pad_l:W+pad_t, :].contiguous()
            local_windows = local_windows[:, pad_t:H+pad_t, pad_l:W+pad_t, :].contiguous()

        bb,hh,ww,cc = local_windows.shape
        local_windows = local_windows.view(bb,hh*ww,cc).contiguous()

        # shortcut and mlp
        local_windows = self.mlp(self.norm(local_windows),hh,ww)
        local_windows = local_windows.view(bb,hh,ww,cc).contiguous() + x_reduction
        local_windows = local_windows.permute(0,3,1,2).contiguous()
        out =  F.interpolate(local_windows,scale_factor=2,mode='bilinear', align_corners=True)
        out = self.conv_out(out)
        return out


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.1, 
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
    
        self.sps = nn.ModuleList([
                LSRFormer(dims[0],8),
                LSRFormer(dims[1],8),
                LSRFormer(dims[2],8),
                LSRFormer(dims[3],8),
            ])
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.sps[i](x) + x
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[64, 128, 256, 512],mode="tiny", **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], mode = "base",**kwargs,)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


class LRDU(nn.Module):
    """
    large receptive detailed upsample
    """
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()

        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor),
            nn.Conv2d(in_c, in_c, 3 ,groups= in_c//4,padding=1), 
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, n_class=6, pretrained = True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        config=[96, 192, 384, 768] # channles of convnext-small
        self.backbone = convnext_small(pretrained,True)

        self.sps = nn.ModuleList([
                LSRFormer(config[2],8),
                LSRFormer(config[1],8),
                LSRFormer(config[0],8),
            ])

        self.Up5 = up_conv(ch_in=config[3], ch_out=config[3]//2)
        self.Up_conv5 = conv_block(ch_in=config[3], ch_out=config[3]//2)

        self.Up4 = up_conv(ch_in=config[2], ch_out=config[2]//2)
        self.Up_conv4 = conv_block(ch_in=config[2], ch_out=config[2]//2)

        self.Up3 = up_conv(ch_in=config[1], ch_out=config[1]//2)
        self.Up_conv3 = conv_block(ch_in=config[1], ch_out=config[1]//2)

        self.Up4x = LRDU(config[0],4)      
        self.convout = nn.Conv2d(config[0], n_class, kernel_size=1, stride=1, padding=0)                
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x128,x64,x32,x16 = self.backbone(x)

        d32 = self.Up5(x16)
        d32 = torch.cat([x32,d32],dim=1)
        d32 = self.Up_conv5(d32)
        d32 = self.sps[0](d32) + d32
 
        d64 = self.Up4(d32)
        d64 = torch.cat([x64,d64],dim=1)
        d64 = self.Up_conv4(d64)
        d64 = self.sps[1](d64) + d64

        d128 = self.Up3(d64)
        d128 = torch.cat([x128,d128],dim=1)
        d128 = self.Up_conv3(d128)
        d128 = self.sps[2](d128) + d128

        d512 = self.Up4x(d128)
        out = self.convout(d512)
        return out


if __name__ == "__main__":

    model = Model(6,False)
    img = torch.rand((1,3,512,512))
    output = model(img)
    print(output.shape)
    
    if 1:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        flops = FlopCountAnalysis(model, img)
        print("FLOPs: %.4f G" % (flops.total()/1e9))

        total_paramters = 0
        for parameter in model.parameters():
            i = len(parameter.size())
            p = 1
            for j in range(i):
                p *= parameter.size(j)
            total_paramters += p
        print("Params: %.4f M" % (total_paramters / 1e6)) 

    """
    FLOPs: 71.0525 G
    Params: 68.0840 M
    """
        