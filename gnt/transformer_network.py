import numpy as np
import torch
import torch.nn as nn

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.shared_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        
        q = self.shared_fc(q)
        with autocast():
            k = self.shared_fc(k)
            v = self.shared_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        torch.cuda.empty_cache()
        return x


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6).to(self.device)
        #self.attn_norm = nn.DataParallel(self.attn_norm)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6).to(self.device)
        #self.ff_norm = nn.DataParallel(self.ff_norm)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate).to(self.device)
        #self.ff = nn.DataParallel(self.ff)
        self.attn = Attention2D(dim, attn_dp_rate)
    
    def forward(self, q, k, pos, mask=None):
        residue = q
        with autocast():
            x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x


# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()
        self.shared_fc = nn.Linear(dim, dim, bias=False)
        if attn_mode in ["pos", "gate"]:
            self.pos_fc = nn.Sequential(
                nn.Linear(pos_dim, pos_dim), nn.ReLU(), nn.Linear(pos_dim, dim // 8)
            )
            self.head_fc = nn.Linear(dim // 8, n_heads)
        if attn_mode == "gate":
            self.gate = nn.Parameter(torch.ones(n_heads))
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.n_heads = n_heads
        self.attn_mode = attn_mode

    def forward(self, x, pos=None, ret_attn=False):
        q = self.shared_fc(x)
        k = self.shared_fc(x)
        v = self.shared_fc(x)

        q = q.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)
        v = v.view(x.shape[0], x.shape[1], self.n_heads, -1).permute(0, 2, 1, 3)

        if self.attn_mode in ["qk", "gate"]:
            attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            attn = torch.softmax(attn, dim=-1)
        elif self.attn_mode == "pos":
            pos = self.pos_fc(pos)
            attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            attn = torch.softmax(attn, dim=-1)
        if self.attn_mode == "gate":
            pos = self.pos_fc(pos)
            pos_attn = self.head_fc(pos[:, :, None, :] - pos[:, None, :, :]).permute(0, 3, 1, 2)
            pos_attn = torch.softmax(pos_attn, dim=-1)
            gate = self.gate.view(1, -1, 1, 1)
            attn = (1.0 - torch.sigmoid(gate)) * attn + torch.sigmoid(gate) * pos_attn
            attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.dp(attn)

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = out.view(x.shape[0], x.shape[1], -1)
        out = self.dp(self.out_fc(out))
        if ret_attn:
            return out, attn
        else:
            return out


# Ray Transformer
class Transformer(nn.Module):
    def __init__(
        self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None
    ):
        super(Transformer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6).to(self.device)
        #self.attn_norm = nn.DataParallel(self.attn_norm)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6).to(self.device)
        #self.ff_norm = nn.DataParallel(self.ff_norm)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate).to(self.device)
        #self.ff = nn.DataParallel(self.ff)
        self.attn = Attention(dim, n_heads, attn_dp_rate, attn_mode, pos_dim).to(self.device)
        #self.attn = nn.DataParallel(self.attn)

    def forward(self, x, pos=None, ret_attn=False):
        residue = x
        x = self.attn_norm(x)#
        x = self.attn(x, pos, ret_attn)
        
        if ret_attn:
            x, attn = x
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        if ret_attn:
            return x, attn.mean(dim=1)[:, 0]
        else:
            return x
        
    
class Conv2d_BN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
    ):
        super(Conv2d_BN,self).__init__()  
        self.conv = torch.nn.Conv2d(
            in_features, out_features, kernel_size, stride, pad, dilation, groups, bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_features)
        #self.bn1 = nn.LayerNorm(dim, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        #self.act_layer = act_layer() if act_layer is not None else nn.Identity()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv(x)))

        return x
          
'''class ResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
    ):
        super(ResBlock,self).__init__()
        hidden_features = int(out_features * 0.5)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv1 = Conv2d_BN(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        
        self.bn1 = nn.BatchNorm2d(hidden_features)
        #self.relu = nn.Hardswish()
        #self.bn1 = nn.LayerNorm(dim, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        
    def forward(self, x):
        identity = x
        with torch.no_grad():
            feat = self.conv1(x)
        with autocast():
            feat = self.relu(self.bn1(self.dwconv(feat)))
            feat = self.conv2(feat)
        del x

        return identity + feat'''

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
 
        #self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        hidden_features = int(outc * 0.25)
        self.conv1 = Conv2d_BN(inc, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features,
            2*kernel_size*kernel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
        )
        hidden_features = 2*kernel_size*kernel_size
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_BN(hidden_features, hidden_features)
        
        # 初始化模型的权重为0
        nn.init.constant_(self.dwconv.weight, 0)
 
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            # 操作过程与p_conv的方法一样
            nn.init.constant_(self.m_conv.weight, 0)
    
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
 
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
    
    def _get_p_0(self, h, w, N, dtype): #卷积核的中心坐标
    # 按图片横纵的尺寸来给每一个像素进行坐标赋值横纵坐标分别为（1， h）、（1， w）
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride),
            indexing='ij')
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p_n(self, N, dtype):#相对坐标
        p_n_x, p_n_y = torch.meshgrid(
        	# 形成p_n的坐标，也就是上图中的蓝色网格[-1， 2）
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            indexing='ij'  # 添加 indexing 参数
            )
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n
    
    #根据位置坐标，得到该位置的像素值
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w) 将特征图 x 展平
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        with autocast():
            index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

            x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    def _reshape_x_offset(self, x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
                   
    def forward(self, x):  # def forward(self, x, y): y在输入前使用class Transformer(nn.Module)作变换,改为offset = self.p_conv(y), 或者forward不输入y，将x作Transformer变换后，再输入self.p_conv
        #offset = self.p_conv(x) 
        identity = x
        offset = self.conv1(x)
        offset = self.relu(self.bn1(self.dwconv(offset)))
        offset = self.conv2(offset)
        
        
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
 
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
 
        if self.padding:
            x = self.zero_padding(x)
 
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype) #p_0 + p_n + offset
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor() #向下取整，detach() 用于阻断梯度传播
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)插值系数：距离权重
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)根据位置坐标，得到该位置的像素值
        x.requires_grad_(True)
        x_q_lt = self._get_x_q(x, q_lt, N) # 左上角的点在原始图片中对应的真实像素值
        x_q_rb = self._get_x_q(x, q_rb, N) # 右下角的点在原始图片中对应的真实像素值
        x_q_lb = self._get_x_q(x, q_lb, N) # 左下角的点在原始图片中对应的真实像素值
        x_q_rt = self._get_x_q(x, q_rt, N) # 右上角的点在原始图片中对应的真实像素值

        # (b, c, h, w, N)##
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                g_rb.unsqueeze(dim=1) * x_q_rb + \
                g_lb.unsqueeze(dim=1) * x_q_lb + \
                g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        out = identity + out
        return out


class GNT(nn.Module):
    def __init__(self, args, N_samples, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(GNT, self).__init__()
        self.args=args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch+3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        ).to(self.device)
        self.rgbfeat_fc = nn.DataParallel(self.rgbfeat_fc)
        self.DeformConv = DeformConv2d(inc=N_samples, outc=N_samples).to(self.device)
        self.DeformConv = self.DeformConv.to(self.device)
            
        self.p_fc= nn.Sequential(
            nn.Linear(args.netwidth*(args.trans_depth+1), args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        ).to(self.device)
        self.p_fc = nn.DataParallel(self.p_fc)
                   
        self.conv2=nn.Conv2d(args.netwidth*2, args.netwidth, kernel_size=1).to(self.device)
        self.conv2 = nn.DataParallel(self.conv2)
        self.conv3=nn.Conv2d(args.netwidth*3, args.netwidth, kernel_size=1).to(self.device)
        self.conv3 = nn.DataParallel(self.conv3)

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])
        self.shared_view_trans = Transformer2D(
            dim=args.netwidth,
            ff_hid_dim=int(args.netwidth * 4),
            ff_dp_rate=0.1,
            attn_dp_rate=0.1,
        ).to(self.device)
        self.shared_view_trans = nn.DataParallel(self.shared_view_trans)
        self.shared_ray_trans = Transformer(
            dim=args.netwidth,
            ff_hid_dim=int(args.netwidth * 4),
            n_heads=4,
            ff_dp_rate=0.1,
            attn_dp_rate=0.1,
        ).to(self.device)

        # 初始化 q_fc 和 Transformer 实例
        for i in range(args.trans_depth):
            self.view_crosstrans.append(self.shared_view_trans)
            self.view_selftrans.append(self.shared_ray_trans)
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                ).to(self.device)
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)
            
        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )  
    '''def parameter_memory_size(self, param):
        return param.element_size() * param.nelement()

    def model_memory_summary(self, model):
        total_memory = 0
        param_memory = {}
        
        for name, param in model.named_parameters(self):
            param_size = self.parameter_memory_size(param)
            param_memory[name] = param_size
            total_memory += param_size

        return param_memory, total_memory
    def tensor_memory_size(self,tensor):
        """计算张量的显存占用"""
        return tensor.element_size() * tensor.nelement()  '''  
    
    def forward(self, args, rgbfeat, ray_diff, mask, pts, ray_d):
        #args, rgbfeat, ray_diff, mask, pts, ray_d = args.to(device), rgbfeat.to(device), ray_diff.to(device), mask.to(device), pts.to(device), ray_d.to(device)
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)
                
        outputs_list=[]
        h_list=[]
        for i_scale in range(args.n_scales):
            rgb_feat=rgbfeat[i_scale] 
            # project rgb features to netwidth
            rgb_feat = self.rgbfeat_fc(rgb_feat)
            
            if args.run_cnn==True and args.n_scales!=1:
                #rgb_feat = self.InvRes(rgb_feat) #CNN
                #rgb_feat = checkpoint.checkpoint(self.DeformConv, rgb_feat)               
                rgb_feat = self.DeformConv(rgb_feat)
            # q_init -> maxpool           
            q = rgb_feat.max(dim=2)[0]#[1024, 32, 64]
            if args.run_cnn==True:
                x=[]
                x.append(q)

            # transformer modules
            for i, (crosstrans, q_fc, selftrans) in enumerate(
                zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
            ):
                # view transformer to update q   
                q = crosstrans(q, rgb_feat, ray_diff, mask)
                #q = checkpoint.checkpoint(crosstrans, q, rgb_feat, ray_diff, mask)
                
                # embed positional information
                if i % 2 == 0:
                    q = torch.cat((q, input_pts, input_views), dim=-1)
                    q = q_fc(q)
                
                # ray transformer
                q = selftrans(q, ret_attn=self.ret_alpha)
                #q = checkpoint.checkpoint(selftrans, q, self.ret_alpha)

                # 'learned' density
                if self.ret_alpha:
                    q, attn = q
                x.append(q) 
                #torch.cuda.empty_cache()
            # normalize & rgb
            if args.run_cnn==True and args.n_scales!=1:
                p=torch.cat(x, dim=-1)
                q=self.p_fc(p)
            
            h = self.norm(q)
            h = h.mean(dim=1)
            h_list.append(h)
            if args.n_scales!=1 and i_scale!=0:
                h = torch.cat(h_list, dim=1)
                h = h.unsqueeze(-1).unsqueeze(-1)
                if h.size(1)==args.netwidth*2:
                    h = self.conv2(h)
                if h.size(1)==args.netwidth*3:
                    h = self.conv3(h)
                h = h.squeeze(-1).squeeze(-1) 
            outputs = self.rgb_fc(h)
            outputs_list.append(outputs)
            
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs_list
