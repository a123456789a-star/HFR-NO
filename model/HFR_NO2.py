import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from timm.layers import trunc_normal_
from model.Embedding import timestep_embedding
from model.spectral_embedding import robust_spectral
import math
from einops import rearrange

activation_dict = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

# High-Frequency Reconstruction Block
class HFR(nn.Module):
    def __init__(self, in_features, out_features, grid_size=7, basis_scale=1.0, use_base_linear=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        if use_base_linear:
            self.base_linear = spectral_norm(nn.Linear(in_features, out_features))
        else:
            self.register_parameter('base_linear', None)
        
        steps = torch.arange(grid_size).float()
        chebyshev_points = -torch.cos((2 * steps + 1) / (2 * grid_size) * math.pi) * 1.5
        self.translation = nn.Parameter(chebyshev_points.view(1, 1, grid_size))
        
        scales = torch.ones(1, 1, grid_size) * basis_scale
        if grid_size > 1:
            scales = scales * torch.logspace(-0.5, 0.5, grid_size).view(1, 1, grid_size)
        self.scale = nn.Parameter(scales, requires_grad=False) 
        
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features * grid_size))
        self.soft_threshold = nn.Parameter(torch.tensor([0.005])) 
        
        self.output_scale = nn.Parameter(torch.zeros([1])) 
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.wavelet_weights, std=0.01)
        if self.base_linear is not None:
            nn.init.zeros_(self.base_linear.weight)
            nn.init.zeros_(self.base_linear.bias)

    def soft_thresholding(self, x, threshold):
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        
        base_out = self.base_linear(F.silu(x_flat)) if self.base_linear is not None else 0
        
        safe_scale = torch.clamp(self.scale.abs(), min=0.1) 
        x_norm = torch.tanh(x_flat) * 2.5 
        x_expanded = (x_norm.unsqueeze(-1) - self.translation) / safe_scale
        
        wavelet_basis = (1 - x_expanded**2) * torch.exp(-0.5 * x_expanded**2)
        wavelet_basis_flat = wavelet_basis.view(x_flat.shape[0], -1)
        
        kan_out = F.linear(wavelet_basis_flat, self.wavelet_weights)
        
        real_threshold = F.softplus(self.soft_threshold)
        kan_out = self.soft_thresholding(kan_out, real_threshold) 
        
        
        out = (base_out + kan_out)* self.output_scale

        return out.view(*original_shape[:-1], self.out_features)

class PhysicsAware(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_norm = nn.LayerNorm(5) 
        self.condenser = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(5, 2)
        )
        self.gate_bias = nn.Parameter(torch.tensor([-8.0, -8.0])) 

    def forward(self, spectral_feature):
        B, H, G, C = spectral_feature.shape
        energy = torch.norm(spectral_feature, p=2, dim=-1).clamp(min=1e-6) 
        
        tail_energy = energy[:, :, G//2:] 
        tail_mean = tail_energy.mean(dim=-1) + 1e-6
        tail_max = tail_energy.max(dim=-1)[0]
        saliency = torch.log1p((tail_max / tail_mean).mean(dim=1, keepdim=True))

        total_sum = energy.sum(dim=-1) + 1e-6
        tail_sum = tail_energy.sum(dim=-1)
        tail_ratio = (tail_sum / total_sum).mean(dim=1, keepdim=True)
        
        mid_energy = energy[:, :, G//4:G//2]
        mid_mean = mid_energy.mean(dim=-1) + 1e-6
        decay_gap = torch.log1p((mid_mean / tail_mean).mean(dim=1, keepdim=True))

        psd_norm = energy / total_sum.unsqueeze(-1)
        entropy = -(psd_norm * torch.log(psd_norm + 1e-8)).sum(dim=-1).mean(dim=1, keepdim=True)
        
        low_energy = energy[:, :, :G//8].sum(dim=-1)
        low_freq_ratio = (low_energy / total_sum).mean(dim=1, keepdim=True)
        feat = torch.stack([saliency, tail_ratio, decay_gap, entropy, low_freq_ratio], dim=-1)

        feat = self.feature_norm(feat)
        raw_gates = self.condenser(feat) + self.gate_bias
        
        gates = torch.sigmoid(raw_gates) 
        
        return gates.view(-1, 1, 1, 2)

class High_low_frequency_fusion(nn.Module):
    def __init__(self, dim_head, freq_num):
        super().__init__()
        self.rec_stream = HFR(in_features=dim_head, out_features=dim_head, grid_size=7, use_base_linear=False)
        self.freq_weight = nn.Parameter(torch.zeros(1, 1, freq_num, 1))
    
        freq_indices = torch.arange(freq_num).float()
        k_norm = freq_indices / freq_num 
        self.register_buffer('k_squared', (k_norm**2).view(1, 1, -1, 1))

    def forward(self, smooth_baseline, gates):
        alpha = gates[..., 0:1] 
        beta = gates[..., 1:2]

        rec_out = self.rec_stream(smooth_baseline)

        weight = torch.sigmoid(self.freq_weight)

        base_mag = smooth_baseline.abs().mean(dim=-1, keepdim=True).detach()
        rec_mag = rec_out.abs().mean(dim=-1, keepdim=True) + 1e-6
        scale_factor = torch.min(torch.tensor(1.0).to(rec_out.device), base_mag / rec_mag * 0.5)
        
        rec_out = alpha * (rec_out * scale_factor * weight)

        beta_safe = torch.clamp(beta, max=0.8) 
        damping_factor = torch.exp(-beta_safe * 4.0 * self.k_squared)
        
        out = (smooth_baseline * damping_factor) + rec_out

        return out

class Mixer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., freq_num=64, H=101, W=31, kernelsize=3): 
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        
        self.H = H
        self.W = W

        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernelsize, 1, kernelsize//2 )  
        # self.in_project_fx = nn.linear(dim, inner_dim)      
        self.mlp_trans_weights = nn.Parameter( torch.empty((dim_head, dim_head)) )
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))
        self.in_project_gates = nn.Linear(dim_head, freq_num)
        for l in [self.in_project_gates]:
            torch.nn.init.orthogonal_(l.weight) 

        self.perceiver = PhysicsAware()
        self.high_low_frequency_fusion = High_low_frequency_fusion(dim_head, freq_num)

        self.layernorm2 = nn.LayerNorm( ( freq_num, dim_head ) )
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        max_freq_num = freq_num + 3
        try:
            BASE_MATRIX = robust_spectral.grid_spectral_points(self.H, self.W, k=max_freq_num)
            BASE_MATRIX = BASE_MATRIX.reshape(-1, max_freq_num).cuda()
        except Exception as e:
            print(f"Warning: using random basis. {e}")
            BASE_MATRIX = torch.randn(self.H * self.W, max_freq_num).cuda()

        inver = BASE_MATRIX[:, :freq_num]
        inver = F.normalize(inver, p=2, dim=-1)
        self.inver = nn.Parameter(inver, requires_grad=False)
        
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()
        
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        eigen_gate = F.softmax(self.in_project_gates(fx_mid) / torch.clamp(self.temperature, min=0.1, max=5), dim=-1) 
        spectral_embedding = self.inver[None, None, :, :]
        eigens = eigen_gate * spectral_embedding 

        spectral_feature = torch.einsum("bhnc,bhng->bhgc", fx_mid, eigens)

        bsize, hsize, gsize, csize = spectral_feature.shape
        smooth_baseline = self.layernorm2(spectral_feature.reshape( -1, gsize, csize )).reshape( bsize, hsize, gsize, csize )
        smooth_baseline = torch.einsum("bhgi,io->bhgo", smooth_baseline, self.mlp_trans_weights)
        gates = self.perceiver(smooth_baseline)
        out_spectral = self.high_low_frequency_fusion(smooth_baseline, gates)

        out_x = torch.einsum("bhgo,bhng->bhno", out_spectral, eigens)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)

class Mlp(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(Mlp, self).__init__()

        if act in activation_dict.keys():
            act = activation_dict[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x
    

class MixerBlock(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            freq_num=32,
            H=85,
            W=85,
            kernelsize=3,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = Mixer(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,dropout=dropout, freq_num=freq_num, H=H, W=W, kernelsize=kernelsize)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx



class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 freq_num=32,
                 ref=8,
                 unified_pos=False,
                 H=85,
                 W=85, 
                 kernelsize=3,
                 ):
        super(Model, self).__init__()
        self.__name__ = "HFR"
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = Mlp(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = Mlp(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([MixerBlock(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      freq_num=freq_num,
                                                      H=H,
                                                      W=W,
                                                      kernelsize=kernelsize,
                                                      last_layer=(layer_id == n_layers - 1))
                                     for layer_id in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx

if __name__ == "__main__":
    from scipy import fftpack
    model = Model(space_dim=2, 
                  n_layers=4, 
                  n_hidden=128, 
                  dropout=0.0,
                  n_head=4, 
                  Time_Input=False, 
                  mlp_ratio=2, 
                  fun_dim=0,
                  out_dim=1, 
                  kernelsize=3, 
                  freq_num=32, 
                  ref=8,
                  unified_pos=True, 
                  H=85, W=85).cuda()
    x = torch.randn(2, 85*85, 2).cuda()
    print(f"Input shape: {x.shape}")
    fx = None
    out = model(x, fx)
    print(f"Output shape: {out.shape}")