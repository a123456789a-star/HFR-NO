import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time
import pandas as pd
from utils.utils import count_params, LpLoss, GaussianNormalizer
import os
import math
from einops import rearrange
from timm.layers import trunc_normal_
from torch.nn.utils import spectral_norm
from datetime import datetime
import argparse

parser = argparse.ArgumentParser('HFR_BloodFlow_Autoregression_Full')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_out', type=int, default=10)
parser.add_argument('--step', type=int, default=1)

parser.add_argument('--data_dir', type=str, default='./data/BloodFlow.mat')
parser.add_argument('--LBO_dir', type=str, default='./data/BloodFlow_LBO_basis/LBO_basis.mat')
parser.add_argument('--CaseName', type=str, default='BloodFlow_NS_Style_Autoreg')
parser.add_argument('--modes', type=int, default=64)

args = parser.parse_args()


def log_print(text, log_file):
    print(text)
    with open(log_file, "a") as f:
        f.write(text + "\n")

def save_args_to_txt(args, log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"{datetime.now()} -- Experiment Parameter Configuration (Autoregression Mode)\n")
        f.write("=" * 40 + "\n")
        for key, value in args.__dict__.items():
            f.write(f"{key:<25}: {value}\n")
        f.write("\n" + "=" * 40 + "\n")

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
    def __init__(self, dim, matrix, heads=8, dim_head=64, dropout=0., freq_num=64, lastlayer=False): 
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.in_temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.out_temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.lastlayer=lastlayer

        self.in_project_x = spectral_norm(nn.Linear(dim, inner_dim))
        self.in_project_fx = spectral_norm(nn.Linear(dim, inner_dim))
        self.mlp_trans_weights = nn.Parameter( torch.empty((dim_head, dim_head)) )
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))
        self.in_project_gates = nn.Linear(dim_head, freq_num)
        for l in [self.in_project_gates]:
            torch.nn.init.orthogonal_(l.weight) 

        self.perceiver = PhysicsAware()
        self.high_low_frequency_fusion = High_low_frequency_fusion(dim_head, freq_num)

        self.layernorm2 = nn.LayerNorm( ( freq_num, dim_head ) )

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.register_buffer('matrix', matrix) 
        self.matrix = F.normalize(self.matrix, p=2, dim=-1)
        
    def forward(self, x):
        B, N, C = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        eigen_gate = F.softmax(self.in_project_gates(fx_mid) / torch.clamp(self.in_temperature, min=0.1, max=5), dim=-1) 
        spectral_embedding = self.matrix[None, None, :, :]
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
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act=nn.GELU(), res=True):
        super(Mlp, self).__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act) for _ in range(n_layers)])

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
            act=nn.GELU(),
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            freq_num=32,
            matrix=None
    ):
        super().__init__()
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = Mixer(hidden_dim, matrix, heads=num_heads, dim_head=hidden_dim // num_heads,dropout=dropout, freq_num=freq_num)

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

class HFR_Model_Unstructured(nn.Module):
    def __init__(self, matrix,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 freq_num=32
                 ):
        super(HFR_Model_Unstructured, self).__init__()
        self.__name__ = 'HFR_Unstructured_WavKAN'
        self.preprocess = Mlp(fun_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=nn.GELU())

        self.n_hidden = n_hidden
        self.blocks = nn.ModuleList([MixerBlock(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=nn.GELU(),
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      freq_num=freq_num,
                                                      matrix=matrix,
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

    def forward(self, fx):
        fx = self.preprocess(fx)
        fx = fx + self.placeholder[None, None, :]

        for block in self.blocks:
            fx = block(fx)

        return fx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_path = f'./checkpoints/{args.CaseName}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = os.path.join(save_path, 'training_log.txt')
    save_args_to_txt(args, log_file)
    
    data = sio.loadmat(args.data_dir)
    LBO_data = sio.loadmat(args.LBO_dir)
    LBO_MATRIX = torch.tensor(LBO_data['LBO_basis']).float()
    
    velocity_x_raw = torch.Tensor(data['vel_x'])
    velocity_y_raw = torch.Tensor(data['vel_y'])
    velocity_z_raw = torch.Tensor(data['vel_z'])
    
    velocity_x = velocity_x_raw.permute(0, 2, 1) 
    velocity_y = velocity_y_raw.permute(0, 2, 1)
    velocity_z = velocity_z_raw.permute(0, 2, 1)
    
    batch_size, time_steps, node_count = velocity_x.shape  
    
    velocity_raw = torch.stack([velocity_x, velocity_y, velocity_z], dim=-1)
    
    boundary_condition_input = torch.Tensor(data['BC_time'])
    
    boundary_condition_raw = boundary_condition_input.unsqueeze(2).repeat(1, 1, node_count, 1)
    
    log_print(f"Data Shapes Corrected:", log_file)
    log_print(f"  Velocity: {velocity_raw.shape} (Batch, Time, Nodes, 3)", log_file)
    log_print(f"  BC      : {boundary_condition_raw.shape} (Batch, Time, Nodes, 6)", log_file)
    
    ntrain = 400
    ntest = 100
    
    train_boundary_condition = boundary_condition_raw[:ntrain]
    train_velocity = velocity_raw[:ntrain]
    test_boundary_condition = boundary_condition_raw[-ntest:]
    test_velocity = velocity_raw[-ntest:]
    
    log_print(f"Train Shape: {train_boundary_condition.shape} [Batch, Time, Nodes, Channels]", log_file)
    log_print(f"Test Shape:  {test_boundary_condition.shape}", log_file)
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_boundary_condition, train_velocity),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_boundary_condition, test_velocity),
        batch_size=args.batch_size, shuffle=False
    )

    fun_dim = (3 * args.T_in) + 6
    out_dim = 3
    
    log_print(f"Building Model (History: {args.T_in}, Input Dim: {fun_dim})...", log_file)
    
    BASE_MATRIX = LBO_MATRIX[:, :args.modes]

    model = HFR_Model_Unstructured(
        matrix=BASE_MATRIX,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        freq_num=args.modes,
        fun_dim=fun_dim,
        out_dim=out_dim,
        dropout=args.dropout
    ).to(device)
    
    log_print(f"Total Parameters: {count_params(model)}", log_file)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    
    myloss = LpLoss(size_average=False)
    best_val_err = float('inf')
    
    log_print("Start Training (Autoregressive Style)...", log_file)
    
    for ep in range(args.epochs):
        model.train()
        train_l2_step = 0
        t1 = time.time()
        
        for boundary_condition_batch, velocity_batch in train_loader:
            boundary_condition_batch = boundary_condition_batch.to(device)
            velocity_batch = velocity_batch.to(device)
            bsz = boundary_condition_batch.shape[0]
            
            history = velocity_batch[:, :args.T_in, :, :].permute(0, 2, 1, 3).reshape(bsz, -1, 3 * args.T_in)
            
            batch_total_loss = 0
            
            for t in range(0, args.T_out, args.step):
                target_idx = args.T_in + t
                
                current_boundary_condition = boundary_condition_batch[:, target_idx, :, :]
                model_input = torch.cat([history, current_boundary_condition], dim=-1)
                
                pred = model(model_input)
                
                target = velocity_batch[:, target_idx, :, :]
                loss = myloss(pred.reshape(bsz, -1), target.reshape(bsz, -1))
                batch_total_loss += loss
                
                history = torch.cat((history[..., 3:], target), dim=-1)
            
            train_l2_step += batch_total_loss.item()
            
            optimizer.zero_grad()
            batch_total_loss.backward()
            if args.dropout < 0.1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        test_l2_step = 0
        
        step_losses_sum = np.zeros(args.T_out)
        
        with torch.no_grad():
            for boundary_condition_batch, velocity_batch in test_loader:
                boundary_condition_batch = boundary_condition_batch.to(device)
                velocity_batch = velocity_batch.to(device)
                bsz = boundary_condition_batch.shape[0]
                
                history = velocity_batch[:, :args.T_in, :, :].permute(0, 2, 1, 3).reshape(bsz, -1, 3 * args.T_in)
                
                for t in range(0, args.T_out, args.step):
                    target_idx = args.T_in + t
                    
                    current_boundary_condition = boundary_condition_batch[:, target_idx, :, :]
                    model_input = torch.cat([history, current_boundary_condition], dim=-1)
                    
                    pred = model(model_input)
                    
                    target = velocity_batch[:, target_idx, :, :]
                    loss = myloss(pred.reshape(bsz, -1), target.reshape(bsz, -1))
                    
                    test_l2_step += loss.item()
                    step_losses_sum[t] += loss.item()
                    
                    history = torch.cat((history[..., 3:], pred), dim=-1)

        train_l2_avg = train_l2_step / ntrain / args.T_out
        test_l2_avg  = test_l2_step / ntest / args.T_out
        
        step_errors_avg = step_losses_sum / ntest
        
        epoch_time = time.time() - t1
        current_lr = optimizer.param_groups[0]['lr']
        
        steps_str = ", ".join([f"{val:.4f}" for val in step_errors_avg])
        
        log_msg = (
            f"Epoch [{ep+1:03d}/{args.epochs}] "
            f"Train Avg: {train_l2_avg:.6f} "
            f"Test Avg: {test_l2_avg:.6f} "
            f"Step Errors: [{steps_str}] "
            f"LR: {current_lr:.2e} "
            f"Time: {epoch_time:.2f}s"
        )
        log_print(log_msg, log_file)
        
        if test_l2_avg < best_val_err:
            best_val_err = test_l2_avg
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            log_print(f"-> Best Model Saved! Err: {best_val_err:.6f}", log_file)

    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))
    
    res_dict = {
        'step_errors': step_errors_avg,
        'final_test_loss': test_l2_avg,
        'best_test_loss': best_val_err
    }
    sio.savemat(os.path.join(save_path, 'result_metrics.mat'), mdict=res_dict)

    log_print("Training Finished.", log_file)

if __name__ == "__main__":
    main(args)