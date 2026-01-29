import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time
import pandas as pd
import os
import math
import scipy.sparse as sp
from lapy import TriaMesh, Solver
from datetime import datetime
from einops import rearrange
from timm.layers import trunc_normal_
from torch.nn.utils import spectral_norm
from utils.utils import count_params, LpLoss, UnitGaussianNormalizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def log_print(text, log_file):
    print(text)
    with open(log_file, "a") as f:
        f.write(text + "\n")

def save_args_to_txt(args, log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"{datetime.now()} -- Experiment Parameter Record\n")
        f.write("=" * 40 + "\n\n")
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
    def __init__(self, dim, external_basis, heads=8, dim_head=64, dropout=0., freq_num=64): 
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        
        self.in_project_fx = nn.Linear(dim, inner_dim)        
        
        self.mlp_trans_weights = nn.Parameter(torch.empty((dim_head, dim_head)))
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))
        self.in_project_gates = nn.Linear(dim_head, freq_num)
        
        self.perceiver = PhysicsAware()
        self.high_low_frequency_fusion = High_low_frequency_fusion(dim_head, freq_num)

        self.layernorm2 = nn.LayerNorm((freq_num, dim_head))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        basis = external_basis[:, :freq_num] 
        basis = F.normalize(basis, p=2, dim=0) 
        self.register_buffer('inver', basis.unsqueeze(0).unsqueeze(0)) 
        
    def forward(self, x):
        B, N, C = x.shape
        
        fx_mid = self.in_project_fx(x)
        fx_mid = fx_mid.view(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        
        eigen_gate = F.softmax(self.in_project_gates(fx_mid) / torch.clamp(self.temperature, min=0.1, max=5), dim=-1) 
        
        spectral_embedding = self.inver 
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

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()
        act_layer = nn.GELU if act == 'gelu' else nn.ReLU
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_layer())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act_layer()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(len(self.linears)):
            if self.res: x = self.linears[i](x) + x
            else: x = self.linears[i](x)
        return self.linear_post(x)

class Mixer_Block_Unstructured(nn.Module):
    def __init__(self, num_heads, hidden_dim, external_basis, dropout=0., freq_num=64):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Mixer(
            hidden_dim, external_basis, heads=num_heads, 
            dim_head=hidden_dim // num_heads, dropout=dropout, freq_num=freq_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, hidden_dim, n_layers=0, res=False)
        
    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx

class HFR_Model_Unstructured(nn.Module):
    def __init__(self, external_basis, space_dim=2, n_layers=4, n_hidden=128, n_head=4, 
                 freq_num=32, fun_dim=1, out_dim=1, dropout=0.0):
        super().__init__()
        
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False)
        
        self.blocks = nn.ModuleList([
            Mixer_Block_Unstructured(
                num_heads=n_head, 
                hidden_dim=n_hidden, 
                external_basis=external_basis, 
                dropout=dropout,
                freq_num=freq_num
            ) for _ in range(n_layers)
        ])
        
        self.last_layer = nn.Linear(n_hidden, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx):
        inp = torch.cat((x, fx), -1)
        feat = self.preprocess(inp)
        
        for block in self.blocks:
            feat = block(feat)
            
        return self.last_layer(feat)

def to_torch_sparse(scipy_matrix, device='cuda'):
    coo = scipy_matrix.tocoo()
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(coo.data)
    shape = coo.shape
    
    return torch.sparse_coo_tensor(i, v, torch.Size(shape)).to(device)

def main(args):  
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, "logs", args.CaseName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    log_file = os.path.join(save_path, 'log.txt')
    save_args_to_txt(args, log_file)
    log_print(f"Saving path: {save_path}", log_file)

    print("\n=============================")
    print("CUDA available: " + str(torch.cuda.is_available()))
    print("=============================\n")

    PATH = args.data_dir
    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    
    max_grad_norm = 1.0 
    
    beta_h1 = 0.1
    
    n_hidden = 128      
    freq_num = 128 
    n_layers = 5    
    n_heads = 4
    
    s = args.size_of_nodes

    log_print("Loading data and calculating LBO...", log_file)
    data = sio.loadmat(PATH)

    k = 128
    Points = np.vstack((data['nodes'].T, np.zeros(s).reshape(1,-1)))
    mesh = TriaMesh(Points.T,data['elements'].T-1)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)
    LBO_TENSOR = torch.from_numpy(LBO_MATRIX).float().cuda() 
    
    log_print("Constructing stiffness matrix for H1 Loss...", log_file)
    stiffness_matrix = fem.stiffness 
    S_tensor = to_torch_sparse(stiffness_matrix, device='cuda')
    
    S_tensor = S_tensor.coalesce()
    
    x_dataIn = torch.Tensor(data['Input'])    
    y_dataIn = torch.Tensor(data['Output'])
    pos_data = torch.Tensor(mesh.v[:, :2]).float().cuda() 
    
    x_data = x_dataIn 
    y_data = y_dataIn 
    
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    log_print(f'Training set shape: {x_train.shape}, Label shape: {y_train.shape}', log_file)
    
    norm_x  = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)
    
    norm_y  = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)

    x_train = x_train.reshape(ntrain, -1, 1)
    x_test = x_test.reshape(ntest, -1, 1)
    y_train = y_train.reshape(ntrain, -1, 1)
    y_test = y_test.reshape(ntest, -1, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)
    
    model = HFR_Model_Unstructured(
        external_basis=LBO_TENSOR, 
        space_dim=2, 
        n_layers=n_layers, 
        n_hidden=n_hidden, 
        n_head=n_heads, 
        freq_num=freq_num,
        fun_dim=1,
        out_dim=1,
        dropout=0.0
    ).cuda()
    
    log_print(f"Model initialization completed. Total parameters: {count_params(model)}", log_file)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    epochs=epochs, steps_per_epoch=len(train_loader))
    
    myloss = LpLoss(size_average=False)
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    
    best_val_err = float('inf')
    best_epoch = 0
    total_time = 0.0

    log_print(f"Start training... (H1 Loss Beta = {beta_h1})\n", log_file)

    for ep in range(epochs):
        model.train()
        train_l2 = 0
        train_h1 = 0 
        time_start = time.perf_counter()
        
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            batch_pos = pos_data.unsqueeze(0).repeat(x.shape[0], 1, 1)
            
            optimizer.zero_grad()
            out = model(batch_pos, x) 
            
            err = out.view(batch_size, -1) - y.view(batch_size, -1)
            l2_term = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            err_t = err.t().contiguous()
            S_err_t = torch.sparse.mm(S_tensor, err_t)
            numerator = torch.sum(err * S_err_t.t(), dim=1)
            
            with torch.no_grad():
                y_flat = y.view(batch_size, -1)
                y_t = y_flat.t().contiguous()
                S_y_t = torch.sparse.mm(S_tensor, y_t)
                denominator = torch.sum(y_flat * S_y_t.t(), dim=1) + 1e-6
            
            h1_rel = (numerator / denominator).mean()
            
            loss = l2_term + beta_h1 * h1_rel
            
            loss.backward() 
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()
            train_h1 += h1_rel.item() 

            scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                batch_pos = pos_data.unsqueeze(0).repeat(x.shape[0], 1, 1)
                
                out = model(batch_pos, x)
                
                out_real = norm_y.decode(out.view(batch_size, -1).cpu())
                y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
                test_l2 += myloss(out_real, y_real).item()                

        train_l2 /= ntrain
        train_h1 /= len(train_loader) 
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        
        time_end = time.perf_counter()
        epoch_time = time_end - time_start
        total_time += epoch_time

        print()
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (
            f"Epoch [{ep+1:03d}/{args.epochs}]\n"
            f"Train Numerical Error (L2): {train_l2:.6f}\n"
            f"Train Derivative Error (H1): {train_h1:.6f}\n"
            f"Test Error (Test)  : {test_l2:.6f}\n"
            f"Current LR         : {current_lr:.8f}\n"
            f"Time cost          : {epoch_time:.2f} s\n"
            f"================================================================\n"
        )
        log_print(log_msg, log_file)
        
        if test_l2 < best_val_err:
            best_val_err = test_l2
            diff_epoch = ep - best_epoch
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            save_msg = f"-> [Update] Best model saved! Error: {best_val_err:.5e}\n"
        else:
            save_msg = f"-> [Keep] Current best: {best_val_err:.5e} (Epoch {best_epoch+1})\n"

        log_print(save_msg, log_file)

    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))
    print("Final model saved.\n")
    log_print("Saving prediction results...\n", log_file)
    print()

    train_loader_final = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=1, shuffle=False)
    test_loader_final = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    
    pre_train = torch.zeros(y_train.shape)
    pre_test = torch.zeros(y_test.shape)
    
    model.eval()
    
    index = 0
    with torch.no_grad():
        for x, y in train_loader_final:
            x, y = x.cuda(), y.cuda()
            batch_pos = pos_data.unsqueeze(0) 
            
            out = model(batch_pos, x)
            out_decoded = norm_y.decode(out.view(1, -1).cpu())
            pre_train[index, :] = out_decoded.view(-1, 1) 
            index += 1
            
    index = 0
    with torch.no_grad():
        for x, y in test_loader_final:
            x, y = x.cuda(), y.cuda()
            batch_pos = pos_data.unsqueeze(0) 

            out = model(batch_pos, x)
            out_decoded = norm_y.decode(out.view(1, -1).cpu())
            pre_test[index, :] = out_decoded.view(-1, 1) 
            index += 1
            
    loss_dict = {'train_error': train_error, 'test_error': test_error}
    
    pred_dict = {
        'pre_test': np.ascontiguousarray(pre_test.cpu().numpy()), 
        'pre_train': np.ascontiguousarray(pre_train.cpu().numpy()),
        'y_test': np.ascontiguousarray(norm_y.decode(y_test.cpu()).numpy()), 
        'y_train': np.ascontiguousarray(norm_y.decode(y_train.cpu()).numpy())
    }
    
    sio.savemat(os.path.join(save_path, 'loss.mat'), mdict=loss_dict)                                                     
    np.savez(os.path.join(save_path, 'pre.npz'), **pred_dict)
    
    print(f"All results saved successfully.\nTotal training time: {total_time:.2f} seconds")
    print(f"Best test error: {best_val_err:.5e} (Epoch {best_epoch+1})")

if __name__ == "__main__":
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            
    for i in range(1): 
        print('====================================')
        print('NO.'+str(i)+' repetition......')
        
        for args in [
                        {'modes': 128,  
                        'width': 32,
                        'size_of_nodes': 2673,
                        'batch_size': 4, 
                        'epochs': 1000,
                        'data_dir': './data/Turbulence',
                        'num_train': 300, 
                        'num_test': 100,
                        'CaseName': 'Turbulence_'+str(i),
                        'lr' : 0.001},
                    ]:
            args = objectview(args)
        main(args)