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
    def __init__(self, dim, mat_output, mat_input, heads=8, dim_head=64, dropout=0., freq_num=64, lastlayer=False): 
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.in_temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.out_temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.lastlayer=lastlayer
        self.mlp_trans_weights = nn.Parameter(torch.empty((dim_head, dim_head)))
        torch.nn.init.kaiming_uniform_(self.mlp_trans_weights, a=math.sqrt(5))

        self.in_project_fx = nn.Linear(dim, inner_dim)     
        self.out_project_gates_spectral = nn.Linear(dim_head, 1)

        self.in_project_gates = nn.Linear(dim_head, freq_num)
        self.out_project_gates = nn.Linear(dim_head, freq_num)

        self.perceiver = PhysicsAware()
        self.high_low_frequency_fusion = High_low_frequency_fusion(dim_head, freq_num)

        self.layernorm2 = nn.LayerNorm((freq_num, dim_head))
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        inbasis = mat_input[:, :freq_num] 
        inbasis = F.normalize(inbasis, p=2, dim=0) 
        self.register_buffer('eigens_input', inbasis.unsqueeze(0).unsqueeze(0)) 

        outbasis = mat_output[:, :freq_num] 
        outbasis = F.normalize(outbasis, p=2, dim=0) 
        self.register_buffer('eigens_output', outbasis.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        B, N, C = x.shape
        
        fx_mid = self.in_project_fx(x)
        fx_mid = fx_mid.view(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        
        in_eigen_gate = F.softmax(self.in_project_gates(fx_mid) / torch.clamp(self.in_temperature, min=0.1, max=5), dim=-1) 
        eigens_input = in_eigen_gate * self.eigens_input
        spectral_feature = torch.einsum("bhnc,bhng->bhgc", fx_mid, eigens_input)

        bsize, hsize, gsize, csize = spectral_feature.shape
        smooth_baseline = self.layernorm2(spectral_feature.reshape( -1, gsize, csize )).reshape( bsize, hsize, gsize, csize )
        smooth_baseline = torch.einsum("bhgi,io->bhgo", smooth_baseline, self.mlp_trans_weights)
        gates = self.perceiver(smooth_baseline)
        out_spectral = self.high_low_frequency_fusion(smooth_baseline, gates)

        out_x = torch.einsum("bhgo,bhng->bhno", out_spectral, self.eigens_output)
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
    def __init__(self, num_heads, hidden_dim, mat_output, mat_input, dropout=0., freq_num=64, lastlayer=False):
        super().__init__()
        self.lastlayer = lastlayer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Mixer(
            hidden_dim, mat_output, mat_input, heads=num_heads, 
            dim_head=hidden_dim // num_heads, dropout=dropout, freq_num=freq_num, lastlayer=lastlayer
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, hidden_dim, n_layers=0, res=False)
        
    def forward(self, fx):
        if self.lastlayer:
            fx = self.Attn(self.ln_1(fx))
            fx = self.mlp(self.ln_2(fx))
        else:
            fx = self.Attn(self.ln_1(fx)) + fx
            fx = self.mlp(self.ln_2(fx)) + fx

        return fx

class HFR_Model_Unstructured(nn.Module):
    def __init__(self, MATRIX_Output, MATRIX_Input, space_dim=2, n_layers=4, n_hidden=128, n_head=4, 
                 freq_num=32, fun_dim=1, out_dim=1, dropout=0.0):
        super().__init__()
        
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False)
        
        self.bridge_block = Mixer_Block_Unstructured(
            num_heads=n_head, 
            hidden_dim=n_hidden, 
            mat_output=MATRIX_Output, 
            mat_input=MATRIX_Input,
            dropout=dropout,
            freq_num=freq_num,
            lastlayer=True 
        )
        
        self.refiner_blocks = nn.ModuleList()
        for i in range(n_layers - 1):
            self.refiner_blocks.append(
                Mixer_Block_Unstructured(
                    num_heads=n_head, 
                    hidden_dim=n_hidden, 
                    mat_output=MATRIX_Output, 
                    mat_input=MATRIX_Output,
                    dropout=dropout,
                    freq_num=freq_num,
                    lastlayer=False
                )
            )
        
        self.num_output_nodes = MATRIX_Output.shape[0]
        self.output_pos_embedding = nn.Parameter(torch.zeros(1, self.num_output_nodes, n_hidden))
        trunc_normal_(self.output_pos_embedding, std=0.02)

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

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x_in = torch.cat((x, grid), dim=-1)
        feat = self.preprocess(x_in) 
        
        feat = self.bridge_block(feat) 
        
        feat = feat + self.output_pos_embedding
        
        for block in self.refiner_blocks:
            feat = block(feat) 
            
        return self.last_layer(feat) 
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

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
    device = torch.device('cuda')
    
    PATH = args.data_dir
    ntrain = args.num_train
    ntest = args.num_test
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    
    max_grad_norm = 0.1
    
    n_hidden = 128     
    freq_num = 128 
    n_layers = 4
    n_heads = 4
    
    log_print("Loading dataset and calculating LBO...", log_file)
    data = sio.loadmat(PATH)

    x_train = torch.Tensor(data['input'][0:ntrain])
    x_test  = torch.Tensor(data['input'][-ntest:])
    
    y_train = torch.Tensor(data['output'][0:ntrain])
    y_test  = torch.Tensor(data['output'][-ntest:])
    
    norm_x  = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)

    
    norm_y  = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
    
    print(ntrain,ntest)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    x_train = x_train.reshape(ntrain,-1,1)
    x_test  = x_test.reshape(ntest,-1,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)
    
    LBO_Output = sio.loadmat('./data/HeatTransfer_LBO_basis/lbe_ev_output.mat')['Eigenvectors']

    BASE_Output = LBO_Output[:,:freq_num]
    MATRIX_Output = torch.Tensor(BASE_Output).to(device)

    LBO_Input = sio.loadmat('./data/HeatTransfer_LBO_basis/lbe_ev_input.mat')['Eigenvectors']
    BASE_Input = LBO_Input[:,:freq_num]
    MATRIX_Input = torch.Tensor(BASE_Input).to(device)

    model = HFR_Model_Unstructured(
        MATRIX_Output=MATRIX_Output,
        MATRIX_Input=MATRIX_Input,
        space_dim=1, 
        n_layers=n_layers, 
        n_hidden=n_hidden, 
        n_head=n_heads, 
        freq_num=freq_num,
        fun_dim=1,
        out_dim=1,
        dropout=0.0
    ).to(device)
    
    log_print(f"Model initialization completed. Total parameters: {count_params(model)}", log_file)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=learning_rate,
                                                    pct_start = 400/3000,
                                                    final_div_factor=10000,
                                                    epochs = epochs, 
                                                    steps_per_epoch=len(train_loader),)
    
    myloss = LpLoss(size_average=False)
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    
    best_val_err = float('inf')
    best_epoch = 0
    total_time = 0.0

    log_print(f"Starting training... \n", log_file)

    for ep in range(epochs):
        model.train()
        train_mse = 0
        train_l2 = 0
        time_start = time.perf_counter()
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x) 

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()   

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_mse += mse.item()

            scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
    
                out = model(x)
                out_real = norm_y.decode(out.view(batch_size, -1).cpu())
                y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
                test_l2 += myloss(out_real, y_real).item()            

        train_mse /= len(train_loader)
        train_l2 /= ntrain
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
            f"Train MSE Error: {train_mse:.6f}\n"
            f"Train L2  Error: {train_l2:.6f}\n"
            f"Test Error (L2)  : {test_l2:.6f}\n"
            f"Current LR       : {current_lr:.8f}\n"
            f"Time cost        : {epoch_time:.2f} s\n"
            f"================================================================\n"
        )
        log_print(log_msg, log_file)
        
        if test_l2 < best_val_err:
            best_val_err = test_l2
            diff_epoch = ep - best_epoch
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            save_msg = f"-> [Update] Best model saved (took {diff_epoch} epochs)! Error: {best_val_err:.5e}\n"
        else:
            save_msg = f"-> [Keep] Current best: {best_val_err:.5e} (Epoch {best_epoch+1})\n"

        log_print(save_msg, log_file)

    torch.save(model.state_dict(), os.path.join(save_path, 'final_model.pt'))
    print("Final model saved.\n")
    log_print("Saving prediction results...\n", log_file)
    print()

    print("Training done...")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=1, shuffle=False)
    pre_train = torch.zeros(y_train.shape)
    y_train   = torch.zeros(y_train.shape)
    
    index = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            
            pre_train[index,:] = out_real
            y_train[index,:]   = y_real
            
            index = index + 1
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)
    y_test   = torch.zeros(y_test.shape)
    x_test   = torch.zeros(x_test.shape[0:2])
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            x_real   = norm_x.decode(x.view(1, -1).cpu())
            
            pre_test[index,:] = out_real
            y_test[index,:] = y_real
            x_test[index,:] = x_real
            
            index = index + 1
            
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
    
    dataframe = pd.DataFrame({'Test_loss' : [test_l2],
                              'Train_loss': [train_l2],
                              'num_paras' : [count_params(model)],
                              'train_time':[total_time]})
    
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error}
    
    pred_dict = {   'pre_test' : pre_test.cpu().detach().numpy(),
                    'pre_train': pre_train.cpu().detach().numpy(),
                    'y_test'   : y_test.cpu().detach().numpy(),
                    'y_train'  : y_train.cpu().detach().numpy(),
                    }
    sio.savemat(sava_path +'NORM_loss.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'NORM_pre.mat', mdict = pred_dict)
    test_l2 = (myloss(y_test, pre_test).item())/ntest
    print('\nTesting error: %.3e'%(test_l2))

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
                        'width': 128,
                        'batch_size': 4, 
                        'epochs': 3000,
                        'data_dir': './data/HeatTransfer.mat',
                        'num_train': 100, 
                        'num_test': 100,
                        'CaseName': 'HeatTransfer_'+str(i),
                        'lr' : 0.0075},
                    ]:
            args = objectview(args)
        main(args)