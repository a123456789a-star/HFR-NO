import os
import argparse
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import *
from utils.testloss import TestLoss
from einops import rearrange
from model.HFR_NO2 import Model
from utils.normalizer import UnitTransformer
import matplotlib.pyplot as plt
from datetime import datetime

parser = argparse.ArgumentParser('HFR')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--n-hidden', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=3)
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max_grad_norm', type=float, default=1)
parser.add_argument('--downsample', type=int, default=5)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--ntrain', type=int, default=1000)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--freq_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HFR_Darcy_experiment')
parser.add_argument('--data_path', type=str, default='./data/darcy')
parser.add_argument('--const_training_loss', type=int, default=0)
args = parser.parse_args()


train_path = args.data_path + '/piececonst_r421_N1024_smooth1.mat' 
test_path  = args.data_path + '/piececonst_r421_N1024_smooth2.mat'
ntrain = 1000
ntest  = 200
epochs = args.epochs
eval   = args.eval

save_name = args.save_name
print(f"Save Name: {save_name}")

const_training_loss = args.const_training_loss

def save_args_to_txt(args, log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write(f"{datetime.now()}--Experiment Parameter Record Table\n")
        f.write("=" * 40 + "\n\n")
        
        for key, value in args.items():
            f.write(f"{key:<25}: {value}\n")
            
        f.write("\n" + "=" * 40 + "\n")

def log_print(text, log_file):
    print(text)
    with open(log_file, "a") as f:
        f.write(text + "\n")

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def central_diff(x: torch.Tensor, h, resolution):
    if x.ndim == 2:
        b, n = x.shape
        x = x.view(b, 1, resolution, resolution)
    elif x.ndim == 3:
        x = rearrange(x, 'b (h w) c -> b c h w', h=resolution, w=resolution)

    x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0.)
    grad_x = (x_padded[:, :, 1:-1, 2:] - x_padded[:, :, 1:-1, :-2]) / (2 * h)
    grad_y = (x_padded[:, :, 2:, 1:-1] - x_padded[:, :, :-2, 1:-1]) / (2 * h)

    grad_x = rearrange(grad_x, 'b c h w -> b (h w) c')
    grad_y = rearrange(grad_y, 'b c h w -> b (h w) c')
    return grad_x, grad_y
def main():
    r = args.downsample
    h = int(((421 - 1) / r) + 1)
    s = h
    dx = 1.0 / s

    train_data = scio.loadmat(train_path)
    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(test_path)
    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)

    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)

    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)

    x_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                               batch_size=args.batch_size, shuffle=False)

    model = Model(space_dim=2, act=args.act,
                  n_layers=args.n_layers,
                  n_hidden=args.n_hidden,
                  dropout=args.dropout,
                  n_head=args.n_heads,
                  Time_Input=False,
                  mlp_ratio=args.mlp_ratio,
                  fun_dim=1,
                  out_dim=1,
                  freq_num=args.freq_num,
                  ref=args.ref,
                  unified_pos=args.unified_pos,
                  H=s, W=s).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    count_parameters(model)

    log_file = f'./logs/{save_name}_log.txt'
    save_args_to_txt(vars(args), log_file)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)

    total_time = 0.0    
    best_val_err = float('inf')
    best_epoch = 0
    for ep in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0
        reg = 0
        time_start = datetime.now()

        for x, fx, y in train_loader:
            x, fx, y = x.cuda(), fx.cuda(), y.cuda()
            optimizer.zero_grad()

            out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1) 
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            l2loss = myloss(out, y)
            gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(-1), dx, s)
            pred_grad_x, pred_grad_y = central_diff(out, dx, s)
            deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)
            loss = 0.1 * deriv_loss + l2loss
            if const_training_loss:
                loss = loss / loss.detach()
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += l2loss.item()
            reg += deriv_loss.item()
            scheduler.step()

        train_loss /= ntrain
        reg /= ntrain

        model.eval()
        rel_err = 0.0
        rel_linf_sum = 0.0

        id = 0
        with torch.no_grad():
            for x, fx, y in test_loader:
                id += 1
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                out = model(x, fx=fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl

                abs_err = torch.abs(out - y)
                abs_target = torch.abs(y)
                
                max_err_val = torch.max(abs_err, dim=1)[0]       
                max_target_val = torch.max(abs_target, dim=1)[0] 
                
                batch_rel_linf = torch.sum(max_err_val / (max_target_val + 1e-9))
                rel_linf_sum += batch_rel_linf.item()

        rel_err = rel_err / ntest
        rel_linf_err = rel_linf_sum / ntest
        time_end = datetime.now()
        epoch_time = (time_end - time_start).total_seconds()
        total_time += epoch_time

        print()
        log_msg = (
            f"Epoch [{ep+1:03d}/{args.epochs}]\n"
            f"Train L2 Loss: {train_loss:.6f}\n"
            f"Train Deriv Loss: {reg:.6f}\n"
            f"val_loss: {rel_err:.6f}\n"
            f"val_linf: {rel_linf_err:.6f}\n"
            f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            f"Epoch time: {epoch_time:.2f} seconds, Average time: {total_time / (ep + 1):.2f} seconds\n"
            f"================================================================\n"
        )

        log_print(log_msg, log_file)
        if rel_err < best_val_err:
            best_val_err = rel_err
            diff_epoch = ep + 1 - best_epoch
            best_epoch = ep + 1
            noIncreaseEpochs = 0
            torch.save(model.state_dict(), f"./checkpoints/{save_name}_BEST.pt")
            info = f"-> New best model saved! Val Score = {best_val_err:.6f}, after {diff_epoch} epochs\n\n"
        else:
            info = f"-> Current best: Epoch {best_epoch}, Val Score = {best_val_err:.6f}\n\n"
            noIncreaseEpochs += 1
        log_print(info, log_file)


    print('save model')
    torch.save(model.state_dict(), os.path.join('./checkpoints', save_name + '.pt'))

if __name__ == "__main__":
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    main()