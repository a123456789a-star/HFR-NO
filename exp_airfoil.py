import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import *
from scipy import fftpack 
from utils.testloss import TestLoss
from model.HFR1 import Model
from datetime import datetime

parser = argparse.ArgumentParser('HFR')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n-hidden', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=3)
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max_grad_norm', type=float, default=1)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--freq_num', type=int, default=32)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HFR_airfoil_experiment')
parser.add_argument('--data_path', type=str, default='./data/airfoil/naca')
parser.add_argument('--ntrain', type=int, default=1000)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
save_name = args.save_name

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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    
    input_x_path = args.data_path + '/NACA_Cylinder_X.npy'
    input_y_path = args.data_path + '/NACA_Cylinder_Y.npy'
    output_sigma_path = args.data_path + '/NACA_Cylinder_Q.npy'

    ntrain = args.ntrain
    ntest = 200

    r1 = args.downsamplex
    r2 = args.downsampley
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    input_x = np.load(input_x_path)
    input_x = torch.tensor(input_x, dtype=torch.float)
    input_y = np.load(input_y_path)
    input_y = torch.tensor(input_y, dtype=torch.float)
    input_data = torch.stack([input_x, input_y], dim=-1)

    output_data = np.load(output_sigma_path)[:, 4]
    output_data = torch.tensor(output_data, dtype=torch.float)
    print(input_data.shape, output_data.shape)

    x_train = input_data[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output_data[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = input_data[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    y_test = output_data[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
    x_train = x_train.reshape(ntrain, -1, 2)
    x_test = x_test.reshape(ntest, -1, 2)
    y_train = y_train.reshape(ntrain, -1)
    y_test = y_test.reshape(ntest, -1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                              batch_size=args.batch_size,
                                              shuffle=False)


    model = Model(space_dim=2, 
                  n_layers=args.n_layers, 
                  n_hidden=args.n_hidden, 
                  dropout=args.dropout,
                  n_head=args.n_heads, 
                  Time_Input=False, 
                  mlp_ratio=args.mlp_ratio, 
                  fun_dim=0,
                  out_dim=1, 
                  kernelsize=1, 
                  freq_num=args.freq_num, 
                  ref=args.ref,
                  unified_pos=args.unified_pos, 
                  H=s1, W=s2).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)
    print(model)
    log_file = f'./logs/{save_name}_log.txt'
    save_args_to_txt(vars(args), log_file)

    best_val_err = float('inf')
    best_epoch = 0

    time = 0

    for ep in range(args.epochs):
        time_start = datetime.now()
        model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            out = model(x, None).squeeze(-1)
            loss = myloss(out, y)
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_l2 += loss.item()
            scheduler.step()

        time_end = datetime.now()
        epoch_time = (time_end - time_start).total_seconds()
        time += epoch_time
        model.eval()
        val_loss = 0
        rel_linf_sum = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x, None).squeeze(-1)
                val_loss += myloss(out, y).item()

                abs_err = torch.abs(out - y)
                abs_target = torch.abs(y)
                
                max_err_val = torch.max(abs_err, dim=1)[0]       
                max_target_val = torch.max(abs_target, dim=1)[0] 
                
                batch_rel_linf = torch.sum(max_err_val / (max_target_val + 1e-9))
                rel_linf_sum += batch_rel_linf.item()


        avg_train = train_l2 / ntrain
        avg_val = val_loss / ntest
        avg_rel_linf = rel_linf_sum / ntest
        
        print()
        log_msg = (
            f"Epoch [{ep+1:03d}/{args.epochs}]\n"
            f"Train L2: {avg_train:.6f}\n"
            f"val_loss: {avg_val:.6f}\n"
            f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            f"Rel_Linf: {avg_rel_linf:.6f}\n"
            f"Epoch time: {epoch_time:.2f} seconds, Average time: {time / (ep + 1):.2f} seconds\n"
            f"================================================================\n"
        )
        
        log_print(log_msg, log_file)

        if avg_val < best_val_err:
            best_val_err = avg_val
            diff_epoch = ep + 1 - best_epoch
            best_epoch = ep + 1
            noIncreaseEpochs = 0
            torch.save(model.state_dict(), f"./checkpoints/{save_name}_BEST.pt")
            info = f"-> New best model saved! Val Score = {best_val_err:.6f}, after {diff_epoch} epochs\n\n"
        else:
            info = f"-> Current best: Epoch {best_epoch}, Val Score = {best_val_err:.6f}\n\n"
            noIncreaseEpochs += 1
        log_print(info, log_file)

    if not os.path.exists('./checkpoints'): os.makedirs('./checkpoints')
    torch.save(model.state_dict(), f"./checkpoints/{args.save_name}_FINAL.pt")

if __name__ == "__main__":
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    main()