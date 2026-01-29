import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
from tqdm import *
from utils.testloss import TestLoss
from utils.normalizer import UnitTransformer
from model.HFR_NO2 import Model
from datetime import datetime

parser = argparse.ArgumentParser('HFR')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n-hidden', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=3)
parser.add_argument('--n-heads', type=int, default=4)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--downsamplex', type=int, default=1)
parser.add_argument('--downsampley', type=int, default=1)
parser.add_argument('--mlp_ratio', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--unified_pos', type=int, default=0)
parser.add_argument('--ref', type=int, default=8)
parser.add_argument('--freq_num', type=int, default=32)
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_name', type=str, default='HFR_plasticity_experiment')
parser.add_argument('--data_path', type=str, default='./data/plasticity')
parser.add_argument('--ntrain', type=int, default=900)

args = parser.parse_args()
eval = args.eval
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
save_name = args.save_name

print(f"Save Name: {save_name}")

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

def random_collate_fn(batch):
    shuffled_batch = [] 
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]

        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)

        t = t[permuted_indices]
        u = u[..., permuted_indices]

        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(0)
            shuffled_t = t.unsqueeze(0)
            shuffled_u = u.unsqueeze(0)
            shuffled_a = a.unsqueeze(0)
        else:
            shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
            shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
            shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
            shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)

    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)

    return shuffled_batch

def main():
    DATA_PATH = args.data_path + '/plas_N987_T20.mat'
    ntrain = args.ntrain

    N = 987
    ntest = 80

    s1 = 101
    s2 = 31
    T = 20
    Deformation = 4

    r1 = 1
    r2 = 1
    s1 = int(((s1 - 1) / r1) + 1)
    s2 = int(((s2 - 1) / r2) + 1)

    data = scio.loadmat(DATA_PATH)
    input = torch.tensor(data['input'], dtype=torch.float)
    output = torch.tensor(data['output'], dtype=torch.float).transpose(-2, -1)
    print(input.shape, output.shape)
    x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain, s1, 1).repeat(1, 1, s2)
    x_train = x_train.reshape(ntrain, -1, 1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, Deformation, T)
    x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest, s1, 1).repeat(1, 1, s2)
    x_test = x_test.reshape(ntest, -1, 1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest, -1, Deformation, T)
    print(x_train.shape, y_train.shape)

    x_normalizer = UnitTransformer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    x_normalizer.cuda()

    x = np.linspace(0, 1, s1)
    y = np.linspace(0, 1, s2)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)

    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    print("Dataloading is over.")

    t = np.linspace(0, 1, T)
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain, 1)
    t_test = t.repeat(ntest, 1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                                batch_size=args.batch_size, shuffle=True, collate_fn=random_collate_fn)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    model = Model(space_dim=2,
                  n_hidden=args.n_hidden,
                  n_layers=args.n_layers,
                  Time_Input=True,
                  n_head=args.n_heads,
                  fun_dim=1,
                  out_dim=Deformation,
                  mlp_ratio=args.mlp_ratio,
                  freq_num=args.freq_num,
                  unified_pos=args.unified_pos,
                  H=s1,
                  W=s2).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    count_parameters(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    log_file = f'./logs/{save_name}_log.txt'
    save_args_to_txt(vars(args), log_file)

    total_time = 0.0    
    best_val_err = float('inf')
    best_epoch = 0
    noIncreaseEpochs = 0 

    for ep in range(args.epochs):
        model.train()
        train_l2_step = 0

        time_start = datetime.now()
        for x, tim, fx, yy in train_loader:
            x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
            bsz = x.shape[0]
            
            for t in range(T):
                y = yy[..., t:t + 1]
                input_T = tim[:, t:t + 1].reshape(bsz, 1) 
                im = model(x, fx, T=input_T)

                loss = myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                train_l2_step += loss.item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            scheduler.step()

        model.eval()
        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for x, tim, fx, yy in test_loader:
                loss = 0
                x, fx, tim, yy = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda()
                bsz = x.shape[0]

                for t in range(T):
                    y = yy[..., t:t + 1]
                    input_T = tim[:, t:t + 1].reshape(bsz, 1)
                    im = model(x, fx, T=input_T)
                    loss += myloss(im.reshape(bsz, -1), y.reshape(bsz, -1))
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                time_end = datetime.now()
        
        epoch_time = (time_end - time_start).total_seconds()
        total_time += epoch_time

        print()
        train_l2 = train_l2_step / ntrain / T
        test_l2_part = test_l2_step / ntest / T
        test_l2_full = test_l2_full / ntest
        log_msg = (
            f"Epoch [{ep+1:03d}/{args.epochs}]\n"
            f"train_step_loss: {train_l2:.6f}\n"
            f"test_step_loss: {test_l2_part:.6f}\n"
            f"test_full_loss: {test_l2_full:.6f}\n"
            f"LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            f"Epoch time: {epoch_time:.2f} s, Average time: {total_time / (ep + 1):.2f} s\n"
            f"================================================================\n"
        )

        log_print(log_msg, log_file)
        if test_l2_full < best_val_err:
            best_val_err = test_l2_full
            diff_epoch = ep + 1 - best_epoch
            best_epoch = ep + 1
            noIncreaseEpochs = 0
            torch.save(model.state_dict(), f"./checkpoints/{save_name}_BEST.pt")
            info = f"-> New best model saved! Val Score = {best_val_err:.6f}, cost {diff_epoch} epochs\n\n"
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