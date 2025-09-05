import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MyDataset
from model import BaselineModel
from utils import print_system_info, format_time


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.01, type=float)
    parser.add_argument('--l2_emb', default=0.001, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # Acceleration options
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP)')
    parser.add_argument('--use_torch_compile', action='store_true', help='Enable torch.compile for model optimization')
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark for performance')
    parser.add_argument('--use_tf32', action='store_true', help='Enable TF32 for faster float32 computations')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    # Enable cuDNN benchmark
    if args.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")
    
    # Enable TF32
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")

    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # Compile collate_fn if enabled
    if args.use_torch_compile:
        train_collate_fn = torch.compile(dataset.collate_fn, mode="reduce-overhead")
        valid_collate_fn = torch.compile(dataset.collate_fn, mode="reduce-overhead")
        print("DataLoader collate functions compiled with torch.compile")
    else:
        train_collate_fn = dataset.collate_fn
        valid_collate_fn = dataset.collate_fn
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=train_collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 对模型参数进行Xavier初始化，但要避免对维度不足的参数进行初始化
    for name, param in model.named_parameters():
        try:
            # 只对维度大于等于2的参数进行Xavier初始化
            if param.dim() >= 2:
                torch.nn.init.xavier_normal_(param.data)
            elif param.dim() == 1:
                # 对一维参数（如偏置项）进行常量初始化
                torch.nn.init.constant_(param.data, 0.0)
        except Exception as e:
            print(f"Warning: Failed to initialize parameter {name}: {e}")

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # Compile model if enabled
    if args.use_torch_compile:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    # Compile components if enabled
    if args.use_torch_compile:
        bce_criterion = torch.compile(bce_criterion)
        print("Loss function compiled with torch.compile")

    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp) if torch.cuda.is_available() else torch.amp.GradScaler('cpu', enabled=args.use_amp)
    if args.use_amp:
        print("Automatic Mixed Precision (AMP) enabled")

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0

    print("Start training")
    print_system_info()
    start_time = time.time()
    global_step = 0
    total_steps = len(train_loader) * args.num_epochs
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
            
        epoch_start_time = time.time()
        epoch_steps = 0

        print(f"Epoch {epoch}/{args.num_epochs}")
        for step, batch in enumerate(train_loader):
            step_start_time = time.time()
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            optimizer.zero_grad()
            
            # Use AMP context manager
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                indices = np.where(next_token_type == 1)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                for param in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(param)

            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step_time = time.time() - step_start_time
            global_step += 1
            epoch_steps += 1
            
            # Calculate speed and estimated time
            elapsed_time = time.time() - start_time
            steps_per_second = global_step / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = total_steps - global_step
            estimated_remaining_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
                        
            log_json = json.dumps({
                'global_step': global_step,
                'epoch': epoch,
                'step': step,
                'loss': loss.item(),
                'step_time': step_time,
                'elapsed_time': elapsed_time,
                'steps_per_second': steps_per_second,
                'estimated_remaining_time': estimated_remaining_time,
                'time': time.time()
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            
            # Print progress every 10 steps
            if step % 10 == 0:
                progress = (step + 1) / len(train_loader) * 100
                print(f"  Step {step+1}/{len(train_loader)} [{progress:.1f}%] - "
                      f"Loss: {loss.item():.4f}, "
                      f"Speed: {steps_per_second:.2f} steps/s, "
                      f"ETA: {format_time(estimated_remaining_time)}")

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Performance/step_time', step_time, global_step)
            writer.add_scalar('Performance/steps_per_second', steps_per_second, global_step)

        # End of epoch statistics
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {format_time(epoch_time)} ({epoch_steps} steps, "
              f"{epoch_steps/epoch_time:.2f} steps/s)")

        model.eval()
        valid_loss_sum = 0
        valid_start_time = time.time()
        valid_steps = 0
        
        with torch.inference_mode():
            print("Validating...")
            for step, batch in enumerate(valid_loader):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                
                # Use AMP context manager for validation
                with torch.amp.autocast('cuda', enabled=args.use_amp):
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                        neg_logits.shape, device=args.device
                    )
                    indices = np.where(next_token_type == 1)
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                valid_loss_sum += loss.item()
                valid_steps += 1
                
        valid_time = time.time() - valid_start_time
        valid_loss_sum /= len(valid_loader)
        print(f"Validation completed in {format_time(valid_time)} ({valid_steps} steps)")
        
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        writer.add_scalar('Performance/validation_time', valid_time, epoch)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    total_training_time = time.time() - start_time
    print(f"Training completed in {format_time(total_training_time)}")
    print("Done")
    writer.close()
    log_file.close()
