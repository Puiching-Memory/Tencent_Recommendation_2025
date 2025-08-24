import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import GenerativeRecommender
from hstu import HSTUModel  # 添加HSTU模型导入

def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # HSTU Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.03, type=float)
    parser.add_argument('--l2_emb', default=0.001, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # 新增：模型类型选择
    parser.add_argument('--model_type', default='hstu', type=str, choices=['generative', 'hstu'],
                        help='选择要使用的模型类型: generative(生成式模型) 或 hstu(HSTU模型)')

    # Acceleration options
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP)')
    parser.add_argument('--use_torch_compile', action='store_true', help='Enable torch.compile for model optimization')
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark for performance')
    parser.add_argument('--use_tf32', action='store_true', help='Enable TF32 for faster float32 computations')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def evaluate(model, dataloader, args, is_validation=False):
    """
    评估模型性能

    Args:
        model: 模型
        dataloader: 数据加载器
        args: 参数
        is_validation: 是否为验证集评估

    Returns:
        results: 评估结果，包括NDCG和HR
    """
    total_ndcg = 0.0
    total_hr = 0.0
    total_num = 0

    for step, batch in enumerate(dataloader):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
        seq = seq.to(args.device)
        pos = pos.to(args.device)
        neg = neg.to(args.device)

        # 根据模型类型执行不同的前向传播
        if args.model_type == 'generative':
            pos_logits, _ = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
        elif args.model_type == 'hstu':
            pos_logits, _ = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )

        # 仅对item token计算评估指标
        loss_mask = (next_token_type == 1).to(args.device)
        pos_logits = pos_logits * loss_mask

        # 计算NDCG和HR
        for i in range(pos_logits.shape[0]):
            pred = pos_logits[i].cpu().numpy()
            target = pos[i].cpu().numpy()
            mask = loss_mask[i].cpu().numpy()

            # 过滤掉padding位置
            pred = pred[mask == 1]
            target = target[mask == 1]

            if len(pred) == 0:
                continue

            # 计算NDCG@10和HR@10
            pred_indices = np.argsort(-pred)[:10]
            target_items = set(target[pred_indices])
            actual_target = set(target)

            # HR@10
            hr = len(target_items.intersection(actual_target)) / len(actual_target) if len(actual_target) > 0 else 0
            total_hr += hr

            # NDCG@10
            dcg = 0.0
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(10, len(actual_target)))])

            for i, idx in enumerate(pred_indices):
                if target[idx] in actual_target:
                    dcg += 1.0 / np.log2(i + 2)

            ndcg = dcg / idcg if idcg > 0 else 0
            total_ndcg += ndcg

            total_num += 1

    return {'ndcg': total_ndcg / total_num if total_num > 0 else 0, 'hr': total_hr / total_num if total_num > 0 else 0}


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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=train_collate_fn, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=valid_collate_fn, pin_memory=True
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 根据选择的模型类型创建相应模型
    if args.model_type == 'generative':
        print("使用生成式模型 (GenerativeRecommender)")
        model = GenerativeRecommender(
            num_users=usernum+1,
            num_items=itemnum+1,
            embedding_dim=args.hidden_units,
            modalities_emb_dims=[32, 1024, 3584, 4096, 3584, 3584],  # 根据MM特征维度设置
            latent_dim=args.hidden_units,
            num_codebooks=4,
            codebook_size=64
        ).to(args.device)
    elif args.model_type == 'hstu':
        print("使用HSTU模型 (HSTUModel)")
        model = HSTUModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    # 初始化嵌入权重（仅适用于HSTUModel）
    if args.model_type == 'hstu':
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
    # 使用weight_decay参数实现L2正则化，替代手动计算L2损失的方式
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)
    
    # Compile components if enabled
    if args.use_torch_compile:
        bce_criterion = torch.compile(bce_criterion)
        print("Loss function compiled with torch.compile")

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    if args.use_amp:
        print("Automatic Mixed Precision (AMP) enabled")

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0

    print("Start training")
    start_time = time.time()
    total_steps = len(train_loader) * args.num_epochs
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
            
        epoch_start_time = time.time()
        epoch_steps = 0

        print(f"Epoch {epoch}/{args.num_epochs}")
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            step_start_time = time.time()
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device, non_blocking=True)
            pos = pos.to(args.device, non_blocking=True)
            neg = neg.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            
            # Use AMP context manager
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.model_type == 'generative':
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, 
                        seq_feat, pos_feat, neg_feat
                    )
                elif args.model_type == 'hstu':
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, 
                        seq_feat, pos_feat, neg_feat
                    )
                    
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                indices = np.where(next_token_type == 1)
                
                # 计算BCE损失
                bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                
                # 添加L2正则化损失
                l2_loss = 0
                if args.model_type == 'hstu':
                    for param in model.item_emb.parameters():
                        l2_loss += args.l2_emb * torch.norm(param)
                
                # 总损失初始化为BCE损失和L2损失
                loss = bce_loss + l2_loss

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
            
            # 统一记录训练指标数据
            train_metrics = {
                'loss': loss.item(),
                'bce_loss': bce_loss.item(),
                'l2_loss': l2_loss.item(),
                'step_time': step_time,
                'elapsed_time': elapsed_time,
                'steps_per_second': steps_per_second,
                'estimated_remaining_time': estimated_remaining_time
            }
            
            log_json = json.dumps({
                'global_step': global_step,
                'epoch': epoch,
                'step': step,
                **train_metrics,
                'time': time.time()
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            
            # Print progress every 10 steps
            if step % 10 == 0:
                progress = (step + 1) / len(train_loader) * 100
                print(f"  Step {step+1}/{len(train_loader)} [{progress:.1f}%] - "
                      f"Loss: {train_metrics['loss']:.4f}, "
                      f"Speed: {train_metrics['steps_per_second']:.2f} steps/s, "
                      f"ETA: {estimated_remaining_time:.0f}s")

            # TensorBoard记录详细信息
            writer.add_scalar('Loss/train', train_metrics['loss'], global_step)
            writer.add_scalar('Loss/BCE', train_metrics['bce_loss'], global_step)
            writer.add_scalar('Performance/step_time', train_metrics['step_time'], global_step)
            writer.add_scalar('Performance/steps_per_second', train_metrics['steps_per_second'], global_step)
            
            # 记录学习率
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'LearningRate/group_{i}', param_group['lr'], global_step)
            
            # 每100步记录一次梯度信息
            if global_step % 100 == 0:
                # 收集所有梯度值
                grad_means = []
                grad_maxs = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_means.append(param.grad.mean().abs().item())
                        grad_maxs.append(param.grad.max().abs().item())
                
                # 记录总体梯度信息
                if grad_means:
                    writer.add_scalar('Gradient/mean', sum(grad_means) / len(grad_means), global_step)
                    writer.add_scalar('Gradient/max', max(grad_maxs), global_step)
            
        # End of epoch statistics
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s ({epoch_steps} steps, "
              f"{epoch_steps/epoch_time:.2f} steps/s)")

        model.eval()
        valid_loss_sum = 0
        valid_bce_loss_sum = 0.0
        valid_steps = 0
        valid_start_time = time.time()
        
        with torch.no_grad():
            print("Validating...")
            for step, batch in enumerate(valid_loader):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device, non_blocking=True)
                pos = pos.to(args.device, non_blocking=True)
                neg = neg.to(args.device, non_blocking=True)

                # Use AMP context manager for validation
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    if args.model_type == 'generative':
                        pos_logits, neg_logits = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                    elif args.model_type == 'hstu':
                        pos_logits, neg_logits = model(
                            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                        )
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                        neg_logits.shape, device=args.device
                    )
                    indices = np.where(next_token_type == 1)
                    bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                    # 验证阶段的损失同样只计算BCE损失
                    loss = bce_loss
                    
                valid_loss_sum += loss.item()
                valid_bce_loss_sum += bce_loss.item()
                valid_steps += 1
                
        valid_time = time.time() - valid_start_time
        valid_loss_sum /= len(valid_loader)
        valid_bce_loss_sum /= len(valid_loader)
        
        # 统一记录验证指标数据
        valid_metrics = {
            'loss': valid_loss_sum,
            'bce_loss': valid_bce_loss_sum,
            'validation_time': valid_time
        }
        
        print(f"Validation completed in {valid_metrics['validation_time']:.2f}s ({valid_steps} steps)")
        
        
        writer.add_scalar('Loss/valid', valid_metrics['loss'], global_step)
        writer.add_scalar('Loss/valid_BCE', valid_metrics['bce_loss'], global_step)
        writer.add_scalar('Performance/validation_time', valid_metrics['validation_time'], epoch)
        
        # Record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate/current', current_lr, epoch)
        print(f"Current learning rate: {current_lr}")

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.2f}s")
    print("Done")
    writer.close()
    log_file.close()