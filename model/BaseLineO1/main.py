import argparse
import subprocess
import sys

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "orjson"])
    import orjson as json
    print("orjson安装成功，使用orjson加载数据")
except subprocess.CalledProcessError:
    print("orjson安装失败，使用默认json库")
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
from utils import print_system_info, format_time, parse_data_path_structure


def get_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有训练参数的对象
    """
    parser = argparse.ArgumentParser()

    # 训练参数配置
    parser.add_argument('--batch_size', default=128, type=int, help='每个批次的样本数量')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率，控制模型参数更新的步长')
    parser.add_argument('--maxlen', default=101, type=int, help='序列最大长度')

    # 模型结构参数
    parser.add_argument('--hidden_units', default=64, type=int, help='隐藏层单元数')
    parser.add_argument('--num_blocks', default=4, type=int, help='Transformer块的数量')
    parser.add_argument('--num_epochs', default=5, type=int, help='训练轮数')
    parser.add_argument('--num_heads', default=4, type=int, help='多头注意力机制中头的数量')
    parser.add_argument('--dropout_rate', default=0.05, type=float, help='Dropout概率，用于防止过拟合')
    parser.add_argument('--l2_emb', default=0.001, type=float, help='L2正则化系数')
    parser.add_argument('--device', default='cuda', type=str, help='训练设备，可选cuda或cpu')
    parser.add_argument('--inference_only', action='store_true', help='是否仅进行推理（验证/测试）')
    parser.add_argument('--state_dict_path', default=None, type=str, help='预训练模型权重文件路径')
    parser.add_argument('--norm_first', action='store_true', help='是否在注意力计算前进行归一化')

    # 性能优化选项
    parser.add_argument('--use_amp', action='store_true', help='启用自动混合精度训练，节省显存并加速训练')
    parser.add_argument('--use_torch_compile', action='store_true', help='启用torch.compile优化模型执行')
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='启用cuDNN基准测试模式，优化卷积计算')
    parser.add_argument('--use_tf32', action='store_true', help='启用TF32格式，提升float32计算性能')

    # 多模态特征ID配置
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)],
                        help='指定使用的多模态嵌入特征ID')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 创建日志和TensorBoard事件文件目录
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    
    # 打开日志文件并创建TensorBoard写入器
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    # 获取训练数据路径
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # 解析命令行参数
    args = get_args()
    
    # 解析并打印数据路径结构
    parse_data_path_structure(data_path)
    
    # 启用cuDNN基准测试模式以提升性能
    if args.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")
    
    # 启用TF32计算以提升float32运算性能
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")

    # 创建数据集实例并划分训练集和验证集
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # 根据配置决定是否编译数据加载器的collate函数
    if args.use_torch_compile:
        train_collate_fn = torch.compile(dataset.collate_fn, mode="reduce-overhead")
        valid_collate_fn = torch.compile(dataset.collate_fn, mode="reduce-overhead")
        print("DataLoader collate functions compiled with torch.compile")
    else:
        train_collate_fn = dataset.collate_fn
        valid_collate_fn = dataset.collate_fn
    
    # 创建训练和验证数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=train_collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_collate_fn
    )
    
    # 获取用户数、物品数以及特征统计信息
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 初始化模型并移动到指定设备
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 对模型参数进行Xavier初始化
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception as e:
            print(e)

    # 将特殊标记（padding）的嵌入向量初始化为0
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # 根据配置决定是否编译模型
    if args.use_torch_compile:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # 设置训练起始轮次
    epoch_start_idx = 1

    # 如果指定了预训练模型路径，则加载模型权重
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 定义损失函数和优化器
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')  # 二分类交叉熵损失
    # 使用weight_decay参数实现L2正则化
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)
    
    # 根据配置决定是否编译损失函数
    if args.use_torch_compile:
        bce_criterion = torch.compile(bce_criterion)
        print("Loss function compiled with torch.compile")

    # 初始化自动混合精度训练的梯度缩放器
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp) if torch.cuda.is_available() else torch.amp.GradScaler('cpu', enabled=args.use_amp)
    if args.use_amp:
        print("Automatic Mixed Precision (AMP) enabled")

    # 初始化训练指标
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
    
    # 开始训练循环
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()  # 设置模型为训练模式
        if args.inference_only:
            break  # 如果仅进行推理，则跳出训练循环
            
        epoch_start_time = time.time()
        epoch_steps = 0

        print(f"Epoch {epoch}/{args.num_epochs}")
        # 遍历训练数据
        for step, batch in enumerate(train_loader):
            step_start_time = time.time()
            
            # 从批次数据中提取各个组成部分
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 将数据移动到指定设备上
            seq = seq.to(args.device, non_blocking=True)
            pos = pos.to(args.device, non_blocking=True)
            neg = neg.to(args.device, non_blocking=True)

            # 清零优化器梯度
            optimizer.zero_grad()
            
            # 使用自动混合精度上下文管理器
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                # 前向传播计算正负样本的预测值
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, 
                    seq_feat, pos_feat, neg_feat
                )
                    
                # 创建正负样本标签
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                
                # 找到需要计算损失的位置（next_token_type == 1的位置）
                indices = np.where(next_token_type == 1)
                
                # 计算二分类交叉熵损失
                bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                                
                # 总损失初始化为BCE损失
                loss = bce_loss

            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            step_time = time.time() - step_start_time
            global_step += 1
            epoch_steps += 1
            
            # 计算训练速度和剩余时间
            elapsed_time = time.time() - start_time
            steps_per_second = global_step / elapsed_time if elapsed_time > 0 else 0
            remaining_steps = total_steps - global_step
            estimated_remaining_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            
            # 收集训练指标数据
            train_metrics = {
                'loss': loss.item(),
                'bce_loss': bce_loss.item(),
                'step_time': step_time,
                'elapsed_time': elapsed_time,
                'steps_per_second': steps_per_second,
                'estimated_remaining_time': estimated_remaining_time
            }
            
            # 将训练指标写入日志文件
            log_json = json.dumps({
                'global_step': global_step,
                'epoch': epoch,
                'step': step,
                **train_metrics,
                'time': time.time()
            })
            log_file.write(log_json + '\n')
            log_file.flush()
            
            # 每10步打印一次训练进度
            if step % 10 == 0:
                progress = (step + 1) / len(train_loader) * 100
                print(f"  Step {step+1}/{len(train_loader)} [{progress:.1f}%] - "
                      f"Loss: {train_metrics['loss']:.4f}, "
                      f"Speed: {train_metrics['steps_per_second']:.2f} steps/s, "
                      f"ETA: {format_time(train_metrics['estimated_remaining_time'])}")

            # 将训练指标写入TensorBoard
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
            
        # 每个epoch结束时的统计信息
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {format_time(epoch_time)} ({epoch_steps} steps, "
              f"{epoch_steps/epoch_time:.2f} steps/s)")

        # 验证阶段
        model.eval()  # 设置模型为评估模式
        valid_loss_sum = 0
        valid_bce_loss_sum = 0.0
        valid_steps = 0
        valid_start_time = time.time()
        
        # 在验证集上进行推理
        with torch.inference_mode():
            print("Validating...")
            for step, batch in enumerate(valid_loader):
                # 从批次数据中提取各个组成部分
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                
                # 将数据移动到指定设备上
                seq = seq.to(args.device, non_blocking=True)
                pos = pos.to(args.device, non_blocking=True)
                neg = neg.to(args.device, non_blocking=True)

                # 使用自动混合精度上下文管理器进行验证
                with torch.amp.autocast('cuda', enabled=args.use_amp):
                    # 前向传播计算正负样本的预测值
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    
                    # 创建正负样本标签
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                        neg_logits.shape, device=args.device
                    )
                    
                    # 找到需要计算损失的位置
                    indices = np.where(next_token_type == 1)
                    
                    # 计算二分类交叉熵损失
                    bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                    
                    # 验证阶段的损失同样只计算BCE损失
                    loss = bce_loss
                    
                # 累加验证损失
                valid_loss_sum += loss.item()
                valid_bce_loss_sum += bce_loss.item()
                valid_steps += 1
                
        # 计算平均验证损失
        valid_time = time.time() - valid_start_time
        valid_loss_sum /= len(valid_loader)
        valid_bce_loss_sum /= len(valid_loader)
        
        # 收集验证指标数据
        valid_metrics = {
            'loss': valid_loss_sum,
            'bce_loss': valid_bce_loss_sum,
            'validation_time': valid_time
        }
        
        print(f"Validation completed in {format_time(valid_metrics['validation_time'])} ({valid_steps} steps)")
        
        # 将验证指标写入TensorBoard
        writer.add_scalar('Loss/valid', valid_metrics['loss'], global_step)
        writer.add_scalar('Loss/valid_BCE', valid_metrics['bce_loss'], global_step)
        writer.add_scalar('Performance/validation_time', valid_metrics['validation_time'], epoch)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate/current', current_lr, epoch)
        print(f"Current learning rate: {current_lr}")

        # 保存模型检查点
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    # 训练完成，打印总耗时
    total_training_time = time.time() - start_time
    print(f"Training completed in {format_time(total_training_time)}")
    print("Done")
    writer.close()
    log_file.close()