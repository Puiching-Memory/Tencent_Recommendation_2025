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
from model_genrecsys import GenerativeRecommender


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Gen-RecSys Model construction
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--num_codebooks', default=4, type=int)
    parser.add_argument('--codebook_size', default=64, type=int)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def extract_multimodal_features(features_batch, mm_emb_ids):
    """
    从特征批次中提取多模态特征
    
    Args:
        features_batch: 特征批次，每个元素为字典
        mm_emb_ids: 多模态特征ID列表
        
    Returns:
        multimodal_features: 多模态特征列表，每个元素为tensor
    """
    batch_size = len(features_batch)
    multimodal_features = []
    
    # 为每个多模态特征ID创建一个特征张量
    for feat_id in mm_emb_ids:
        # 获取第一个样本的特征维度来确定特征维度
        sample_feature = features_batch[0][feat_id]
        if isinstance(sample_feature, np.ndarray):
            feat_dim = sample_feature.shape[0]
        else:
            # 如果不是数组，跳过这个特征
            continue
            
        # 创建该特征的批次张量
        feat_tensor = torch.zeros(batch_size, feat_dim)
        for i, features in enumerate(features_batch):
            if feat_id in features and features[feat_id] is not None:
                if isinstance(features[feat_id], np.ndarray):
                    feat_tensor[i] = torch.from_numpy(features[feat_id])
                else:
                    feat_tensor[i] = torch.tensor(features[feat_id])
            # 如果特征不存在，保持为0
        
        multimodal_features.append(feat_tensor)
    
    return multimodal_features


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 初始化生成推荐系统模型
    model = GenerativeRecommender(
        num_users=usernum,
        num_items=itemnum,
        embedding_dim=args.embedding_dim,
        modalities_emb_dims=[32],  # 根据实际多模态特征维度调整
        latent_dim=args.latent_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size
    ).to(args.device)

    # 初始化模型参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 定义损失函数和优化器
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 解析批次数据
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 将数据移到指定设备
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            # 从特征中提取多模态特征
            pos_multimodal = extract_multimodal_features(pos_feat, args.mm_emb_id)
            neg_multimodal = extract_multimodal_features(neg_feat, args.mm_emb_id)
            
            # 构造用户和物品ID
            # 使用序列中的最后一个有效用户作为用户ID
            user_ids = seq[:, -1]  # 简化处理，实际应根据业务逻辑确定用户ID
            user_ids = torch.clamp(user_ids, 0, usernum - 1)  # 确保用户ID在有效范围内
            
            # 正样本和负样本物品ID
            pos_item_ids = pos[:, -1]  # 取最后一个位置作为正样本
            neg_item_ids = neg[:, -1]  # 取最后一个位置作为负样本
            pos_item_ids = torch.clamp(pos_item_ids, 0, itemnum - 1)  # 确保物品ID在有效范围内
            neg_item_ids = torch.clamp(neg_item_ids, 0, itemnum - 1)  # 确保物品ID在有效范围内
            
            # 前向传播
            pos_predictions, pos_losses = model(user_ids, pos_item_ids, pos_multimodal)
            neg_predictions, neg_losses = model(user_ids, neg_item_ids, neg_multimodal)
            
            # 构造标签
            pos_labels = torch.ones_like(pos_predictions, device=args.device)
            neg_labels = torch.zeros_like(neg_predictions, device=args.device)
            
            # 计算损失
            pos_loss = bce_criterion(pos_predictions, pos_labels)
            neg_loss = bce_criterion(neg_predictions, neg_labels)
            rqvae_loss = pos_losses['rqvae_loss'] + neg_losses['rqvae_loss']
            loss = pos_loss + neg_loss + 0.01 * rqvae_loss  # RQ-VAE损失权重可调

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        # 验证阶段
        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                
                # 验证集数据处理逻辑与训练集相同
                pos_multimodal = extract_multimodal_features(pos_feat, args.mm_emb_id)
                neg_multimodal = extract_multimodal_features(neg_feat, args.mm_emb_id)
                
                user_ids = seq[:, -1]
                user_ids = torch.clamp(user_ids, 0, usernum - 1)
                
                pos_item_ids = pos[:, -1]
                neg_item_ids = neg[:, -1]
                pos_item_ids = torch.clamp(pos_item_ids, 0, itemnum - 1)
                neg_item_ids = torch.clamp(neg_item_ids, 0, itemnum - 1)
                
                pos_predictions, pos_losses = model(user_ids, pos_item_ids, pos_multimodal)
                neg_predictions, neg_losses = model(user_ids, neg_item_ids, neg_multimodal)
                
                pos_labels = torch.ones_like(pos_predictions, device=args.device)
                neg_labels = torch.zeros_like(neg_predictions, device=args.device)
                
                pos_loss = bce_criterion(pos_predictions, pos_labels)
                neg_loss = bce_criterion(neg_predictions, neg_labels)
                rqvae_loss = pos_losses['rqvae_loss'] + neg_losses['rqvae_loss']
                loss = pos_loss + neg_loss + 0.01 * rqvae_loss
                
                valid_loss_sum += loss.item()
                
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()