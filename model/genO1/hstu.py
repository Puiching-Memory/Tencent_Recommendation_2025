"""
基于论文"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"的HSTU模型实现
该实现适配了当前项目的特征处理和数据格式
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from dataset import save_emb


class HSTUAttention(torch.nn.Module):
    """
    HSTU中的注意力机制实现，基于论文"Actions Speak Louder than Words: 
    Trillion-Parameter Sequential Transducers for Generative Recommendations"
    """
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(HSTUAttention, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.scale = self.head_dim ** -0.5
        
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()
        
        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用PyTorch内置的scaled_dot_product_attention（支持FlashAttention）
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, 
                dropout_p=self.dropout_rate if self.training else 0.0, 
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None
            )
        else:
            # 降级到标准注意力机制实现
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate if self.training else 0.0)
            attn_output = torch.matmul(attn_weights, V)
        
        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        
        # 最终的线性变换
        output = self.out_linear(attn_output)
        
        return output


class HSTULayer(torch.nn.Module):
    """
    HSTU的基本层，包含注意力机制和前馈网络
    """
    def __init__(self, hidden_units, num_heads, dropout_rate, ff_dim=None):
        super(HSTULayer, self).__init__()
        
        self.hidden_units = hidden_units
        if ff_dim is None:
            ff_dim = 4 * hidden_units
            
        # 注意力机制
        self.attention = HSTUAttention(hidden_units, num_heads, dropout_rate)
        self.attn_layer_norm = torch.nn.LayerNorm(hidden_units)
        
        # 前馈网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_units, ff_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ff_dim, hidden_units),
            torch.nn.Dropout(dropout_rate)
        )
        self.ffn_layer_norm = torch.nn.LayerNorm(hidden_units)
        
    def forward(self, x, mask=None):
        # 注意力子层
        attn_output = self.attention(x, x, x, mask)
        x = self.attn_layer_norm(x + attn_output)
        
        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_output)
        
        return x


class PackedSwiGLUFFN(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        multiple_of=256,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # 如果未指定hidden_dim，则使用默认值（通常是dim的4倍）
        if hidden_dim is None:
            hidden_dim = 4 * dim
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        else:
            hidden_dim = int(2 * hidden_dim / 3)
            # custom dim factor multiplier
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w13 = torch.nn.Linear(dim, 2 * hidden_dim, bias=False, **factory_kwargs)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)


class HSTUModel(torch.nn.Module):
    """
    基于Hierarchical Sequential Transduction Unit的生成式推荐模型
    
    该模型采用HSTU架构处理用户行为序列，通过自注意力机制捕获序列中的长期依赖关系，
    并利用多层HSTU层逐步提取更高层次的序列表示。该实现基于论文"Actions Speak Louder than Words: 
    Trillion-Parameter Sequential Transducers for Generative Recommendations"。
    
    Attributes:
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """
    
    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):
        super(HSTUModel, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.args = args
        
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        
        self._init_feat_info(feat_statistics, feat_types)
        
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )
        
        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)
        
        # HSTU层
        self.hstu_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        for _ in range(args.num_blocks):
            new_hstu_layer = HSTULayer(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.hstu_layers.append(new_hstu_layer)
            
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            # 使用PackedSwiGLUFFN替换原有的PointWiseFeedForward
            new_fwd_layer = PackedSwiGLUFFN(args.hidden_units, device=args.device)
            self.forward_layers.append(new_fwd_layer)
        
        # 初始化嵌入层
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
            
    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        将序列特征转换为PyTorch张量
        
        Args:
            seq_feature (list): 序列特征列表，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k (str): 特征ID
            
        Returns:
            torch.Tensor: 特征值张量，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 对于Array类型特征，需要进行padding处理后转换为张量
            max_array_len = 0
            max_seq_len = 0

            # 计算最大序列长度和最大数组长度用于padding
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            # 在目标设备上直接创建张量以避免后续数据迁移
            batch_data = torch.zeros((batch_size, max_seq_len, max_array_len), dtype=torch.int64, device=self.dev)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    # 将数据直接转换为张量并放置在目标设备上
                    batch_data[i, j, :actual_len] = torch.tensor(item_data[:actual_len], dtype=torch.int64, device=self.dev)

            return batch_data
        else:
            # Sparse类型特征直接转换为张量
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            # 在目标设备上直接创建张量
            batch_data = torch.zeros((batch_size, max_seq_len), dtype=torch.int64, device=self.dev)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                # 将数据直接转换为张量并放置在目标设备上
                batch_data[i] = torch.tensor(seq_data, dtype=torch.int64, device=self.dev)

            return batch_data

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        将序列特征转换为嵌入向量表示
        
        Args:
            seq (torch.Tensor): 序列ID张量
            feature_array (list): 特征列表，每个元素为当前时刻的特征字典
            mask (torch.Tensor, optional): 掩码，1表示item，2表示user
            include_user (bool): 是否处理用户特征
            
        Returns:
            torch.Tensor: 序列特征的嵌入表示
        """
        seq = seq.to(self.dev)
        # 预计算嵌入表示
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # 批量处理所有特征类型
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # 批量处理每种特征类型
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        # 多模态特征处理 - 使用向量化操作替代循环以提高效率
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])
            
            # 在GPU上直接创建张量，避免CPU-GPU数据传输
            batch_emb_data = torch.zeros((batch_size, seq_len, emb_dim), 
                                         dtype=torch.float32, device=self.dev)
            
            # 使用向量化方法收集数据
            indices_list = []
            values_list = []
            
            for i in range(batch_size):
                for j, item in enumerate(feature_array[i]):
                    if k in item:
                        indices_list.append([i, j])
                        values_list.append(item[k])
            
            # 批量赋值
            if indices_list:
                indices_tensor = torch.tensor(indices_list, device=self.dev)
                # 将值转换为张量并放置在目标设备上
                values_tensor = torch.tensor(values_list, dtype=torch.float32, device=self.dev)
                batch_emb_data[indices_tensor[:, 0], indices_tensor[:, 1]] = values_tensor
            
            item_feat_list.append(self.emb_transform[k](batch_emb_data))

        # 特征融合
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        将用户行为序列转换为特征表示
        
        Args:
            log_seqs (torch.Tensor): 用户序列ID张量
            mask (torch.Tensor): token类型掩码，1表示item token，2表示user token
            seq_feature (list): 序列特征列表，每个元素为当前时刻的特征字典
            
        Returns:
            torch.Tensor: 序列的嵌入表示，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        # 序列嵌入缩放
        seqs *= self.item_emb.embedding_dim**0.5
        # 位置编码
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # 注意力掩码计算优化 - 使用缓存和高效的张量操作
        # 缓存下三角矩阵以避免重复创建
        if not hasattr(self, '_tril_cache') or self._tril_cache.size(-1) != maxlen:
            self._tril_cache = torch.tril(torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev))
        
        # 使用缓存的下三角矩阵并进行高效的张量操作
        attention_mask_tril = self._tril_cache.unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask_pad = (mask != 0).unsqueeze(1).expand(-1, maxlen, -1)
        attention_mask = attention_mask_tril & attention_mask_pad

        # HSTU编码器层
        for i in range(len(self.hstu_layers)):
            if self.norm_first:
                # Pre-LN架构
                x = self.hstu_layers[i](seqs, attn_mask=attention_mask)
                seqs = seqs + x
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                # Post-LN架构
                x = self.hstu_layers[i](seqs, attn_mask=attention_mask)
                seqs = self.hstu_layers[i](seqs + x)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        前向传播函数，计算正负样本的logits
        
        Args:
            user_item (torch.Tensor): 用户序列ID张量
            pos_seqs (torch.Tensor): 正样本序列ID张量
            neg_seqs (torch.Tensor): 负样本序列ID张量
            mask (torch.Tensor): token类型掩码，1表示item token，2表示user token
            next_mask (torch.Tensor): 下一个token类型掩码，1表示item token，2表示user token
            next_action_type (torch.Tensor): 下一个token动作类型，0表示曝光，1表示点击
            seq_feature (list): 序列特征列表，每个元素为当前时刻的特征字典
            pos_feature (list): 正样本特征列表，每个元素为当前时刻的特征字典
            neg_feature (list): 负样本特征列表，每个元素为当前时刻的特征字典
            
        Returns:
            tuple: (pos_logits, neg_logits) 正负样本logits，形状为 [batch_size, maxlen]
        """
        # 获取序列特征表示
        log_feats = self.log2feats(user_item, mask, seq_feature)
        # 构造损失计算掩码，仅对item token计算损失
        loss_mask = (next_mask == 1).to(self.dev)

        # 计算正负样本嵌入表示
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # 计算logits（点积相似度）
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        # 应用损失掩码
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        预测函数，计算用户序列的表征向量
        
        Args:
            log_seqs (torch.Tensor): 用户序列ID张量
            seq_feature (list): 序列特征列表，每个元素为当前时刻的特征字典
            mask (torch.Tensor): token类型掩码，1表示item token，2表示user token
            
        Returns:
            torch.Tensor: 用户序列的表征向量，形状为 [batch_size, hidden_units]
        """
        # 获取序列特征表示
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        # 取序列最后一个时刻的表征作为用户表征
        final_feat = log_feats[:, -1, :]

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding用于检索
        
        Args:
            item_ids (list): 候选item ID（re-id形式）
            retrieval_ids (list): 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict (dict): 训练集所有item特征字典，key为特征ID，value为特征值
            save_path (str): 保存路径
            batch_size (int): 批次大小
        """
        all_embs = []
        all_embs_tensor = []  # 用于存储PyTorch张量以避免不必要的设备迁移

        # 分批处理以节省内存
        for start_idx in range(0, len(item_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            # 特征转换
            batch_feat = torch.stack([torch.tensor(f, dtype=object) for f in batch_feat])

            # 计算嵌入表示
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)
            
            # 直接保存张量，避免转换为numpy
            all_embs_tensor.append(batch_emb.detach())

        # 合并所有批次的结果并保存
        # 使用torch.cat替代np.concatenate以保持张量操作一致性
        final_ids = torch.tensor(retrieval_ids, dtype=torch.uint64, device='cpu').reshape(-1, 1)
        final_embs = torch.cat(all_embs_tensor, dim=0).cpu()
        
        # 转换为numpy用于保存
        final_embs_np = final_embs.numpy().astype(np.float32)
        final_ids_np = final_ids.numpy()
        
        save_emb(final_embs_np, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids_np, Path(save_path, 'id.u64bin'))