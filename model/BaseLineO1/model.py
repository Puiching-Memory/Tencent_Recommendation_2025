"""
model.py
从路径导入模块
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    """
    基于Flash Attention优化的多头注意力机制
    
    这个类实现了多头注意力机制，并在支持的情况下使用PyTorch 2.0的Flash Attention优化，
    以提高计算效率。注意力机制是Transformer模型的核心组件，用于计算序列中不同位置之间的相关性。
    """
    
    def __init__(self, hidden_units, num_heads, dropout_rate):
        """
        初始化多头注意力层
        
        Args:
            hidden_units: 隐藏单元数量，即特征向量的维度
            num_heads: 注意力头的数量，需要能整除hidden_units
            dropout_rate: Dropout比率，用于防止过拟合
        """
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads  # 每个注意力头的维度
        self.dropout_rate = dropout_rate

        # 确保隐藏单元数可以被头数整除
        assert hidden_units % num_heads == 0, "hidden_units必须能被num_heads整除"

        # 定义查询(Q)、键(K)、值(V)的线性变换层
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播函数
        
        Args:
            query: 查询向量，形状为[batch_size, seq_len, hidden_units]
            key: 键向量，形状为[batch_size, seq_len, hidden_units]
            value: 值向量，形状为[batch_size, seq_len, hidden_units]
            attn_mask: 注意力掩码，用于屏蔽某些位置的注意力计算
            
        Returns:
            output: 注意力计算结果
            None: 为了接口兼容性返回None
        """
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 重塑为多头格式: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 检查是否有PyTorch 2.0的Flash Attention支持
        if hasattr(F, 'scaled_dot_product_attention'):
            # 使用PyTorch 2.0内置的Flash Attention优化实现
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制实现
            scale = (self.head_dim) ** -0.5  # 缩放因子，防止梯度消失
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # 计算注意力分数

            # 应用掩码，将被屏蔽位置的分数设为负无穷
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            # 通过softmax函数得到注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            # 应用dropout
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            # 计算加权和
            attn_output = torch.matmul(attn_weights, V)

        # 重新整理形状为[batch_size, seq_len, hidden_units]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PackedSwiGLUFFN(torch.nn.Module):
    """
    基于SwiGLU激活函数的前馈神经网络
    
    SwiGLU是一种改进的激活函数，结合了SiLU（Sigmoid-weighted Linear Unit）和GLU（Gated Linear Unit），
    在大型语言模型中表现出色。这个实现使用了打包的权重矩阵以提高计算效率。
    """
    
    def __init__(
        self,
        dim,  # 输入维度
        hidden_dim=None,  # 隐藏层维度
        multiple_of=256,  # 隐藏层维度的倍数约束
        ffn_dim_multiplier=None,  # 隐藏层维度的倍数因子
        device=None,
        dtype=None,
        dropout_rate=0.0,  # Dropout比率
    ):
        """
        初始化前馈网络
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # 如果未指定隐藏层维度，则根据输入维度计算
        if hidden_dim is None:
            hidden_dim = 4 * dim
            # 自定义维度倍数因子
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        else:
            hidden_dim = int(2 * hidden_dim / 3)
            # 自定义维度倍数因子
            if ffn_dim_multiplier is not None:
                hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # 定义权重矩阵，将两个线性变换打包到一个矩阵中以提高效率
        self.w13 = torch.nn.Linear(dim, 2 * hidden_dim, bias=False, **factory_kwargs)
        self.w2 = torch.nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
        # 可选的dropout层
        self.dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入张量
            
        Returns:
            output: 经过SwiGLU前馈网络处理后的输出
        """
        # 将权重矩阵的输出分为两部分
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        # 应用SwiGLU激活函数: Swish(x1) * x3
        output = self.w2(F.silu(x1) * x3)
        # 应用dropout（如果设置了dropout率）
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class BaselineModel(torch.nn.Module):
    """
    双塔推荐模型基线实现
    
    这是一个基于Transformer架构的双塔推荐模型，包含用户塔和物品塔。
    用户塔处理用户的历史行为序列，物品塔处理物品的特征信息。
    最终通过计算用户表示和物品表示的相似度来进行推荐。
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        """
        初始化模型
        
        Args:
            user_num: 用户数量
            item_num: 物品数量
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表
            args: 全局参数配置
        """
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first  # 是否先进行LayerNorm
        self.maxlen = args.maxlen  # 序列最大长度
        
        # 定义物品和用户的嵌入表（Embedding Table）
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        # 位置嵌入，用于表示序列中每个位置的信息
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        # 嵌入层的dropout
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # 存储不同特征类型的嵌入层和变换层
        self.sparse_emb = torch.nn.ModuleDict()  # 稀疏特征嵌入
        self.emb_transform = torch.nn.ModuleDict()  # 嵌入特征变换

        # 注意力层和前馈网络层
        self.attention_layernorms = torch.nn.ModuleList()  # 注意力层的LayerNorm
        self.attention_layers = torch.nn.ModuleList()      # 注意力层
        self.forward_layernorms = torch.nn.ModuleList()    # 前馈网络的LayerNorm
        self.forward_layers = torch.nn.ModuleList()        # 前馈网络

        # 初始化特征信息
        self._init_feat_info(feat_statistics, feat_types)

        # 计算用户和物品特征的总维度
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
            args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT))
            + len(self.ITEM_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        # 用户和物品的全连接层，用于整合所有特征
        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        # 最后的LayerNorm层
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 创建多个Transformer块
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PackedSwiGLUFFN(args.hidden_units, dropout_rate=args.dropout_rate, device=args.device)
            self.forward_layers.append(new_fwd_layer)

        # 为不同类型的稀疏特征创建嵌入层
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        # 为嵌入特征创建线性变换层
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        初始化特征信息，将特征按类型分组
        
        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        # 将特征按类型分组存储
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}      # 用户稀疏特征
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']                                 # 用户连续特征
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}      # 物品稀疏特征
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']                                 # 物品连续特征
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}        # 用户数组特征
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}        # 物品数组特征
        
        # 多模态嵌入特征的维度映射
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}

    def feat2tensor(self, seq_feature, k):
        """
        将序列特征转换为张量格式
        
        Args:
            seq_feature: 序列特征列表，每个元素为当前时刻的特征字典
            k: 特征ID
            
        Returns:
            batch_data: 特征值的张量，形状根据特征类型而定
        """
        batch_size = len(seq_feature)

        # 处理数组类型特征
        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 计算最大数组长度和序列长度
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                # 计算当前序列中数组的最大长度
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            # 创建用于存储数据的数组
            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 处理稀疏类型特征
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        将特征转换为嵌入向量
        
        Args:
            seq: 序列ID
            feature_array: 特征列表，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征
            
        Returns:
            seqs_emb: 序列特征的嵌入向量
        """
        seq = seq.to(self.dev)
        # 预计算嵌入向量
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

        # 如果包含用户特征，则也处理用户特征
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

                # 根据特征类型应用不同的处理方式
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    # 对数组特征求和
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # 连续特征直接添加维度
                    feat_list.append(tensor_feature.unsqueeze(2))

        # 处理嵌入特征
        for k in self.ITEM_EMB_FEAT:
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # 预分配张量
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            # 填充嵌入特征数据
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # 批量转换并转移到GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # 合并特征
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
            log_seqs: 用户行为序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征列表，每个元素为当前时刻的特征字典
            
        Returns:
            log_feats: 序列的特征表示，形状为[batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        # 获取序列嵌入
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        # 缩放嵌入向量
        seqs *= self.item_emb.embedding_dim**0.5
        # 添加位置嵌入
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0  # 对填充值位置置零
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # 构造注意力掩码
        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)  # 下三角矩阵，确保只能看到前面的token
        attention_mask_pad = (mask != 0).to(self.dev)  # 填充位置掩码
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        # 通过多个Transformer块处理序列
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                # 先进行LayerNorm的处理方式
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                # 标准的处理方式
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        前向传播函数，用于模型训练
        
        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码
            next_mask: 下一个token类型掩码
            next_action_type: 下一个token动作类型
            seq_feature: 序列特征列表
            pos_feature: 正样本特征列表
            neg_feature: 负样本特征列表
            
        Returns:
            pos_logits: 正样本logits
            neg_logits: 负样本logits
        """
        # 获取用户序列的特征表示
        log_feats = self.log2feats(user_item, mask, seq_feature)
        # 创建损失计算掩码，只计算item位置的损失
        loss_mask = (next_mask == 1).to(self.dev)

        # 获取正负样本的嵌入向量
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # 计算正负样本的logits（通过点积计算相似度）
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        # 应用损失掩码
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        预测函数，用于生成用户表示
        
        Args:
            log_seqs: 用户行为序列ID
            seq_feature: 序列特征列表
            mask: token类型掩码
            
        Returns:
            final_feat: 用户序列的最终表示
        """
        # 获取用户序列的特征表示
        log_feats = self.log2feats(log_seqs, mask, seq_feature)
        # 取最后一个位置的表示作为用户的最终表示
        final_feat = log_feats[:, -1, :]

        return final_feat
    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in range(0, len(item_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
