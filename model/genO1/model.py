from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import save_emb


class HSTUAttention(torch.nn.Module):
    """
    HSTU中的注意力机制实现
    """
    def __init__(self, embedding_dim, num_heads, dropout_rate):
        super(HSTUAttention, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout_rate = dropout_rate
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.q_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.out_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.scale = self.head_dim ** -0.5

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
        
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
        
        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        # 最终的线性变换
        output = self.out_linear(attn_output)
        
        return output


class HSTULayer(torch.nn.Module):
    """
    HSTU的基本层，包含注意力机制和前馈网络
    """
    def __init__(self, embedding_dim, num_heads, dropout_rate, ff_dim=None):
        super(HSTULayer, self).__init__()
        
        self.embedding_dim = embedding_dim
        if ff_dim is None:
            ff_dim = 4 * embedding_dim
            
        # 注意力机制
        self.attention = HSTUAttention(embedding_dim, num_heads, dropout_rate)
        self.attn_layer_norm = torch.nn.LayerNorm(embedding_dim)
        
        # 前馈网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ff_dim, embedding_dim),
            torch.nn.Dropout(dropout_rate)
        )
        self.ffn_layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        # 注意力子层
        attn_output = self.attention(x, x, x, mask)
        x = self.attn_layer_norm(x + attn_output)
        
        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_output)
        
        return x


class InteractionTransformerLayer(torch.nn.Module):
    """
    交互变换层，用于处理用户-物品交互
    """
    def __init__(self, embedding_dim, num_heads, dropout_rate, ff_dim=None):
        super(InteractionTransformerLayer, self).__init__()
        
        self.embedding_dim = embedding_dim
        if ff_dim is None:
            ff_dim = 4 * embedding_dim
            
        # 注意力机制
        self.attention = HSTUAttention(embedding_dim, num_heads, dropout_rate)
        self.attn_layer_norm = torch.nn.LayerNorm(embedding_dim)
        
        # 前馈网络
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, ff_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ff_dim, embedding_dim),
            torch.nn.Dropout(dropout_rate)
        )
        self.ffn_layer_norm = torch.nn.LayerNorm(embedding_dim)
        
        # 位置编码
        self.positional_encoding = torch.nn.Parameter(torch.randn(200, embedding_dim))

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        # 添加位置编码
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # 注意力子层
        attn_output = self.attention(x, x, x, mask)
        x = self.attn_layer_norm(x + attn_output)
        
        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.ffn_layer_norm(x + ffn_output)
        
        return x


class GenerativeRecommender(torch.nn.Module):
    """
    基于Hierarchical Sequential Transduction Unit的生成式推荐模型
    
    该模型采用HSTU架构处理用户行为序列，通过自注意力机制捕获序列中的长期依赖关系，
    并利用多层HSTU层逐步提取更高层次的序列表示。
    
    Attributes:
        item_embedding: 物品嵌入表
        user_embedding: 用户嵌入表
        positional_embedding: 位置嵌入
        hstu_layers: HSTU层堆栈
        output_projection: 输出投影层
        embedding_dim: 嵌入维度
        maxlen: 最大序列长度
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        maxlen: int = 100,
        feat_statistics: dict = None,
        feat_types: dict = None
    ):
        super(GenerativeRecommender, self).__init__()
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.maxlen = maxlen
        self.feat_statistics = feat_statistics or {}
        self.feat_types = feat_types or {}
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 嵌入层
        self.item_embedding = torch.nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = torch.nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.positional_embedding = torch.nn.Embedding(maxlen, embedding_dim)
        
        # 特征嵌入
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()
        
        # 初始化特征信息
        self._init_feat_info()
        
        # 特征处理网络
        user_dim = embedding_dim * (len(self.USER_SPARSE_FEAT) + 1)  # +1 for user_id embedding
        item_dim = embedding_dim * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_EMB_FEAT))  # +1 for item_id embedding
        
        self.user_dnn = torch.nn.Sequential(
            torch.nn.Linear(user_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        
        self.item_dnn = torch.nn.Sequential(
            torch.nn.Linear(item_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        
        # HSTU层 - 用于处理用户行为序列
        self.hstu_layers = torch.nn.ModuleList([
            HSTULayer(embedding_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 交互变换层 - 用于处理用户-物品交互
        self.interaction_layers = torch.nn.ModuleList([
            InteractionTransformerLayer(embedding_dim, num_heads, dropout_rate)
            for _ in range(2)
        ])
        
        # LayerNorm和Dropout
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # 输出层
        self.output_projection = torch.nn.Linear(embedding_dim, num_items + 1)
        
        # 初始化稀疏特征嵌入
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, embedding_dim, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, embedding_dim, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], embedding_dim)
            
        # 特征交互模块
        # 动态计算特征交互的输入维度
        # 计算特征数量（包括用户/物品ID嵌入）
        total_features = 1 + len(self.ITEM_SPARSE_FEAT) + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_CONTINUAL_FEAT) + len(self.ITEM_EMB_FEAT)
        # 计算两两组合的数量
        interaction_pairs = total_features * (total_features - 1) // 2
        # 计算交互层输入维度（每个特征的维度 + 两两交互的维度）
        interaction_input_dim = total_features * embedding_dim
        feature_interaction_input_dim = interaction_pairs * embedding_dim * 2
        
        # 只有当存在特征交互时才创建交互模块
        if feature_interaction_input_dim > 0:
            self.feature_interaction = torch.nn.Sequential(
                torch.nn.Linear(feature_interaction_input_dim, embedding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim),
                torch.nn.ReLU()
            )
        else:
            self.feature_interaction = None

    def _init_feat_info(self):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table
        """
        self.USER_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feat_types.get('user_sparse', [])}
        self.USER_CONTINUAL_FEAT = self.feat_types.get('user_continual', [])
        self.ITEM_SPARSE_FEAT = {k: self.feat_statistics[k] for k in self.feat_types.get('item_sparse', [])}
        self.ITEM_CONTINUAL_FEAT = self.feat_types.get('item_continual', [])
        self.USER_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feat_types.get('user_array', [])}
        self.ITEM_ARRAY_FEAT = {k: self.feat_statistics[k] for k in self.feat_types.get('item_array', [])}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in self.feat_types.get('item_emb', [])}  # 记录的是不同多模态特征的维度

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def _process_features(self, seq, feature_array, is_user=False):
        """
        处理特征，将其转换为嵌入表示
        
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            is_user: 是否处理用户特征
            
        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if is_user:
            user_embedding = self.user_embedding(seq)
            feat_list = [user_embedding]
        else:
            item_embedding = self.item_embedding(seq)
            feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', feat_list),
        ]

        if is_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', feat_list),
                ]
            )

        # batch-process each feature type
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

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            feat_list.append(self.emb_transform[k](tensor_feature))

        # 特征交互
        interactions = []
        for i in range(len(feat_list)):
            for j in range(i+1, len(feat_list)):
                interactions.append(torch.cat([feat_list[i], feat_list[j]], dim=-1))
        
        if interactions and self.feature_interaction is not None:
            interaction_input = torch.cat(interactions, dim=-1)
            interaction_output = self.feature_interaction(interaction_input)
            all_emb = torch.cat(feat_list + [interaction_output], dim=-1)
        else:
            all_emb = torch.cat(feat_list, dim=-1)
        
        # 特征整合
        if is_user:
            all_emb = torch.relu(self.user_dnn(all_emb))
        else:
            all_emb = torch.relu(self.item_dnn(all_emb))
        return all_emb

    def _create_attention_mask(self, seq):
        """
        创建注意力掩码
        
        Args:
            seq: 输入序列
            
        Returns:
            attention_mask: 注意力掩码
        """
        batch_size, seq_len = seq.size()
        # 创建因果掩码（causal mask）
        ones_matrix = torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.dev)
        causal_mask = torch.tril(ones_matrix)
        # 创建padding掩码
        padding_mask = (seq != 0).unsqueeze(1)  # [batch_size, 1, seq_len]
        # 合并掩码
        attention_mask = causal_mask.unsqueeze(0) & padding_mask  # [batch_size, seq_len, seq_len]
        return attention_mask

    def forward(
        self, user_item, pos_seqs=None, neg_seqs=None, mask=None, next_mask=None, 
        next_action_type=None, seq_feature=None, pos_feature=None, neg_feature=None
    ):
        """
        前向传播函数
        
        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典
            
        Returns:
            logits: 预测logits
        """
        device = user_item.device
        batch_size, seq_len = user_item.size()
        
        # 处理序列特征
        seq_emb = self._process_features(user_item, seq_feature, is_user=False)
        
        # 添加位置编码
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        positional_emb = self.positional_embedding(positions)
        seq_emb = seq_emb + positional_emb
        seq_emb = self.dropout(seq_emb)
        
        # 创建注意力掩码
        attention_mask = self._create_attention_mask(user_item)
        
        # 通过HSTU层处理用户行为序列
        hidden = seq_emb
        for layer in self.hstu_layers:
            hidden = layer(hidden, attention_mask)
            
        # 通过交互变换层进一步处理
        for layer in self.interaction_layers:
            hidden = layer(hidden, attention_mask)
            
        # LayerNorm和输出投影
        hidden = self.layer_norm(hidden)
        logits = self.output_projection(hidden)
        
        if pos_seqs is not None and neg_seqs is not None:
            # 训练模式，返回正负样本logits
            pos_emb = self._process_features(pos_seqs, pos_feature, is_user=False)
            neg_emb = self._process_features(neg_seqs, neg_feature, is_user=False)
            
            pos_logits = (hidden * pos_emb).sum(dim=-1)
            neg_logits = (hidden * neg_emb).sum(dim=-1)
            
            return pos_logits, neg_logits
        else:
            # 推理模式，返回最终logits
            return logits

    def predict(self, seq, seq_feat, mask=None):
        """
        预测用户下一个可能交互的物品
        
        Args:
            seq: 用户行为序列 [batch_size, seq_len]
            seq_feat: 序列特征
            mask: 掩码
            
        Returns:
            user_repr: 用户表示 [batch_size, embedding_dim]
        """
        logits = self.forward(seq, seq_feature=seq_feat)
        # 取最后一个时间步的表示作为用户表示
        user_repr = logits[:, -1, :]
        return user_repr

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

            batch_emb = self._process_features(item_seq, [batch_feat], is_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))