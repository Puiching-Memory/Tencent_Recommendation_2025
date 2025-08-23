from pathlib import Path
import json
import pickle
import time
import torch

from dataset import MyDataset,load_mm_emb

# 测试数据集类
class MyTestDataset(MyDataset):

    def __init__(self, data_dir, args):
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # 加载测试偏移量数据
        with open(self.data_dir / 'predict_seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        
        # 测试数据集不需要缓存，因为通常只访问一次
        # self.user_data = self._load_all_user_data()
        
    # 测试数据集不需要LRU缓存，因为通常只访问一次
    def _load_user_data(self, uid):
        """根据uid从文件中加载测试用户数据，不使用缓存"""
        start_time = time.time()
        with open(self.data_dir / "predict_seq.jsonl", 'rb') as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            user_data = json.loads(line)
        load_time = time.time() - start_time
        # 可选：记录时间到日志或用于统计
        return user_data

    # 处理冷启动特征
    def _process_cold_start_feat(self, feat):
        process_start_time = time.time()
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [
                    0 if isinstance(v, str) else v for v in feat_value
                ]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        process_time = time.time() - process_start_time
        # 可选：记录时间到日志或用于统计
        return processed_feat

    # 获取单个测试用户的数据并进行处理
    def __getitem__(self, uid):
        start_time = time.time()
        user_sequence = self._load_user_data(uid)
        load_user_data_time = time.time() - start_time

        seq_init_start = time.time()
        # 初始化返回数据
        seq = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        token_type = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        seq_feat = [self.feature_default_value] * (self.maxlen + 1)
        user_id = None
        seq_init_time = time.time() - seq_init_start

        # 处理空序列情况
        if not user_sequence:
            return seq, token_type, seq_feat, user_id

        ext_user_seq_start = time.time()
        # 构建扩展用户序列和获取user_id
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            
            # 提取user_id
            if u:
                user_id = u if isinstance(u, str) else self.indexer_u_rev[u]
                    
            # 处理用户特征
            if u and user_feat:
                ext_user_sequence.append((u, user_feat, 2))
                
            # 处理物品特征
            if i and item_feat:
                # 处理训练时未见过的物品
                i = 0 if i > self.itemnum else i
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))
        ext_user_seq_time = time.time() - ext_user_seq_start

        # 处理空序列情况
        if not ext_user_sequence:
            return seq, token_type, seq_feat, user_id

        fill_seq_start = time.time()
        idx = self.maxlen

        # 从后向前填充序列
        sequence_length = min(len(ext_user_sequence), self.maxlen + 1)
        for j in range(sequence_length):
            if idx == -1:
                break
                
            record_tuple = ext_user_sequence[-1 - j]
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
        fill_seq_time = time.time() - fill_seq_start

        total_time = time.time() - start_time
        # 可选：在这里可以记录各部分的时间用于分析性能瓶颈

        return seq, token_type, seq_feat, user_id

    # 返回测试数据集大小
    def __len__(self):
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        # 如果temp是列表，返回列表长度
        if isinstance(temp, list):
            return len(temp)
        # 如果temp是字典，返回字典长度
        return len(temp)

    # 测试数据批处理函数
    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id
