import json
import pickle
import struct
from pathlib import Path
import os
from functools import lru_cache
import warnings

import torch

# 用户序列数据集类
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, args):
        super().__init__()
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

        # 预加载偏移量数据
        with open(self.data_dir / 'seq_offsets.pkl', 'rb') as f:
            self.seq_offsets = pickle.load(f)

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        
    # 使用LRU缓存来替代一次性加载所有用户数据到内存中
    @lru_cache(maxsize=10000)
    def _load_user_data(self, uid):
        """根据uid从文件中加载用户数据，使用LRU缓存机制"""
        with open(self.data_dir / "seq.jsonl", 'rb') as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()
            user_data = json.loads(line)
        return user_data

    # 生成负采样样本
    def _random_neq(self, l, r, s):
        # 预先将排除项转换为集合类型以提高查找效率
        excluded_items = set(str(t) for t in s if str(t) in self.item_feat_dict)
        excluded_items.add("0")  # 排除0
        
        # 生成批量候选样本以提高效率
        candidates = torch.randint(l, r, (100,))  # 一次生成多个候选
        
        for candidate in candidates:
            t = candidate.item()
            if str(t) not in excluded_items and str(t) in self.item_feat_dict:
                return t
                
        # 如果批量采样失败，回退到逐个采样
        warnings.warn("Batch sampling failed, falling back to individual sampling. "
                      "This may impact performance. "
                      "Consider adjusting the sampling strategy or data distribution.")
        t = torch.randint(l, r, (1,)).item()
        while str(t) in excluded_items or str(t) not in self.item_feat_dict:
            t = torch.randint(l, r, (1,)).item()
        return t

    # 获取单个用户的数据并进行处理
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        # 初始化返回数据
        seq = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        pos = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        neg = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        token_type = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        next_token_type = torch.zeros([self.maxlen + 1], dtype=torch.int32)
        next_action_type = torch.zeros([self.maxlen + 1], dtype=torch.int32)

        # 初始化特征列表
        seq_feat = [self.feature_default_value] * (self.maxlen + 1)
        pos_feat = [self.feature_default_value] * (self.maxlen + 1)
        neg_feat = [self.feature_default_value] * (self.maxlen + 1)

        # 处理空序列情况
        if not user_sequence:
            return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

        # 构建扩展用户序列
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.append((u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 处理空序列情况
        if not ext_user_sequence:
            return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

        # 获取序列最后一个元素作为next元素
        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        # 构建物品ID集合用于负采样
        ts = set(record[0] for record in ext_user_sequence if record[2] == 1 and record[0])

        # 从后向前填充序列
        sequence_length = min(len(ext_user_sequence) - 1, self.maxlen + 1)
        
        for j in range(sequence_length):
            if idx == -1:
                break
                
            record_tuple = ext_user_sequence[-2 - j]
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            
            # 填充特征
            feat = self.fill_missing_feat(feat, i)
            
            seq[idx] = i
            token_type[idx] = type_
            
            if next_type is not None:
                next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            
            # 处理正负样本
            if next_type == 1 and next_i != 0:
                next_feat = self.fill_missing_feat(next_feat, next_i)
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                
            nxt = record_tuple
            idx -= 1

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    # 返回数据集大小
    def __len__(self):
        return len(self.seq_offsets)

    # 初始化特征信息
    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = torch.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=torch.float32
            )

        # 预计算所有特征ID以提高fill_missing_feat函数的性能
        self.all_feat_ids = [feat_id for feat_type in feat_types.values() for feat_id in feat_type]
        
        return feat_default_value, feat_types, feat_statistics

    # 填充缺失特征
    def fill_missing_feat(self, feat, item_id):
        if feat is None:
            feat = {}
            
        # 复制已有特征
        filled_feat = dict(feat)
        
        # 处理缺失字段
        missing_fields = set(self.all_feat_ids) - set(feat.keys())
        
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
            
        # 处理多模态特征
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                emb_value = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
                # 使用推荐的方法创建张量副本
                emb_value = emb_value.detach().clone()
                filled_feat[feat_id] = emb_value

        return filled_feat

    # 批处理函数
    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        # 将序列数据转换为张量
        seq = torch.stack(seq)
        pos = torch.stack(pos)
        neg = torch.stack(neg)
        token_type = torch.stack(token_type)
        next_token_type = torch.stack(next_token_type)
        next_action_type = torch.stack(next_action_type)
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


# 保存嵌入向量到二进制文件
def save_emb(emb, save_path):
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


# 加载多模态嵌入向量
def load_mm_emb(mm_path, feat_ids):
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {feat_id: {} for feat_id in feat_ids}
    
    # 单线程处理所有文件
    total_feat_ids = len(feat_ids)
    print(f"Loading multi-modal embeddings for {total_feat_ids} feature IDs...")
    
    for feat_idx, feat_id in enumerate(feat_ids):
        try:
            base_path = Path(mm_path, f'emb_{feat_id}_{SHAPE_DICT[feat_id]}')
            json_files = list(base_path.glob('*.json'))
            print(f"Processing feature ID {feat_id} ({feat_idx+1}/{total_feat_ids}): {len(json_files)} files found")
        except Exception as e:
            print(f"Error collecting tasks for feature {feat_id}: {e}")
            continue
            
        total_files = len(json_files)
        for file_idx, json_file in enumerate(json_files):
            # 打印文件处理进度
            if total_files > 10:  # 只有当文件数量较多时才打印详细进度
                if (file_idx + 1) % max(1, total_files // 10) == 0 or file_idx == total_files - 1:
                    print(f"  Processing files for feature {feat_id}: {file_idx + 1}/{total_files} ({(file_idx + 1) / total_files * 100:.1f}%)")
            elif total_files > 1:
                print(f"  Processing file {file_idx + 1}/{total_files} for feature {feat_id}")
                
            # 处理单个文件，提取嵌入向量
            file_emb_dict = {}
            try:
                with open(json_file, 'r', encoding='utf-8') as file:
                    line_count = 0
                    for line in file:
                        data_dict_origin = json.loads(line.strip())
                        line_count += 1
                            
                        # 获取 anonymous_cid
                        anonymous_cid = data_dict_origin['anonymous_cid']
                        
                        # 检查 'emb' 键是否存在，如果不存在则跳过该记录
                        if 'emb' not in data_dict_origin:
                            # 当找不到emb时应该放过，后续会有填充
                            continue
                        
                        insert_emb = data_dict_origin['emb']
                        # 使用推荐的方法创建张量副本
                        insert_emb = torch.tensor(insert_emb, dtype=torch.float32).detach().clone()
                        
                        data_dict = {anonymous_cid: insert_emb}
                        file_emb_dict.update(data_dict)

            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                continue
            mm_emb_dict[feat_id].update(file_emb_dict)
            
        print(f"Finished processing feature ID {feat_id} ({feat_idx+1}/{total_feat_ids}) - Total embeddings: {len(mm_emb_dict[feat_id])}")

    print("Finished loading all multi-modal embeddings")
    return mm_emb_dict