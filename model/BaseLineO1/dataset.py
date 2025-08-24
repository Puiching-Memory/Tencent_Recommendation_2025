import pickle
import struct
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import time

import subprocess
import sys

# Attempt to use orjson for faster JSON parsing

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "orjson"])
    import orjson as json
    print("orjson安装成功，使用orjson加载数据")
except subprocess.CalledProcessError:
    print("orjson安装失败，使用默认json库")
    import json

import numpy as np
import torch


class MyDataset(torch.utils.data.Dataset):
    """
    用户行为序列数据集类，用于训练推荐模型
    该类继承自PyTorch的Dataset基类，实现了推荐系统训练所需的数据加载和预处理功能

    Args:
        data_dir (str): 数据文件所在目录路径
        args (object): 配置参数对象，需包含maxlen(序列最大长度)和mm_emb_id(多模态特征ID列表)属性

    Attributes:
        data_dir (Path): 数据文件目录路径对象
        maxlen (int): 用户行为序列的最大长度，超出部分截断，不足部分补零
        item_feat_dict (dict): 物品特征字典，key为物品ID，value为该物品的特征信息
        mm_emb_ids (list): 需要加载的多模态特征ID列表
        mm_emb_dict (dict): 多模态特征嵌入向量字典
        itemnum (int): 物品总数
        usernum (int): 用户总数
        indexer_i_rev (dict): 物品索引反向映射字典，将内部ID映射回原始物品ID
        indexer_u_rev (dict): 用户索引反向映射字典，将内部ID映射回原始用户ID
        indexer (dict): 索引字典，包含用户、物品及特征的索引信息
        feature_default_value (dict): 特征默认值字典，用于填充缺失特征
        feature_types (dict): 特征类型分类字典，按用户/物品和特征类型分类
        feat_statistics (dict): 特征统计信息字典，记录各类特征的取值数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集，加载所有必要数据文件并构建特征信息
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        # 使用orjson或默认json加载物品特征字典
        with open(Path(data_dir, "item_feat_dict.json"), 'rb') as f:
            self.item_feat_dict = json.loads(f.read())
        # 加载多模态嵌入特征
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        # 加载索引信息
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        # 构建反向索引映射
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # 初始化特征相关信息
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户行为序列数据文件和偏移量索引文件
        偏移量文件用于快速定位和读取特定用户的数据，避免全文件扫描
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        根据用户ID加载该用户的行为序列数据

        Args:
            uid (int): 用户内部ID

        Returns:
            list: 用户行为序列数据，每个元素为(用户ID,物品ID,用户特征,物品特征,行为类型,时间戳)元组
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        从指定范围内随机生成一个不在给定集合中的整数，用于负采样
        负采样是推荐系统训练中的重要技术，通过构造负例来提升模型效果

        Args:
            l (int): 随机数下界（包含）
            r (int): 随机数上界（不包含）
            s (set): 需要避开的数字集合

        Returns:
            int: 不在集合s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取指定用户的数据，进行预处理并返回模型所需格式
        这是Dataset类的核心方法，PyTorch在训练时会自动调用

        Args:
            uid (int): 用户内部ID

        Returns:
            tuple: 包含以下元素的元组:
                - seq: 用户历史行为序列（物品ID）
                - pos: 正样本序列（真实下一行为物品ID）
                - neg: 负样本序列（随机采样物品ID）
                - token_type: 序列中每个位置的类型标识（1表示物品，2表示用户）
                - next_token_type: 下一位置的类型标识
                - next_action_type: 下一行为的类型
                - seq_feat: 序列特征信息列表
                - pos_feat: 正样本特征信息列表
                - neg_feat: 负样本特征信息列表
        """
        # 加载用户行为序列数据
        user_sequence = self._load_user_data(uid)

        # 扩展用户序列，将用户特征和物品特征分别标记类型
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                # 用户特征插入到序列前端，类型标记为2
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                # 物品特征添加到序列后端，类型标记为1
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 初始化返回数据结构
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)           # 用户行为序列
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)           # 正样本序列
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)           # 负样本序列
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)    # token类型序列
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 下一token类型
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32) # 下一行为类型
        seq_feat = np.empty([self.maxlen + 1], dtype=object)        # 序列特征
        pos_feat = np.empty([self.maxlen + 1], dtype=object)        # 正样本特征
        neg_feat = np.empty([self.maxlen + 1], dtype=object)        # 负样本特征

        # 获取序列最后一个元素作为初始"下一个"元素
        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        # 收集物品ID集合，用于负采样时避免重复
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # 从后向前填充序列（左填充），确保长度不超过maxlen+1
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            
            # 填充缺失特征
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            
            # 填充序列数据
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            
            # 如果下一元素是物品且非空，则构造正负样本
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                # 负采样：随机选择一个不在用户历史行为中的物品
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 将None值替换为默认特征值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集大小，即用户总数

        Returns:
            int: 数据集中用户的总数
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征相关信息，包括默认值、类型分类和统计信息
        特征按类型分为：稀疏特征(sparse)、数组特征(array)、嵌入特征(emb)和连续特征(continual)

        Returns:
            tuple: 包含以下三个字典的元组:
                - feat_default_value: 各特征的默认值
                - feat_types: 特征类型分类
                - feat_statistics: 特征统计信息
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        
        # 定义用户和物品的各类特征ID
        feat_types['user_sparse'] = ['103', '104', '105', '109']  # 用户稀疏特征
        feat_types['item_sparse'] = [  # 物品稀疏特征
            '100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116'
        ]
        feat_types['item_array'] = []    # 物品数组特征（暂无）
        feat_types['user_array'] = ['106', '107', '108', '110']  # 用户数组特征
        feat_types['item_emb'] = self.mm_emb_ids   # 物品嵌入特征
        feat_types['user_continual'] = []  # 用户连续特征（暂无）
        feat_types['item_continual'] = []  # 物品连续特征（暂无）

        # 设置各类特征的默认值和统计信息
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
            # 多模态嵌入特征默认值为零向量，维度根据第一个嵌入向量确定
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        填充特征中的缺失值，确保所有特征都有值

        Args:
            feat (dict): 原始特征字典，可能包含缺失值
            item_id (int): 物品ID，用于获取多模态嵌入特征

        Returns:
            dict: 填充缺失值后的特征字典
        """
        # 如果特征为空，则初始化为空字典
        if feat is None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        # 收集所有应该存在的特征ID
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        # 找出缺失的特征字段
        missing_fields = set(all_feat_ids) - set(feat.keys())
        # 用默认值填充缺失字段
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        # 处理多模态嵌入特征
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        自定义批处理函数，用于将多个样本组合成一个批次
        PyTorch DataLoader在构建批次时会调用此函数

        Args:
            batch (list): 由多个__getitem__返回的样本组成的列表

        Returns:
            tuple: 组合后的批次数据元组
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集类，继承自MyDataset类
    用于模型推理阶段，处理待预测的用户数据
    """

    def __init__(self, data_dir, args):
        """
        初始化测试数据集
        """
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        """
        加载测试数据文件和偏移量索引文件
        """
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动问题中的特征缺失
        冷启动指训练集中未出现但在测试集中出现的用户或物品
        对于字符串类型的特征值，统一替换为0

        Args:
            feat (dict): 原始特征字典

        Returns:
            dict: 处理后的特征字典
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                # 处理列表类型的特征值
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                # 字符串类型的特征值替换为0
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取指定用户的测试数据

        Args:
            uid (int): 用户在数据文件中的行号

        Returns:
            tuple: 包含以下元素的元组:
                - seq: 用户历史行为序列（物品ID）
                - token_type: 序列中每个位置的类型标识
                - seq_feat: 序列特征信息列表
                - user_id: 原始用户ID字符串
        """
        # 加载用户行为序列数据
        user_sequence = self._load_user_data(uid)

        # 扩展用户序列，处理冷启动特征
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:
                    # 如果是字符串，则为原始用户ID
                    user_id = u
                else:
                    # 如果是数字，则为内部ID，需要映射回原始ID
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 对于训练时未见过的物品，保留creative_id（通常远大于itemnum）
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        # 初始化返回数据结构
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        # 从后向前填充序列（左填充）
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        # 将None值替换为默认特征值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        返回测试数据集大小，即用户总数

        Returns:
            int: 测试数据集中用户的总数
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        自定义批处理函数，用于将多个测试样本组合成一个批次

        Args:
            batch (list): 由多个__getitem__返回的样本组成的列表

        Returns:
            tuple: 组合后的批次数据元组
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id

def save_emb(emb, save_path):
    """
    将嵌入向量保存为二进制文件，提高加载效率

    Args:
        emb (numpy.ndarray): 要保存的嵌入向量矩阵，形状为[num_points, num_dimensions]
        save_path (str): 保存文件路径
    """
    num_points = emb.shape[0]      # 向量数量
    num_dimensions = emb.shape[1]  # 向量维度
    print(f'正在保存 {save_path}')
    with open(Path(save_path), 'wb') as f:
        # 先写入向量数量和维度信息
        f.write(struct.pack('II', num_points, num_dimensions))
        # 再写入嵌入向量数据
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    并行加载多模态嵌入特征文件
    利用多进程提高大数据量加载效率

    Args:
        mm_path (str): 多模态特征文件目录路径
        feat_ids (list): 需要加载的特征ID列表

    Returns:
        dict: 多模态嵌入特征字典，格式为{特征ID: {物品ID: 嵌入向量}}
    """
    start_time = time.time()
    # 各特征ID对应的嵌入向量维度
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}

    # 使用进程池并行加载不同的特征ID
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        # 提交所有加载任务
        future_to_feat = {
            executor.submit(_load_single_feat_emb, mm_path, feat_id, SHAPE_DICT[feat_id]): feat_id 
            for feat_id in feat_ids
        }
        
        # 收集加载结果
        for future in as_completed(future_to_feat):
            feat_id, emb_dict = future.result()
            mm_emb_dict[feat_id] = emb_dict
    
    elapsed_time = time.time() - start_time
    print(f"完成所有多模态嵌入特征加载，耗时 {elapsed_time:.2f} 秒")
    return mm_emb_dict


def _load_single_feat_emb(mm_path, feat_id, shape):
    """
    加载单个多模态特征的嵌入向量
    
    Args:
        mm_path (str): 多模态特征文件目录路径
        feat_id (str): 特征ID
        shape (int): 嵌入向量维度
        
    Returns:
        tuple: (特征ID, 嵌入向量字典)
    """
    emb_dict = {}
    if feat_id != '81':
        try:
            base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
            json_files = list(base_path.glob('*.json'))
            print(f"正在加载特征 {feat_id}，共 {len(json_files)} 个文件")
            
            total_line_count = 0
            for i, json_file in enumerate(json_files):
                with open(json_file, 'r', encoding='utf-8') as file:
                    line_count = 0
                    for line in file:
                        data_dict_origin = json.loads(line.strip())
                        insert_emb = data_dict_origin['emb']
                        if isinstance(insert_emb, list):
                            insert_emb = np.array(insert_emb, dtype=np.float32)
                        data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                        emb_dict.update(data_dict)
                        line_count += 1
                    total_line_count += line_count
                    # 每处理100个文件或处理完所有文件时打印进度
                    if (i + 1) % 100 == 0 or (i + 1) == len(json_files):
                        print(f"  已处理 {i + 1}/{len(json_files)} 个文件，特征ID: {feat_id}")
        except Exception as e:
            print(f"特征加载出错: {e}")
    if feat_id == '81':
        # 特殊处理特征81，从pickle文件加载
        with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
            emb_dict = pickle.load(f)
    print(f'完成特征 #{feat_id} 的嵌入向量加载，共 {len(emb_dict)} 个向量')
    return feat_id, emb_dict