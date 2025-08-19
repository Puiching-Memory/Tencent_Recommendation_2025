import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model_genrecsys import GenerativeRecommender


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


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


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    
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
    
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        
        # 构造生成模型所需的输入
        user_ids = seq[:, -1]  # 简化处理，实际应根据业务逻辑确定用户ID
        user_ids = torch.clamp(user_ids, 0, usernum - 1)  # 确保用户ID在有效范围内
        
        # 从特征中提取多模态特征
        seq_multimodal = extract_multimodal_features(seq_feat, args.mm_emb_id)
        
        # TODO: 实际实现需要根据数据格式调整，这里仅为示例
        logits, _ = model(user_ids, None, seq_multimodal)  # 仅获取用户嵌入
        
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    # ANN 检索
    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        + " --dataset_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
        + " --dataset_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
        + " --query_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
        + " --result_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    os.system(ann_cmd)

    # 取出top-k
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list