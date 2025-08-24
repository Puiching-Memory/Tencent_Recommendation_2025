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
from model import GenerativeRecommender
from hstu import HSTUModel  # 添加HSTU模型导入
from utils import parse_data_path_structure

def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Inference params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--maxlen', default=101, type=int)

    # HSTU Model construction
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=4, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--dropout_rate', default=0.03, type=float)
    parser.add_argument('--norm_first', action='store_true', help='Enable normalization first in transformer layers', default=False)
    parser.add_argument('--device', default='cuda', type=str)

    # 新增：模型类型选择
    parser.add_argument('--model_type', default='hstu', type=str, choices=['generative', 'hstu'],
                        help='选择要使用的模型类型: generative(生成式模型) 或 hstu(HSTU模型)')

    # Acceleration options for inference
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP)',default=True)
    parser.add_argument('--use_torch_compile', action='store_true', help='Enable torch.compile for model optimization',default=True)
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark for performance',default=True)
    parser.add_argument('--use_tf32', action='store_true', help='Enable TF32 for faster float32 computations',default=True)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


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
        retrieve_id2creative_id: 检索ID到creative_id的映射
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = os.path.join(os.environ.get('EVAL_DATA_PATH'), "candidate.jsonl")
    item_ids = []
    creative_ids = []
    retrieval_ids = []
    features = []
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
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn, pin_memory=True
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    
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

    model.eval()

    # Compile model if enabled
    if args.use_torch_compile:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type, seq_feat, user = batch
        seq = seq.to(args.device)
        with torch.no_grad():
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.model_type == 'generative':
                        user_emb = model.predict(seq, seq_feat, token_type)
                    elif args.model_type == 'hstu':
                        user_emb = model.predict(seq, seq_feat, token_type)
            else:
                if args.model_type == 'generative':
                    user_emb = model.predict(seq, seq_feat, token_type)
                elif args.model_type == 'hstu':
                    user_emb = model.predict(seq, seq_feat, token_type)
        all_embs.append(user_emb.cpu().numpy())
        user_list.extend(user)

    all_embs = np.vstack(all_embs)

    save_path = Path(os.environ.get('EVAL_RESULT_PATH'))
    save_path.mkdir(parents=True, exist_ok=True)
    save_emb(all_embs, Path(save_path, 'embedding.fbin'))
    with open(Path(save_path, 'id.txt'), 'w') as f:
        for user_id in user_list:
            f.write(str(user_id) + '\n')


if __name__ == '__main__':
    infer()