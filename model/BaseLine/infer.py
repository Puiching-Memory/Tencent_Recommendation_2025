import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MyTestDataset, save_emb
from model import BaselineModel


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
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # Training acceleration (默认开启)
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision', default=True)
    parser.add_argument('--use_compile', action='store_true', help='Compile model with torch.compile', default=True)
    parser.add_argument('--enable_tf32', action='store_true', help='Enable TF32 format for faster computations', default=True)
    parser.add_argument('--cudnn_deterministic', action='store_true', help='Use deterministic CuDNN operations (slower but reproducible)')

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
    
    # Enable TF32 for faster training on Ampere GPUs (默认启用)
    if args.enable_tf32 is not False and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for faster training")
    
    # Enable CuDNN benchmark for faster training
    if torch.cuda.is_available():
        if args.cudnn_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print("CuDNN deterministic mode enabled for reproducible results")
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("CuDNN benchmark enabled for faster training")
            
    data_path = os.environ.get('EVAL_DATA_PATH')
    print(f"Loading test dataset from {data_path}")
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn, persistent_workers=True
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    print(f"Dataset loaded: {len(test_dataset)} users, {itemnum} items")
    
    print(f"Creating model with hidden_units={args.hidden_units}, num_blocks={args.num_blocks}, num_heads={args.num_heads}")
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    
    # Compile model for faster execution (默认启用)
    if args.use_compile is not False:
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            
    model.eval()
    print("Model set to evaluation mode")

    ckpt_path = get_ckpt_path()
    print(f"Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    print("Checkpoint loaded successfully")
    
    all_embs = []
    user_list = []
    
    # Scaler for AMP (默认启用)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp is not False)) if torch.cuda.is_available() else torch.amp.GradScaler('cpu', enabled=(args.use_amp is not False))
    if args.use_amp is not False:
        print("Automatic Mixed Precision (AMP) enabled")
        
    print(f"Starting inference with batch_size={args.batch_size}")
    start_time = time.time()
    
    for step, batch in enumerate(test_loader):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        
        # Use AMP context manager (默认启用)
        with torch.amp.autocast('cuda', enabled=(args.use_amp is not False)):
            logits = model.predict(seq, seq_feat, token_type)
            
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id
        
        # Print progress every 10 steps
        if step % 10 == 0:
            elapsed_time = time.time() - start_time
            progress = (step + 1) / len(test_loader) * 100
            steps_per_second = (step + 1) / elapsed_time if elapsed_time > 0 else 0
            
            # Calculate estimated remaining time
            remaining_steps = len(test_loader) - (step + 1)
            estimated_remaining_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            
            # Format time for display
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    return f"{seconds/60:.1f}m"
                else:
                    return f"{seconds/3600:.1f}h"
            
            print(f"  Step {step+1}/{len(test_loader)} [{progress:.1f}%] - "
                  f"Speed: {steps_per_second:.2f} steps/s, "
                  f"ETA: {format_time(estimated_remaining_time)}")

    total_time = time.time() - start_time
    print(f"Inference completed in {total_time:.2f} seconds ({len(test_loader)/total_time:.2f} steps/s)")

    # 生成候选库的embedding 以及 id文件
    print("Generating candidate embeddings")
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    print("Candidate embeddings generated")
    
    # 保存query文件
    print("Saving query embeddings")
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    print("Query embeddings saved")
    
    # ANN 检索
    print("Performing ANN search")
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
    print(f"Running ANN command: {ann_cmd}")
    os.system(ann_cmd)
    print("ANN search completed")

    # 取出top-k
    print("Processing results")
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for i, top10 in enumerate(top10s_retrieved):
        # Print progress every 10 steps
        if i % 10 == 0:
            progress = (i + 1) / len(top10s_retrieved) * 100
            print(f"  Processing results: {i+1}/{len(top10s_retrieved)} [{progress:.1f}%]")
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    print("Results processed")

    return top10s, user_list
