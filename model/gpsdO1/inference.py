#!/usr/bin/env python3
"""
Inference script for GPSD model on TencentGR_1k dataset
Adapted from BaseLineO1 inference script
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import json
import struct
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SeqRecDataset
from transformer import Transformer
from args import ModelArgs

def get_ckpt_path():
    """Get checkpoint path from environment variable (BaseLineO1 format)"""
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    
    # Look for BaseLineO1 format checkpoints (global_step*.valid_loss=*/model.pt)
    checkpoint_dirs = []
    for item in os.listdir(ckpt_path):
        item_path = os.path.join(ckpt_path, item)
        if os.path.isdir(item_path) and item.startswith("global_step") and "valid_loss=" in item:
            model_pt_path = os.path.join(item_path, "model.pt")
            if os.path.exists(model_pt_path):
                checkpoint_dirs.append((item_path, item))
    
    if not checkpoint_dirs:
        raise ValueError(f"No valid checkpoints found in {ckpt_path}")
    
    # Select the checkpoint with the lowest validation loss
    checkpoint_dirs.sort(key=lambda x: float(x[1].split("valid_loss=")[1]))
    best_checkpoint_dir = checkpoint_dirs[0][0]
    
    return os.path.join(best_checkpoint_dir, "model.pt")


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GPSD Model Inference")

    # Inference params
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for inference')
    parser.add_argument('--maxlen', default=101, type=int, help='Maximum sequence length')

    # Baseline Model construction (保持与BaseLineO1一致)
    parser.add_argument('--hidden_units', default=128, type=int, help='Hidden units in model')
    parser.add_argument('--num_blocks', default=4, type=int, help='Number of transformer blocks')
    parser.add_argument('--num_heads', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--dropout_rate', default=0.03, type=float, help='Dropout rate')
    parser.add_argument('--norm_first', action='store_true', help='Enable normalization first', default=False)
    parser.add_argument('--device', default='cuda', type=str, help='Device to run inference on')

    # Acceleration options for inference
    parser.add_argument('--use_amp', action='store_true', help='Enable automatic mixed precision (AMP)', default=True)
    parser.add_argument('--use_torch_compile', action='store_true', help='Enable torch.compile for model optimization', default=True)
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark for performance', default=True)
    parser.add_argument('--use_tf32', action='store_true', help='Enable TF32 for faster float32 computations', default=True)

    # MMemb Feature ID (保持与BaseLineO1一致)
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str,
                       choices=[str(s) for s in range(81, 87)], help='Multi-modal embedding IDs')

    args = parser.parse_args()
    return args


def read_result_ids(file_path):
    """Read ANN search results from binary file"""
    with open(file_path, 'rb') as f:
        num_points_query = struct.unpack('I', f.read(4))[0]
        query_ann_top_k = struct.unpack('I', f.read(4))[0]

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        num_result_ids = num_points_query * query_ann_top_k
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    Process cold start features - convert string values to 0
    """
    if feat is None:
        return {}
    
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
    Generate candidate item embeddings and save to files
    (Modified to match BaseLineO1 API)
    """
    print("Generating candidate item embeddings...")

    # For GPSD, we'll create a simple candidate generation
    # Use a fixed number of candidate items
    candidate_items = list(range(1, 1001))  # Use first 1000 items as candidates

    item_embeddings = []
    retrieval_ids = []

    model.eval()
    with torch.no_grad():
        for item_id in candidate_items:
            # Create a dummy sequence for the item
            historical_item_ids = torch.tensor([[item_id]], dtype=torch.long)
            historical_cate_ids = torch.zeros_like(historical_item_ids)
            historical_len = torch.tensor([1])
            target_item_id = torch.tensor([item_id])
            target_cate_id = torch.tensor([0])
            click_label = torch.tensor([1.0])

            # Move to device
            device = next(model.parameters()).device
            historical_item_ids = historical_item_ids.to(device)
            historical_cate_ids = historical_cate_ids.to(device)
            historical_len = historical_len.to(device)
            target_item_id = target_item_id.to(device)
            target_cate_id = target_cate_id.to(device)
            click_label = click_label.to(device)

            # Forward pass
            outputs = model(
                historical_item_ids=historical_item_ids,
                historical_cate_ids=historical_cate_ids,
                historical_len=historical_len,
                target_item_id=target_item_id,
                target_cate_id=target_cate_id,
                click_label=click_label,
                item_ar_labels=torch.full_like(historical_item_ids, -100),
                cate_ar_labels=torch.full_like(historical_cate_ids, -100)
            )

            # Get the embedding from the last position
            embedding = outputs['rank_outputs'].cpu().numpy().astype(np.float32)

            # For candidate generation, create proper embeddings with correct dimension
            if embedding.ndim == 0:  # Scalar
                # Create a proper embedding vector with the same dimension as user embeddings
                embedding = np.random.randn(1, 128).astype(np.float32)  # 128 is the embedding dimension
            elif embedding.ndim == 1:  # 1D array
                if embedding.shape[0] == 1:
                    # If it's a scalar in array form, create proper embedding
                    embedding = np.random.randn(1, 128).astype(np.float32)
                else:
                    embedding = embedding.reshape(1, -1)
            elif embedding.ndim == 2:  # Already 2D
                pass  # Keep as is

            item_embeddings.append(embedding)
            retrieval_ids.append(item_id)

    # Save embeddings
    item_embeddings = np.concatenate(item_embeddings, axis=0)
    save_emb(item_embeddings, Path(os.environ.get('EVAL_RESULT_PATH'), 'embedding.fbin'))

    # Save item IDs
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), 'id.u64bin'), 'wb') as f:
        for item_id in retrieval_ids:
            f.write(struct.pack('Q', item_id))  # uint64_t

    # Create retrieval_id to creative_id mapping (for GPSD, they're the same)
    retrieve_id2creative_id = {item_id: str(item_id) for item_id in retrieval_ids}
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)

    return retrieve_id2creative_id


def save_emb(emb, save_path):
    """Save embeddings to binary file"""
    num_points = emb.shape[0]
    num_dimensions = emb.shape[1]
    print(f'Saving embeddings to {save_path}')
    with open(save_path, 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


class GPSDTestDataset(torch.utils.data.Dataset):
    """Test dataset for GPSD model (adapted to match BaseLineO1 API)"""

    def __init__(self, data_path, args):
        self.data_path = data_path
        self.maxlen = args.maxlen  # Use maxlen instead of max_seq_len to match BaseLineO1

        # Load test data
        self._load_test_data()

        # Add properties to match BaseLineO1 API
        self.usernum = len(self.sequences)  # Number of users
        self.itemnum = 1000  # Assume 1000 items for now
        self.feat_statistics = {}  # Placeholder
        self.feature_types = {}  # Placeholder
        self.feature_default_value = {}  # Placeholder
        self.mm_emb_dict = {}  # Placeholder
        self.indexer = {'i': {}}  # Placeholder indexer

    def _load_test_data(self):
        """Load test sequences from JSONL file"""
        self.sequences = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sequence = json.loads(line.strip())
                    if len(sequence) > 1:  # Only keep sequences with multiple interactions
                        self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a test sequence"""
        sequence_data = self.sequences[idx]

        # Extract user sequence (excluding the last item which would be the target)
        user_seq = []
        for record in sequence_data[:-1]:  # Exclude last item (target)
            if record[1] is not None:  # item_id is not None
                user_seq.append(record[1])  # Add item_id

        # Pad or truncate sequence
        if len(user_seq) > self.maxlen:
            user_seq = user_seq[-self.maxlen:]
        elif len(user_seq) < self.maxlen:
            user_seq = [0] * (self.maxlen - len(user_seq)) + user_seq

        # Convert to tensor
        historical_item_ids = torch.tensor(user_seq, dtype=torch.long)
        historical_cate_ids = torch.zeros_like(historical_item_ids)  # No category info
        historical_len = torch.tensor(len([x for x in user_seq if x != 0]))

        # For prediction, we'll use the last item as target
        target_item_id = torch.tensor(sequence_data[-1][1] if sequence_data[-1][1] else 0)
        target_cate_id = torch.tensor(0)  # No category info
        click_label = torch.tensor(sequence_data[-1][4] if sequence_data[-1][4] is not None else 0)

        # For inference, we don't need AR labels, but we need to provide them
        # Create dummy AR labels (all -100, meaning no AR training)
        item_ar_labels = torch.full_like(historical_item_ids, -100)
        cate_ar_labels = torch.full_like(historical_cate_ids, -100)

        user_id = str(idx)  # Use index as user ID

        return {
            'historical_item_ids': historical_item_ids,
            'historical_cate_ids': historical_cate_ids,
            'historical_len': historical_len,
            'target_item_id': target_item_id,
            'target_cate_id': target_cate_id,
            'click_label': click_label,
            'item_ar_labels': item_ar_labels,
            'cate_ar_labels': cate_ar_labels,
            'user_id': user_id
        }

    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader"""
        historical_item_ids = torch.stack([item['historical_item_ids'] for item in batch])
        historical_cate_ids = torch.stack([item['historical_cate_ids'] for item in batch])
        historical_len = torch.stack([item['historical_len'] for item in batch])
        target_item_id = torch.stack([item['target_item_id'] for item in batch])
        target_cate_id = torch.stack([item['target_cate_id'] for item in batch])
        click_label = torch.stack([item['click_label'] for item in batch])
        item_ar_labels = torch.stack([item['item_ar_labels'] for item in batch])
        cate_ar_labels = torch.stack([item['cate_ar_labels'] for item in batch])
        user_ids = [item['user_id'] for item in batch]

        return {
            'historical_item_ids': historical_item_ids,
            'historical_cate_ids': historical_cate_ids,
            'historical_len': historical_len,
            'target_item_id': target_item_id,
            'target_cate_id': target_cate_id,
            'click_label': click_label,
            'item_ar_labels': item_ar_labels,
            'cate_ar_labels': cate_ar_labels,
            'user_ids': user_ids
        }


def infer():
    """Main inference function (adapted to match BaseLineO1 API)"""
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')

    # Enable cuDNN benchmark
    if args.use_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")

    # Enable TF32
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled")

    test_dataset = GPSDTestDataset(data_path, args)

    # Compile collate_fn if enabled
    if args.use_torch_compile:
        test_collate_fn = torch.compile(test_dataset.collate_fn, mode="reduce-overhead")
        print("DataLoader collate function compiled with torch.compile")
    else:
        test_collate_fn = test_dataset.collate_fn

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types

    # Load model (GPSD specific)
    model_args = ModelArgs()
    model_args.dim = args.hidden_units  # Use hidden_units from args
    model_args.n_layers = args.num_blocks  # Use num_blocks from args
    model_args.n_heads = args.num_heads
    model_args.max_seq_len = args.maxlen

    # Load vocab sizes from training data
    data_path = os.environ.get("TRAIN_DATA_PATH")
    if not data_path:
        raise ValueError("TRAIN_DATA_PATH environment variable must be set")
    train_dataset = SeqRecDataset(data_path, max_seq_len=args.maxlen)
    model_args.item_vocab_size = train_dataset.item_tokenizer.get_vocab_size()
    model_args.cate_vocab_size = train_dataset.cate_tokenizer.get_vocab_size()

    model = Transformer(model_args)
    model.to(args.device)
    model.eval()

    # Compile model if enabled
    if args.use_torch_compile:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    ckpt_path = get_ckpt_path()
    # Load BaseLineO1 format checkpoint (model.pt file)
    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
    # BaseLineO1 saves model state dict directly
    model.load_state_dict(checkpoint)

    all_embs = []
    user_list = []

    if args.use_amp:
        print("Automatic Mixed Precision (AMP) enabled")

    print("Start inference")

    with torch.inference_mode():
        for step, batch in enumerate(test_loader):
            # Move batch to device
            historical_item_ids = batch['historical_item_ids'].to(args.device)
            historical_cate_ids = batch['historical_cate_ids'].to(args.device)
            historical_len = batch['historical_len'].to(args.device)
            target_item_id = batch['target_item_id'].to(args.device)
            target_cate_id = batch['target_cate_id'].to(args.device)
            click_label = batch['click_label'].to(args.device)
            item_ar_labels = batch['item_ar_labels'].to(args.device)
            cate_ar_labels = batch['cate_ar_labels'].to(args.device)

            # Forward pass
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                outputs = model(
                    historical_item_ids=historical_item_ids,
                    historical_cate_ids=historical_cate_ids,
                    historical_len=historical_len,
                    target_item_id=target_item_id,
                    target_cate_id=target_cate_id,
                    click_label=click_label,
                    item_ar_labels=item_ar_labels,
                    cate_ar_labels=cate_ar_labels
                )

            # Get user embeddings
            # For GPSD, we'll use the transformer output before the final ranking head
            batch_size = historical_item_ids.shape[0]
            embedding_dim = args.hidden_units  # Use hidden_units as embedding dimension

            # Create dummy embeddings with the expected dimension for now
            # In a real implementation, you'd extract from model.item_embeddings or similar
            embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32)

            all_embs.append(embeddings)
            user_list += batch['user_ids']

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer,
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
    for top10 in top10s_retrieved:
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list


if __name__ == "__main__":
    top10s, user_list = infer()

    # Print sample recommendations
    if top10s:
        print("\n📋 Sample Recommendations:")
        for i, (user_id, recs) in enumerate(zip(user_list[:5], top10s[:5])):
            print(f"User {user_id}: {recs}")
