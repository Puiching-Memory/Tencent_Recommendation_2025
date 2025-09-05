import random
from typing import Union
import os
import torch
from torch.utils.data import Dataset
import pickle
import json
import numpy as np
from common import get_dataset

class ItemTokenizer(object):
    def __init__(self, table_path: str=None):
        self.path = table_path
        self.unk_token = 0
        self.tokenize_map = {}
        self.detokenize_map = {}
        if table_path:
            self._build_map()
        else:
            # For TencentGR_1k, items are already re-id, so we don't need tokenizer
            self.tokenize_map = None
            self.detokenize_map = None

    def _build_map(self):
        with open(self.path, 'r') as file:
            while True:
                data = file.readline()
                if data == '':
                    break
                data = data.split(',')
                item_id, item_index = str(data[0]), int(data[1])
                self.tokenize_map[item_id] = item_index
                self.detokenize_map[item_index] = item_id

    def get_vocab_size(self):
        if self.tokenize_map is None:
            # For TencentGR_1k, assume max item id is around 60000
            return 60001
        return len(self.detokenize_map) + 1

    def tokenize(self, item: Union[str, int]):
        if self.tokenize_map is None:
            # For TencentGR_1k, items are already integers
            return int(item) if item is not None else 0
        item = str(item)
        if item in self.tokenize_map:
            return self.tokenize_map[item]
        return self.unk_token

    def detokenize(self, id: int):
        if self.detokenize_map is None:
            return str(id)
        if id in self.detokenize_map:
            return self.detokenize_map[id]
        return "<unk>"

    def in_vocab(self, item: Union[str, int]):
        if self.tokenize_map is None:
            return item is not None and 0 <= int(item) < self.get_vocab_size()
        item = str(item)
        return item in self.tokenize_map

    def save(self, path:str):
        if self.tokenize_map is not None:
            with open(path, 'wb') as f:
                obj = {"tokenize_map": self.tokenize_map, "detokenize_map": self.detokenize_map}
                pickle.dump(obj, f)


class CateTokenizer(ItemTokenizer):
    def __init__(self, table_path: str=None):
        super(CateTokenizer, self).__init__(table_path)


def pad_or_truncate(to_pad, max_seq_len, padding_value):
    orig_length = len(to_pad)
    if orig_length > max_seq_len:
        return to_pad[-max_seq_len:], max_seq_len
    else:
        to_pad.extend([padding_value] * (max_seq_len-len(to_pad)))
        return to_pad, orig_length


def preprocess_tencent_data(sequence_data, max_seq_len, item_tokenizer, cate_tokenizer, use_ar_on_rank_samples, include_target_for_ar, modelling_style="ar"):
    """
    Preprocess TencentGR_1k sequence data
    sequence_data: list of [user_id, item_id, user_feature, item_feature, action_type, timestamp]
    """
    # Filter out user profile records (where item_id is None)
    item_records = [record for record in sequence_data if record[1] is not None]
    
    if len(item_records) < 2:
        return None
    
    # Sort by timestamp
    item_records.sort(key=lambda x: x[5])
    
    target_record = item_records[-1]  # Last interaction as target
    historical_records = item_records[:-1]  # Previous interactions as history
    
    target_item_id = item_tokenizer.tokenize(target_record[1])
    target_cate_id = 0  # TencentGR_1k doesn't have category info, use 0 as default
    click_label = int(target_record[4]) if target_record[4] is not None else 0  # action_type as click label, default to 0 if None
    
    historical_item_ids = []
    historical_cate_ids = []
    item_ar_labels = []
    cate_ar_labels = []
    
    for record in historical_records:
        item_id = item_tokenizer.tokenize(record[1])
        cate_id = 0  # No category info
        if item_id == item_tokenizer.unk_token:
            continue
        if historical_item_ids:
            item_ar_labels.append(item_id)
            cate_ar_labels.append(cate_id)
        historical_item_ids.append(item_id)
        historical_cate_ids.append(cate_id)
    
    if include_target_for_ar and click_label in (0, 1):
        item_ar_labels.append(target_item_id)
        cate_ar_labels.append(target_cate_id)
    else:
        item_ar_labels.append(-100)
        cate_ar_labels.append(-100)

    if len(historical_item_ids) < 1:
        return None

    if not use_ar_on_rank_samples and click_label in (0, 1):
        item_ar_labels = [-100] * len(item_ar_labels)
        cate_ar_labels = [-100] * len(cate_ar_labels)
    elif modelling_style == "mlm":
        masked_historical_item_ids = [0 if random.random() < 0.3 else x for x in historical_item_ids]
        masked_historical_cate_ids = [0 if random.random() < 0.3 else x for x in historical_cate_ids]
        item_ar_labels = [x if mx == 0 and x > 0 else -100 for x, mx in zip(historical_item_ids, masked_historical_item_ids)]
        cate_ar_labels = [x if mx == 0 and x > 0 else -100 for x, mx in zip(historical_cate_ids, masked_historical_cate_ids)]
        historical_item_ids = masked_historical_item_ids
        historical_cate_ids = masked_historical_cate_ids

    historical_item_ids, historical_len = pad_or_truncate(historical_item_ids, max_seq_len, 0)
    historical_cate_ids, _ = pad_or_truncate(historical_cate_ids, max_seq_len, 0)
    item_ar_labels, _ = pad_or_truncate(item_ar_labels, max_seq_len, -100)
    cate_ar_labels, _ = pad_or_truncate(cate_ar_labels, max_seq_len, -100)

    return {
        "target_item_id": torch.tensor(target_item_id),
        "target_cate_id": torch.tensor(target_cate_id),
        "click_label": torch.tensor(click_label),
        "historical_item_ids": torch.tensor(historical_item_ids, dtype=torch.int64),
        "historical_cate_ids": torch.tensor(historical_cate_ids, dtype=torch.int64),
        "historical_len": torch.tensor(historical_len),
        "item_ar_labels": torch.tensor(item_ar_labels),
        "cate_ar_labels": torch.tensor(cate_ar_labels)
    }


class SeqRecDataset(Dataset):
    def __init__(self, table_path, max_seq_len=100, modelling_style="ar", use_ar_on_rank_samples=False, include_target_for_ar=True, data_cols="user_id,item_id,cate_id,is_clicked,seq_value,seq_len", split="train", train_ratio=0.8, seed=42):
        self.table_path = table_path
        self.data_cols = data_cols
        self.max_seq_len = max_seq_len
        self.include_target_for_ar = include_target_for_ar
        self.use_ar_on_rank_samples = use_ar_on_rank_samples
        self.modelling_style = modelling_style
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        
        # Load TencentGR_1k dataset (JSONL format)
        self.item_tokenizer = ItemTokenizer()  # No tokenizer file needed
        self.cate_tokenizer = CateTokenizer()  # No tokenizer file needed
        self.data = self._load_tencent_data(table_path)
        self._split_data()

    def _load_tencent_data(self, data_path):
        """Load TencentGR_1k data from JSONL file"""
        if os.path.isdir(data_path):
            data_path = os.path.join(data_path, 'seq.jsonl')
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sequence = json.loads(line.strip())
                    if len(sequence) > 1:  # Only keep sequences with multiple interactions
                        data.append(sequence)
        return data

    def _split_data(self):
        """Split data into train and validation sets"""
        import random
        random.seed(self.seed)
        
        # Shuffle data
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        # Split indices
        train_size = int(len(indices) * self.train_ratio)
        if self.split == "train":
            self.indices = indices[:train_size]
        elif self.split == "val":
            self.indices = indices[train_size:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")
        
        print(f"Split '{self.split}': {len(self.indices)} samples out of {len(self.data)} total")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        if "TencentGR_1k" in self.table_path:
            # TencentGR_1k format
            sequence_data = self.data[actual_index]
            features = preprocess_tencent_data(
                sequence_data, 
                max_seq_len=self.max_seq_len, 
                item_tokenizer=self.item_tokenizer, 
                cate_tokenizer=self.cate_tokenizer, 
                modelling_style=self.modelling_style, 
                use_ar_on_rank_samples=self.use_ar_on_rank_samples, 
                include_target_for_ar=self.include_target_for_ar
            )
        else:
            # Original format - removed to simplify
            raise NotImplementedError("Only TencentGR_1k dataset is supported")
        return features

    def save_tokenizers(self, ckpt_dir):
        self.item_tokenizer.save(os.path.join(ckpt_dir, "item.tokenizer.pkl"))
        self.cate_tokenizer.save(os.path.join(ckpt_dir, "cate.tokenizer.pkl"))


if __name__ == '__main__':
    import os
    from torch.utils.data import DataLoader
    
    # Test with TencentGR_1k dataset
    data_path = os.environ.get("TRAIN_DATA_PATH")
    if os.path.exists(data_path):
        dataset = SeqRecDataset(data_path, modelling_style="ar", max_seq_len=50)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample data:")
        for batch in dataloader:
            print("Batch keys:", list(batch.keys()))
            print("Target item ids:", batch['target_item_id'])
            print("Historical item ids shape:", batch['historical_item_ids'].shape)
            print("Click labels:", batch['click_label'])
            break
    else:
        print(f"TencentGR_1k data not found at {data_path}")
