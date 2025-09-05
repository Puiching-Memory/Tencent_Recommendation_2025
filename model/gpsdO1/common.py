import os
import random
from datetime import datetime
from tqdm import tqdm
import subprocess
import json


def get_file_line_count(file_path):
    line_count = 0
    buffer_size = 1024 * 1024
    with open(file_path, 'r') as file:
        buffer = file.read(buffer_size)
        while buffer:
            line_count += buffer.count('\n')
            buffer = file.read(buffer_size)
    return line_count


def create_batch_from_iterator(iterator, processor, batch_size):
    batch = []
    for item in iterator:
        x = processor(item)
        if x is None:
            continue
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_temp_file(ext=''):
    directory = os.environ.get("CACHE_DIR", "/tmp/")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    file_name = datetime.now().strftime("%Y%m%d_%H:%M:%S_") + str(random.randint(100000000, 999999999)) + ext
    return os.path.join(directory, file_name)


def count_lines_using_subprocess(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)
    num_lines = int(result.stdout.split()[0])
    return num_lines


def get_dataset(table_path, col_names="", mute=True):
    if os.path.isdir(table_path):
        # Find the JSONL file in the directory
        files = [f for f in os.listdir(table_path) if f.endswith('.jsonl')]
        if files:
            table_path = os.path.join(table_path, files[0])
        else:
            raise ValueError(f"No JSONL file found in directory {table_path}")
    
    if not table_path.endswith('.jsonl'):
        raise ValueError(f"Unsupported file format: {table_path}. Only JSONL is supported.")
    
    # Load as JSONL
    data = []
    with open(table_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sequence = json.loads(line.strip())
                if len(sequence) > 1:  # Only keep sequences with multiple interactions
                    data.append(sequence)
    return data


def latest_checkpoint(model_dir):
    """Find the latest checkpoint, supporting both GPSD and BaseLineO1 formats"""
    max_step = -1
    latest_ckpt_path = None
    
    for fname in os.listdir(model_dir):
        ckpt_path = os.path.join(model_dir, fname)
        
        # Handle BaseLineO1 format: global_step{step}.valid_loss={loss}/model.pt
        if os.path.isdir(ckpt_path) and fname.startswith('global_step') and 'valid_loss=' in fname:
            try:
                step_str = fname.split('global_step')[1].split('.')[0]
                step = int(step_str)
                model_pt_path = os.path.join(ckpt_path, 'model.pt')
                if os.path.exists(model_pt_path) and step > max_step:
                    max_step = step
                    latest_ckpt_path = model_pt_path
            except (ValueError, IndexError):
                continue
        
        # Handle old GPSD format: checkpoint-{step}.pth
        elif fname.startswith('checkpoint-') and fname.endswith('.pth'):
            try:
                step = int(fname.replace('checkpoint-', '').replace('.pth', ''))
                if step > max_step:
                    max_step = step
                    latest_ckpt_path = ckpt_path
            except ValueError:
                continue
    
    return latest_ckpt_path
