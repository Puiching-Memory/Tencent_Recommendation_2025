import psutil
import torch
import sys
import os
from pathlib import Path

def print_system_info():
    """
    获取并打印系统资源信息
    """
    # 获取CPU信息
    cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
    cpu_count_logical = psutil.cpu_count(logical=True)  # 逻辑核心数
    cpu_percent = psutil.cpu_percent(interval=1)  # CPU使用率
    memory = psutil.virtual_memory()  # 内存信息
    
    system_info = {}
    system_info['cpu_physical_cores'] = cpu_count
    system_info['cpu_logical_cores'] = cpu_count_logical
    system_info['cpu_usage_percent'] = cpu_percent
    system_info['total_memory_gb'] = round(memory.total / (1024**3), 2)
    system_info['available_memory_gb'] = round(memory.available / (1024**3), 2)
    system_info['memory_usage_percent'] = memory.percent
    system_info['python_version'] = sys.version.split()[0]
    system_info['torch_version'] = torch.__version__
    
    # 获取GPU信息
    system_info['gpu_count'] = torch.cuda.device_count()
    system_info['gpu_names'] = []
    system_info['gpu_memory_gb'] = []
    system_info['gpu_memory_used_gb'] = []
    system_info['cuda_version'] = 'N/A'
    if torch.cuda.is_available():
        system_info['cuda_version'] = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            system_info['gpu_names'].append(gpu_name)
            system_info['gpu_memory_gb'].append(round(gpu_memory_total, 2))
            system_info['gpu_memory_used_gb'].append(round(gpu_memory_allocated, 2))
    
    # 打印系统资源信息
    print("=" * 50)
    print("系统资源信息")
    print("=" * 50)

    print(f"Python 版本: {system_info.get('python_version', 'N/A')}")
    print(f"PyTorch 版本: {system_info.get('torch_version', 'N/A')}")
    print(f"CPU 物理核心数: {system_info.get('cpu_physical_cores', 'N/A')}")
    print(f"CPU 逻辑核心数: {system_info.get('cpu_logical_cores', 'N/A')}")
    print(f"CPU 使用率: {system_info.get('cpu_usage_percent', 'N/A')}%")
    print(f"总内存: {system_info.get('total_memory_gb', 'N/A')} GB")
    print(f"可用内存: {system_info.get('available_memory_gb', 'N/A')} GB")
    print(f"内存使用率: {system_info.get('memory_usage_percent', 'N/A')}%")
    print(f"CUDA 版本: {system_info.get('cuda_version', 'N/A')}")

    print(f"GPU 数量: {system_info.get('gpu_count', 0)}")
    if system_info['gpu_count'] > 0:
        for i, (name, memory_total, memory_used) in enumerate(zip(system_info['gpu_names'], system_info['gpu_memory_gb'], system_info['gpu_memory_used_gb'])):
            print(f"  GPU {i}: {name} ({memory_total} GB 总显存, {memory_used} GB 已使用)")

    try:
        import orjson
        print(f"orjson 版本: {orjson.__version__}")
    except Exception as e:
        print(e)

    print("=" * 50)

def parse_data_path_structure(data_path):
    """
    解析data_path目录结构并打印
    
    Args:
        data_path: 数据目录路径
    """
    if not data_path:
        print("警告: TRAIN_DATA_PATH 环境变量未设置")
        return
        
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"警告: 数据目录 {data_path} 不存在")
        return
    
    print(f"数据目录结构 ({data_path}):")
    print("-" * 50)
    
    try:
        # 列出根目录下的文件和文件夹
        items = list(data_path.iterdir())
        files = [item for item in items if item.is_file()]
        dirs = [item for item in items if item.is_dir()]
        
        # 打印文件
        if files:
            print("文件:")
            for file in sorted(files):
                size = file.stat().st_size
                print(f"  {file.name} ({format_file_size(size)})")
        
        # 打印目录
        if dirs:
            print("目录:")
            for dir_item in sorted(dirs):
                try:
                    sub_items = list(dir_item.iterdir())
                    sub_files = [item for item in sub_items if item.is_file()]
                    print(f"  {dir_item.name}/ ({len(sub_files)} files)")
                    
                    # 特别处理 creative_emb 目录，统计每个emb子目录的文件数量和大小
                    if dir_item.name == "creative_emb":
                        print("    creative_emb子目录统计:")
                        try:
                            emb_dirs = [item for item in sub_items if item.is_dir()]
                            for emb_dir in sorted(emb_dirs):
                                try:
                                    emb_files = list(emb_dir.iterdir())
                                    # 计算该目录下所有文件的总大小
                                    total_size = sum(f.stat().st_size for f in emb_files if f.is_file())
                                    print(f"      {emb_dir.name}/ ({len(emb_files)} files, {format_file_size(total_size)})")
                                except PermissionError:
                                    print(f"      {emb_dir.name}/ (无法访问)")
                        except Exception as e:
                            print(f"      无法读取creative_emb子目录: {e}")
                            
                except PermissionError:
                    print(f"  {dir_item.name}/ (无法访问)")
                    
    except Exception as e:
        print(f"解析目录结构时出错: {e}")
        
    print("-" * 50)

def format_file_size(size_bytes):
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节大小
        
    Returns:
        格式化后的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
        
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
        
    return f"{size_bytes:.1f}{size_names[i]}"

# Time formatting function
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"