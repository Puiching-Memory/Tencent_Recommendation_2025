import psutil
import torch
import sys

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

    print("=" * 50)
