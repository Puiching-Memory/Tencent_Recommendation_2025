# =============================================================================
# 🚀 GPSD Training Launcher for TencentGR_1k Dataset
# =============================================================================

import os
import sys
import argparse
import subprocess
import time

# =============================================================================
# � 配置参数
# =============================================================================

# 数据路径配置
DATA_PATH = os.environ.get("TRAIN_DATA_PATH")
if not DATA_PATH:
    raise ValueError("TRAIN_DATA_PATH environment variable must be set")
CHECKPOINT_PATH = "./checkpoints"
LOG_PATH = "./logs"
TF_EVENTS_PATH = "./tf_events"

# 训练参数配置
TRAINING_CONFIG = {
    # 基本训练参数
    "epochs": 5,                    # 训练轮数
    "batch_size": 128,              # 批次大小
    "learning_rate": 0.001,         # 学习率

    # 模型参数
    "model_dim": 128,               # 模型维度
    "model_n_layers": 4,            # Transformer层数
    "model_n_heads": 8,             # 注意力头数
    "model_dropout": 0.1,           # Dropout率

    # 训练控制参数
    "log_steps": 5,                 # 日志记录步数
    "eval_steps": 50,               # 评估步数
    "warmup_steps": 50,             # 预热步数
    "weight_decay": 0.01,           # 权重衰减
    "max_grads_norm": 1.0,          # 梯度裁剪

    # 数据加载参数
    "num_data_workers": 2,          # 数据加载进程数
    "prefetch_factor": 32,          # 预取因子

    # 其他参数
    "use_bf16": False,              # 是否使用BF16
    "random_seed": 17,              # 随机种子
}

# =============================================================================
# 🛠️ 核心功能函数
# =============================================================================

def print_banner():
    """显示启动器横幅"""
    print("🚀 GPSD Training Launcher for TencentGR_1k Dataset")
    print("=" * 55)
    print()

def show_help():
    """显示帮助信息"""
    print("GPSD Training Launcher for TencentGR_1k Dataset")
    print("=" * 55)
    print()
    print("快速使用:")
    print("  python launcher.py                    # 使用默认配置")
    print("  python launcher.py --test            # 测试配置")
    print("  python launcher.py --clean           # 清理项目目录")
    print()
    print("示例:")
    print("  python launcher.py --test")
    print()

def set_environment_variables(config):
    """设置环境变量"""
    print("🔧 设置环境变量...")

    # 数据和路径
    os.environ['TRAIN_DATA_PATH'] = DATA_PATH
    os.environ['TRAIN_CKPT_PATH'] = CHECKPOINT_PATH
    os.environ['TRAIN_LOG_PATH'] = LOG_PATH
    os.environ['TRAIN_TF_EVENTS_PATH'] = TF_EVENTS_PATH

    # 训练参数
    os.environ['TRAIN_EPOCHS'] = str(config['epochs'])
    os.environ['TRAIN_BATCH_SIZE'] = str(config['batch_size'])
    os.environ['TRAIN_LEARNING_RATE'] = str(config['learning_rate'])
    os.environ['TRAIN_MIN_LEARNING_RATE'] = "0.00001"
    os.environ['TRAIN_WARMUP_STEPS'] = str(config['warmup_steps'])
    os.environ['TRAIN_WEIGHT_DECAY'] = str(config['weight_decay'])
    os.environ['TRAIN_MAX_GRADS_NORM'] = str(config['max_grads_norm'])
    os.environ['TRAIN_LOG_STEPS'] = str(config['log_steps'])
    os.environ['TRAIN_EVAL_STEPS'] = str(config['eval_steps'])
    os.environ['TRAIN_GRAD_ACCUMULATE_STEPS'] = "1"

    # 模型参数
    os.environ['MODEL_DIM'] = str(config['model_dim'])
    os.environ['MODEL_N_LAYERS'] = str(config['model_n_layers'])
    os.environ['MODEL_N_HEADS'] = str(config['model_n_heads'])
    os.environ['MODEL_DROPOUT'] = str(config['model_dropout'])
    os.environ['MODEL_MAX_SEQ_LEN'] = "50"

    # 训练选项
    os.environ['TRAIN_USE_BF16'] = str(config['use_bf16']).lower()
    os.environ['TRAIN_SHUFFLE'] = "true"
    os.environ['TRAIN_RESTORE'] = "true"
    os.environ['TRAIN_RANDOM_SEED'] = str(config['random_seed'])

    # 数据加载
    os.environ['TRAIN_NUM_DATA_WORKERS'] = str(config['num_data_workers'])
    os.environ['TRAIN_PREFETCH_FACTOR'] = str(config['prefetch_factor'])

    print("✅ 环境变量配置完成!")

def show_configuration(config_name, config):
    """显示当前配置"""
    print()
    print(f"📊 训练配置: {config_name}")
    print("=" * 60)
    print(f"配置名称  : {config_name}")
    print(f"数据路径   : {DATA_PATH}")
    print(f"检查点路径 : {CHECKPOINT_PATH}")
    print(f"日志路径   : {LOG_PATH}")
    print(f"事件路径   : {TF_EVENTS_PATH}")
    print(f"训练轮数   : {config['epochs']}")
    print(f"批次大小   : {config['batch_size']}")
    print(f"学习率     : {config['learning_rate']}")
    print(f"模型维度   : {config['model_dim']}")
    print(f"模型层数   : {config['model_n_layers']}")
    print(f"注意力头数 : {config['model_n_heads']}")
    print(f"Dropout率  : {config['model_dropout']}")
    print()

def test_configuration():
    """测试配置和依赖"""
    print("🧪 测试配置...")

    # 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据文件未找到: {DATA_PATH}")
        return False
    print(f"✅ 数据文件已找到: {DATA_PATH}")

    # 检查Python
    try:
        result = subprocess.run([sys.executable, "--version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Python可用: {result.stdout.strip()}")
        else:
            print("❌ Python未找到或无法工作")
            return False
    except Exception as e:
        print(f"❌ Python错误: {e}")
        return False

    # 检查必需模块
    required_modules = ['torch', 'numpy', 'loguru']
    for module in required_modules:
        try:
            result = subprocess.run([sys.executable, "-c", f"import {module}"],
                                  capture_output=True)
            if result.returncode == 0:
                print(f"✅ 模块 '{module}' 可用")
            else:
                print(f"❌ 模块 '{module}' 不可用")
                return False
        except Exception as e:
            print(f"❌ 模块 '{module}' 错误: {e}")
            return False

    print("🎉 配置测试通过!")
    return True

def clean_project():
    """清理项目目录"""
    print("🧹 清理项目目录...")

    import shutil
    from pathlib import Path

    dirs_to_clean = [
        CHECKPOINT_PATH,
        LOG_PATH,
        TF_EVENTS_PATH,
        "__pycache__"
    ]

    for item in dirs_to_clean:
        path = Path(item)
        if path.is_dir():
            print(f"删除目录: {path}")
            shutil.rmtree(path)
        elif path.exists():
            print(f"删除文件: {path}")
            path.unlink()

    print("✅ 项目目录已清理!")

def run_training():
    """运行训练过程"""
    print("🚀 开始GPSD训练...")
    print()

    start_time = time.time()

    try:
        # 构建命令参数
        cmd_args = [
            sys.executable, "train.py",
            "--epochs", os.environ['TRAIN_EPOCHS'],
            "--batch_size", os.environ['TRAIN_BATCH_SIZE'],
            "--learning_rate", os.environ['TRAIN_LEARNING_RATE'],
            "--min_learning_rate", os.environ['TRAIN_MIN_LEARNING_RATE'],
            "--warmup_steps", os.environ['TRAIN_WARMUP_STEPS'],
            "--weight_decay", os.environ['TRAIN_WEIGHT_DECAY'],
            "--max_grads_norm", os.environ['TRAIN_MAX_GRADS_NORM'],
            "--log_steps", os.environ['TRAIN_LOG_STEPS'],
            "--eval_steps", os.environ['TRAIN_EVAL_STEPS'],
            "--grad_accumulate_steps", os.environ['TRAIN_GRAD_ACCUMULATE_STEPS'],
            "--model_dim", os.environ['MODEL_DIM'],
            "--model_n_layers", os.environ['MODEL_N_LAYERS'],
            "--model_n_heads", os.environ['MODEL_N_HEADS'],
            "--model_dropout", os.environ['MODEL_DROPOUT'],
            "--model_max_seq_len", os.environ['MODEL_MAX_SEQ_LEN'],
            "--data_path", os.environ['TRAIN_DATA_PATH'],
            "--num_data_workers", os.environ['TRAIN_NUM_DATA_WORKERS'],
            "--prefetch_factor", os.environ['TRAIN_PREFETCH_FACTOR'],
            "--ckpt_path", os.environ['TRAIN_CKPT_PATH'],
            "--log_path", os.environ['TRAIN_LOG_PATH'],
            "--tf_events_path", os.environ['TRAIN_TF_EVENTS_PATH']
        ]

        # 添加布尔参数
        if os.environ['TRAIN_USE_BF16'].lower() == 'true':
            cmd_args.append('--use_bf16')
        if os.environ['TRAIN_SHUFFLE'].lower() == 'true':
            cmd_args.append('--shuffle')
        if os.environ['TRAIN_RESTORE'].lower() == 'true':
            cmd_args.append('--restore')

        print(f"命令: {' '.join(cmd_args)}")
        print()

        # 运行训练
        result = subprocess.run(cmd_args, cwd=os.getcwd())

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print()
            print("✅ 训练成功完成!")
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"⏱️  总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print()
            print(f"📁 检查点保存位置: {CHECKPOINT_PATH}/tencentgr/")
            print("📊 训练日志保存在 logs/ 目录中")
        else:
            print()
            print(f"❌ 训练失败，返回码: {result.returncode}")

        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️  用户中断训练")
        return 1
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        return 1

# =============================================================================
# 🎯 主函数
# =============================================================================

def main():
    """主入口点"""
    parser = argparse.ArgumentParser(description="GPSD Training Launcher", add_help=False)
    parser.add_argument("--test", action="store_true",
                      help="测试配置而不训练")
    parser.add_argument("--clean", action="store_true",
                      help="清理项目目录")
    parser.add_argument("-h", "--help", action="store_true",
                      help="显示帮助信息")

    args = parser.parse_args()

    # 处理帮助
    if args.help:
        show_help()
        return 0

    # 处理清理
    if args.clean:
        clean_project()
        return 0

    # 获取配置
    config = TRAINING_CONFIG.copy()

    # 显示横幅
    print_banner()

    # 设置环境变量
    set_environment_variables(config)

    # 显示配置
    show_configuration("default", TRAINING_CONFIG)

    # 如果请求测试配置
    if args.test:
        if not test_configuration():
            return 1
        return 0

    # 运行训练
    return run_training()

if __name__ == "__main__":
    sys.exit(main())
