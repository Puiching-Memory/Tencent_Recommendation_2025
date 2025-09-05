import os
import re
import sys
import time
import contextlib
from pathlib import Path
import torch
import torch.nn as nn
import math
from typing import Type
from loguru import logger
from adamw_bf16 import AdamWBF16
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.classification import BinaryAUROC

from data import SeqRecDataset
from transformer import ModelArgs, Transformer
from common import latest_checkpoint


class SpeedTracker(object):
    def __init__(self, buffer_size=10000):
        self.times = []
        self.buffer_size = buffer_size

    def update(self):
        self.times.append(time.time())
        if len(self.times) > 2 * self.buffer_size:
            self.times = self.times[-self.buffer_size:]

    def get_speed(self, steps=100):        
        if len(self.times) < 2:
            return 0
        steps = min(len(self.times)-1, steps)
        return steps / (self.times[-1]-self.times[-steps-1])

def collate_fn(batch):
    # filter None samples
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)

@torch.no_grad()
def eval(
        model:torch.nn.Module,
        summary_writer:SummaryWriter,
        data_path:str=None,
        batch_size:int=128,
        num_data_workers:int=2,
        prefetch_factor:int=32,
        log_steps:int=100,
        global_step:int=0,
        max_steps:int=200,
        drop_last:bool=False,
        split:str="val"
    ):
    if data_path is None:
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Use the same device as the model
    device = next(model.parameters()).device

    dataset = SeqRecDataset(data_path, split=split, train_ratio=0.8)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_data_workers, pin_memory=True, drop_last=drop_last, prefetch_factor=prefetch_factor, collate_fn=collate_fn)
    logger.info(f"eval dataset: {len(dataset)} eval dataloader: {len(dataloader)}")
    model.eval()
    rank_metric = BinaryAUROC()
    rank_loss_sum = 0
    steps = 0
    total_steps = min(max_steps, len(dataloader)) if max_steps > 0 else len(dataloader)
    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)
        ret = model(**batch)
        rank_predicts = ret["rank_outputs"]
        rank_loss = ret["rank_loss"]
        rank_labels = batch["click_label"]
        steps += 1
        if world_size > 1:
            rank_predicts_gather_list = [torch.zeros_like(rank_predicts) for _ in range(world_size)]
            rank_labels_gather_list = [torch.zeros_like(rank_labels) for _ in range(world_size)]
            dist.all_gather(rank_predicts_gather_list, rank_predicts)
            dist.all_gather(rank_labels_gather_list, rank_labels)
            rank_predicts = torch.cat(rank_predicts_gather_list, dim=0)
            rank_labels = torch.cat(rank_labels_gather_list, dim=0)
            dist.all_reduce(rank_loss, op=dist.ReduceOp.SUM)
            rank_loss /= world_size
        rank_loss_sum += rank_loss
        rank_mask = (rank_labels == 0) | (rank_labels == 1)
        rank_metric.update(rank_predicts[rank_mask], rank_labels[rank_mask])
        if local_rank == 0 and steps % log_steps == 0:
            rank_auc = rank_metric.compute()
            rank_loss = rank_loss_sum/steps
            logger.info(f"eval steps: {steps}/{total_steps}, rank_loss: {rank_loss:.4f} rank_auc: {rank_auc:.4f}")
        if max_steps > 0 and steps >= max_steps:
            break
    del dataloader
    rank_auc = rank_metric.compute()
    rank_loss = rank_loss_sum/steps
    if local_rank == 0:
        logger.info(f"eval metrics: global_step: {global_step} rank_loss: {rank_loss:.4f} rank_auc: {rank_auc:.4f}")
    if rank == 0:
        summary_writer.add_scalar("eval/rank_loss", rank_loss.to(torch.float32), global_step)
        summary_writer.add_scalar("eval/rank_auc", rank_auc.to(torch.float32), global_step)
    model.train()
    return rank_auc.to(torch.float32).item()


def train(
        epoch:int=None,
        batch_size:int=None,
        data_path:str=None,
        num_data_workers:int=None,
        prefetch_factor:int=None,
        ckpt_root_dir:str=None,
        ckpt_name:str=None,
        learning_rate:float=None,
        min_learning_rate:float=None,
        warmup_steps:int=None,
        adam_betas:tuple=(0.9, 0.98),
        weight_decay:float=None,
        max_grads_norm:float=None,
        shuffle:bool=None,
        restore:bool=None,
        use_bf16:bool=None,
        log_steps:int=None,
        model_cls:Type=Transformer,
        frozen_params:str=None,
        load_ckpt:str=None,
        load_params:str=".*",
        random_seed:int=None,
        eval_steps:int=None,
        grad_accumulate_steps:int=None
        ):
    # Read parameters from environment variables with defaults
    epoch = int(os.environ.get("TRAIN_EPOCHS", epoch or 5))
    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", batch_size or 128))
    data_path = os.environ.get("TRAIN_DATA_PATH") or data_path
    if data_path is None:
        raise ValueError("data_path must be provided either as argument or environment variable TRAIN_DATA_PATH")
    num_data_workers = int(os.environ.get("TRAIN_NUM_DATA_WORKERS", num_data_workers or 2))
    prefetch_factor = int(os.environ.get("TRAIN_PREFETCH_FACTOR", prefetch_factor or 32))
    ckpt_root_dir = os.environ.get("TRAIN_CKPT_PATH", "./checkpoints")
    learning_rate = float(os.environ.get("TRAIN_LEARNING_RATE", learning_rate or 1e-3))
    min_learning_rate = float(os.environ.get("TRAIN_MIN_LEARNING_RATE", min_learning_rate or 1e-5)) if os.environ.get("TRAIN_MIN_LEARNING_RATE") else None
    warmup_steps = int(os.environ.get("TRAIN_WARMUP_STEPS", warmup_steps or 50))
    weight_decay = float(os.environ.get("TRAIN_WEIGHT_DECAY", weight_decay or 0.01))
    max_grads_norm = float(os.environ.get("TRAIN_MAX_GRADS_NORM", max_grads_norm or 1.0))
    shuffle = os.environ.get("TRAIN_SHUFFLE", str(shuffle or True)).lower() == "true"
    restore = os.environ.get("TRAIN_RESTORE", str(restore or True)).lower() == "true"
    use_bf16 = os.environ.get("TRAIN_USE_BF16", str(use_bf16 or False)).lower() == "true"
    log_steps = int(os.environ.get("TRAIN_LOG_STEPS", log_steps or 5))
    random_seed = int(os.environ.get("TRAIN_RANDOM_SEED", random_seed or 17))
    eval_steps = int(os.environ.get("TRAIN_EVAL_STEPS", eval_steps or 50))
    grad_accumulate_steps = int(os.environ.get("TRAIN_GRAD_ACCUMULATE_STEPS", grad_accumulate_steps or 1))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    checkpoint_dir = os.path.join(ckpt_root_dir, ckpt_name)
    log_path = os.environ.get('TRAIN_LOG_PATH', checkpoint_dir)
    tf_events_path = os.environ.get('TRAIN_TF_EVENTS_PATH', checkpoint_dir)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    Path(tf_events_path).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(log_path, f'train-{rank}.log'), 'w')
    writer = SummaryWriter(tf_events_path)
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        summary_writer = writer
    else:
        summary_writer = None
    logger.add(log_file)
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')
    logger.info(f"WORLD_SIZE: {world_size} RANK: {rank} LOCAL_RANK: {local_rank}")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
    logger.info("checkpoint_dir: {}".format(checkpoint_dir))
    # init dist
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # build datasets
    train_dataset = SeqRecDataset(data_path, split="train", train_ratio=0.8)
    val_dataset = SeqRecDataset(data_path, split="val", train_ratio=0.8)
    if rank == 0:
        train_dataset.save_tokenizers(checkpoint_dir)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_data_workers, pin_memory=torch.cuda.is_available(), drop_last=True, prefetch_factor=prefetch_factor, collate_fn=collate_fn)
    logger.info(f"train dataset: {len(train_dataset)} val dataset: {len(val_dataset)} train dataloader: {len(dataloader)}")
    # build model args
    model_args = ModelArgs()
    model_args.item_vocab_size=train_dataset.item_tokenizer.get_vocab_size()
    model_args.cate_vocab_size=train_dataset.cate_tokenizer.get_vocab_size()
    # build model
    model_: nn.Module = model_cls(model_args, seed=random_seed)
    model_ = model_.to(device)
    # count params
    n_sparse_params = n_dense_params = 0
    for name, param in model_.named_parameters():
        if name.startswith("item_embeddings"):
            n_sparse_params += param.numel()
        else:
            n_dense_params += param.numel()
    logger.info(f"#Sparse: {n_sparse_params} #Dense: {n_dense_params}")

    if world_size > 1:
        model = DDP(model_, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
    else:
        model = model_

    if use_bf16:
        model_ = model_.to(torch.bfloat16)
        logger.info(f"Convert model to bfloat16.")

    # build optimizer
    opt_cls = AdamWBF16 if use_bf16 else torch.optim.AdamW
    opt = opt_cls(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=weight_decay,
    )

    # set frozen params
    if frozen_params:
        logger.info(f"frozen_param: {frozen_params}")
        pattern = re.compile(frozen_params)
        for name, param in model_.named_parameters():
            if pattern.match(name):
                param.requires_grad = False
                logger.info(f"Frozen param: {name}")

    # prepare training
    global_step = 0
    start_epoch = 0
    batch_index = 0
    epoch_steps = len(dataloader) // grad_accumulate_steps
    total_steps = epoch_steps * epoch
    rank_metric = BinaryAUROC()

    restored = False
    if restore:
        latest_ckpt = latest_checkpoint(checkpoint_dir)
        if latest_ckpt:
            # Handle BaseLineO1 format checkpoint (model.pt file)
            if latest_ckpt.endswith("model.pt"):
                ckpt = torch.load(latest_ckpt)
                # BaseLineO1 saves model state dict directly
                ret = model_.load_state_dict(ckpt, strict=False)
                logger.info(f"Loaded BaseLineO1 format checkpoint from {latest_ckpt}")
                logger.info(f"Missing keys: {ret.missing_keys}")
                logger.info(f"Unexpected keys: {ret.unexpected_keys}")
                # Note: Cannot restore optimizer state and training progress from BaseLineO1 format
                global_step = 0
                start_epoch = 0
            else:
                # Handle old GPSD format checkpoint
                ckpt = torch.load(open(latest_ckpt, 'rb'))
                model.load_state_dict(ckpt["model_state_dict"])
                opt.load_state_dict(ckpt["optimizer_state_dict"])
                global_step = ckpt["steps"]
                start_epoch = ckpt["epoch"]
                del ckpt
            restored = True

    if load_ckpt and not restored:
        if os.path.isdir(load_ckpt):
            load_ckpt = latest_checkpoint(load_ckpt)
        if load_ckpt.endswith("model.pt"):
            # Handle BaseLineO1 format checkpoint
            ckpt = torch.load(load_ckpt)
            ret = model_.load_state_dict(ckpt, strict=False)
            logger.info(f"Loaded BaseLineO1 format checkpoint from {load_ckpt}")
            logger.info(f"Missing keys: {ret.missing_keys}")
            logger.info(f"Unexpected keys: {ret.unexpected_keys}")
        else:
            # Handle old GPSD format checkpoint
            ckpt = torch.load(open(load_ckpt, 'rb'))
            ckpt["model_state_dict"] = {re.sub('^module.', '', k): v for k, v in ckpt["model_state_dict"].items()} # remove wrapper
            pattern = re.compile(load_params)
            for key in list(ckpt["model_state_dict"].keys()):
                if not pattern.match(key):
                    logger.info(f"Excluded key during loading: {key}.")
                    ckpt["model_state_dict"].pop(key)
            ret = model_.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Missing keys: {ret.missing_keys}")
            logger.info(f"unexpected keys: {ret.unexpected_keys}")
            logger.info(f"Load from {load_ckpt} successfully.")
            del ckpt

    # set learning rate
    def schedule_lr(global_step, total_steps):
        peak_lr, min_lr = learning_rate, min_learning_rate
        assert min_lr is None or min_lr < peak_lr
        if warmup_steps > 0 and global_step < warmup_steps:
            # apply warm up
            lr = peak_lr * (global_step+1) / warmup_steps
        else:
            if min_lr is None:
                lr = peak_lr
            else:
                # apply cosine annealing
                cur_steps = min(global_step + 1 - warmup_steps, total_steps)
                max_steps = total_steps - warmup_steps
                lr = min_lr + (peak_lr - min_lr) * (1 + torch.cos(torch.tensor(cur_steps/max_steps*math.pi))) / 2.0
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        return lr

    speed_tracker = SpeedTracker()
    for ep in range(start_epoch, epoch):
        model.train()
        for batch in dataloader:
            batch_index += 1
            is_accumlate_step = batch_index % grad_accumulate_steps != 0
            for k in batch:
                batch[k] = batch[k].to(device)
            with model.no_sync() if is_accumlate_step and world_size > 1 else contextlib.nullcontext():
                ret = model(**batch)
                loss = ret['loss']
                item_ar_loss = ret['item_ar_loss']
                cate_ar_loss = ret['cate_ar_loss']
                rank_loss = ret['rank_loss']
                loss.backward()
                if is_accumlate_step:
                    continue
            if grad_accumulate_steps > 1:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= grad_accumulate_steps
            if max_grads_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grads_norm)
            lr = schedule_lr(global_step, total_steps)
            opt.step()
            opt.zero_grad()
            global_step += 1
            speed_tracker.update()

            # update metrics
            rank_mask = (batch["click_label"] == 0) | (batch["click_label"] == 1)
            rank_metric.update(ret["rank_outputs"][rank_mask], batch["click_label"][rank_mask])

            if ((global_step-1) % log_steps == 0):
                if local_rank == 0:
                    rank_auc = rank_metric.compute() if torch.any(rank_mask) else torch.tensor(0.5)
                    logger.info(f"epoch: {ep+1}/{epoch} global_step: {global_step}/{total_steps}, loss: {loss:.4f}, item_ar_loss: {item_ar_loss:.4f}, cate_ar_loss: {cate_ar_loss:.4f}, rank_loss: {rank_loss:.4f}, rank_auc: {rank_auc:.4f}, speed: {speed_tracker.get_speed():.2f}steps/s, learning_rate: {lr:.6f}")
                if rank == 0:
                    rank_auc = rank_metric.compute() if torch.any(rank_mask) else torch.tensor(0.5)
                    summary_writer.add_scalar("train/loss", loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/item_ar_loss", item_ar_loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/cate_ar_loss", cate_ar_loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/rank_loss", rank_loss.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/rank_auc", rank_auc.to(torch.float32), global_step)
                    summary_writer.add_scalar("train/steps_per_second", speed_tracker.get_speed(), global_step)
                    summary_writer.add_scalar("train/learning_rate", lr, global_step)
                    for name, param in model_.named_parameters():
                        summary_writer.add_scalar(f"param_norm/{name}", param.norm().to(torch.float32), global_step)
                rank_metric.reset()
            
            # eval during epoch
            if eval_steps > 0 and global_step % eval_steps == 0:
                eval(model=model, summary_writer=summary_writer, data_path=data_path, global_step=global_step, batch_size=batch_size, split="val")
        # eval and save after epoch
        eval_auc = eval(model=model, summary_writer=summary_writer, data_path=data_path, global_step=global_step, batch_size=batch_size, split="val")
        if rank == 0:
            # Create checkpoint directory with BaseLineO1 format
            checkpoint_subdir = f"global_step{global_step}.valid_loss={eval_auc:.4f}"
            checkpoint_path = os.path.join(ckpt_root_dir, checkpoint_subdir)
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model state dict only (following BaseLineO1 format)
            torch.save(model_.state_dict(), os.path.join(checkpoint_path, "model.pt"))
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        batch_index = 0 # reset

    if rank == 0:
        summary_writer.close()

    # destory dist
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPSD Training Launcher", add_help=False)
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=0.00001, help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--max_grads_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--log_steps', type=int, default=10, help='Log steps')
    parser.add_argument('--eval_steps', type=int, default=100, help='Evaluation steps')
    parser.add_argument('--grad_accumulate_steps', type=int, default=2, help='Gradient accumulation steps')
    
    # Model parameters
    parser.add_argument('--model_dim', type=int, default=8, help='Model dimension')
    parser.add_argument('--model_n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--model_n_heads', type=int, default=1, help='Number of heads')
    parser.add_argument('--model_dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--model_max_seq_len', type=int, default=101, help='Max sequence length')
    
    # Training options
    parser.add_argument('--use_bf16', action='store_true', help='Use BF16')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Shuffle data')
    parser.add_argument('--restore', action='store_true', default=True, help='Restore from checkpoint')
    parser.add_argument('--random_seed', type=int, default=17, help='Random seed')
    
    # Data
    parser.add_argument('--num_data_workers', type=int, default=0, help='Number of data workers')
    parser.add_argument('--prefetch_factor', type=int, default=1, help='Prefetch factor')
    
    # Paths
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints", help='Checkpoint path')
    parser.add_argument('--log_path', type=str, default="./logs", help='Log path')
    parser.add_argument('--tf_events_path', type=str, default="./tf_events", help='TensorBoard events path')
    
    # Launcher options
    parser.add_argument("--test", action="store_true", help="Test configuration without training")
    parser.add_argument("--clean", action="store_true", help="Clean project directory")
    parser.add_argument("-h", "--help", action="store_true", help="Show help information")
    
    args = parser.parse_args()
    
    # Handle help
    if args.help:
        print("GPSD Training Launcher for TencentGR_1k Dataset")
        print("=" * 55)
        print()
        print("Quick usage:")
        print("  python train.py                    # Use default configuration")
        print("  python train.py --test            # Test configuration")
        print("  python train.py --clean           # Clean project directory")
        print()
        print("Examples:")
        print("  python train.py --test")
        print()
        parser.print_help()
        sys.exit(0)
    
    # Handle clean
    if args.clean:
        print("🧹 Cleaning project directory...")
        import shutil
        from pathlib import Path
        
        dirs_to_clean = [
            args.ckpt_path,
            args.log_path,
            args.tf_events_path,
            "__pycache__"
        ]
        
        for item in dirs_to_clean:
            path = Path(item)
            if path.is_dir():
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
            elif path.exists():
                print(f"Removing file: {path}")
                path.unlink()
        
        print("✅ Project directory cleaned!")
        sys.exit(0)
    
    # Handle test
    if args.test:
        print("🧪 Testing configuration...")
        
        # Check data file
        data_path = os.environ.get("TRAIN_DATA_PATH") or args.data_path
        if data_path is None:
            print("❌ Data path not provided. Use --data_path or set TRAIN_DATA_PATH environment variable.")
            sys.exit(1)
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            sys.exit(1)
        print(f"✅ Data file found: {data_path}")
        
        # Check Python
        try:
            import subprocess
            result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Python available: {result.stdout.strip()}")
            else:
                print("❌ Python not found or not working")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Python error: {e}")
            sys.exit(1)
        
        # Check required modules
        required_modules = ['torch', 'numpy', 'loguru']
        for module in required_modules:
            try:
                result = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True)
                if result.returncode == 0:
                    print(f"✅ Module '{module}' available")
                else:
                    print(f"❌ Module '{module}' not available")
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Module '{module}' error: {e}")
                sys.exit(1)
        
        print("🎉 Configuration test passed!")
        sys.exit(0)
        
    # Show configuration
    data_path = os.environ.get("TRAIN_DATA_PATH") or args.data_path
    if data_path is None:
        print("❌ Data path not provided. Use --data_path or set TRAIN_DATA_PATH environment variable.")
        sys.exit(1)
    log_path = os.environ.get("TRAIN_LOG_PATH") or args.log_path
    tf_events_path = os.environ.get("TRAIN_TF_EVENTS_PATH") or args.tf_events_path
    print("🚀 GPSD Training Launcher for TencentGR_1k Dataset")
    print("=" * 55)
    print()
    print("📊 Training Configuration")
    print("=" * 60)
    print(f"Data path          : {data_path}")
    print(f"Checkpoint path    : {args.ckpt_path}")
    print(f"Log path           : {log_path}")
    print(f"TensorBoard path   : {tf_events_path}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"Model dimension    : {args.model_dim}")
    print(f"Model layers       : {args.model_n_layers}")
    print(f"Attention heads    : {args.model_n_heads}")
    print(f"Dropout rate       : {args.model_dropout}")
    print()
    
    logger.remove()
    logger.add(sys.stdout, format="[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>] <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
    train(data_path=args.data_path, ckpt_name="tencentgr")