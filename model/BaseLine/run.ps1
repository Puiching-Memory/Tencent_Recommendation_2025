# 设置必要的环境变量
$env:TRAIN_LOG_PATH = "logs"
$env:TRAIN_TF_EVENTS_PATH = "tf_events"
$env:TRAIN_DATA_PATH = "data/TencentGR_1k"
$env:TRAIN_CKPT_PATH = "checkpoints"

# 创建必要的目录
New-Item -ItemType Directory -Path $env:TRAIN_LOG_PATH -Force | Out-Null
New-Item -ItemType Directory -Path $env:TRAIN_TF_EVENTS_PATH -Force | Out-Null
New-Item -ItemType Directory -Path $env:TRAIN_DATA_PATH -Force | Out-Null
New-Item -ItemType Directory -Path $env:TRAIN_CKPT_PATH -Force | Out-Null

# 运行主程序
python -u ./model/BaseLine/main.py --batch_size 16 --mm_emb_id 81 82 83 84 85 86