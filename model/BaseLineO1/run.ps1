# 获取脚本所在目录作为工作目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# 显示脚本目录
Write-Host "Working directory: $ScriptDir"

# 设置必要的环境变量
$env:TRAIN_LOG_PATH = Join-Path $ScriptDir "logs"
$env:TRAIN_TF_EVENTS_PATH = Join-Path $ScriptDir "tf_events"
$env:TRAIN_DATA_PATH = Join-Path $ScriptDir "..\..\data\TencentGR_1k"
$env:TRAIN_CKPT_PATH = Join-Path $ScriptDir "checkpoints"

# 显示环境变量设置
Write-Host "TRAIN_LOG_PATH: $env:TRAIN_LOG_PATH"
Write-Host "TRAIN_TF_EVENTS_PATH: $env:TRAIN_TF_EVENTS_PATH"
Write-Host "TRAIN_DATA_PATH: $env:TRAIN_DATA_PATH"
Write-Host "TRAIN_CKPT_PATH: $env:TRAIN_CKPT_PATH"

# 进入脚本所在目录
Set-Location $ScriptDir

# 运行python脚本
python -u main.py --use_amp --use_torch_compile --use_cudnn_benchmark --use_tf32 --batch_size 2