import subprocess
import sys
import os

def run_command(command, cwd=None):
    """执行命令并打印输出"""
    print(f"执行命令: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        print("标准输出:")
        print(result.stdout)
        if result.stderr:
            print("标准错误:")
            print(result.stderr)
        print(f"命令执行成功，返回码: {result.returncode}\n")
        return result
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}")
        print("标准输出:")
        print(e.stdout)
        print("标准错误:")
        print(e.stderr)
        print()
        return e

def run_command_in_venv(command):
    """在虚拟环境中执行命令"""
    # 直接使用虚拟环境中的python和pip
    venv_command = f"venv/bin/{command}"
    return run_command(venv_command)

# 创建虚拟环境
print("1. 创建虚拟环境...")
run_command("python3 -m venv venv")

# 检查虚拟环境中的Python版本
print("2. 检查虚拟环境中的Python版本...")
run_command_in_venv("python --version")

# 拉取GitHub仓库
print("3. 拉取GitHub仓库...")
run_command("git clone https://github.com/meta-recsys/generative-recommenders.git")

print("脚本执行完成")