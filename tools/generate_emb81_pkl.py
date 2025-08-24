import json
import pickle
import numpy as np
from pathlib import Path
import os

def generate_emb81_pkl():
    """
    生成emb_81_32.pkl文件，该文件是特征81的嵌入向量字典
    """
    # 定义路径
    data_dir = Path(__file__).parent / ".." / "data" / "TencentGR_1k" / "creative_emb"
    emb_81_dir = data_dir / "emb_81_32"
    output_file = data_dir / "emb_81_32.pkl"
    
    print(f"数据目录: {data_dir}")
    print(f"特征81目录: {emb_81_dir}")
    print(f"输出文件: {output_file}")
    
    # 检查输入目录是否存在
    if not emb_81_dir.exists():
        print(f"错误: 目录 {emb_81_dir} 不存在")
        return False
        
    # 收集所有part文件
    part_files = list(emb_81_dir.glob("part-*"))
    if not part_files:
        print(f"错误: 在 {emb_81_dir} 中未找到part文件")
        return False
        
    print(f"找到 {len(part_files)} 个part文件")
    
    # 创建嵌入向量字典
    emb_dict = {}
    
    # 逐个处理part文件
    for i, part_file in enumerate(part_files):
        print(f"处理文件 {i+1}/{len(part_files)}: {part_file.name}")
        try:
            with open(part_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        # 提取匿名创意ID和嵌入向量
                        anonymous_cid = data['anonymous_cid']
                        emb = data['emb']
                        
                        # 确保嵌入向量是numpy数组格式
                        if isinstance(emb, list):
                            emb = np.array(emb, dtype=np.float32)
                            
                        # 添加到字典中
                        emb_dict[anonymous_cid] = emb
                    except json.JSONDecodeError as e:
                        print(f"  警告: 在 {part_file.name} 的第 {line_num+1} 行JSON解析错误: {e}")
                    except KeyError as e:
                        print(f"  警告: 在 {part_file.name} 的第 {line_num+1} 行缺少键: {e}")
        except Exception as e:
            print(f"处理文件 {part_file.name} 时出错: {e}")
    
    print(f"总共加载了 {len(emb_dict)} 个嵌入向量")
    
    # 保存为pkl文件
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(emb_dict, f)
        print(f"成功生成 {output_file}")
        return True
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return False

if __name__ == "__main__":
    generate_emb81_pkl()