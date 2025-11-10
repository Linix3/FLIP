#!/usr/bin/env python3
"""
花卉分类模型预测脚本（简化版 - 随机生成预测结果）

使用方法:
    python ./code/predict.py <测试集文件夹> <输出文件路径>

示例:
    python ./code/predict.py ./unified_flower_dataset/images/test ./results/submission.csv

输出格式:
    CSV文件包含三列: filename, category_id, confidence
    - filename: 测试图片文件名
    - category_id: 预测的类别ID (对应花卉类别编号)
    - confidence: 预测置信度 (0-1之间)
"""

import os
import argparse
import pandas as pd
import random
from pathlib import Path


# 所有可能的类别ID
CATEGORY_IDS = [
    # 164-245 范围
    164, 165, 166, 167, 169, 171, 172, 173, 174, 176, 177, 178, 179, 180,
    183, 184, 185, 186, 188, 189, 190, 192, 193, 194, 195, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
    243, 244, 245,
    # 1734-1833 范围
    1734, 1743, 1747, 1749, 1750, 1751, 1759, 1765, 1770, 1772, 1774, 1776,
    1777, 1780, 1784, 1785, 1786, 1789, 1796, 1797, 1801, 1805, 1806, 1808,
    1818, 1827, 1833
]


def get_image_files(img_dir, img_extensions=None):
    """获取目录中的所有图片文件"""
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    img_dir_path = Path(img_dir)
    image_files = []

    # 获取所有图片文件
    for ext in img_extensions:
        image_files.extend(img_dir_path.glob(f'*{ext}'))
        image_files.extend(img_dir_path.glob(f'*{ext.upper()}'))

    # 只保留文件名，并排序确保一致性
    image_files = sorted([f.name for f in image_files])

    return image_files


def generate_random_predictions(image_files, category_ids, min_confidence=0.5, max_confidence=0.99):
    """为每张图片生成随机预测结果"""
    predictions = []

    for filename in image_files:
        # 随机选择类别
        category_id = random.choice(category_ids)

        # 生成随机置信度（在指定范围内）
        confidence = random.uniform(min_confidence, max_confidence)

        predictions.append({
            'filename': filename,
            'category_id': category_id,
            'confidence': confidence
        })

    return predictions


def main():
    parser = argparse.ArgumentParser(description='花卉分类模型预测（随机生成）')

    # 位置参数：测试集文件夹和输出文件
    parser.add_argument('test_img_dir', type=str,
                        help='测试图片目录')
    parser.add_argument('output_path', type=str,
                        help='预测结果输出路径 (CSV文件)')

    # 可选参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（用于可重复性）')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                        help='最小置信度 (默认: 0.5)')
    parser.add_argument('--max_confidence', type=float, default=0.99,
                        help='最大置信度 (默认: 0.99)')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print(f'测试集目录: {args.test_img_dir}')
    print(f'输出文件: {args.output_path}')
    print(f'随机种子: {args.seed}')
    print(f'置信度范围: [{args.min_confidence}, {args.max_confidence}]')
    print()

    # 检查测试集目录是否存在
    if not os.path.exists(args.test_img_dir):
        print(f"错误: 测试集目录不存在: {args.test_img_dir}")
        return

    # 获取所有图片文件
    print("正在扫描测试集目录...")
    image_files = get_image_files(args.test_img_dir)

    if not image_files:
        print(f"错误: 在目录 {args.test_img_dir} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片")
    print()

    # 生成随机预测
    print("正在生成随机预测...")
    predictions = generate_random_predictions(
        image_files,
        CATEGORY_IDS,
        args.min_confidence,
        args.max_confidence
    )

    # 创建 DataFrame
    results_df = pd.DataFrame(predictions)

    # 按照文件名排序
    results_df = results_df.sort_values('filename').reset_index(drop=True)

    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    # 保存结果
    results_df.to_csv(args.output_path, index=False)
    print(f"预测结果已保存到: {args.output_path}")
    print()   

    print("预测完成!")


if __name__ == '__main__':
    main()
