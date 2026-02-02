"""
测试数据集拆分脚本
从 rag_data 中每个类别随机抽取 20% 的图片移动到 test_data 目录
"""

import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config


def get_category_folders(data_dir: Path) -> list[Path]:
    """获取所有类别文件夹"""
    folders = []
    for item in sorted(data_dir.iterdir()):
        if item.is_dir() and item.name[0].isdigit():
            folders.append(item)
    return folders


def get_image_files(folder: Path) -> list[Path]:
    """获取文件夹中的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    images = []
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in image_extensions:
            images.append(item)
    return sorted(images)


def split_test_data(
    rag_data_dir: Path = None,
    test_data_dir: Path = None,
    split_ratio: float = None,
    random_seed: int = None,
    dry_run: bool = False
) -> dict:
    """
    拆分测试数据集
    
    Args:
        rag_data_dir: RAG 数据目录
        test_data_dir: 测试数据目录
        split_ratio: 测试集比例
        random_seed: 随机种子
        dry_run: 是否只预览不执行
        
    Returns:
        dict: 拆分结果信息
    """
    # 使用配置默认值
    rag_data_dir = Path(rag_data_dir or config.RAG_DATA_DIR)
    test_data_dir = Path(test_data_dir or config.TEST_DATA_DIR)
    split_ratio = split_ratio or config.TEST_SPLIT_RATIO
    random_seed = random_seed if random_seed is not None else config.RANDOM_SEED
    
    # 设置随机种子
    random.seed(random_seed)
    
    print(f"RAG 数据目录: {rag_data_dir}")
    print(f"测试数据目录: {test_data_dir}")
    print(f"测试集比例: {split_ratio * 100:.0f}%")
    print(f"随机种子: {random_seed}")
    print(f"模式: {'预览' if dry_run else '执行'}")
    print("-" * 50)
    
    # 获取所有类别文件夹
    category_folders = get_category_folders(rag_data_dir)
    
    if not category_folders:
        print(f"错误: 在 {rag_data_dir} 中未找到类别文件夹")
        return {}
    
    # 统计信息
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "rag_data_dir": str(rag_data_dir),
            "test_data_dir": str(test_data_dir),
            "split_ratio": split_ratio,
            "random_seed": random_seed
        },
        "categories": {},
        "total_train": 0,
        "total_test": 0
    }
    
    # 处理每个类别
    for folder in category_folders:
        category_name = folder.name
        images = get_image_files(folder)
        total_count = len(images)
        
        if total_count == 0:
            print(f"警告: {category_name} 中没有图片文件")
            continue
        
        # 计算测试集数量（至少 1 张）
        test_count = max(1, int(total_count * split_ratio))
        train_count = total_count - test_count
        
        # 随机选择测试集图片
        test_images = random.sample(images, test_count)
        
        print(f"{category_name}:")
        print(f"  总数: {total_count}, 训练集: {train_count}, 测试集: {test_count}")
        
        # 创建测试数据目录
        test_category_dir = test_data_dir / category_name
        
        if not dry_run:
            test_category_dir.mkdir(parents=True, exist_ok=True)
            
            # 移动测试集图片
            for img in test_images:
                dest = test_category_dir / img.name
                shutil.move(str(img), str(dest))
        
        # 记录结果
        results["categories"][category_name] = {
            "original_count": total_count,
            "train_count": train_count,
            "test_count": test_count,
            "test_files": [img.name for img in test_images]
        }
        results["total_train"] += train_count
        results["total_test"] += test_count
    
    print("-" * 50)
    print(f"总计: 训练集 {results['total_train']} 张, 测试集 {results['total_test']} 张")
    
    # 保存清单文件
    if not dry_run:
        manifest_path = test_data_dir / "test_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n测试清单已保存到: {manifest_path}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="拆分测试数据集")
    parser.add_argument("--rag-data-dir", type=str, help="RAG 数据目录")
    parser.add_argument("--test-data-dir", type=str, help="测试数据目录")
    parser.add_argument("--split-ratio", type=float, default=0.2, help="测试集比例 (默认: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不执行移动操作")
    
    args = parser.parse_args()
    
    results = split_test_data(
        rag_data_dir=args.rag_data_dir,
        test_data_dir=args.test_data_dir,
        split_ratio=args.split_ratio,
        random_seed=args.seed,
        dry_run=args.dry_run
    )
    
    if results and not args.dry_run:
        print("\n✅ 测试数据集拆分完成!")
        print("\n⚠️  提醒: 请运行以下命令重新构建 RAG 索引:")
        print("   python scripts/build_rag_index.py")


if __name__ == "__main__":
    main()
