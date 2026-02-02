#!/usr/bin/env python
"""
构建 RAG 索引脚本
从 data/rag_data 目录中的分类文件夹构建视觉检索索引
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.visual_rag import VisualRAG
from src.config import config


def main():
    parser = argparse.ArgumentParser(
        description="构建 Visual RAG 索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/build_rag_index.py                    # 使用默认目录
  python scripts/build_rag_index.py --data-dir ./data/rag_data
  python scripts/build_rag_index.py --batch-size 16   # 减小批量大小以节省显存
        """
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        help=f"数据目录路径 (默认: {config.RAG_DATA_DIR})"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help=f"索引输出目录 (默认: {config.EMBEDDINGS_DIR / 'rag_index'})"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="批量编码大小 (默认: 32, 显存不足时可减小)"
    )
    
    args = parser.parse_args()
    
    # 确定数据目录
    data_dir = Path(args.data_dir) if args.data_dir else config.RAG_DATA_DIR
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Visual RAG 索引构建工具")
    print("=" * 60)
    print(f"数据目录: {data_dir}")
    print(f"批量大小: {args.batch_size}")
    print()
    
    # 创建 RAG 实例
    if args.output_dir:
        rag = VisualRAG(index_path=args.output_dir)
    else:
        rag = VisualRAG()
    
    # 构建索引
    rag.build_index_from_folders(data_dir, batch_size=args.batch_size)
    
    # 显示统计信息
    print("\n" + "=" * 60)
    print("索引构建完成!")
    print("=" * 60)
    print(f"总病例数: {len(rag.cases)}")
    print("\n类别分布:")
    distribution = rag.get_category_distribution()
    for cat_name, count in sorted(distribution.items()):
        print(f"  - {cat_name}: {count} 张")
    
    print(f"\n索引保存位置: {rag.index_path}")


if __name__ == "__main__":
    main()
