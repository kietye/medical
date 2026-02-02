#!/usr/bin/env python
"""
RAG 查询测试脚本
测试 Visual RAG 检索功能
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.visual_rag import get_rag, CATEGORY_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="测试 Visual RAG 检索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/test_rag_query.py --image ./test.jpg
  python scripts/test_rag_query.py --image ./test.jpg --top-k 10
  python scripts/test_rag_query.py --image ./test.jpg --category 1  # 只检索子宫内膜癌类别

类别列表:
  0: 粘膜下子宫肌瘤
  1: 子宫内膜癌
  2: 子宫内膜息肉
  3: 子宫内膜息肉样增生
  4: 子宫内膜增生不伴不典型增生
  5: 宫内异物
  6: 子宫颈息肉
  7: 子宫内膜不典型增生
        """
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="查询图像路径"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="返回的结果数量 (默认: 5)"
    )
    
    parser.add_argument(
        "--category", "-c",
        type=int,
        default=None,
        help="筛选特定类别 (0-7)"
    )
    
    parser.add_argument(
        "--show-stats", "-s",
        action="store_true",
        help="显示索引统计信息"
    )
    
    args = parser.parse_args()
    
    # 检查图像文件
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"错误: 图像文件不存在: {image_path}")
        sys.exit(1)
    
    # 加载 RAG
    print("加载 RAG 索引...")
    rag = get_rag()
    
    if len(rag.cases) == 0:
        print("错误: RAG 索引为空，请先运行 build_rag_index.py 构建索引")
        sys.exit(1)
    
    # 显示统计信息
    if args.show_stats:
        print("\n索引统计信息:")
        print(f"  总病例数: {len(rag.cases)}")
        distribution = rag.get_category_distribution()
        for cat_name, count in sorted(distribution.items()):
            print(f"  - {cat_name}: {count} 张")
        print()
    
    # 执行检索
    print(f"\n查询图像: {image_path}")
    if args.category is not None:
        cat_name = CATEGORY_NAMES.get(args.category, f"类别{args.category}")
        print(f"筛选类别: {cat_name}")
    print(f"返回前 {args.top_k} 个结果...")
    print()
    
    results = rag.retrieve(image_path, top_k=args.top_k, filter_category=args.category)
    
    if not results:
        print("未找到匹配结果")
        return
    
    # 显示结果
    print("=" * 70)
    print("检索结果")
    print("=" * 70)
    
    formatted = rag.format_for_prompt(results)
    for item in formatted:
        print(f"\n排名 #{item['rank']}")
        print(f"  诊断: {item['diagnosis']}")
        print(f"  类别ID: {item['category_id']}")
        print(f"  相似度: {item['similarity']}")
        print(f"  病理结果: {item['pathology_result']}")
    
    # 显示诊断分布
    print("\n" + "-" * 70)
    diagnosis_counts = {}
    for result in results:
        diag = result.case.diagnosis
        diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1
    
    print("检索结果诊断分布:")
    for diag, count in sorted(diagnosis_counts.items(), key=lambda x: -x[1]):
        print(f"  - {diag}: {count} ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
