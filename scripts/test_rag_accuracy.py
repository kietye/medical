"""
RAG 检索准确率测试脚本
使用测试数据集评估 Visual RAG 的分类准确率

此脚本仅测试 RAG 向量检索的准确率（基于 CLIP 相似度 + 多数投票）
✅ 完全本地运行，不调用 LLM API，免费

如需测试完整 LLM 分析流程，请使用 test_pipeline_accuracy.py
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.visual_rag import VisualRAG, CATEGORY_NAMES, get_category_id_from_folder


def get_test_images(test_data_dir: Path) -> list[tuple[Path, int, str]]:
    """
    获取测试集中的所有图片及其真实类别
    
    Returns:
        list of (image_path, category_id, category_name)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    test_images = []
    
    for category_folder in sorted(test_data_dir.iterdir()):
        if not category_folder.is_dir():
            continue
        
        category_id = get_category_id_from_folder(category_folder.name)
        if category_id < 0:
            continue
        
        category_name = CATEGORY_NAMES.get(category_id, category_folder.name)
        
        for img_path in category_folder.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                test_images.append((img_path, category_id, category_name))
    
    return test_images


def compute_confusion_matrix(predictions: list[dict], num_classes: int = 8) -> list[list[int]]:
    """
    计算混淆矩阵
    
    Args:
        predictions: 预测结果列表
        num_classes: 类别数量
        
    Returns:
        混淆矩阵 (actual x predicted)
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    
    for pred in predictions:
        actual = pred["actual_category_id"]
        predicted = pred["predicted_category_id"]
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            matrix[actual][predicted] += 1
    
    return matrix


def run_accuracy_test(
    test_data_dir: Path = None,
    top_k: int = 5,
    output_path: Path = None,
    verbose: bool = False
) -> dict:
    """
    运行准确率测试
    
    Args:
        test_data_dir: 测试数据目录
        top_k: 预测时使用的 top-k 值
        output_path: 结果输出路径
        verbose: 是否输出详细信息
        
    Returns:
        dict: 测试结果
    """
    test_data_dir = Path(test_data_dir or config.TEST_DATA_DIR)
    
    if not test_data_dir.exists():
        print(f"错误: 测试数据目录不存在 {test_data_dir}")
        print("请先运行 split_test_data.py 拆分测试数据集")
        return {}
    
    print(f"测试数据目录: {test_data_dir}")
    print(f"Top-K: {top_k}")
    print("-" * 50)
    
    # 加载 RAG 索引
    print("加载 RAG 索引...")
    rag = VisualRAG()
    if not rag.load_index():
        print("错误: 无法加载 RAG 索引")
        print("请先运行 build_rag_index.py 构建索引")
        return {}
    
    # 获取测试图片
    test_images = get_test_images(test_data_dir)
    print(f"找到 {len(test_images)} 张测试图片")
    
    if not test_images:
        print("错误: 测试数据目录中没有图片")
        return {}
    
    # 按类别统计
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []
    
    # 测试每张图片
    print("\n开始测试...")
    for img_path, actual_id, actual_name in tqdm(test_images, desc="测试进度"):
        try:
            predicted_id, confidence, details = rag.predict_category(img_path, top_k=top_k)
            
            is_correct = (predicted_id == actual_id)
            
            # 记录结果
            predictions.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "actual_category_id": actual_id,
                "actual_category_name": actual_name,
                "predicted_category_id": predicted_id,
                "predicted_category_name": CATEGORY_NAMES.get(predicted_id, "未知"),
                "confidence": confidence,
                "is_correct": is_correct
            })
            
            # 更新统计
            category_stats[actual_id]["total"] += 1
            if is_correct:
                category_stats[actual_id]["correct"] += 1
            
            if verbose and not is_correct:
                print(f"  ❌ {img_path.name}: 预测={CATEGORY_NAMES.get(predicted_id, '未知')}, "
                      f"实际={actual_name}, 置信度={confidence:.2%}")
        
        except Exception as e:
            print(f"  错误: 处理 {img_path.name} 时出错: {e}")
            predictions.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "actual_category_id": actual_id,
                "actual_category_name": actual_name,
                "predicted_category_id": -1,
                "predicted_category_name": "错误",
                "confidence": 0.0,
                "is_correct": False,
                "error": str(e)
            })
            category_stats[actual_id]["total"] += 1
    
    # 计算总体准确率
    total_correct = sum(1 for p in predictions if p["is_correct"])
    total_samples = len(predictions)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 计算混淆矩阵
    confusion_matrix = compute_confusion_matrix(predictions)
    
    # 构建结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "test_data_dir": str(test_data_dir),
            "top_k": top_k
        },
        "overall": {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_samples
        },
        "per_category": {
            CATEGORY_NAMES.get(cat_id, f"类别{cat_id}"): {
                "category_id": cat_id,
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "total": stats["total"]
            }
            for cat_id, stats in sorted(category_stats.items())
        },
        "confusion_matrix": {
            "matrix": confusion_matrix,
            "labels": [CATEGORY_NAMES.get(i, f"类别{i}") for i in range(8)]
        },
        "predictions": predictions
    }
    
    # 输出结果
    print("\n" + "=" * 50)
    print("📊 测试结果")
    print("=" * 50)
    print(f"\n整体准确率: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    print("\n各类别准确率:")
    for cat_name, stats in results["per_category"].items():
        acc = stats["accuracy"]
        correct = stats["correct"]
        total = stats["total"]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cat_name:20s}: {bar} {acc:.1%} ({correct}/{total})")
    
    # 保存结果
    if output_path is None:
        output_path = config.OUTPUT_DIR / f"accuracy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_path}")
    
    return results


def print_confusion_matrix(matrix: list[list[int]], labels: list[str]):
    """打印混淆矩阵"""
    print("\n混淆矩阵 (行=实际, 列=预测):")
    
    # 简化标签
    short_labels = [l[:6] for l in labels]
    
    # 表头
    header = "        " + " ".join(f"{l:>6s}" for l in short_labels)
    print(header)
    print("-" * len(header))
    
    # 每行
    for i, row in enumerate(matrix):
        row_str = f"{short_labels[i]:6s} |" + " ".join(f"{v:>6d}" for v in row)
        print(row_str)


def main():
    parser = argparse.ArgumentParser(description="RAG 准确率测试")
    parser.add_argument("--test-data-dir", type=str, help="测试数据目录")
    parser.add_argument("--top-k", type=int, default=5, help="预测使用的 top-k 值 (默认: 5)")
    parser.add_argument("--output", type=str, help="结果输出路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--show-confusion-matrix", action="store_true", help="显示混淆矩阵")
    
    args = parser.parse_args()
    
    results = run_accuracy_test(
        test_data_dir=args.test_data_dir,
        top_k=args.top_k,
        output_path=args.output,
        verbose=args.verbose
    )
    
    if results and args.show_confusion_matrix:
        print_confusion_matrix(
            results["confusion_matrix"]["matrix"],
            results["confusion_matrix"]["labels"]
        )


if __name__ == "__main__":
    main()
