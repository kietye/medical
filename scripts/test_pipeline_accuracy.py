"""
全链路准确率测试脚本
使用完整的 Pipeline（LLM 分析）评估分类准确率

⚠️ 注意：此脚本会调用 LLM API，会产生费用！
预估费用：约 ¥10-30 元（取决于测试样本数和模型）
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.pipeline import BiopsyNavigationPipeline
from src.visual_rag import CATEGORY_NAMES, get_category_id_from_folder


# 类别名称与关键词映射（用于从 LLM 响应中提取类别）
CATEGORY_KEYWORDS = {
    0: ["粘膜下子宫肌瘤", "子宫肌瘤", "肌瘤", "平滑肌瘤"],
    1: ["子宫内膜癌", "内膜癌", "癌", "恶性"],
    2: ["子宫内膜息肉", "内膜息肉"],
    3: ["息肉样增生", "息肉样", "增生性息肉"],
    4: ["增生不伴不典型", "子宫内膜增生不伴不典型增生", "单纯性增生", "复杂性增生不伴"],
    5: ["宫内异物", "异物", "IUD", "避孕环", "残留"],
    6: ["子宫颈息肉", "宫颈息肉", "颈管息肉"],
    7: ["不典型增生", "子宫内膜不典型增生", "不典型", "非典型增生"],
}


def extract_category_from_llm_response(raw_response: str, analysis_dict: dict) -> tuple[int, str]:
    """
    从 LLM 响应中提取预测的类别
    
    Args:
        raw_response: LLM 原始响应文本
        analysis_dict: 解析后的 JSON 分析结果
        
    Returns:
        tuple: (预测类别ID, 预测类别名称)
    """
    # 首先尝试从 differential_diagnosis 中提取最可能的诊断
    if analysis_dict and "differential_diagnosis" in analysis_dict:
        diagnoses = analysis_dict["differential_diagnosis"]
        if diagnoses:
            # 取第一个（最可能的）诊断
            first_diag = diagnoses[0]
            if isinstance(first_diag, dict):
                diag_name = first_diag.get("diagnosis", "")
            else:
                diag_name = str(first_diag)
            
            # 匹配类别
            for cat_id, keywords in CATEGORY_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in diag_name:
                        return cat_id, CATEGORY_NAMES.get(cat_id, diag_name)
    
    # 如果解析失败，尝试从原始响应中匹配关键词
    # 按照优先级顺序检查（更具体的关键词优先）
    priority_order = [7, 4, 3, 2, 1, 6, 5, 0]  # 不典型增生优先于普通增生
    
    for cat_id in priority_order:
        keywords = CATEGORY_KEYWORDS[cat_id]
        for keyword in keywords:
            if keyword in raw_response:
                return cat_id, CATEGORY_NAMES.get(cat_id, keyword)
    
    # 无法识别
    return -1, "未知"


def get_test_images(test_data_dir: Path, limit: int = None) -> list[tuple[Path, int, str]]:
    """
    获取测试集中的图片
    
    Args:
        test_data_dir: 测试数据目录
        limit: 限制每个类别的测试图片数量（用于快速测试）
        
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
        
        count = 0
        for img_path in sorted(category_folder.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                test_images.append((img_path, category_id, category_name))
                count += 1
                if limit and count >= limit:
                    break
    
    return test_images


def run_pipeline_accuracy_test(
    test_data_dir: Path = None,
    output_path: Path = None,
    use_sam: bool = False,
    use_rag: bool = True,
    use_cot: bool = False,
    limit_per_category: int = None,
    verbose: bool = False
) -> dict:
    """
    运行全链路准确率测试
    
    Args:
        test_data_dir: 测试数据目录
        output_path: 结果输出路径
        use_sam: 是否使用 SAM 分割
        use_rag: 是否使用 RAG 检索
        use_cot: 是否使用 Chain-of-Thought
        limit_per_category: 限制每个类别的测试数量（用于快速测试）
        verbose: 是否输出详细信息
        
    Returns:
        dict: 测试结果
    """
    test_data_dir = Path(test_data_dir or config.TEST_DATA_DIR)
    
    if not test_data_dir.exists():
        print(f"错误: 测试数据目录不存在 {test_data_dir}")
        print("请先运行 split_test_data.py 拆分测试数据集")
        return {}
    
    print("=" * 60)
    print("🔬 全链路准确率测试 (Pipeline + LLM)")
    print("=" * 60)
    print(f"测试数据目录: {test_data_dir}")
    print(f"SAM 分割: {'启用' if use_sam else '禁用'}")
    print(f"RAG 检索: {'启用' if use_rag else '禁用'}")
    print(f"CoT 模式: {'启用' if use_cot else '禁用'}")
    if limit_per_category:
        print(f"每类别限制: {limit_per_category} 张")
    print("-" * 60)
    
    # 初始化 Pipeline
    print("\n初始化 Pipeline...")
    pipeline = BiopsyNavigationPipeline(
        use_sam=use_sam,
        use_rag=use_rag,
        use_cot=use_cot
    )
    
    # 获取测试图片
    test_images = get_test_images(test_data_dir, limit=limit_per_category)
    print(f"测试图片数量: {len(test_images)}")
    
    if not test_images:
        print("错误: 测试数据目录中没有图片")
        return {}
    
    # 预估费用
    estimated_cost = len(test_images) * 0.02  # 假设每张图约 ¥0.02
    print(f"\n⚠️  预估 API 费用: ¥{estimated_cost:.2f}")
    
    # 按类别统计
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []
    errors = []
    
    # 创建临时输出目录（避免保存大量文件）
    temp_output_dir = config.OUTPUT_DIR / "pipeline_test_temp"
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试每张图片
    print("\n开始测试...")
    for img_path, actual_id, actual_name in tqdm(test_images, desc="测试进度"):
        try:
            # 调用 Pipeline 进行分析
            result = pipeline.process(img_path, output_dir=temp_output_dir)
            
            # 从 LLM 响应中提取预测类别
            predicted_id, predicted_name = extract_category_from_llm_response(
                result.raw_response,
                result.analysis
            )
            
            is_correct = (predicted_id == actual_id)
            
            # 记录结果
            predictions.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "actual_category_id": actual_id,
                "actual_category_name": actual_name,
                "predicted_category_id": predicted_id,
                "predicted_category_name": predicted_name,
                "is_correct": is_correct,
                "llm_diagnosis": result.analysis.get("differential_diagnosis", [])[:3] if result.analysis else [],
                "raw_response_preview": result.raw_response[:500] if result.raw_response else ""
            })
            
            # 更新统计
            category_stats[actual_id]["total"] += 1
            if is_correct:
                category_stats[actual_id]["correct"] += 1
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"  {status} {img_path.name}: 实际={actual_name}, 预测={predicted_name}")
        
        except Exception as e:
            print(f"  ❌ 错误: 处理 {img_path.name} 时出错: {e}")
            errors.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "error": str(e)
            })
            category_stats[actual_id]["total"] += 1
    
    # 计算总体准确率
    total_correct = sum(1 for p in predictions if p["is_correct"])
    total_samples = len(predictions)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # 构建结果
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "test_data_dir": str(test_data_dir),
            "use_sam": use_sam,
            "use_rag": use_rag,
            "use_cot": use_cot,
            "limit_per_category": limit_per_category
        },
        "overall": {
            "accuracy": overall_accuracy,
            "correct": total_correct,
            "total": total_samples,
            "errors": len(errors)
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
        "predictions": predictions,
        "errors": errors
    }
    
    # 输出结果
    print("\n" + "=" * 60)
    print("📊 测试结果")
    print("=" * 60)
    print(f"\n整体准确率: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    if errors:
        print(f"错误数量: {len(errors)}")
    print("\n各类别准确率:")
    for cat_name, stats in results["per_category"].items():
        acc = stats["accuracy"]
        correct = stats["correct"]
        total = stats["total"]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {cat_name:20s}: {bar} {acc:.1%} ({correct}/{total})")
    
    # 保存结果
    if output_path is None:
        output_path = config.OUTPUT_DIR / f"pipeline_accuracy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_path}")
    
    # 清理临时文件（可选）
    # import shutil
    # shutil.rmtree(temp_output_dir, ignore_errors=True)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="全链路准确率测试 (Pipeline + LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（每类别 2 张）
  python scripts/test_pipeline_accuracy.py --limit 2
  
  # 完整测试
  python scripts/test_pipeline_accuracy.py
  
  # 启用 RAG + CoT
  python scripts/test_pipeline_accuracy.py --rag --cot

⚠️  注意：此脚本会调用 LLM API，会产生费用！
"""
    )
    parser.add_argument("--test-data-dir", type=str, help="测试数据目录")
    parser.add_argument("--output", type=str, help="结果输出路径")
    parser.add_argument("--sam", action="store_true", help="启用 SAM 分割")
    parser.add_argument("--no-rag", action="store_true", help="禁用 RAG 检索")
    parser.add_argument("--rag", action="store_true", default=True, help="启用 RAG 检索 (默认)")
    parser.add_argument("--cot", action="store_true", help="启用 Chain-of-Thought")
    parser.add_argument("--limit", type=int, help="每个类别的测试图片数量限制（用于快速测试）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    
    args = parser.parse_args()
    
    # 确定 RAG 设置
    use_rag = not args.no_rag
    
    results = run_pipeline_accuracy_test(
        test_data_dir=args.test_data_dir,
        output_path=args.output,
        use_sam=args.sam,
        use_rag=use_rag,
        use_cot=args.cot,
        limit_per_category=args.limit,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
