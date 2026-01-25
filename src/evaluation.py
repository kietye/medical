"""
评估模块
计算 IoU、准确率等指标
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    """边界框"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def area(self) -> int:
        return self.width * self.height


def calculate_iou(pred_box: Box, gt_box: Box) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)
    
    Args:
        pred_box: 预测框
        gt_box: 真实框 (Ground Truth)
        
    Returns:
        float: IoU 值 (0-1)
    """
    # 计算交集
    x1 = max(pred_box.x, gt_box.x)
    y1 = max(pred_box.y, gt_box.y)
    x2 = min(pred_box.x2, gt_box.x2)
    y2 = min(pred_box.y2, gt_box.y2)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    union = pred_box.area + gt_box.area - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    计算分类准确率
    
    Args:
        predictions: 预测标签列表
        ground_truth: 真实标签列表
        
    Returns:
        float: 准确率 (0-1)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("预测和真实标签数量不一致")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def calculate_region_match(pred_region_id: int, gt_region_ids: list[int]) -> bool:
    """
    检查预测的区域是否在真实区域列表中
    
    Args:
        pred_region_id: 预测的区域 ID
        gt_region_ids: 真实区域 ID 列表
        
    Returns:
        bool: 是否匹配
    """
    return pred_region_id in gt_region_ids


def alignment_score(explanations: list[str], expert_ratings: list[int]) -> dict:
    """
    计算对齐分数
    
    Args:
        explanations: AI 生成的解释文本列表
        expert_ratings: 专家评分列表 (1-5)
        
    Returns:
        dict: 包含平均分、最高分、最低分的字典
    """
    if not expert_ratings:
        return {"mean": 0, "max": 0, "min": 0}
    
    ratings = np.array(expert_ratings)
    return {
        "mean": float(np.mean(ratings)),
        "max": int(np.max(ratings)),
        "min": int(np.min(ratings)),
        "std": float(np.std(ratings))
    }


class EvaluationMetrics:
    """评估指标收集器"""
    
    def __init__(self):
        self.iou_scores = []
        self.region_matches = []
        self.predictions = []
        self.ground_truths = []
        self.expert_ratings = []
    
    def add_iou(self, pred_box: Box, gt_box: Box):
        """添加一个 IoU 样本"""
        iou = calculate_iou(pred_box, gt_box)
        self.iou_scores.append(iou)
        return iou
    
    def add_region_match(self, pred_id: int, gt_ids: list[int]):
        """添加一个区域匹配样本"""
        match = calculate_region_match(pred_id, gt_ids)
        self.region_matches.append(match)
        return match
    
    def add_classification(self, prediction: str, ground_truth: str):
        """添加一个分类样本"""
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)
    
    def add_rating(self, rating: int):
        """添加一个专家评分"""
        self.expert_ratings.append(rating)
    
    def summary(self) -> dict:
        """生成评估摘要"""
        summary = {}
        
        if self.iou_scores:
            summary["iou"] = {
                "mean": float(np.mean(self.iou_scores)),
                "std": float(np.std(self.iou_scores)),
                "max": float(np.max(self.iou_scores)),
                "min": float(np.min(self.iou_scores)),
                "count": len(self.iou_scores)
            }
        
        if self.region_matches:
            summary["region_match_rate"] = sum(self.region_matches) / len(self.region_matches)
        
        if self.predictions and self.ground_truths:
            summary["accuracy"] = calculate_accuracy(self.predictions, self.ground_truths)
        
        if self.expert_ratings:
            summary["alignment"] = alignment_score([], self.expert_ratings)
        
        return summary
