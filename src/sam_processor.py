"""
SAM (Segment Anything Model) 处理模块
实现 Set-of-Mark (SoM) 标记集提示法
"""

import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
from dataclasses import dataclass

from .config import config


@dataclass
class Region:
    """分割区域"""
    id: int
    mask: np.ndarray
    area: int
    bbox: tuple  # (x1, y1, x2, y2)
    centroid: tuple  # (x, y)
    score: float


class SAMProcessor:
    """SAM 分割处理器"""
    
    def __init__(self, model_type: str = None, checkpoint_path: str | Path = None):
        """
        初始化 SAM 处理器
        
        Args:
            model_type: 模型类型 (vit_h, vit_l, vit_b)
            checkpoint_path: 模型权重路径
        """
        self.model_type = model_type or config.SAM_MODEL_TYPE
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else config.sam_checkpoint_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = None
        self.mask_generator = None
        
    def load_model(self):
        """加载 SAM 模型"""
        if self.sam is not None:
            return
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM 模型权重不存在: {self.checkpoint_path}\n"
                f"请运行 python scripts/download_sam.py 下载模型"
            )
        
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        print(f"加载 SAM 模型: {self.model_type} on {self.device}")
        self.sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        self.sam.to(device=self.device)
        
        # 创建自动掩码生成器
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,  # 过滤太小的区域
        )
        print("SAM 模型加载完成")
    
    def segment(self, image_path: str | Path) -> list[Region]:
        """
        对图像进行分割
        
        Args:
            image_path: 图像路径
            
        Returns:
            list[Region]: 分割区域列表
        """
        self.load_model()
        
        # 读取图像
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # 生成掩码
        masks = self.mask_generator.generate(image)
        
        # 按面积排序，取前 20 个最大的区域
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:20]
        
        regions = []
        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]
            bbox = mask_data["bbox"]  # (x, y, w, h)
            
            # 计算质心
            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0:
                centroid = (int(np.mean(x_indices)), int(np.mean(y_indices)))
            else:
                centroid = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
            
            regions.append(Region(
                id=i + 1,
                mask=mask,
                area=mask_data["area"],
                bbox=(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                centroid=centroid,
                score=mask_data.get("predicted_iou", 0.0)
            ))
        
        return regions
    
    def draw_marks(
        self, 
        image_path: str | Path, 
        regions: list[Region],
        output_path: str | Path = None,
        show_masks: bool = True,
        alpha: float = 0.3
    ) -> Image.Image:
        """
        在图像上绘制标记
        
        Args:
            image_path: 原始图像路径
            regions: 分割区域列表
            output_path: 输出路径（可选）
            show_masks: 是否显示掩码覆盖
            alpha: 掩码透明度
            
        Returns:
            Image.Image: 标记后的图像
        """
        # 读取原始图像
        image = Image.open(image_path).convert("RGBA")
        
        # 创建掩码叠加层
        if show_masks:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # 颜色列表
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
                (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 255),
            ]
            
            for region in regions:
                color = colors[(region.id - 1) % len(colors)]
                mask_img = Image.fromarray((region.mask * 255).astype(np.uint8))
                colored_mask = Image.new("RGBA", image.size, (*color, int(255 * alpha)))
                overlay.paste(colored_mask, mask=mask_img)
            
            image = Image.alpha_composite(image, overlay)
        
        # 转换回 RGB 用于绘制文字
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体，失败则使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # 绘制标记编号
        for region in regions:
            cx, cy = region.centroid
            text = str(region.id)
            
            # 绘制白色背景圆圈
            radius = 18
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(255, 255, 255),
                outline=(0, 0, 0),
                width=2
            )
            
            # 绘制编号文字
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (cx - text_width // 2, cy - text_height // 2 - 2),
                text,
                fill=(0, 0, 0),
                font=font
            )
        
        # 保存输出
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
        
        return image
    
    def process(self, image_path: str | Path, output_path: str | Path = None) -> tuple[Image.Image, list[Region]]:
        """
        完整处理流程：分割 + 标记
        
        Args:
            image_path: 输入图像路径
            output_path: 输出路径（可选）
            
        Returns:
            tuple: (标记后的图像, 区域列表)
        """
        regions = self.segment(image_path)
        marked_image = self.draw_marks(image_path, regions, output_path)
        return marked_image, regions


# 便捷函数
def get_processor() -> SAMProcessor:
    """获取默认 SAM 处理器"""
    return SAMProcessor()
