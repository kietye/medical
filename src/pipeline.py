"""
主流程模块
整合 SAM 分割 + LLM 分析
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.panel import Panel

from .config import config
from .sam_processor import SAMProcessor, Region
from .llm_client import LLMClient, AnalysisResult
from .prompts import BASIC_ANALYSIS_PROMPT, COT_ANALYSIS_PROMPT
from .visual_rag import VisualRAG

console = Console()


@dataclass
class NavigationResult:
    """导航结果"""
    image_path: str
    marked_image_path: str
    regions: list[dict]
    analysis: dict
    raw_response: str
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, output_path: str | Path):
        """保存结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class BiopsyNavigationPipeline:
    """活检导航主流程"""
    
    def __init__(
        self,
        use_sam: bool = None,
        use_rag: bool = False,
        use_cot: bool = False
    ):
        """
        初始化流程
        
        Args:
            use_sam: 是否使用 SAM 分割 (默认读取 config.SAM_ENABLED)
            use_rag: 是否使用 Visual RAG
            use_cot: 是否使用 Chain-of-Thought
        """
        # 如果未指定，从配置读取
        self.use_sam = use_sam if use_sam is not None else config.SAM_ENABLED
        self.use_rag = use_rag
        self.use_cot = use_cot
        
        self.sam_processor = SAMProcessor() if self.use_sam else None
        self.llm_client = LLMClient()
        self.visual_rag = None
        
        if use_rag:
            self.visual_rag = VisualRAG()
            self.visual_rag.load_index()
    
    def process(
        self, 
        image_path: str | Path,
        output_dir: str | Path = None
    ) -> NavigationResult:
        """
        处理单张图像
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            
        Returns:
            NavigationResult: 导航结果
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = image_path.stem
        
        console.print(Panel(f"[bold blue]处理图像: {image_path.name}[/bold blue]"))
        
        # Step 1: SAM 分割 (可选)
        regions = []
        marked_image_path = ""
        
        if self.use_sam:
            console.print("[yellow]► 步骤 1: SAM 分割...[/yellow]")
            marked_image_path = output_dir / f"{image_name}_marked_{timestamp}.png"
            _, regions_obj = self.sam_processor.process(image_path, marked_image_path)
            regions = [{"id": r.id, "area": r.area, "centroid": r.centroid} for r in regions_obj]
            console.print(f"  [green]✓ 检测到 {len(regions)} 个区域[/green]")
            analysis_image = marked_image_path
        else:
            analysis_image = image_path
        
        # Step 2: RAG 检索（可选）
        reference_cases = []
        if self.use_rag and self.visual_rag:
            console.print("[yellow]► 步骤 2: RAG 检索...[/yellow]")
            results = self.visual_rag.retrieve(image_path, top_k=3)
            reference_cases = self.visual_rag.format_for_prompt(results)
            console.print(f"  [green]✓ 检索到 {len(reference_cases)} 个相似病例[/green]")
        
        # Step 3: 构建 Prompt
        if self.use_cot:
            prompt = COT_ANALYSIS_PROMPT
        else:
            prompt = BASIC_ANALYSIS_PROMPT
        
        # 如果有参考病例，添加到 prompt
        if reference_cases:
            from .prompts import get_rag_prompt
            prompt = get_rag_prompt(reference_cases, use_cot=self.use_cot)
        
        # Step 4: LLM 分析
        console.print("[yellow]► 步骤 3: LLM 分析...[/yellow]")
        
        # 同时发送原图和标记图，让模型既能看到完整细节，又能参考区域编号
        if self.use_sam and marked_image_path:
            # 原图 + 标记图
            images_to_analyze = [str(image_path), str(marked_image_path)]
            console.print("  [dim]发送: 原图 + SAM标记图[/dim]")
        else:
            images_to_analyze = [str(image_path)]
        
        analysis_result = self.llm_client.analyze(
            images=images_to_analyze,
            prompt=prompt
        )
        console.print("  [green]✓ 分析完成[/green]")
        
        # 解析 JSON 结果
        analysis_dict = {}
        try:
            # 尝试从响应中提取 JSON
            raw = analysis_result.raw_response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                analysis_dict = json.loads(raw[json_start:json_end])
        except json.JSONDecodeError:
            console.print("  [yellow]⚠ JSON 解析失败，返回原始文本[/yellow]")
        
        # 构建结果
        result = NavigationResult(
            image_path=str(image_path),
            marked_image_path=str(marked_image_path),
            regions=regions,
            analysis=analysis_dict,
            raw_response=analysis_result.raw_response,
            timestamp=timestamp
        )
        
        # 保存结果
        result_path = output_dir / f"{image_name}_result_{timestamp}.json"
        result.save(result_path)
        console.print(f"  [green]✓ 结果已保存: {result_path}[/green]")
        
        # 打印摘要
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: NavigationResult):
        """打印结果摘要"""
        console.print("\n" + "=" * 50)
        console.print("[bold green]分析结果摘要[/bold green]")
        console.print("=" * 50)
        
        if result.analysis:
            # 形态描述
            if "morphology_description" in result.analysis:
                console.print(f"\n[bold]形态描述:[/bold]\n{result.analysis['morphology_description'][:200]}...")
            
            # 活检建议
            if "recommended_biopsy" in result.analysis:
                biopsy = result.analysis["recommended_biopsy"]
                console.print(f"\n[bold]活检建议:[/bold]")
                console.print(f"  主要位置: 区域 #{biopsy.get('primary_id', 'N/A')}")
                console.print(f"  置信度: {biopsy.get('confidence', 'N/A')}")
                console.print(f"  理由: {biopsy.get('reasoning', 'N/A')[:100]}...")
            
            # 鉴别诊断
            if "differential_diagnosis" in result.analysis:
                console.print(f"\n[bold]鉴别诊断:[/bold]")
                for diag in result.analysis["differential_diagnosis"][:3]:
                    if isinstance(diag, dict):
                        console.print(f"  - {diag.get('diagnosis', 'N/A')} ({diag.get('probability', 'N/A')})")
                    else:
                        console.print(f"  - {diag}")
        else:
            console.print("\n[yellow]未能解析结构化结果，请查看原始响应[/yellow]")
        
        console.print("\n" + "=" * 50)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="宫腔镜智能活检导航系统")
    parser.add_argument("--image", "-i", required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--no-sam", action="store_true", help="禁用 SAM 分割")
    parser.add_argument("--sam", action="store_true", help="强制启用 SAM 分割 (覆盖 .env 配置)")
    parser.add_argument("--rag", action="store_true", help="启用 Visual RAG")
    parser.add_argument("--cot", action="store_true", help="启用 Chain-of-Thought")
    
    args = parser.parse_args()
    
    # 处理 SAM 开关逻辑：命令行参数优先于配置文件
    if args.no_sam:
        use_sam = False
    elif args.sam:
        use_sam = True
    else:
        use_sam = None  # 使用 config.SAM_ENABLED
    
    pipeline = BiopsyNavigationPipeline(
        use_sam=use_sam,
        use_rag=args.rag,
        use_cot=args.cot
    )
    
    result = pipeline.process(args.image, args.output)
    
    return result


if __name__ == "__main__":
    main()
