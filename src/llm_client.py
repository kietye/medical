"""
LLM 客户端模块
支持 Qwen API (OpenAI 兼容格式) 和 OpenAI API
"""

import base64
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI

from .config import config


@dataclass
class AnalysisResult:
    """分析结果"""
    raw_response: str
    morphology_description: str = ""
    suspicious_regions: list = None
    recommended_biopsy: dict = None
    differential_diagnosis: list = None
    
    def __post_init__(self):
        if self.suspicious_regions is None:
            self.suspicious_regions = []
        if self.differential_diagnosis is None:
            self.differential_diagnosis = []


class LLMClient:
    """大语言模型客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        初始化 LLM 客户端
        
        Args:
            api_key: API Key，默认从配置读取
            base_url: API Base URL，默认从配置读取
            model: 模型名称，默认从配置读取
        """
        self.api_key = api_key or config.get_api_key()
        self.base_url = base_url or config.get_base_url()
        self.model = model or config.DEFAULT_MODEL
        
        if not self.api_key:
            raise ValueError("未配置 API Key，请在 .env 文件中设置 QWEN_API_KEY 或 OPENAI_API_KEY")
        
        # 初始化 OpenAI 客户端 (兼容 Qwen)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _encode_image(self, image_path: str | Path) -> str:
        """将图像编码为 base64"""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_media_type(self, image_path: str | Path) -> str:
        """获取图像 MIME 类型"""
        suffix = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/jpeg")
    
    def analyze(
        self, 
        images: list[str | Path], 
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> AnalysisResult:
        """
        分析图像
        
        Args:
            images: 图像路径列表
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            
        Returns:
            AnalysisResult: 分析结果
        """
        # 构建消息内容
        content = []
        
        # 添加图像
        for image_path in images:
            image_data = self._encode_image(image_path)
            media_type = self._get_image_media_type(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }
            })
        
        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        
        # 调用 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 提取响应
        raw_response = response.choices[0].message.content
        
        return AnalysisResult(raw_response=raw_response)
    
    def test_connection(self) -> bool:
        """测试 API 连接"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello, respond with 'OK'"}],
                max_tokens=10
            )
            return "OK" in response.choices[0].message.content.upper()
        except Exception as e:
            print(f"API 连接测试失败: {e}")
            return False


# 便捷函数
def get_client() -> LLMClient:
    """获取默认 LLM 客户端"""
    return LLMClient()
