"""
配置管理模块
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class Config:
    """全局配置"""
    
    # 项目根目录
    ROOT_DIR: Path = Path(__file__).parent.parent
    
    # 数据目录
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    ANNOTATED_DIR: Path = DATA_DIR / "annotated"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    RAG_DATA_DIR: Path = DATA_DIR / "rag_data"
    TEST_DATA_DIR: Path = DATA_DIR / "test_data"
    
    # 测试配置
    TEST_SPLIT_RATIO: float = 0.2  # 20% 作为测试集
    RANDOM_SEED: int = 42  # 可复现的随机种子
    
    # 模型目录
    MODELS_DIR: Path = ROOT_DIR / "models"
    SAM_DIR: Path = MODELS_DIR / "sam"
    
    # 输出目录
    OUTPUT_DIR: Path = ROOT_DIR / "output"
    
    # API 配置
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_BASE_URL: str = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # 模型配置
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen-vl-max")
    SAM_ENABLED: bool = os.getenv("SAM_ENABLED", "true").lower() in ("true", "1", "yes")
    SAM_MODEL_TYPE: str = os.getenv("SAM_MODEL_TYPE", "vit_h")
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "false").lower() in ("true", "1", "yes")
    
    # SAM 模型权重文件名映射
    SAM_CHECKPOINTS: dict = None
    
    def __post_init__(self):
        self.SAM_CHECKPOINTS = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        
        # 创建必要的目录
        for dir_path in [self.RAW_DIR, self.ANNOTATED_DIR, self.EMBEDDINGS_DIR, 
                         self.SAM_DIR, self.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def sam_checkpoint_path(self) -> Path:
        """获取 SAM 模型权重路径"""
        return self.SAM_DIR / self.SAM_CHECKPOINTS.get(self.SAM_MODEL_TYPE, "sam_vit_h_4b8939.pth")
    
    def get_api_key(self) -> str:
        """获取可用的 API Key"""
        return self.QWEN_API_KEY or self.OPENAI_API_KEY
    
    def get_base_url(self) -> str | None:
        """获取 API Base URL"""
        if self.QWEN_API_KEY:
            return self.QWEN_BASE_URL
        return None  # OpenAI 使用默认 URL


# 全局配置实例
config = Config()
