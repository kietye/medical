"""
Visual RAG 模块
实现视觉检索增强生成
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from PIL import Image

from .config import config


@dataclass
class CaseMetadata:
    """病例元数据"""
    image_id: str
    image_path: str
    diagnosis: str
    features: list[str]
    biopsy_region: str
    pathology_result: str = ""
    difficulty_level: str = "medium"
    annotator: str = ""


@dataclass
class RetrievalResult:
    """检索结果"""
    case: CaseMetadata
    similarity: float
    image: Image.Image = None


class VisualRAG:
    """视觉检索增强模块"""
    
    def __init__(self, index_path: str | Path = None):
        """
        初始化 Visual RAG
        
        Args:
            index_path: 向量索引路径
        """
        self.index_path = Path(index_path) if index_path else config.EMBEDDINGS_DIR / "index"
        self.cases: list[CaseMetadata] = []
        self.embeddings: np.ndarray = None
        self.encoder = None
        
    def _load_encoder(self):
        """加载图像编码器"""
        if self.encoder is not None:
            return
            
        from sentence_transformers import SentenceTransformer
        
        print("加载图像编码器: clip-ViT-B-32")
        self.encoder = SentenceTransformer("clip-ViT-B-32")
        print("编码器加载完成")
    
    def _encode_image(self, image_path: str | Path) -> np.ndarray:
        """编码图像为向量"""
        self._load_encoder()
        image = Image.open(image_path).convert("RGB")
        embedding = self.encoder.encode(image)
        return embedding
    
    def add_case(self, image_path: str | Path, metadata: dict):
        """
        添加病例到索引
        
        Args:
            image_path: 图像路径
            metadata: 病例元数据
        """
        # 创建元数据
        case = CaseMetadata(
            image_id=metadata.get("image_id", Path(image_path).stem),
            image_path=str(image_path),
            diagnosis=metadata.get("diagnosis", "未知"),
            features=metadata.get("features", []),
            biopsy_region=metadata.get("biopsy_region", "未标注"),
            pathology_result=metadata.get("pathology_result", ""),
            difficulty_level=metadata.get("difficulty_level", "medium"),
            annotator=metadata.get("annotator", "")
        )
        
        # 编码图像
        embedding = self._encode_image(image_path)
        
        # 添加到列表
        self.cases.append(case)
        if self.embeddings is None:
            self.embeddings = embedding.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        print(f"添加病例: {case.image_id}")
    
    def build_index(self, data_dir: str | Path = None):
        """
        从数据目录构建索引
        
        Args:
            data_dir: 标注数据目录
        """
        data_dir = Path(data_dir) if data_dir else config.ANNOTATED_DIR
        
        # 遍历 JSON 标注文件
        json_files = list(data_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 个标注文件")
        
        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            image_path = data.get("image_path", "")
            if not Path(image_path).exists():
                # 尝试相对路径
                image_path = config.RAW_DIR / Path(image_path).name
            
            if Path(image_path).exists():
                self.add_case(image_path, data.get("expert_annotation", data))
            else:
                print(f"警告: 图像不存在 {image_path}")
        
        # 保存索引
        self.save_index()
    
    def save_index(self, path: str | Path = None):
        """保存索引到磁盘"""
        path = Path(path) if path else self.index_path
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存元数据
        cases_data = [asdict(case) for case in self.cases]
        with open(path / "cases.json", "w", encoding="utf-8") as f:
            json.dump(cases_data, f, ensure_ascii=False, indent=2)
        
        # 保存向量
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        
        print(f"索引已保存到: {path}")
    
    def load_index(self, path: str | Path = None):
        """从磁盘加载索引"""
        path = Path(path) if path else self.index_path
        
        cases_path = path / "cases.json"
        embeddings_path = path / "embeddings.npy"
        
        if not cases_path.exists():
            print(f"索引不存在: {path}")
            return False
        
        # 加载元数据
        with open(cases_path, "r", encoding="utf-8") as f:
            cases_data = json.load(f)
        self.cases = [CaseMetadata(**data) for data in cases_data]
        
        # 加载向量
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        print(f"已加载 {len(self.cases)} 个病例")
        return True
    
    def retrieve(self, query_image: str | Path, top_k: int = 3) -> list[RetrievalResult]:
        """
        检索相似病例
        
        Args:
            query_image: 查询图像路径
            top_k: 返回前 K 个结果
            
        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        if len(self.cases) == 0:
            print("警告: 索引为空，请先构建索引")
            return []
        
        # 编码查询图像
        query_embedding = self._encode_image(query_image)
        
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取 Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            case = self.cases[idx]
            results.append(RetrievalResult(
                case=case,
                similarity=float(similarities[idx]),
                image=Image.open(case.image_path) if Path(case.image_path).exists() else None
            ))
        
        return results
    
    def format_for_prompt(self, results: list[RetrievalResult]) -> list[dict]:
        """
        将检索结果格式化为 Prompt 所需格式
        
        Args:
            results: 检索结果列表
            
        Returns:
            list[dict]: 格式化后的病例信息
        """
        formatted = []
        for result in results:
            case = result.case
            formatted.append({
                "diagnosis": case.diagnosis,
                "features": ", ".join(case.features),
                "biopsy_region": case.biopsy_region,
                "pathology_result": case.pathology_result,
                "similarity": f"{result.similarity:.2%}"
            })
        return formatted


# 便捷函数
def get_rag() -> VisualRAG:
    """获取默认 RAG 实例"""
    rag = VisualRAG()
    rag.load_index()
    return rag
