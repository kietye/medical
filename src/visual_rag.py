"""
Visual RAG 模块
实现视觉检索增强生成，支持从分类文件夹构建索引
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from PIL import Image
from typing import Optional
from tqdm import tqdm

from .config import config


@dataclass
class CaseMetadata:
    """病例元数据"""
    image_id: str
    image_path: str
    diagnosis: str
    category_id: int = 0
    features: list[str] = field(default_factory=list)
    biopsy_region: str = ""
    pathology_result: str = ""
    difficulty_level: str = "medium"
    annotator: str = ""


@dataclass
class RetrievalResult:
    """检索结果"""
    case: CaseMetadata
    similarity: float
    image: Optional[Image.Image] = None


# 类别映射
CATEGORY_NAMES = {
    0: "粘膜下子宫肌瘤",
    1: "子宫内膜癌",
    2: "子宫内膜息肉",
    3: "子宫内膜息肉样增生",
    4: "子宫内膜增生不伴不典型增生",
    5: "宫内异物",
    6: "子宫颈息肉",
    7: "子宫内膜不典型增生",
}


def get_category_id_from_folder(folder_name: str) -> int:
    """从文件夹名解析类别ID"""
    # 文件夹格式: "0 粘膜下子宫肌瘤"
    try:
        return int(folder_name.split()[0])
    except (ValueError, IndexError):
        return -1


def get_category_name_from_folder(folder_name: str) -> str:
    """从文件夹名解析类别名称"""
    # 文件夹格式: "0 粘膜下子宫肌瘤"
    parts = folder_name.split(maxsplit=1)
    if len(parts) > 1:
        return parts[1]
    return folder_name


class VisualRAG:
    """视觉检索增强模块"""
    
    def __init__(self, index_path: str | Path = None):
        """
        初始化 Visual RAG
        
        Args:
            index_path: 向量索引路径
        """
        self.index_path = Path(index_path) if index_path else config.EMBEDDINGS_DIR / "rag_index"
        self.cases: list[CaseMetadata] = []
        self.embeddings: np.ndarray = None
        self.encoder = None
        self._faiss_index = None
        
    def _load_encoder(self):
        """加载图像编码器"""
        if self.encoder is not None:
            return
            
        from sentence_transformers import SentenceTransformer
        
        print("加载图像编码器: clip-ViT-B-32")
        self.encoder = SentenceTransformer("clip-ViT-B-32")
        print("编码器加载完成")
    
    def _encode_image(self, image_path: str | Path) -> np.ndarray:
        """编码单个图像为向量"""
        self._load_encoder()
        image = Image.open(image_path).convert("RGB")
        embedding = self.encoder.encode(image)
        return embedding
    
    def _encode_images_batch(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """批量编码图像为向量，提高效率"""
        self._load_encoder()
        
        all_embeddings = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(image_paths), batch_size), 
                      desc="编码图像", total=total_batches):
            batch_paths = image_paths[i:i + batch_size]
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"警告: 无法加载图像 {path}: {e}")
                    # 创建一个空白图像作为占位符
                    images.append(Image.new("RGB", (224, 224), color="black"))
            
            batch_embeddings = self.encoder.encode(images, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
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
            category_id=metadata.get("category_id", 0),
            features=metadata.get("features", []),
            biopsy_region=metadata.get("biopsy_region", ""),
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
    
    def build_index_from_folders(self, data_dir: str | Path = None, batch_size: int = 32):
        """
        从分类文件夹构建索引
        
        Args:
            data_dir: 数据目录，包含按类别分类的子文件夹
            batch_size: 批量编码的大小
        """
        data_dir = Path(data_dir) if data_dir else config.RAG_DATA_DIR
        
        if not data_dir.exists():
            print(f"错误: 数据目录不存在 {data_dir}")
            return
        
        print(f"从目录构建索引: {data_dir}")
        
        # 收集所有图像和元数据
        all_image_paths = []
        all_metadata = []
        
        # 遍历类别文件夹
        category_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
        
        for folder in category_folders:
            category_id = get_category_id_from_folder(folder.name)
            category_name = get_category_name_from_folder(folder.name)
            
            if category_id < 0:
                print(f"跳过无效文件夹: {folder.name}")
                continue
            
            # 获取该类别下的所有图像
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
            images = set()  # Use set to avoid duplicates (Windows is case-insensitive)
            for ext in image_extensions:
                images.update(folder.glob(f"*{ext}"))
                images.update(folder.glob(f"*{ext.upper()}"))
            images = sorted(images)  # Convert back to sorted list
            
            print(f"类别 [{category_id}] {category_name}: 找到 {len(images)} 张图像")
            
            for img_path in images:
                all_image_paths.append(img_path)
                all_metadata.append({
                    "image_id": img_path.stem,
                    "image_path": str(img_path),
                    "diagnosis": category_name,
                    "category_id": category_id,
                    "features": [],
                    "biopsy_region": "",
                    "pathology_result": category_name,
                })
        
        if not all_image_paths:
            print("错误: 未找到任何图像")
            return
        
        print(f"\n共找到 {len(all_image_paths)} 张图像，开始批量编码...")
        
        # 批量编码所有图像
        self.embeddings = self._encode_images_batch(all_image_paths, batch_size=batch_size)
        
        # 创建病例元数据
        self.cases = [
            CaseMetadata(
                image_id=meta["image_id"],
                image_path=meta["image_path"],
                diagnosis=meta["diagnosis"],
                category_id=meta["category_id"],
                features=meta["features"],
                biopsy_region=meta["biopsy_region"],
                pathology_result=meta["pathology_result"],
            )
            for meta in all_metadata
        ]
        
        print(f"编码完成! 共 {len(self.cases)} 个病例")
        
        # 构建 FAISS 索引
        self._build_faiss_index()
        
        # 保存索引
        self.save_index()
    
    def _build_faiss_index(self):
        """构建 FAISS 索引以加速检索"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        
        try:
            import faiss
            
            # 归一化向量
            embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings_norm.astype(np.float32)
            
            # 创建内积索引（归一化后等同于余弦相似度）
            dimension = embeddings_norm.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dimension)
            self._faiss_index.add(embeddings_norm)
            
            print(f"FAISS 索引构建完成, 维度: {dimension}, 向量数: {self._faiss_index.ntotal}")
        except ImportError:
            print("警告: FAISS 未安装，将使用 NumPy 进行暴力搜索")
            self._faiss_index = None
    
    def build_index(self, data_dir: str | Path = None):
        """
        从数据目录构建索引（兼容旧接口）
        
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
        
        # 构建 FAISS 索引
        self._build_faiss_index()
        
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
        
        # 保存 FAISS 索引
        if self._faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self._faiss_index, str(path / "faiss.index"))
            except Exception as e:
                print(f"警告: 无法保存 FAISS 索引: {e}")
        
        # 保存类别统计
        category_stats = {}
        for case in self.cases:
            cat_name = CATEGORY_NAMES.get(case.category_id, case.diagnosis)
            category_stats[cat_name] = category_stats.get(cat_name, 0) + 1
        
        with open(path / "stats.json", "w", encoding="utf-8") as f:
            json.dump({
                "total_cases": len(self.cases),
                "category_stats": category_stats,
                "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"索引已保存到: {path}")
    
    def load_index(self, path: str | Path = None) -> bool:
        """从磁盘加载索引"""
        path = Path(path) if path else self.index_path
        
        cases_path = path / "cases.json"
        embeddings_path = path / "embeddings.npy"
        faiss_path = path / "faiss.index"
        
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
        
        # 加载 FAISS 索引
        if faiss_path.exists():
            try:
                import faiss
                self._faiss_index = faiss.read_index(str(faiss_path))
                print(f"FAISS 索引已加载, 向量数: {self._faiss_index.ntotal}")
            except Exception as e:
                print(f"警告: 无法加载 FAISS 索引: {e}")
                self._faiss_index = None
        
        print(f"已加载 {len(self.cases)} 个病例")
        return True
    
    def retrieve(self, query_image: str | Path, top_k: int = 5, 
                 filter_category: int = None) -> list[RetrievalResult]:
        """
        检索相似病例
        
        Args:
            query_image: 查询图像路径
            top_k: 返回前 K 个结果
            filter_category: 可选，筛选特定类别
            
        Returns:
            list[RetrievalResult]: 检索结果列表
        """
        if len(self.cases) == 0:
            print("警告: 索引为空，请先构建索引")
            return []
        
        # 编码查询图像
        query_embedding = self._encode_image(query_image)
        query_embedding = query_embedding.astype(np.float32)
        
        # 归一化
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # 使用 FAISS 检索（如果可用）
        if self._faiss_index is not None:
            # FAISS 检索
            actual_k = min(top_k * 3 if filter_category is not None else top_k, len(self.cases))
            similarities, indices = self._faiss_index.search(
                query_norm.reshape(1, -1), actual_k
            )
            similarities = similarities[0]
            indices = indices[0]
        else:
            # NumPy 暴力搜索
            embeddings_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            similarities = np.dot(embeddings_norm, query_norm)
            indices = np.argsort(similarities)[::-1]
            similarities = similarities[indices]
        
        # 构建结果
        results = []
        for idx, sim in zip(indices, similarities):
            if idx < 0 or idx >= len(self.cases):
                continue
            case = self.cases[idx]
            
            # 类别过滤
            if filter_category is not None and case.category_id != filter_category:
                continue
            
            results.append(RetrievalResult(
                case=case,
                similarity=float(sim),
                image=None  # 延迟加载图像
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_with_images(self, query_image: str | Path, top_k: int = 5,
                             filter_category: int = None) -> list[RetrievalResult]:
        """
        检索相似病例并加载图像
        
        Args:
            query_image: 查询图像路径
            top_k: 返回前 K 个结果
            filter_category: 可选，筛选特定类别
            
        Returns:
            list[RetrievalResult]: 检索结果列表（包含图像）
        """
        results = self.retrieve(query_image, top_k, filter_category)
        
        for result in results:
            if Path(result.case.image_path).exists():
                result.image = Image.open(result.case.image_path)
        
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
        for i, result in enumerate(results):
            case = result.case
            formatted.append({
                "rank": i + 1,
                "diagnosis": case.diagnosis,
                "category_id": case.category_id,
                "features": ", ".join(case.features) if case.features else "未标注",
                "biopsy_region": case.biopsy_region or "未标注",
                "pathology_result": case.pathology_result or case.diagnosis,
                "similarity": f"{result.similarity:.2%}"
            })
        return formatted
    
    def get_category_distribution(self) -> dict:
        """获取索引中的类别分布"""
        distribution = {}
        for case in self.cases:
            cat_name = CATEGORY_NAMES.get(case.category_id, case.diagnosis)
            distribution[cat_name] = distribution.get(cat_name, 0) + 1
        return distribution
    
    def predict_category(self, query_image: str | Path, top_k: int = 5) -> tuple[int, float, dict]:
        """
        预测图像的类别
        
        基于 top-k 检索结果的多数投票
        
        Args:
            query_image: 查询图像路径
            top_k: 用于投票的检索结果数量
            
        Returns:
            tuple: (预测类别ID, 置信度, 详细信息)
                - 预测类别ID: int
                - 置信度: float (0-1)
                - 详细信息: dict 包含投票统计等
        """
        results = self.retrieve(query_image, top_k=top_k)
        
        if not results:
            return -1, 0.0, {"error": "无法检索到相似病例"}
        
        # 统计每个类别的投票数（按相似度加权）
        category_votes = {}
        category_counts = {}
        
        for result in results:
            cat_id = result.case.category_id
            similarity = result.similarity
            
            category_votes[cat_id] = category_votes.get(cat_id, 0.0) + similarity
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        # 选择加权投票最高的类别
        predicted_category = max(category_votes.keys(), key=lambda k: category_votes[k])
        
        # 计算置信度（加权投票比例）
        total_weight = sum(category_votes.values())
        confidence = category_votes[predicted_category] / total_weight if total_weight > 0 else 0.0
        
        # 详细信息
        details = {
            "predicted_category_id": predicted_category,
            "predicted_category_name": CATEGORY_NAMES.get(predicted_category, "未知"),
            "confidence": confidence,
            "top_k": top_k,
            "actual_retrieved": len(results),
            "category_votes": {
                CATEGORY_NAMES.get(k, f"类别{k}"): {
                    "count": category_counts[k],
                    "weighted_score": category_votes[k]
                }
                for k in category_votes
            },
            "top_results": [
                {
                    "image_id": r.case.image_id,
                    "category_id": r.case.category_id,
                    "category_name": CATEGORY_NAMES.get(r.case.category_id, "未知"),
                    "similarity": r.similarity
                }
                for r in results[:5]
            ]
        }
        
        return predicted_category, confidence, details


# 便捷函数
def get_rag() -> VisualRAG:
    """获取默认 RAG 实例"""
    rag = VisualRAG()
    rag.load_index()
    return rag


def build_rag_index(data_dir: str | Path = None, batch_size: int = 32) -> VisualRAG:
    """构建 RAG 索引"""
    rag = VisualRAG()
    rag.build_index_from_folders(data_dir, batch_size)
    return rag

