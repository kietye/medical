# 宫腔镜智能活检导航系统

基于 Set-of-Mark (SoM) + Visual RAG 的宫腔镜下子宫内膜病变智能活检导航系统。

## 环境要求

- Python 3.10+
- CUDA 12.1+ (推荐)
- 显存 >= 8GB

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 SAM 模型
python scripts/download_sam.py

# 3. 配置 API
cp .env.example .env
# 编辑 .env 填入 API Key

# 4. 运行测试
python scripts/test_api.py

# 5. 运行示例
python -m src.pipeline --image data/raw/example.jpg
```

## 项目结构

```
├── data/                   # 数据目录
│   ├── raw/                # 原始宫腔镜图像
│   ├── annotated/          # 专家标注 JSON
│   └── embeddings/         # 向量索引
├── models/                 # 模型权重
│   └── sam/               # SAM 模型
├── src/                   # 源代码
│   ├── config.py          # 配置管理
│   ├── sam_processor.py   # SoM 分割
│   ├── llm_client.py      # LLM 客户端
│   ├── visual_rag.py      # RAG 检索
│   └── pipeline.py        # 主流程
└── scripts/               # 工具脚本
```

## 技术路线

1. **Set-of-Mark (SoM)**: 使用 SAM 分割 + 编号标记
2. **Visual RAG**: 相似病例检索增强
3. **Chain-of-Thought**: 分步推理，提供可解释性
