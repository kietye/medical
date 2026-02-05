# 宫腔镜智能活检导航系统

基于 Set-of-Mark (SoM) + Visual RAG 的宫腔镜下子宫内膜病变智能活检导航系统。

## 环境要求

- Python 3.10+
- CUDA 12.1+ (推荐)
- 显存 >= 8GB

## 快速开始

1. **环境准备 (推荐使用虚拟环境)**

```bash
# 创建虚拟环境 (若尚未创建)
python -m venv .venv

# 激活虚拟环境 (Windows)
.\.venv\Scripts\activate

# 激活虚拟环境 (Linux/macOS)
# source .venv/bin/activate
```

2. **安装依赖**

```bash
pip install -r requirements.txt
```
3. **下载 SAM 模型**

```bash
python scripts/download_sam.py
```

4. **配置 API**

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
```

5. **运行测试**

```bash
python scripts/test_api.py
```

6. **运行示例**

```bash
python -m src.pipeline --image data/raw/example.jpg
```

## 项目结构

```
├── data/                   # 数据目录
│   ├── raw/                # 原始宫腔镜图像
│   ├── annotated/          # 专家标注 JSON
│   ├── embeddings/         # 向量索引
│   ├── rag_data/           # RAG 训练数据（80%）
│   │   ├── 0 粘膜下子宫肌瘤/
│   │   ├── 1 子宫内膜癌/
│   │   └── ...
│   └── test_data/          # 测试数据（20%）
│       ├── 0 粘膜下子宫肌瘤/
│       └── ...
├── models/                 # 模型权重
│   └── sam/               # SAM 模型
├── src/                   # 源代码
│   ├── config.py          # 配置管理
│   ├── sam_processor.py   # SoM 分割
│   ├── llm_client.py      # LLM 客户端
│   ├── visual_rag.py      # RAG 检索
│   └── pipeline.py        # 主流程
└── scripts/               # 工具脚本
    ├── build_rag_index.py     # 构建 RAG 索引
    ├── split_test_data.py     # 拆分测试数据集
    ├── test_rag_accuracy.py   # RAG 检索准确率测试
    └── test_pipeline_accuracy.py # 全链路 LLM 准确率测试
```

## Visual RAG 使用指南

### 1. 构建 RAG 索引

首次使用需要从分类数据构建向量索引：

```bash
# 使用默认设置
python scripts/build_rag_index.py

# 指定批量大小（显存不足时减小）
python scripts/build_rag_index.py --batch-size 16

# 查看帮助
python scripts/build_rag_index.py --help
```

索引将保存到 `data/embeddings/rag_index/` 目录。

### 2. 测试 RAG 检索

```bash
# 基本检索
python scripts/test_rag_query.py --image path/to/test.jpg

# 返回更多结果
python scripts/test_rag_query.py --image path/to/test.jpg --top-k 10

# 筛选特定类别（0-7）
python scripts/test_rag_query.py --image path/to/test.jpg --category 1

# 显示索引统计
python scripts/test_rag_query.py --image path/to/test.jpg --show-stats
```

### 3. 启用 RAG 进行分析

```bash
# 使用 RAG 增强分析
python -m src.pipeline --image data/raw/example.jpg --rag

# 同时启用 RAG 和 Chain-of-Thought
python -m src.pipeline --image data/raw/example.jpg --rag --cot
```

### 4. RAG 数据类别

| 类别ID | 诊断名称                   |
| ------ | -------------------------- |
| 0      | 粘膜下子宫肌瘤             |
| 1      | 子宫内膜癌                 |
| 2      | 子宫内膜息肉               |
| 3      | 子宫内膜息肉样增生         |
| 4      | 子宫内膜增生不伴不典型增生 |
| 5      | 宫内异物                   |
| 6      | 子宫颈息肉                 |
| 7      | 子宫内膜不典型增生         |

## 准确率测试

### 1. 拆分测试数据集

首次使用需要从 RAG 数据中拆分 20% 作为测试集：

```bash
# 预览拆分结果（不实际移动）
python scripts/split_test_data.py --dry-run

# 执行拆分
python scripts/split_test_data.py

# 重新构建 RAG 索引（排除测试集）
python scripts/build_rag_index.py
```

### 2. RAG 检索准确率测试（免费，本地运行）

```bash
python scripts/test_rag_accuracy.py
python scripts/test_rag_accuracy.py --show-confusion-matrix
```

### 3. 全链路 LLM 准确率测试（调用 API，有费用）

```bash
# 快速测试（每类别 2 张，约 ¥0.3-0.5）
python scripts/test_pipeline_accuracy.py --limit 2

# 完整测试（596 张，约 ¥10-20）
python scripts/test_pipeline_accuracy.py

# 启用 RAG + CoT
python scripts/test_pipeline_accuracy.py --rag --cot --limit 3

# 启用 CoT
python scripts/test_pipeline_accuracy.py --no-rag --cot --limit 3
```

---

## 技术路线

1. **Set-of-Mark (SoM)**: 使用 SAM 分割 + 编号标记
2. **Visual RAG**: 相似病例检索增强
3. **Chain-of-Thought**: 分步推理，提供可解释性
