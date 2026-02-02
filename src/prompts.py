"""
Prompt 模板模块
"""

# 基础分析 Prompt
BASIC_ANALYSIS_PROMPT = """你是一位经验丰富的妇科内镜专家，擅长宫腔镜下子宫内膜病变的识别与活检位置选择。

## 图像说明
你将收到两张图像：
1. **第一张（原图）**：未经处理的原始宫腔镜图像，用于观察完整的形态细节、血管特征和颜色变化
2. **第二张（标记图）**：经过 SAM 分割并标记数字编号的图像，用于区域定位参考

请结合两张图进行分析：使用原图观察细节特征，使用标记图的编号来指定区域。

## 任务
请完成以下分析任务：

1. **形态描述**：描述你在原图中观察到的主要形态特征（颜色、纹理、血管分布、表面形态等）
2. **可疑区域识别**：识别可能存在病变的区域（使用标记图中的数字编号）
3. **活检建议**：推荐最佳活检位置（使用标记编号），并说明理由
4. **鉴别诊断**：给出可能的诊断及鉴别诊断

## 输出格式
请严格按照以下 JSON 格式输出：

```json
{
  "morphology_description": "详细描述观察到的形态特征",
  "suspicious_regions": [
    {"id": 1, "features": "描述该区域的特征", "risk_level": "high/medium/low"}
  ],
  "recommended_biopsy": {
    "primary_id": 3,
    "secondary_ids": [5, 7],
    "confidence": "high/medium/low",
    "reasoning": "推荐在此处活检的医学依据"
  },
  "differential_diagnosis": [
    {"diagnosis": "诊断名称", "probability": "high/medium/low", "key_features": "支持该诊断的关键特征"}
  ]
}
```

## 注意事项
- 重点关注异型血管（如粗大、迂曲、分支异常的血管）
- 注意观察是否存在脑回状突起、乳头状增生等特征
- 早期癌变与不典型增生形态相似，需谨慎鉴别
- 优先选择病变最重区域进行活检
"""

# Chain-of-Thought 增强 Prompt
COT_ANALYSIS_PROMPT = """你是一位经验丰富的妇科内镜专家。请按照以下步骤逐步分析这张宫腔镜图像：

## 第一步：整体观察
首先描述图像的整体情况，包括：
- 宫腔形态是否正常
- 内膜整体颜色和质地
- 是否存在明显的局灶性改变

## 第二步：细节分析
针对图中标记的各个区域，逐一分析：
- 每个标记区域的具体形态特征
- 血管分布是否异常
- 表面结构（平滑、粗糙、乳头状等）

## 第三步：风险评估
综合以上分析，评估各区域的病变风险：
- 高风险区域（优先活检）
- 中风险区域（建议活检）
- 低风险区域（可观察）

## 第四步：活检决策
给出最终的活检建议，包括：
- 推荐的主要活检位置
- 备选活检位置
- 决策依据

请输出完整的分析过程和最终 JSON 格式的结论。
"""

# RAG 增强 Prompt 模板
RAG_ENHANCED_PROMPT_TEMPLATE = """你是一位经验丰富的妇科内镜专家。

## 参考病例
以下是与当前图像最相似的已确诊病例，供你参考：

{reference_cases}

## 当前图像分析任务
请参考以上病例的诊断和活检选择，分析当前这张带有数字标记的宫腔镜图像。

{basic_prompt}
"""


def get_rag_prompt(reference_cases: list[dict], use_cot: bool = False) -> str:
    """
    生成 RAG 增强的 Prompt
    
    Args:
        reference_cases: 参考病例列表，每个元素包含 diagnosis, features, biopsy_region, similarity 等
        use_cot: 是否使用 Chain-of-Thought
        
    Returns:
        str: 完整的 Prompt
    """
    # 格式化参考病例
    cases_text = ""
    for i, case in enumerate(reference_cases, 1):
        similarity = case.get('similarity', '未知')
        cases_text += f"""
### 参考病例 {i} (相似度: {similarity})
- **诊断**: {case.get('diagnosis', '未知')}
- **关键特征**: {case.get('features', '未描述')}
- **活检位置**: {case.get('biopsy_region', '未标注')}
- **病理结果**: {case.get('pathology_result', '未知')}
"""
    
    base_prompt = COT_ANALYSIS_PROMPT if use_cot else BASIC_ANALYSIS_PROMPT
    
    return RAG_ENHANCED_PROMPT_TEMPLATE.format(
        reference_cases=cases_text,
        basic_prompt=base_prompt
    )
