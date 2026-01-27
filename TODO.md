# 🚀 项目优化 TODO

> 创新性工程方法提高 LLM 识别病例图片能力
> 创建时间: 2026-01-28

---

## 📊 当前项目状态

已实现的功能：
- [x] Set-of-Mark (SAM) 分割与标记
- [x] Visual RAG 视觉检索增强
- [x] Chain-of-Thought 思维链推理
- [x] 基础 LLM 分析流程
- [x] 评估指标框架 (IoU, Accuracy)

---

## 🔬 一、Prompt Engineering 优化

### 1.1 Multi-Persona Debate (多角色辩论) 🌟高优先级
**原理**: 让 LLM 分饰不同角色（如"激进派肿瘤科"vs"保守派病理科"），通过对抗性推理减少误判

**实现思路**:
```python
# src/prompts.py 新增
DEBATE_PROMPT = """
你现在将进行一场内部辩论。

【角色 A: 肿瘤外科医生】
偏向激进，倾向于关注任何可疑特征，宁可多取不漏诊。

【角色 B: 保守派病理专家】  
偏向保守，强调证据充分，避免过度诊断。

【裁判】
综合双方观点，给出最终建议。

请按顺序输出三段论述，最后输出 JSON 结论。
"""
```

**TODO**:
- [ ] 在 `prompts.py` 中添加 `DEBATE_PROMPT` 模板
- [ ] 在 `pipeline.py` 中添加 `--debate` 命令行参数
- [ ] 实现辩论逻辑，解析多角色输出

---

### 1.2 Coarse-to-Fine (由粗到细策略) 🌟中优先级
**原理**: 先全局扫描找可疑区，再自动裁剪放大局部区域精细分析

**实现思路**:
```python
# 第一阶段：全局分析，识别可疑区域
# 第二阶段：裁剪可疑区域，放大2-4倍
# 第三阶段：局部精细分析，确认最终建议
```

**TODO**:
- [ ] 在 `sam_processor.py` 添加 `crop_region()` 方法，支持区域裁剪
- [ ] 新建 `src/coarse_to_fine.py` 模块
- [ ] 实现两阶段推理流程
- [ ] 添加裁剪图像的多分辨率拼接

---

### 1.3 Visual ICL (视觉上下文学习) 🌟高优先级
**原理**: 在 Prompt 中直接嵌入几组对比鲜明的典型图像作为范例

**与现有 RAG 的区别**:
- RAG: 动态检索相似图像
- ICL: 固定的专家精选教学图像对（如："这种血管是癌" vs "这种是息肉"）

**TODO**:
- [ ] 创建 `data/teaching_cases/` 目录存放教学图像
- [ ] 准备 10-20 张对比鲜明的典型病例
- [ ] 实现 `get_icl_prompt()` 函数，支持多图像输入
- [ ] 在 `llm_client.py` 中支持多张参考图像并发输入

---

### 1.4 Structured Output Enforcement (结构化输出强制)
**原理**: 使用 JSON Schema/Function Calling 强制 LLM 输出符合预期格式的结构化数据

**TODO**:
- [ ] 定义 JSON Schema 描述输出格式
- [ ] 集成 OpenAI Function Calling / Structured Output 功能
- [ ] 减少 JSON 解析失败率

---

## 🧠 二、模型能力增强

### 2.1 Medical Knowledge Base Injection (医学知识库注入) 🌟高优先级
**原理**: 将医学指南标准（如"异型血管"、"脑回状突起"的定义）注入 Prompt

**TODO**:
- [ ] 整理宫腔镜病变特征描述标准（30-50 条）
- [ ] 创建 `data/knowledge_base/features.json` 知识库
- [ ] 实现知识库自动加载和 Prompt 注入
- [ ] 支持按病变类型动态选择相关知识条目

---

### 2.2 Multi-Model Ensemble (多模型集成) 🌟中优先级
**原理**: 同时调用多个 LLM (GPT-4V, Claude 3.5, Qwen-VL) 并综合结果

**TODO**:
- [ ] 在 `config.py` 支持多模型配置
- [ ] 在 `llm_client.py` 实现多模型并行调用
- [ ] 实现投票机制或加权融合策略
- [ ] 添加模型一致性分数作为置信度指标

---

### 2.3 Self-Consistency Decoding (自一致性解码)
**原理**: 同一图像多次采样（temperature > 0），选择一致性最高的答案

**TODO**:
- [ ] 在 `llm_client.py` 添加 `multi_sample()` 方法
- [ ] 实现多次采样结果的一致性计算
- [ ] 当一致性低时自动提高采样次数

---

## 🖼️ 三、图像预处理优化

### 3.1 Visual Prompting Enhancement (视觉提示增强)
**原理**: 在图像上添加网格、标尺、热力层等视觉辅助

**TODO**:
- [ ] 实现 5×5 网格叠加功能
- [ ] 添加比例尺标注
- [ ] 实现可疑区域高亮（基于 SAM 分割结果）
- [ ] 支持自定义网格密度

---

### 3.2 Image Preprocessing Pipeline (图像预处理流水线)
**原理**: 标准化图像质量，增强病变特征

**TODO**:
- [ ] 自动白平衡校正
- [ ] CLAHE 直方图均衡化增强对比度
- [ ] 去除宫腔镜边缘黑边
- [ ] 图像去噪（特别是内镜反光区域）

---

### 3.3 Attention Map Visualization (注意力图可视化)
**原理**: 可视化 LLM 关注的图像区域，提高可解释性

**TODO**:
- [ ] 研究 GPT-4V/Qwen-VL 是否支持注意力输出
- [ ] 如不支持，使用 GradCAM 等技术对开源模型做可视化
- [ ] 将注意力热力图叠加到原图，辅助医生理解

---

## 📈 四、评估与反馈

### 4.1 Human-in-the-Loop Refinement (人机协作优化) 🌟高优先级
**原理**: 收集医生对 AI 输出的反馈，持续优化 Prompt

**TODO**:
- [ ] 设计反馈收集界面（Web UI）
- [ ] 实现反馈数据存储格式
- [ ] 基于反馈自动优化 Prompt 模板
- [ ] 建立"难例库"用于测试模型边界

---

### 4.2 Active Learning Pipeline (主动学习流水线)
**原理**: 自动识别模型不确定的样本，优先请专家标注

**TODO**:
- [ ] 实现不确定性度量（多次采样方差、置信度分数）
- [ ] 自动筛选高不确定性样本
- [ ] 集成标注工具（如 LabelStudio）

---

### 4.3 A/B Testing Framework (A/B 测试框架) 🌟中优先级
**原理**: 对比不同方法的效果（医生 vs 医生+AI）

**TODO**:
- [ ] 设计双盲测试流程
- [ ] 实现测试数据随机分组
- [ ] 统计显著性检验（McNemar's test, paired t-test）
- [ ] 自动生成对比报告

---

## 🔧 五、工程质量优化

### 5.1 Async Processing (异步处理)
**TODO**:
- [ ] 使用 `asyncio` 重构 LLM 调用
- [ ] 实现请求队列和并发限制
- [ ] 添加自动重试和熔断机制

---

### 5.2 Caching Layer (缓存层)
**TODO**:
- [ ] 图像 Embedding 缓存（避免重复计算）
- [ ] LLM 响应缓存（相同输入复用结果）
- [ ] 使用 Redis 或本地 SQLite 实现

---

### 5.3 Logging & Monitoring (日志与监控)
**TODO**:
- [ ] 添加结构化日志 (JSON 格式)
- [ ] 记录每次推理的延迟、Token 消耗
- [ ] 实现成本追踪（API 费用）

---

### 5.4 Testing & CI/CD (测试与持续集成)
**TODO**:
- [ ] 添加单元测试 (`pytest`)
- [ ] 添加集成测试（端到端流程）
- [ ] 设置 GitHub Actions CI 流水线

---

## 🎯 六、创新性研究方向

### 6.1 Uncertainty-Guided Biopsy (不确定性引导活检)
**原理**: 高不确定性 = 高信息量区域，优先活检

**TODO**:
- [ ] 计算每个区域的诊断不确定性
- [ ] 将不确定性作为活检优先级因素
- [ ] 论文创新点：不确定性感知的活检策略

---

### 6.2 Cross-Modal Reasoning (跨模态推理)
**原理**: 结合患者病史、年龄、症状等文本信息增强诊断

**TODO**:
- [ ] 设计患者信息输入接口
- [ ] 在 Prompt 中集成患者背景
- [ ] 实现图像+文本联合推理

---

### 6.3 Temporal Analysis (时序分析)
**原理**: 对同一患者的多次宫腔镜图像进行时序对比分析

**TODO**:
- [ ] 支持多时间点图像输入
- [ ] 实现病变演变追踪
- [ ] 辅助判断治疗效果

---

### 6.4 Federated Learning Readiness (联邦学习准备)
**原理**: 为未来多中心数据协作做准备

**TODO**:
- [ ] 数据格式标准化
- [ ] 模型评估指标标准化
- [ ] 预留联邦学习接口

---

## 📋 优先级排序

### 🔴 高优先级 (近期实施)
1. Multi-Persona Debate (多角色辩论)
2. Visual ICL (视觉上下文学习)
3. Medical Knowledge Base (医学知识库注入)
4. Human-in-the-Loop (人机协作反馈)

### 🟡 中优先级 (中期规划)
5. Coarse-to-Fine (由粗到细)
6. Multi-Model Ensemble (多模型集成)
7. A/B Testing Framework
8. Image Preprocessing Pipeline

### 🟢 低优先级 (长期探索)
9. Uncertainty-Guided Biopsy
10. Cross-Modal Reasoning
11. Temporal Analysis
12. Federated Learning

---

## 📚 参考文献

1. **Set-of-Mark**: Yang et al., "Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V"
2. **Visual RAG**: Lu et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
3. **Chain-of-Thought**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
4. **Self-Consistency**: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

---

*最后更新: 2026-01-28*
