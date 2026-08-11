# Hyper-Spectrum 基础模型（HS-FM）设计方案

日期：2026-08-11
状态：设计定稿（用户已批准架构路线与阶段划分；实现计划另行编写）
前置调研：`duranze/hyperdata-papers` 仓 `surveys/2026-08-11-spectral-foundation-model-survey/`
（全模态谱学预训练大模型调研综述，本方案所有「调研结论」均指该文档，
主仓镜像在 hyper-data 仓 `docs/dev/research/hyper-spectrum/`）

## 0. 一句话

做第一个跨「分子谱 ↔ 材料谱」边界的**谱原生自监督基础模型**（统一
Transformer 主干 + 物理轴 Fourier 编码 + 掩码重建/跨模态对比预训练），
首发 XAS + Raman/IR + XRD 三族，单卡起步、分阶段规模化，每阶段产出可
接入 hyper-spectrum 运行时（M0 合同）的权重资产；encoder 成熟后加
LLM 桥接轨。

## 1. 定位与北极星（已批准：研究+平台双轨分阶段）

- **研究北极星**：两个调研确认的空白，各自可成一篇论文——
  1. 首个跨分子谱/材料谱边界的谱原生自监督基础模型（现状：谱原生自监督
     × 广模态覆盖是无人占据的交集；没有任何模型同时吃过 XRD/XAS 与
     IR/Raman/NMR/MS）；
  2. 首个跨模态谱基准 HS-Bench（现状：SpecX/SpectrumBench 纯分子，
     无 XRD/XAS/XPS/EELS/HSI 覆盖）。
- **平台北极星**：每阶段产出以能力声明接入 hyper-spectrum 运行时的资产
  （统一去噪轨自研权重、hub 谱 embedding 检索），接入形态与 XASDenoise
  / HyperSIGMA 固定权重工具同构——平台持有评测器，模型不自报分数。
- **失败兜底**：统一主干若不达 SOTA，语料库 + HS-Bench + 单模态
  baseline 本身就有独立发表与平台价值；LLM 桥接轨（§4）不依赖统一主干
  拿到 SOTA 才成立。

## 2. 数据模型与 Tokenizer（技术核心）

### 2.1 输入规范 = M0 SpectrumSample 扩展

在既有 SpectrumSample（显式轴/单位/掩码/通道/稀疏复数表示/来源）基础上
合并 Multimodal Universe schema 的四个字段：

| 字段 | 语义 | 为什么 |
|---|---|---|
| `provenance` | project/survey(instrument)/release 三元组 | 仪器嵌入（§3.2）的查找键 |
| `ivar` | 逐点逆方差 | 损失加权（§3.3）；不同信噪比仪器混训的前提 |
| `lsf_sigma` | **逐点线扩展函数宽度（分辨率）** | 全场公认缺口——没有任何已发表 FM 显式建模分辨率；HyperData 在采集端就能落此字段，是结构性优势 |
| 原生物理轴数组 | 不重采样、不齐轴 | 「原生网格」路线（天文侧已验证）的数据前提 |

沿用 M0 铁律：不按形状猜轴、3D 立方体不压平、归一化在模型外且参数
随样本持久化。

### 2.2 双形态统一 token 化（共享同一 λ-Fourier 编码词汇）

- **密集 1D 谱**（Raman / IR / XAS）：原生网格 patch（起步 32 点，P0
  消融定），patch 内 flux + `ivar` + `lsf_sigma` 三通道；位置编码 =
  连续物理轴（cm⁻¹ / eV）的 log 间隔 sin/cos Fourier 特征，与 patch
  嵌入相加（Universal Spectral Tokenization 配方）。**绝不用像素序号
  位置编码**（OmniSpectra 消融：去掉物理轴编码重建损失 ×1.8）。
- **稀疏峰集**（XRD 峰表；P3 扩展到 MS/NMR）：每峰一个 token——峰位
  Fourier 特征（同一词汇）+ 强度嵌入；**成对峰位差作为注意力偏置**
  （DreaMS 配方；XRD 的 2θ 差↔晶面间距关系与 MS 的中性丢失物理同构）。
- **条件嵌入**：模态嵌入 + 逐仪器嵌入（AION 式：同模态不同仪器嵌入
  不同，键 = `provenance`）。
- **幅度与形状解耦**：每样本幅度因子（log 均值）单独 token 化，防止
  定标差异主导表征。
- 坏点/缺口：patch 内 ≥50% 无效 → patch 级掩码，注意力中剔除。

### 2.3 明确不做（YAGNI）

- P0–P2 不做 2D 相关谱与 3D 立方体 token（留 P3；M0 合同已保证立方体
  不压平，接口不堵死）。
- 不做谱转文本序列化（调研结论：有损序列化是信号级任务的硬上限；
  LLM 能力走 §4 桥接轨）。

## 3. 主干与预训练目标

### 3.1 主干

- 50–100M 参数 encoder-decoder Transformer（DreaMS 116M / SpectraFM 8M
  均在单卡量级验证成功；5090 单卡可训）。
- encoder 出口加 **K-query 汇聚层**（CARL 配方，K≈8）：任意长 token 序列
  压到定长谱向量——同时解决「token 随波段数爆炸」与「embedding 检索需要
  定长向量」两个问题。
- **解码器支持任意输出物理轴查询**（UST 配方）：解码时给目标轴的
  Fourier 嵌入即可在任意波长/能量点重建——模型天然是重采样器、跨仪器
  翻译器、去噪器（直接对接 M0 统一去噪评测轨，zero-shot）。

### 3.2 预训练目标（按已验证优先级）

1. **掩码连续物理跨度重建**（主目标）：掩码连续物理区间（≈2.5 个
   patch 宽）而非随机点——随机逐点掩码可被插值平凡破解；损失逐点
   `ivar` 加权；峰集形态的峰位预测用**序数回归**（UltraNMR 配方，对
   模拟-实验系统性偏移鲁棒）。
2. **跨模态对比**：同一材料/分子的不同模态谱互为正样本（XAS↔XRD 来自
   MP 同一晶体结构；IR↔Raman↔UV-Vis 来自 QM9S 同一分子；Raman↔XRD
   来自 RRUFF 同一矿物标本）。这是「跨分子/材料边界」叙事的关键监督。
   潜空间按 SpecCLIP CLIP-split 切分：共享子空间 + 模态/仪器私有子空间
   （纯对比对齐会损毁仪器特有信息，有文档化失败案例）。
3. **教师蒸馏（辅助、非必需路径）**：Vib2Mol（Apache-2.0）/ OmniXAS
   （BSD-3）/ DreaMS（MIT）冻结 embedding 作额外回归目标加速冷启动，
   权重随训练线性衰减到 0；任何阶段可整体关闭（消融必须含无教师组）。

### 3.3 显式对照（必须打赢的笨基线）

- 公共网格重采样 + 4 层 MLP（天文侧实测与 43M FM 打平的配方）；
- TabPFN（波长盲表格 FM，现居 Raman/NIR 基准第一）；
- 各族现有单模态 SOTA（OmniXAS / CPICANN / DSCF 等）。

## 4. LLM 桥接轨（本次并入；P2 启动、P3 交付）

调研澄清：谱转文本路线（SpectraLLM 式）的问题不是「不产生表征」，而是
①峰表序列化在进模型前不可逆丢弃线形/基线/弱峰（信号级任务失明，
SpecX/SpectrumLab 实证：MLLM 缺谱 grounding）；②SFT 目标把表征锚定在
小分子结构解析；③32B 级推理成本做不了平台百万级检索。

正确关系是**先后而非取舍**：LLaVA 式桥接——**HS-FM encoder（冻结）+
projector + 开源 LLM**，谱以连续 token 进 LLM，不经文本序列化。

- 交付：谱问答/结构解析/agent 集成能力；与 hyper-spectrum agent CLI
  和 HyperData 内建对话 agent 天然衔接。
- 论文叙事：encoder 一篇；桥接后在 SpectraLLM 自己的任务（谱→SMILES）
  上对打第二篇。
- 依赖：仅需 P1 encoder 收敛，不需其达 SOTA。

## 5. 语料战略

### 5.1 模拟谱主粮（许可全部干净）

| 语料 | 模态 | 规模 | 许可 | 跨模态配对键 |
|---|---|---|---|---|
| Materials Project XAS | XAS K-edge | 50 万+ | 开放 | **mp-id（同结构可模拟 XRD → XAS↔XRD 正样本）** |
| QM9S | IR/Raman/UV-Vis | 13.4 万分子 | figshare 开放 | **同一分子三谱** |
| SimXRD-4M | XRD | 400 万图 | MIT | 结构 → 与 MP 桥接 |
| XASDataLibrary + MP 扩展 | XAS 43 元素 | 32.9 万配对 | P0 逐条核实后方可入池（XASDataLibrary 仓标 NOASSERTION） | Uni-XAS 同源 |

### 5.2 实验谱护城河

- opXRD（9.3 万实验图，CC-BY）、RRUFF（**矿物 Raman+XRD 同体配对**，
  跨模态对比的实验级正样本）、XASDataLibrary 实验谱；
- **HyperData 在吞的实验语料**（XAS/EM 等）：按 GeMS 式质量分档 + LSH
  去冗流水线入库——语料工程本身是可复制方法论，也是平台差异化。
- 模拟-实验域差策略（调研已验证清单）：序数回归损失 + 物理知情增强
  （XRD-AutoAnalyzer 的应变/峰位误差/畴尺寸展宽/织构/杂峰配方）+
  随机轴偏移增强 + 模拟预训练→实验微调。

### 5.3 许可红线

不碰 NIST 库及任何 CC-NC/ND 污染源；Alberts 多模态数据集两处许可标注
（CDLA-Sharing vs CC BY-NC-ND）核实清楚前不引入；引用许可结论一律读
原始 LICENSE 文件（GitHub 检测器会把非商用误报为 NOASSERTION）。

## 6. 评测：HS-Bench（先于训练建设）

- 把已有 per-domain research packet（XAS / XRD / spectroscopy 任务包）
  统一到一个评测合同：**固定骨架式/成分式划分**（NMRGym 教训：随机划分
  数字虚高 6 倍）、逐样本指标、失败样本留存；正式评分由 ACE 承接。
- 任务面：去噪（接 M0 现有轨）、XRD 相/空间群分类、XAS 氧化态/配位数
  回归、Raman 物种鉴定、**跨模态检索**（谱→同源异模态谱）。
- **未见仪器/多余波段压力测试**（GeoCrossBench 式）作为一等任务——
  遥感侧实测全部 FM 跌 2–4 倍、无人通过；第一个通过的谱模型本身就是
  发表点。
- 评测可信度规则继承平台现状：平台持有评测器与划分，不采信模型侧
  自报分数。

## 7. 阶段计划

| 阶段 | 周期 | 内容 | 算力 | 出口判据 |
|---|---|---|---|---|
| **P0** | ~1 月 | 语料落库 + HS-Bench 建成 + tokenizer 消融（物理轴 Fourier vs 序号编码；patch 尺寸；单模态小模型复现 DreaMS/UST 配方） | 5090 单卡 | 消融证明物理轴编码显著有效（参照 OmniSpectra 1.8× 量级）；HS-Bench 三个笨基线数字落档 |
| **P1** | ~2 月 | 三族统一预训练 50–100M；跨模态对比开启；教师蒸馏（含无教师消融组） | 单~双卡 | 统一模型 ≥2 族不劣于自家单模态 baseline；跨模态检索显著优于随机；负迁移族有归因分析 |
| **P2** | 与 P1 后半并行 | 平台接入：embedding 进 LanceDB 检索、去噪权重以 tool.yaml 注册进运行时；LLM 桥接轨启动（projector 对齐） | 单卡 | hub 真实数据上可用；去噪轨过 M0 评测 |
| **P3** | 申请算力后 | 规模化（数亿模拟谱、参数上探）+ 扩模态（NMR/MS 峰集、EELS、2D-NMR、HSI 立方体）+ 桥接轨交付 | 集群 | encoder 论文投稿；SpectraLLM 任务对打 |

P0/P1 的消融与基准证据即 P3 集群申请材料的口径。

## 8. 风险与对策

1. **三族负迁移**：P1 出口判据卡死（统一 vs 单模态逐族对照）；模态
   采样比例可调；最坏退化为共享 tokenizer + 分族头。
2. **模拟-实验域差**：§5.2 策略清单；HS-Bench 强制含实验测试集
   （opXRD/RRUFF/hub 实验数据）。
3. **跨模态配对稀疏**（实验级配对少）：模拟配对为主 + RRUFF 实验配对
   验证；配对不足时对比目标降权，不阻塞主目标。
4. **单卡算力天花板**：50–100M + 混合精度 + 序列打包在 5090 可行
   （DreaMS 前例）；数据流水线按集群规模预设计（webdataset 分片）。
5. **评测可信度**（MassSpecGym 首年 17/26 篇翻车的教训）：划分与
   评测器平台持有、泄漏检查进 CI、失败样本留存。

## 9. 与现有系统的关系

- **M0 统一去噪合同**：SpectrumSample 是 tokenizer 输入规范的基座，
  本方案只增字段不改语义；HS-FM 去噪能力以 ModelCapabilities 声明接入，
  与 XASDenoise/HyperSIGMA 并列成为统一去噪轨的一个工具。
- **ACE Benchmark**：HS-Bench 正式评分、排行、失败样本留存由 ACE 承接。
- **HyperData**：语料供给端（质量分档入库流水线）+ 消费端（embedding
  检索进 LanceDB、hub 文件详情页谱相似检索）。
- **hyperdata-papers 仓**：调研综述已入 `surveys/`；论文写作在该仓
  延续。

## 10. 如何验证本设计

- P0 消融表（物理轴编码 / patch 尺寸 / 教师有无）是设计正确性的第一
  批实证，全部在 HS-Bench 上出数。
- 每阶段出口判据（§7 表）即验收标准；任何一条不过即回到设计层修订，
  不带病进入下一阶段。
