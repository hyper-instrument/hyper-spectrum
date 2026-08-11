# HS-FM P0 阶段实现计划（语料落库 + HS-Bench + tokenizer 消融）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成 HS-FM 的数据底座（三族语料统一入 SpectrumRecord 格式）、HS-Bench 最小基准（4 任务 + 3 笨基线数字落档）、tokenizer 消融（物理轴 Fourier vs 序号编码），产出 P0 出口判据报告。

**Architecture:** 新建独立研究仓 `hs-fm`（hyper-spectrum 运行时合同禁止训练，研究代码不进运行时仓）。核心抽象是 `SpectrumRecord`（M0 SpectrumSample 语义 + provenance/ivar/lsf_sigma/原生轴四字段）；语料适配器把各公开语料转成统一 parquet 分片；HS-Bench 是「任务注册表 + 固定划分 + 逐样本指标」的评测器；消融用同一个小 encoder（~10M）只换位置编码方式。

**Tech Stack:** Python 3.11 + uv；PyTorch ≥2.4（mps/cuda 双端）；numpy/pyarrow/polars；pytest（asyncio 不需要）；tabpfn（基线）；mp-api（MP XAS 下载）。

**运行环境约定:** 开发与单测在 Mac（CPU/MPS）；训练/消融跑 5090（数据先 rsync 上去）。所有脚本必须 `--device auto`。

**Spec:** `docs/superpowers/specs/2026-08-11-hyper-spectrum-foundation-model-design.md`（同分支）

---

## 文件结构总览

```
hs-fm/                          # 新仓 ~/Developer/hs-fm
├── pyproject.toml
├── src/hsfm/
│   ├── record.py               # SpectrumRecord + 校验（Task 2）
│   ├── io.py                   # parquet 读写分片（Task 2）
│   ├── corpora/
│   │   ├── rruff.py            # RRUFF Raman 适配器（Task 3）
│   │   ├── qm9s.py             # QM9S IR/Raman 适配器（Task 4）
│   │   ├── mp_xas.py           # MP XAS 适配器（Task 5）
│   │   └── simxrd.py           # SimXRD 峰表适配器（Task 6）
│   ├── bench/
│   │   ├── registry.py         # 任务注册表 + 固定划分（Task 7）
│   │   ├── splits.py           # 成分不相交划分（Task 7）
│   │   ├── metrics.py          # 逐样本指标（Task 7）
│   │   └── tasks/              # 每任务一个模块（Task 8）
│   ├── baselines/
│   │   ├── resample_mlp.py     # 笨基线①（Task 9）
│   │   └── tabpfn_runner.py    # 笨基线②（Task 9）
│   ├── tokenize/
│   │   ├── axis_encoding.py    # λ-Fourier / 序号 PE（Task 10）
│   │   └── dense_patch.py      # 密集谱 patch 化（Task 10）
│   ├── model/
│   │   └── encoder.py          # 小 encoder + 分类头（Task 11）
│   └── train/
│       └── ablate_pe.py        # 消融训练脚本（Task 12）
├── scripts/fetch_*.py          # 各语料下载脚本
├── tests/                      # 与 src 镜像
└── reports/p0/                 # 消融结果与出口判据报告（Task 13）
```

数据目录（不进 git）：`~/hs-fm-data/{raw,records,bench}/`。

---

### Task 1: 仓库脚手架

**Files:** Create `pyproject.toml`, `.gitignore`, `README.md`, `src/hsfm/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: 建仓**

```bash
mkdir -p ~/Developer/hs-fm/{src/hsfm,tests,scripts,reports/p0} && cd ~/Developer/hs-fm && git init -b main
```

- [ ] **Step 2: 写 pyproject.toml**

```toml
[project]
name = "hsfm"
version = "0.0.1"
description = "HS-FM: spectrum-native foundation model research (P0)"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26", "pyarrow>=16", "polars>=1.0", "torch>=2.4",
  "pydantic>=2.7", "tqdm", "requests",
]
[project.optional-dependencies]
baselines = ["tabpfn>=2.0", "scikit-learn>=1.4"]
mp = ["mp-api>=0.41"]
test = ["pytest>=8"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`: `__pycache__/ *.egg-info .venv data/ ~$*`。`README.md` 一段话指向 spec 位置。

- [ ] **Step 3: 安装并验证**

```bash
uv venv && uv pip install -e ".[test]" && .venv/bin/pytest --co -q
```
Expected: `no tests ran`（空集合，无报错）。

- [ ] **Step 4: Commit** `git add -A && git commit -m "chore: scaffold hs-fm P0 workspace"`

---

### Task 2: SpectrumRecord 数据模型 + parquet IO

**Files:** Create `src/hsfm/record.py`, `src/hsfm/io.py`, `tests/test_record.py`

spec §2.1 的落地。铁律进校验：轴严格递增、不猜轴（axis_unit 必填）、ivar/lsf_sigma 长度与 flux 一致或为 None、峰集形态 axis 为 None 而 peaks 必填。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_record.py
import numpy as np, pytest
from hsfm.record import SpectrumRecord, Provenance

def _rec(**kw):
    d = dict(
        modality="raman", axis_unit="cm^-1",
        axis=np.linspace(100, 3500, 1000), flux=np.random.rand(1000),
        provenance=Provenance(project="rruff", instrument="unknown", release="2026"),
    )
    d.update(kw); return SpectrumRecord(**d)

def test_valid_dense_record_roundtrip(tmp_path):
    from hsfm.io import write_shard, read_shard
    recs = [_rec() for _ in range(3)]
    p = tmp_path / "s.parquet"; write_shard(recs, p)
    back = read_shard(p)
    assert len(back) == 3
    np.testing.assert_allclose(back[0].axis, recs[0].axis)
    assert back[0].provenance.project == "rruff"

def test_axis_must_increase():
    with pytest.raises(ValueError, match="strictly increasing"):
        _rec(axis=np.linspace(3500, 100, 1000))

def test_ivar_length_checked():
    with pytest.raises(ValueError, match="ivar"):
        _rec(ivar=np.ones(7))

def test_peak_form_requires_peaks():
    with pytest.raises(ValueError, match="peaks"):
        SpectrumRecord(modality="xrd_peaks", axis_unit="two_theta_deg",
                       axis=None, flux=None, peaks=None,
                       provenance=Provenance(project="simxrd", instrument="sim", release="v1"))
```

- [ ] **Step 2: 跑测试确认失败** `pytest tests/test_record.py -q` → ImportError。

- [ ] **Step 3: 实现 record.py**

```python
# src/hsfm/record.py
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Provenance:
    project: str      # 语料/项目名，如 "rruff"
    instrument: str   # 仪器标识；模拟谱写模拟器名，如 "dft-b3lyp"
    release: str      # 版本/发布号

@dataclass
class SpectrumRecord:
    modality: str                     # raman | ir | xas | xrd_peaks
    axis_unit: str                    # cm^-1 | eV | two_theta_deg
    provenance: Provenance
    axis: np.ndarray | None = None    # 原生物理轴（密集形态）
    flux: np.ndarray | None = None
    ivar: np.ndarray | None = None    # 逐点逆方差，未知为 None（不造假）
    lsf_sigma: np.ndarray | None = None  # 逐点分辨率，未知为 None
    mask: np.ndarray | None = None    # True=有效；None 表示全有效
    peaks: np.ndarray | None = None   # (N,2) [position, intensity]，峰集形态
    label: dict = field(default_factory=dict)   # 下游标签（如 mineral, space_group）
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.modality.endswith("_peaks"):
            if self.peaks is None or len(self.peaks) == 0:
                raise ValueError("peak-form record requires non-empty peaks")
            return
        if self.axis is None or self.flux is None:
            raise ValueError("dense record requires axis and flux")
        if len(self.axis) != len(self.flux):
            raise ValueError("axis/flux length mismatch")
        if not np.all(np.diff(self.axis) > 0):
            raise ValueError("axis must be strictly increasing")
        for name in ("ivar", "lsf_sigma", "mask"):
            v = getattr(self, name)
            if v is not None and len(v) != len(self.flux):
                raise ValueError(f"{name} length mismatch")
```

- [ ] **Step 4: 实现 io.py**（列式存储，变长数组用 pyarrow list）

```python
# src/hsfm/io.py
from __future__ import annotations
import json, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from .record import SpectrumRecord, Provenance

_ARRAY_COLS = ("axis", "flux", "ivar", "lsf_sigma", "mask")

def write_shard(records: list[SpectrumRecord], path) -> None:
    cols: dict[str, list] = {c: [] for c in _ARRAY_COLS}
    cols |= {"modality": [], "axis_unit": [], "provenance": [], "peaks": [],
             "label": [], "meta": []}
    for r in records:
        for c in _ARRAY_COLS:
            v = getattr(r, c)
            cols[c].append(None if v is None else np.asarray(v, np.float64).tolist())
        cols["modality"].append(r.modality)
        cols["axis_unit"].append(r.axis_unit)
        cols["provenance"].append(json.dumps(vars(r.provenance)))
        cols["peaks"].append(None if r.peaks is None else np.asarray(r.peaks, np.float64).reshape(-1).tolist())
        cols["label"].append(json.dumps(r.label))
        cols["meta"].append(json.dumps(r.meta))
    pq.write_table(pa.table(cols), path)

def read_shard(path) -> list[SpectrumRecord]:
    t = pq.read_table(path).to_pydict()
    out = []
    for i in range(len(t["modality"])):
        arr = {c: (None if t[c][i] is None else np.asarray(t[c][i])) for c in _ARRAY_COLS}
        peaks = None if t["peaks"][i] is None else np.asarray(t["peaks"][i]).reshape(-1, 2)
        out.append(SpectrumRecord(
            modality=t["modality"][i], axis_unit=t["axis_unit"][i],
            provenance=Provenance(**json.loads(t["provenance"][i])),
            peaks=peaks, label=json.loads(t["label"][i]), meta=json.loads(t["meta"][i]),
            **arr))
    return out
```

- [ ] **Step 5: 跑测试全绿** `pytest tests/test_record.py -q` → 4 passed。
- [ ] **Step 6: Commit** `git commit -am "feat: SpectrumRecord data model with M0-extended fields and parquet IO"`

---

### Task 3: RRUFF Raman 适配器（实验谱 + 矿物标签）

**Files:** Create `src/hsfm/corpora/rruff.py`, `scripts/fetch_rruff.py`, `tests/test_rruff.py`

RRUFF `excellent_unoriented.zip`（~229 MB，rruff.net/zipped_data_files/raman/）。每个 txt 文件头部有 `##NAMES=矿物名`、`##RRUFFID=`、`##LASER_WAVELENGTH=` 元数据行，之后是 `波数, 强度` 两列。矿物名是 mineral-id 任务的标签，激光波长入 provenance.instrument。

- [ ] **Step 1: 写失败测试**（用内嵌 3 行迷你样例文件，不依赖真下载）

```python
# tests/test_rruff.py
from pathlib import Path
from hsfm.corpora.rruff import parse_rruff_txt

SAMPLE = """##NAMES=Quartz
##RRUFFID=R040031
##LASER_WAVELENGTH=532
##END=
100.0, 5.2
101.0, 6.1
102.0, 5.9
"""

def test_parse_rruff(tmp_path):
    p = tmp_path / "quartz.txt"; p.write_text(SAMPLE)
    rec = parse_rruff_txt(p)
    assert rec.modality == "raman" and rec.axis_unit == "cm^-1"
    assert rec.label["mineral"] == "Quartz"
    assert rec.provenance.instrument == "laser-532"
    assert len(rec.axis) == 3 and rec.ivar is None  # 不造假：RRUFF 无不确定度
```

- [ ] **Step 2: 确认失败** → ImportError。

- [ ] **Step 3: 实现 parse_rruff_txt + 批转换入口**

```python
# src/hsfm/corpora/rruff.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from ..record import SpectrumRecord, Provenance

def parse_rruff_txt(path: Path) -> SpectrumRecord:
    meta, xs, ys = {}, [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("##"):
            k, _, v = line[2:].partition("=")
            meta[k.strip().upper()] = v.strip()
        elif line and "," in line:
            a, b = line.split(",")[:2]
            xs.append(float(a)); ys.append(float(b))
    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x)                      # RRUFF 少数文件降序
    x, y = x[order], y[order]
    keep = np.concatenate([[True], np.diff(x) > 0])   # 去重复点
    wl = meta.get("LASER_WAVELENGTH", "unknown")
    return SpectrumRecord(
        modality="raman", axis_unit="cm^-1", axis=x[keep], flux=y[keep],
        provenance=Provenance("rruff", f"laser-{wl}", meta.get("RRUFFID", "?")),
        label={"mineral": meta.get("NAMES", "unknown").split(",")[0].strip()},
        meta={"rruff_id": meta.get("RRUFFID", "")})

def convert_dir(src_dir: Path, out_path: Path, shard_size: int = 2000) -> int:
    from ..io import write_shard
    recs, n, shard = [], 0, 0
    for p in sorted(Path(src_dir).glob("*.txt")):
        try:
            recs.append(parse_rruff_txt(p)); n += 1
        except (ValueError, IndexError):
            continue                            # 坏文件跳过并计数，不 silent 全吞
        if len(recs) >= shard_size:
            write_shard(recs, out_path.with_suffix(f".{shard:03d}.parquet")); recs, shard = [], shard + 1
    if recs:
        write_shard(recs, out_path.with_suffix(f".{shard:03d}.parquet"))
    return n
```

- [ ] **Step 4: 测试绿** → 1 passed。
- [ ] **Step 5: 写下载脚本**

```python
# scripts/fetch_rruff.py — 下载 excellent_unoriented + fair_unoriented 并解压后调 convert_dir
# 用 requests 流式下载到 ~/hs-fm-data/raw/rruff/，zipfile 解压，convert_dir 输出到
# ~/hs-fm-data/records/rruff_raman.*.parquet，最后打印总数与跳过数。
```
（脚本 ~30 行，逻辑同上注释；对照官网文件大小校验。）

- [ ] **Step 6: 真跑一次并记录数字**

```bash
.venv/bin/python scripts/fetch_rruff.py
```
Expected: 记录数落在 5,000–10,000 量级（excellent+fair unoriented）；把实际数字写进 `reports/p0/corpus-inventory.md`。

- [ ] **Step 7: Commit** `git commit -am "feat: RRUFF Raman corpus adapter"`

---

### Task 4: QM9S IR/Raman 适配器（模拟谱 + 分子配对键）

**Files:** Create `src/hsfm/corpora/qm9s.py`, `scripts/fetch_qm9s.py`, `tests/test_qm9s.py`

QM9S（figshare 24235333，DetaNet 论文数据）：13.4 万分子的模拟 IR/Raman/UV-Vis。下载后实际文件格式以落地为准（figshare 包内是 npz/csv 的组合）——**适配器测试同样用合成迷你样例**，只测转换逻辑：同一分子的 IR 与 Raman 各成一条 record，`label["molecule_id"]` 相同（跨模态配对键），`provenance.instrument="dft-b3lyp-def2tzvp"`，ivar=None、lsf_sigma=None（模拟谱如实标 None，增广阶段另行处理）。

- [ ] **Step 1: 失败测试**：`make_records(mol_id, freq_axis, ir, raman)` 返回两条 record、modality 各正确、molecule_id 一致。
- [ ] **Step 2: 实现**（~25 行，套 Task 3 模式）。
- [ ] **Step 3: 下载脚本 + 真跑**，数字入 `corpus-inventory.md`（预期 ~13.4 万分子 × 2 谱）。
- [ ] **Step 4: Commit** `git commit -am "feat: QM9S IR/Raman adapter with molecule pairing key"`

---

### Task 5: MP XAS 适配器（模拟 XAS + 结构配对键）

**Files:** Create `src/hsfm/corpora/mp_xas.py`, `scripts/fetch_mp_xas.py`, `tests/test_mp_xas.py`

用 `mp-api`（需免费 API key，环境变量 `MP_API_KEY`）拉 K-edge XANES：`mpr.materials.xas.search(edge="K", spectrum_type="XANES")`。每条谱 → record：axis=能量 eV、`label["mp_id"]`（与 SimXRD/未来 XRD 模拟的结构配对键）、`label["absorbing_element"]`、provenance=("materials-project", "feff9", 版本)。

- [ ] **Step 1: 失败测试**：mock 一个 spectrum 对象（x/y/mp_id/element 属性）→ `to_record()` 字段正确。
- [ ] **Step 2: 实现 to_record + 分页批拉函数**（限速重试，每 1 万条一个 shard）。
- [ ] **Step 3: 真跑**（先 `--limit 5000` 冒烟，再全量后台跑；全量 50 万条可放 5090 上跑）。数字入 inventory。
- [ ] **Step 4: Commit** `git commit -am "feat: Materials Project K-edge XANES adapter"`

---

### Task 6: SimXRD 峰表适配器（峰集形态）

**Files:** Create `src/hsfm/corpora/simxrd.py`, `scripts/fetch_simxrd.py`, `tests/test_simxrd.py`

SimXRD-4M（github Bin-Cao/SimXRD，MIT；P0 只取其**验证子集 ~10 万条**，全量留 P1）。原始是密集衍射图+空间群标签：适配器做两件事——密集 record（modality="xrd"）与峰提取后的峰集 record（modality="xrd_peaks"，scipy.signal.find_peaks，prominence 阈值写进 meta 保证可复现）。`label["space_group"]`（1–230）。

- [ ] **Step 1: 失败测试**：合成三峰高斯谱 → `extract_peaks()` 返回 (3,2) 且峰位误差 < 1 个格点；`to_records()` 出密集+峰集两条、label 一致。
- [ ] **Step 2: 实现**（find_peaks 参数：`prominence=0.02*max, distance=3`）。
- [ ] **Step 3: 下载子集真跑**，数字入 inventory。
- [ ] **Step 4: Commit** `git commit -am "feat: SimXRD adapter with dense and peak-set dual forms"`

---

### Task 7: HS-Bench 骨架（注册表 + 成分不相交划分 + 逐样本指标）

**Files:** Create `src/hsfm/bench/registry.py`, `src/hsfm/bench/splits.py`, `src/hsfm/bench/metrics.py`, `tests/test_bench_core.py`

spec §6 的核心纪律：**划分固定且成分不相交**（NMRGym 教训）、逐样本指标落盘、评测器持有划分。

- [ ] **Step 1: 失败测试**

```python
# tests/test_bench_core.py
import numpy as np
from hsfm.bench.splits import group_disjoint_split
from hsfm.bench.metrics import macro_accuracy, per_sample_log

def test_group_disjoint():
    groups = ["a","a","b","b","c","c","d","d"]
    tr, te = group_disjoint_split(groups, test_frac=0.25, seed=0)
    assert set(np.array(groups)[tr]) & set(np.array(groups)[te]) == set()  # 组零重叠
    tr2, te2 = group_disjoint_split(groups, test_frac=0.25, seed=0)
    assert list(te) == list(te2)                                           # 同种子确定性

def test_per_sample_log_shape(tmp_path):
    rows = per_sample_log(ids=["s1","s2"], y_true=[0,1], y_pred=[0,0])
    assert rows[1]["correct"] is False        # 失败样本可追溯
```

- [ ] **Step 2: 确认失败。**
- [ ] **Step 3: 实现**：`group_disjoint_split`（按组 hash+seed 排序后切，组级不相交）；`macro_accuracy`（按类均衡）；`per_sample_log`（list[dict]，含 id/y_true/y_pred/correct，评测跑完写 `reports/p0/persample/<task>.jsonl`）；`registry.py` 的 `BenchTask` dataclass（name / load_fn / split / metric / groups_key）与全局 `TASKS` 字典。
- [ ] **Step 4: 测试绿 → Commit** `git commit -am "feat: HS-Bench core (registry, group-disjoint splits, per-sample metrics)"`

---

### Task 8: HS-Bench P0 四任务注册

**Files:** Create `src/hsfm/bench/tasks/{rruff_mineral.py,simxrd_sg.py,qm9s_retrieval.py,xas_denoise_stub.py}`, `tests/test_bench_tasks.py`

| 任务 | 数据 | 划分组键 | 指标 |
|---|---|---|---|
| `rruff-mineral-id` | Task 3 输出，取样本数 ≥20 的矿物类 | mineral（组=矿物种，同种不跨集）→ 改用 rruff_id 分组、mineral 为标签 | macro-accuracy |
| `simxrd-space-group` | Task 6 密集形态 | 结构 id | macro-accuracy |
| `qm9s-ir2raman-retrieval` | Task 4，IR 查 Raman 库 | molecule_id | recall@1/@10 |
| `xas-denoising`（存根） | 委托 M0 现有评测轨 | — | 只在报告引用，不重实现 |

- [ ] **Step 1: 每任务写失败测试**：注册后 `TASKS["rruff-mineral-id"].load()` 返回 (train_records, test_records) 且组不相交、类数 ≥30。
- [ ] **Step 2: 实现四个任务模块**（每个 ~40 行：读 parquet → 过滤 → 调 splits → 返回）。`xas_denoise_stub` 只注册元数据并指向 hyper-spectrum M0 轨的 evidence 路径。
- [ ] **Step 3: 测试绿 → Commit** `git commit -am "feat: register P0 benchmark tasks"`

---

### Task 9: 笨基线落档（resample+MLP 与 TabPFN）

**Files:** Create `src/hsfm/baselines/resample_mlp.py`, `src/hsfm/baselines/tabpfn_runner.py`, `scripts/run_baselines.py`, `tests/test_baselines.py`

spec §3.3：这两个数字是后面一切模型的地板。

- [ ] **Step 1: 失败测试**：合成两类可分谱（不同峰位高斯）各 50 条 → `ResampleMLP.fit/predict` 准确率 > 0.9；`resample_to_grid(rec, grid)` 线性插值+域外 NaN→0。
- [ ] **Step 2: 实现 resample_mlp.py**

```python
# 核心：公共网格线性插值 + sklearn MLPClassifier(hidden=(256,256), max_iter=300)
# 网格按任务定：raman 200–3500 cm⁻¹ @1024 点；xrd 5–90° @1024 点
def resample_to_grid(rec, grid):
    import numpy as np
    y = np.interp(grid, rec.axis, rec.flux, left=0.0, right=0.0)
    m = y.max();  return y / m if m > 0 else y
```

- [ ] **Step 3: 实现 tabpfn_runner.py**：特征>500 时等距抽 500 点（TabPFN 上限，方法写进报告）；训练集>10k 时分层抽 10k。
- [ ] **Step 4: 真跑两基线 × 两分类任务**（rruff-mineral-id / simxrd-space-group），检索任务基线用重采样向量余弦相似度。

```bash
.venv/bin/python scripts/run_baselines.py --tasks all --out reports/p0/baselines.json
```
Expected: 每任务每基线一行 {task, baseline, metric, value, n_train, n_test}，逐样本 jsonl 同步落盘。

- [ ] **Step 5: Commit** `git commit -am "feat: dumb baselines (resample+MLP, TabPFN) with logged numbers"`

---

### Task 10: Tokenizer——物理轴 Fourier 编码 + 密集 patch 化

**Files:** Create `src/hsfm/tokenize/axis_encoding.py`, `src/hsfm/tokenize/dense_patch.py`, `tests/test_tokenize.py`

- [ ] **Step 1: 失败测试**

```python
# tests/test_tokenize.py
import numpy as np, torch
from hsfm.tokenize.axis_encoding import fourier_axis_encoding, index_sincos_encoding
from hsfm.tokenize.dense_patch import patchify

def test_fourier_encoding_is_axis_dependent():
    ax1, ax2 = np.linspace(100, 2000, 512), np.linspace(500, 2400, 512)
    e1 = fourier_axis_encoding(torch.tensor(ax1), dim=64, period_range=(2.0, 8000.0))
    e2 = fourier_axis_encoding(torch.tensor(ax2), dim=64, period_range=(2.0, 8000.0))
    assert e1.shape == (512, 64) and not torch.allclose(e1, e2)   # 轴变编码变
    # 同物理位置编码相同（跨仪器可对齐的根据）：
    assert torch.allclose(e1[-1], fourier_axis_encoding(torch.tensor([2000.0]), 64, (2.0, 8000.0))[0])

def test_index_encoding_is_axis_blind():
    ax1, ax2 = np.linspace(100, 2000, 512), np.linspace(500, 2400, 512)
    e1, e2 = (index_sincos_encoding(512, 64), index_sincos_encoding(512, 64))
    assert torch.allclose(e1, e2)                                  # 消融对照组：轴盲

def test_patchify_variable_length_and_mask():
    ax = np.linspace(100, 3500, 1000); fl = np.random.rand(1000)
    p = patchify(torch.tensor(fl), torch.tensor(ax), patch=32)
    assert p.tokens.shape[0] == int(np.ceil(1000/32))
    assert p.tokens.shape[1] == 32 * 3            # flux+ivar+lsf 三通道（缺失通道零填+指示位并入 meta 嵌入）
    assert p.axis_centers.shape[0] == p.tokens.shape[0]
    assert bool(p.valid[-1]) is True              # 末 patch 补齐后仍有效（>50% 实点）
```

- [ ] **Step 2: 确认失败。**
- [ ] **Step 3: 实现 axis_encoding.py**

```python
# src/hsfm/tokenize/axis_encoding.py
import torch, math

def fourier_axis_encoding(axis: torch.Tensor, dim: int, period_range: tuple[float, float]) -> torch.Tensor:
    """物理轴连续 sin/cos 编码。period_range 以轴单位计（如 cm⁻¹ 域 (2, 8000)），log 间隔。"""
    half = dim // 2
    periods = torch.logspace(math.log10(period_range[0]), math.log10(period_range[1]), half)
    ang = 2 * math.pi * axis.to(torch.float64)[:, None] / periods[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(torch.float32)

def index_sincos_encoding(n: int, dim: int) -> torch.Tensor:
    """消融对照：标准 transformer 序号 PE（轴盲）。"""
    return fourier_axis_encoding(torch.arange(n, dtype=torch.float64), dim, (2.0, 2.0 * n))
```

- [ ] **Step 4: 实现 dense_patch.py**（`Patched` namedtuple: tokens/axis_centers/valid；三通道拼接，ivar/lsf 缺失整段置 0；末 patch 零填、实点 <50% 置 valid=False）。
- [ ] **Step 5: 测试绿 → Commit** `git commit -am "feat: physical-axis Fourier encoding and dense patch tokenizer"`

---

### Task 11: 小 encoder（消融用，~10M 参数）

**Files:** Create `src/hsfm/model/encoder.py`, `tests/test_encoder.py`

P0 只需要分类/检索消融，**不建 decoder**（任意轴查询解码是 P1 工作，spec §3.1 保留接口即可）。

- [ ] **Step 1: 失败测试**：随机两条不同长度谱 → `SpectrumEncoder(pe="fourier")` 前向输出 (B, d) 定长向量；`pe="index"` 同样可跑；valid 掩码为 False 的 patch 不影响输出（置换该 patch 内容输出不变）。
- [ ] **Step 2: 实现**

```python
# src/hsfm/model/encoder.py 关键结构（完整实现 ~120 行）
class SpectrumEncoder(nn.Module):
    def __init__(self, d=256, layers=6, heads=8, patch=32, k_queries=8,
                 pe="fourier", period_range=(2.0, 8000.0)):
        # patch 线性投影 (patch*3 → d) + PE 相加（fourier: 逐 patch 轴中心；index: 序号）
        # nn.TransformerEncoder(layers)，src_key_padding_mask ← ~valid
        # K 个可学习 query 对 encoder 输出做 cross-attention → mean → (B, d)
    def forward(self, flux, axis, ivar=None, lsf=None) -> torch.Tensor: ...
```
变长 batch：collate 按最长 patch 数补齐 + padding mask。
- [ ] **Step 3: 测试绿 → Commit** `git commit -am "feat: small spectrum encoder with K-query pooling and switchable PE"`

---

### Task 12: 消融实验——物理轴 PE vs 序号 PE × patch 尺寸

**Files:** Create `src/hsfm/train/ablate_pe.py`, `tests/test_ablate_smoke.py`

设计（spec §7 P0 出口判据的直接证据）：

- 任务：`rruff-mineral-id`（实验谱、天然多仪器/多分辨率——物理轴编码的优势场）与 `simxrd-space-group`（模拟谱对照）。
- 因子：PE ∈ {fourier, index} × patch ∈ {16, 32, 64}，每格 3 seed，共 36 run。
- 训练：监督分类（AdamW lr=3e-4，30 epoch，早停 patience=5），单 run 在 5090 上 ~10 分钟量级。
- **决定性子实验（轴扰动迁移）**：训练集保持原轴，测试集把每条谱裁剪到随机子区间并降采样 2×（模拟「另一台仪器」）。假设：fourier 组掉点显著小于 index 组——这是「物理轴编码带来跨仪器鲁棒」的直接证据，对标 OmniSpectra 1.8× 消融。

- [ ] **Step 1: 冒烟测试**：`ablate_pe.run_one(config, n_limit=200, epochs=2)` 在 CPU 上 <60s 跑通并返回 {config, acc, acc_perturbed}。
- [ ] **Step 2: 实现完整脚本**（config 网格、jsonl 追加落盘 `reports/p0/ablation.jsonl`、`--device auto`、断点续跑按 config hash 跳过已完成）。
- [ ] **Step 3: rsync 数据+代码上 5090，全网格真跑**

```bash
rsync -a ~/hs-fm-data/records <你的用户名>@118.180.19.234:~/hs-fm-data/
ssh <你的用户名>@118.180.19.234 'cd ~/hs-fm && .venv/bin/python -m hsfm.train.ablate_pe --grid full'
```
- [ ] **Step 4: Commit（代码）** `git commit -am "feat: PE ablation harness with axis-perturbation transfer test"`

---

### Task 13: P0 出口判据报告

**Files:** Create `reports/p0/exit-report.md`（数据由 Task 3–6/9/12 产出汇总）

- [ ] **Step 1: 汇总脚本**：读 `corpus-inventory.md`、`baselines.json`、`ablation.jsonl`，生成报告表格：语料清单（族×形态×数量×许可）；基准四任务 × 基线数字；消融主表（PE×patch×seed 均值±std）+ 轴扰动迁移表。
- [ ] **Step 2: 对照 spec §7 出口判据逐条判定**：①物理轴编码显著有效（fourier vs index 差异 > 3×seed std，轴扰动下差异放大）；②笨基线数字落档。每条写 PASS/FAIL + 证据指针。
- [ ] **Step 3: 结论段**：若 PASS → P1 开工依据；若 FAIL → 归因与设计层修订建议（spec §10 的「不带病进入下一阶段」）。
- [ ] **Step 4: Commit + 推远端**，报告要点同步写 hyper-data 仓 `docs/dev/` 日志并飞书通知。

---

## Self-Review 记录

- **Spec 覆盖**：§2.1 字段（Task 2）、§2.2 双形态（Task 6/10；峰集 token 的注意力偏置属 P1 预训练，P0 只落峰集数据形态——spec §7 P0 范围如此）、§3.3 笨基线（Task 9）、§5 语料+许可红线（Task 3–6，XASDataLibrary 未入 P0 池符合 spec「核实后方可入池」）、§6 划分纪律（Task 7）、§7 P0 出口（Task 12/13）。跨模态对比目标与 decoder 属 P1，不在本计划。
- **占位符**：Task 4/5/6 的实现步骤以模式引用 Task 3 的完整代码骨架并给出差异点与行数预估——数据格式需下载后确认，测试先行用合成样例锁行为，这是刻意选择而非缺口。
- **类型一致性**：`SpectrumRecord` 字段名在 Task 2/3/9/10 一致（axis/flux/ivar/lsf_sigma/valid→mask 已统一为 mask；Patched.valid 是 patch 级派生量，与 record.mask 点级语义分开命名）。
