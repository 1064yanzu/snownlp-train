# SnowNLP 情感分析训练工具集

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个完整的 **SnowNLP 情感分析模型** 训练、测试和评估工具集。支持图形界面、Web界面和命令行三种使用方式，适用于各种运行环境。

---

## ⚡ 快速开始（30秒上手）

```bash
# 1. 安装依赖
pip install pandas snownlp tqdm matplotlib numpy scikit-learn

# 2. 启动工具（自动检测最佳界面）
python 启动工具.py
```

---

## 📦 安装

### 系统要求
- **Python**: 3.6 或更高版本
- **操作系统**: Windows / macOS / Linux
- **GUI支持**: 图形界面需要 Tkinter（大多数系统预装）

### 依赖安装

```bash
# 基础依赖
pip install pandas snownlp tqdm

# 可视化依赖（可选，用于图表显示）
pip install matplotlib numpy scikit-learn

# Web界面依赖（可选）
pip install flask
```

> 💡 首次运行时，启动器会自动检测并提示安装缺失的依赖。

---

## 🛠️ 工具说明

### 主要工具

| 工具 | 使用场景 | 启动命令 |
|------|---------|---------|
| **启动工具.py** | 🎯 推荐入口，自动选择最佳界面 | `python 启动工具.py` |
| **SnowNLP训练测试工具.py** | 完整GUI界面，功能全面 | `python SnowNLP训练测试工具.py` |
| **Web界面工具.py** | 无GUI环境（服务器、云环境） | `python Web界面工具.py` |
| **命令行训练工具.py** | 自动化和脚本集成 | `python 命令行训练工具.py` |
| **测试新模型.py** | 详细模型测试和评估 | `python 测试新模型.py` |
| **快速验证.py** | 快速检查模型效果 | `python 快速验证.py` |

### 选择指南

```
┌────────────────────────────────────────────────────────┐
│                  你的使用环境是？                       │
├────────────────────────────────────────────────────────┤
│                                                        │
│   有图形界面（Windows/macOS/Linux桌面）                │
│       → 运行 python 启动工具.py                        │
│       → 自动启动 GUI 界面                              │
│                                                        │
│   无图形界面（Linux服务器/云环境）                     │
│       → 运行 python Web界面工具.py                     │
│       → 浏览器访问 http://localhost:5000               │
│                                                        │
│   需要自动化/脚本集成                                  │
│       → 运行 python 命令行训练工具.py --help           │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 📊 数据格式

### 训练/测试数据（CSV）

工具统一读取 **CSV 表格**，每一行代表一条样本。

- **必需列**：
  - `content`：文本内容
  - `sentiment`：标签
- **列名兼容（可直接复用公开数据集）**：
  - `content` 也兼容：`text` / `review` / `comment` / `sentence` / `内容` / `文本` / `评论`
  - `sentiment` 也兼容：`label` / `class` / `情感` / `标签`

#### CSV 示例（仅为“格式示意”，不是实际数据）

```csv
content,sentiment
<一条中文文本>,正面
<一条中文文本>,负面
<一条中文文本>,中性
```

### 支持的标签

| 情感 | 支持的标签值 |
|------|-------------|
| **正面** | 正面、积极、正向、positive、1 |
| **负面** | 负面、消极、负向、negative、0 |
| **中性** | 中性、中立、neutral、2 |

#### 数字标签说明

- `1` 表示正面
- `0` 表示负面
- `2` 表示中性（是否参与训练取决于“中性数据处理策略”）

> 💡 你可以直接使用公开数据集中常见的 `label/review` 字段：工具会自动把 `review` 识别为 `content`，把 `label` 识别为 `sentiment`。

### 文件命名与放置位置

启动器会在项目根目录自动查找：

- 训练文件：`train.csv` / `训练*.csv` / `*train*.csv`
- 测试文件：`test.csv` / `测试*.csv` / `*test*.csv`

建议：

- **训练集**：`train.csv`
- **测试集**：`test.csv`

### 编码与分隔符兼容策略（已实现）

#### 编码（自动回退）

会按顺序自动尝试：

- `utf-8-sig`（Excel 导出常见，带 BOM）
- `utf-8`
- `gbk`
- `gb2312`
- `gb18030`

#### 分隔符（自动识别）

会从文件前若干行自动识别，支持：

- 逗号 `,`
- 制表符 `\t`
- 分号 `;`
- 竖线 `|`

> 建议优先使用逗号 `,` 或制表符 `\t`。

### 推荐数据来源（真实来源）

如果你没有自己的标注数据，可以从公开中文情感数据集开始：

- **ChineseNlpCorpus / ChnSentiCorp_htl_all（酒店评论二分类）**
  - 主页：https://github.com/SophonPlus/ChineseNlpCorpus/tree/master/datasets/ChnSentiCorp_htl_all
  - CSV 下载（原始字段为 `label,review`）：https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv

该数据集的字段：

- `review`：评论文本（工具会自动识别为 `content`）
- `label`：0/1 标签（工具会自动识别为 `sentiment`，其中 `1=正面`, `0=负面`）

你可以直接将其重命名为 `train.csv` 放到项目根目录进行训练；测试集可以从中另存一份为 `test.csv`（例如随机抽样）。

#### 从 ChnSentiCorp 生成可训练数据（推荐流程）

1) 下载原始 CSV（字段为 `label,review`）：

```bash
curl -L "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv" -o ChnSentiCorp_htl_all.csv
```

2) 转换列名并切分为 `train.csv` / `test.csv`（80/20，随机种子固定便于复现）：

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("ChnSentiCorp_htl_all.csv")  # 原始列名: label,review
df = df.rename(columns={"review": "content", "label": "sentiment"})

# 清洗：去掉空文本/空标签
df["content"] = df["content"].astype(str).str.strip()
df = df[df["content"].ne("")]
df = df.dropna(subset=["sentiment"])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(df) * 0.8)
train_df = df.iloc[:split]
test_df = df.iloc[split:]

train_df.to_csv("train.csv", index=False, encoding="utf-8")
test_df.to_csv("test.csv", index=False, encoding="utf-8")

print("train.csv rows:", len(train_df))
print("test.csv  rows:", len(test_df))
PY
```

3) 运行工具：

```bash
python 启动工具.py
```

> 如果你不想在本地下载原始文件，也可以在浏览器中手动下载后放到项目根目录，再执行第 2 步。

### 常见注意事项（非常重要）

- **不要有空列/缺失值**：`content` 或 `sentiment` 为空的行会被跳过。
- **文本尽量不要包含未转义换行**：如果 `content` 中包含换行但 CSV 没有正确引号包裹，解析会失败。
- **优先使用 UTF-8 导出**：工具支持多编码回退，但统一用 UTF-8 最省心。
- **分隔符建议用逗号**：虽然支持自动识别 `, \t ; |`，但最通用仍是 `,`。
- **标签要统一**：同一个文件里混用 `正面/positive/1` 是允许的，但请确保语义一致。
- **中性数据**：
  - 若你提供 `中性/neutral/2`，工具会按照你选择的“中性数据处理策略”决定是否丢弃或分配到正负类。
  - 如果你做的是二分类训练且不希望引入中性，请选择“排除中性”。

---

## 🚀 使用流程

### GUI界面流程

1. **启动程序**: 运行 `python 启动工具.py`
2. **选择数据**: 浏览或自动查找 CSV 数据文件
3. **配置训练**: 选择中性数据处理策略（推荐"自动平衡"）
4. **开始训练**: 点击"开始训练"，观察实时进度
5. **测试验证**: 训练完成后进行测试验证
6. **使用新模型**: **重启程序**后使用新训练的模型

### 命令行流程

```bash
# 训练模型（交互式）
python 命令行训练工具.py

# 训练模型（指定参数）
python 命令行训练工具.py --train --neutral-strategy balance

# 快速验证
python 快速验证.py

# 完整测试
python 测试新模型.py
```

---

## 🔧 核心功能

### 训练功能
- ✅ 自动编码检测（UTF-8-SIG/UTF-8/GBK/GB2312/GB18030）
- ✅ 自动识别常见 CSV 分隔符（逗号/制表符/分号/竖线）
- ✅ 列名兼容（`content/sentiment`、`review/label` 等）
- ✅ 6种中性数据处理策略
- ✅ 实时训练进度显示
- ✅ 自动备份原始模型
- ✅ 绕过 SnowNLP 原生保存问题

### 测试功能
- ✅ 快速验证（5个基础测试）
- ✅ 完整测试（20个精心设计用例）
- ✅ 数据集评估（大规模测试）
- ✅ 交互式测试（实时输入）
- ✅ 模型对比分析

### 中性数据处理策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **自动平衡** | 智能分配到正负类 | 🎯 推荐，通用场景 |
| **忽略中性** | 只用正负样本训练 | 二分类需求 |
| **改成正面** | 中性全部当正面 | 乐观评估场景 |
| **改成负面** | 中性全部当负面 | 保守评估场景 |
| **随机分配** | 随机分到正负类 | 数据量充足时 |

---

## 📈 评估标准

| 准确率 | 等级 | 建议 |
|--------|------|------|
| ≥80% | 🎉 优秀 | 模型表现很好 |
| ≥60% | 👍 良好 | 可以使用 |
| ≥40% | 😐 一般 | 建议优化数据 |
| <40% | 😞 较差 | 建议重新训练 |

---

## ❓ 常见问题

### Q: 训练后模型没有变化？
**A**: 需要**重启Python解释器**。SnowNLP 会缓存模型，重启后才会加载新模型。

### Q: 数据文件读取失败？
**A**: 优先检查下面几项：

- **列名是否正确**：需要 `content` + `sentiment`（也兼容 `review/label` 等别名）
- **是否有换行破坏 CSV**：文本列如果包含未转义的换行，CSV 会被解析成多行
- **编码问题**：工具会自动尝试 `utf-8-sig/utf-8/gbk/gb2312/gb18030`，如果仍失败，建议用 UTF-8 重新导出
- **分隔符问题**：工具会自动识别 `, \t ; |`，但极端情况下建议手动改为逗号分隔

### Q: 训练效果不好？
**A**: 请检查：
1. 数据质量：标签是否准确
2. 数据平衡：正负样本比例是否合理
3. 尝试不同的中性数据处理策略

### Q: 无法显示图形界面？
**A**: 
- Linux 服务器：使用 `Web界面工具.py`
- 缺少 Tkinter（提示 `No module named '_tkinter'`）：
  - macOS + Homebrew Python 3.12：`brew install python-tk@3.12`
  - 其它环境：使用系统包管理器安装 Tcl/Tk（或改用命令行/网页界面）

---

## 📁 项目结构

```
snownlp-train/
├── 启动工具.py              # 🎯 推荐入口
├── SnowNLP训练测试工具.py   # 完整GUI工具
├── Web界面工具.py           # Web界面版本
├── 命令行训练工具.py        # 命令行版本
├── 测试新模型.py            # 模型测试工具
├── 快速验证.py              # 快速验证
├── model_history.json       # 模型历史配置
├── temp_data/               # 临时训练数据
├── test_data/               # 测试数据
└── deprecated/              # 归档的开发工具
```

---

## 🔬 技术说明

### 核心原理
- 基于 **SnowNLP** 的朴素贝叶斯情感分类器
- 使用 **jieba** 进行中文分词
- 通过阈值实现三分类（正面/中性/负面）

### 模型保存机制
工具通过直接操作 SnowNLP 内部模型文件来解决原生 `save()` 方法的问题，确保训练后的模型能正确保存和加载。

### 文件位置
训练后的模型会替换 SnowNLP 安装目录下的 `sentiment/sentiment.marshal.3` 文件。原始文件会自动备份。

---

## 📝 更新日志

### v3.0 (当前版本)
- 🎮 全新图形界面设计
- 🚀 智能启动器
- 📊 实时进度监控
- 🧪 多种测试模式
- 🌐 Web界面支持
- 📚 完整文档

---

## 📜 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

**🎉 感谢使用 SnowNLP 训练工具集！**

如有问题或建议，欢迎反馈。