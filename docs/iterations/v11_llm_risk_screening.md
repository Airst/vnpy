# V11 Step 1: LLM 事件风险排查 + 股票黑名单机制

> 创建日期：2026-04-24
> 目标：在模型截面信号基础上，通过 LLM 识别模型无法感知的基本面/事件风险，建立黑名单机制自动排除问题股票

---

## 一、问题诊断

### 1.1 核心矛盾

4 月分析揭示了截面模型的系统性边界：

| 股票 | 模型信号 | 市场真相 | 模型盲区 |
|:---|:---|:---|:---|
| 603209 兴通股份 | 1.73（高分） | 横盘 +1.86%，扣非净利-14% | 非经常性损益美化业绩 |
| 002486 嘉麟杰 | 无信号（已剔除） | -6.12% | 实控人被拘留 |
| 002453 华软科技 | 无信号（已剔除） | -8.70% | 连续亏损+基本面恶化 |
| 000862 银星能源 | 1.68（上升趋势） | +4.77% 6连阳 | 绿电政策+国企改革催化 |

**根本问题**：截面因子模型是历史模式识别器，无法感知：
1. 公司治理风险（实控人负面事件、监管处罚）
2. 盈利质量（非经常性损益 vs 主业利润）
3. 行业/政策催化（绿电政策、资产注入）

这些信息的共同点：它们不存在于日频 OHLCV 和财务比率中，存在于**公告和新闻**中。

### 1.2 解决思路

LLM 不替代模型——模型管截面选股，LLM 管事件风险排查。两者职责正交。

LLM 输出两类决策：
- **黑名单**（blacklist）：自动排除，默认 1 个月后到期
- **关注**（watch）：降权但不排除

---

## 二、架构设计

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                              │
│  training.py                                                 │
│    │                                                         │
│    ├── Step 0: Load blacklist → filter candidates            │
│    │           ↓                                              │
│    │   FundamentalSelector.get_candidate_symbols()            │
│    │   → 自动排除 blacklisted stocks                          │
│    │                                                         │
│    ├── Step 1-N: Normal training (clean dataset)             │
│    │                                                         │
│    └── Step N+1: Generate signals → save to parquet          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 SIGNAL POST-PROCESSING                        │
│                 (daily, before strategy execution)            │
│                                                              │
│  signal_processor.py                                         │
│    │                                                         │
│    ├── Load latest signals + blacklist                       │
│    │                                                         │
│    ├── Filter: remove blacklisted stocks                     │
│    │                                                         │
│    ├── For top-N candidates (score > threshold):             │
│    │   │                                                     │
│    │   ├── [MANUAL mode] Export task file                    │
│    │   │   → User opens OpenClaw, installs tushare skill     │
│    │   │   → OpenClaw analyzes each stock                    │
│    │   │   → Saves result JSON                               │
│    │   │   → signal_processor reads results                  │
│    │   │                                                     │
│    │   ├── [AUTO mode] HTTP call to OpenClaw API             │
│    │   │   → OpenClaw uses tushare + web-search skills       │
│    │   │   → Returns structured risk assessment              │
│    │   │                                                     │
│    │   └── Apply: blacklist/watch flags to signal scores     │
│    │                                                     │
│    └── Output filtered signals for strategy execution        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 文件 | 职责 |
|:---|:---|:---|
| **BlacklistManager** | `core/blacklist_manager.py` | 黑名单 CRUD、过期管理、持久化 |
| **OpenClawClient** | `core/llm/openclaw_client.py` | OpenClaw HTTP API 客户端 |
| **RiskScreener** | `core/llm/risk_screener.py` | LLM 风险排查编排逻辑 |
| **Prompts** | `core/llm/prompts.py` | LLM 提示词模板 |
| **SignalProcessor** | `core/signal_processor.py` | 信号后处理过滤器 |

### 2.3 数据流（手动模式，先实现这个）

```
Training 完成
    │
    ├── signal_processor.py exports task file:
    │   core/alpha_db/llm_tasks/task_2026-04-24.json
    │   {
    │     "task_type": "risk_screening",
    │     "stocks_to_check": [
    │       {"vt_symbol": "603209.SSE", "score": 1.73, "reason": "top signal"},
    │       ...
    │     ],
    │     "date": "2026-04-24"
    │   }
    │
    ▼
用户操作 OpenClaw:
    ├── Load task file
    ├── For each stock:
    │   ├── tushare: 获取近30天公告
    │   ├── tushare: 获取最新财务指标
    │   └── LLM 分析 → 风险评估
    └── Save result:
        core/alpha_db/llm_tasks/result_2026-04-24.json
        {
          "results": [
            {
              "vt_symbol": "603209.SSE",
              "risk_level": "watch",
              "reason": "扣非净利同比-13.98%，卖船收益美化报表",
              "confidence": 0.85,
              "sources": ["Q1 2026 财报", "业绩说明会公告"],
              "action": {
                "blacklist": false,
                "signal_multiplier": 0.5,
                "comment": "主业承压但无重大风险，降权处理"
              }
            }
          ]
        }
    │
    ▼
signal_processor.py reads result_2026-04-24.json
    ├── Apply signal_multiplier to affected stocks
    ├── Add blacklisted stocks to blacklist.json
    ├── Save filtered signals
    └── Strategy uses filtered signals
```

---

## 三、方案设计（会改/新增的文件和详细设计）

### 3.1 文件清单

```
新增:
  core/blacklist_manager.py       # 黑名单管理器
  core/alpha_db/blacklist.json    # 黑名单数据文件（初始）
  core/alpha_db/llm_tasks/        # LLM 任务/结果目录
  core/llm/__init__.py
  core/llm/openclaw_client.py     # OpenClaw 客户端
  core/llm/prompts.py             # 提示词模板
  core/llm/risk_screener.py       # 风险排查编排

修改:
  core/selector/selector.py       # FundamentalSelector 集成黑名单
  core/alpha/engine.py            # Engine 训练前过滤黑名单
  training.py                     # 可选的 signal_processor 集成
```

---

### 3.2 黑名单管理器 (`core/blacklist_manager.py`)

```python
class BlacklistManager:
    """
    Stock blacklist with automatic expiry.
    
    Features:
    - CRUD operations for blacklist entries
    - Automatic expiry (default 30 days)
    - Persistence to JSON file
    - Manual override (add/remove/clear)
    - Support for both LLM-auto and manual entry
    """
    
    DEFAULT_EXPIRY_DAYS = 30
    
    def __init__(self, data_path: str = "core/alpha_db/blacklist.json"):
        self.data_path = data_path
        self.entries: Dict[str, BlacklistEntry] = {}
        self._load()
    
    def add(self, vt_symbol, reason, source="manual", expiry_days=30):
        """Add stock to blacklist. Returns entry."""
    
    def remove(self, vt_symbol):
        """Manually remove stock from blacklist."""
    
    def is_blacklisted(self, vt_symbol, date=None) -> bool:
        """Check if stock is currently blacklisted."""
    
    def get_active_blacklist(self, date=None) -> List[str]:
        """Get list of currently blacklisted vt_symbols."""
    
    def get_all_entries(self) -> List[BlacklistEntry]:
        """Get all entries (including expired) for audit."""
    
    def expire_check(self, date=None):
        """Mark expired entries, keep for history."""
    
    def to_list(self) -> List[dict]:
        """Export as list for LLM / API use."""
```

**BlacklistEntry 数据结构：**
```python
@dataclass
class BlacklistEntry:
    vt_symbol: str          # "002486.SZSE"
    reason: str             # "实控人李兆廷被公安局拘留"
    source: str             # "llm_auto" | "manual"
    added_date: str         # "2026-04-24"
    expiry_date: str        # "2026-05-24"
    priority: str           # "blacklist" | "watch"
```

**blacklist.json 文件格式：**
```json
{
  "version": 1,
  "last_updated": "2026-04-24T10:30:00",
  "entries": {
    "002486.SZSE": {
      "vt_symbol": "002486.SZSE",
      "reason": "实控人李兆廷被公安局拘留，公司存在重大治理风险",
      "source": "llm_auto",
      "added_date": "2026-04-24",
      "expiry_date": "2026-05-24",
      "priority": "blacklist"
    }
  }
}
```

---

### 3.3 Selector 集成黑名单

修改 `FundamentalSelector`，在两个位置加入黑名单过滤：

**位置 1：`get_candidate_symbols()` — 选股池级别过滤**

```python
# 现有: return symbols (主板 stocks)
# 新增: 排除 blacklisted stocks

def get_candidate_symbols(self) -> List[str]:
    symbols = [...]  # 现有逻辑
    
    # NEW: 过滤黑名单
    from core.blacklist_manager import BlacklistManager
    bm = BlacklistManager()
    active_blacklist = set(bm.get_active_blacklist())
    
    symbols = [s for s in symbols if s not in active_blacklist]
    
    return symbols
```

效果：黑名单股票从根源上不出现在训练数据中。下一次 `python training.py -v9 -t` 时自动生效，模型永远不会学到这些股票的因子模式。

**位置 2：`filter_polars()` — DataFrame 级别过滤**

```python
@staticmethod
def filter_polars(df: pl.DataFrame, blacklist: List[str] = None) -> pl.DataFrame:
    # 现有过滤: EP>0, turnover>1%, ln_cap>=11.5
    
    # NEW: 黑名单过滤
    if blacklist and "vt_symbol" in df.columns:
        df = df.filter(~pl.col("vt_symbol").is_in(blacklist))
    
    return df
```

---

### 3.4 Engine 集成黑名单

修改 `AlphaEngine.start_training()` 或相应的数据准备阶段：

```python
# 在准备训练数据集之前
from core.blacklist_manager import BlacklistManager
bm = BlacklistManager()
active_blacklist = bm.get_active_blacklist()

# 从数据集中移除黑名单股票
df_calc = df_calc.filter(~pl.col("vt_symbol").is_in(active_blacklist))
```

---

### 3.5 OpenClaw 客户端 (`core/llm/openclaw_client.py`)

OpenClaw 是一个本地 LLM 编排工具，类似桌面 AI 助手。设计两种交互模式：

**模式 A：HTTP API 调用（自动化）**

```python
class OpenClawClient:
    """
    HTTP client for OpenClaw local LLM orchestrator.
    
    OpenClaw endpoints (假设):
    - POST /api/task         提交分析任务
    - GET  /api/task/{id}    查询任务状态
    - GET  /api/task/{id}/result  获取结果
    """
    
    def __init__(self, endpoint="http://localhost:8910", api_key=None):
        self.endpoint = endpoint
        self.api_key = api_key
    
    def submit_risk_screening(self, stocks: List[dict]) -> str:
        """提交风险排查任务，返回 task_id"""
    
    def get_result(self, task_id: str) -> dict:
        """获取分析结果"""
    
    def screen_stocks(self, stocks: List[dict], timeout=300) -> List[dict]:
        """同步执行：提交 → 等待 → 返回结果"""
```

**模式 B：文件交互（手动/离线）**

```python
class TaskFileExporter:
    """导出任务文件供 OpenClaw 手动操作"""
    
    def export_risk_screening_task(
        self, 
        stocks: List[dict], 
        output_path: str
    ) -> str:
        """生成 JSON 任务文件"""
        task = {
            "task_type": "risk_screening",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "instructions": "对以下股票进行事件风险排查。使用 tushare 获取公告/财务数据，分析是否存在风险。",
            "stocks": [
                {
                    "vt_symbol": s["vt_symbol"],
                    "current_score": s.get("score"),
                    "check_items": [
                        "近30天公告中是否有监管处罚、高管变动、诉讼",
                        "最新季度扣非净利润同比变化",
                        "是否有非经常性损益美化业绩",
                        "大股东是否减持/质押异常",
                        "行业政策是否有负面变化"
                    ]
                }
                for s in stocks
            ],
            "output": {
                "format": "json",
                "schema": {
                    "vt_symbol": "string",
                    "risk_level": "normal | watch | blacklist",
                    "reason": "string (1-2 sentences in Chinese)",
                    "confidence": "float (0-1)",
                    "action": {
                        "blacklist": "boolean",
                        "signal_multiplier": "float (0=exclude, 0.5=reduce, 1.0=normal)",
                        "expiry_days": "int (default 30)"
                    }
                },
                "save_path": f"core/alpha_db/llm_tasks/result_{datetime.now().strftime('%Y-%m-%d')}.json"
            }
        }
```

---

### 3.6 LLM 提示词模板 (`core/llm/prompts.py`)

```python
RISK_SCREENING_SYSTEM = """你是一个 A 股量化选股系统的风险排查助手。你的任务是分析给定股票的近期公告和财务数据，识别模型无法捕捉的事件风险。

## 排查维度（按重要性排序）

1. **公司治理风险**：实控人/高管被查、监管处罚、信息披露违规、股权冻结
2. **盈利质量**：扣非净利润 vs 归母净利润差异 >20% = 非经常性损益依赖
3. **主业恶化**：扣非净利润同比连续 2 季度下降 >15%
4. **财务异常**：应收账款/存货异常增长、经营现金流为负
5. **股东行为**：大股东减持、股权质押比例 >50%
6. **行业风险**：行业政策重大利空、关税影响

## 输出规则

- risk_level: "normal"（无风险）/ "watch"（关注，降权 0.5）/ "blacklist"（排除）
- blacklist 判断标准：触及以上维度第 1 或第 5 项直接 blacklist；第 2+3+4 同时触发时 blacklist
- watch 判断标准：触及以上维度第 2 或第 3 项单项时 watch
- 必须有具体数据支撑（公告日期、财务数字）
- confidence < 0.6 时默认降为 watch 而非 blacklist
"""
```

---

### 3.7 信号处理器 (`core/signal_processor.py`)

```python
class SignalProcessor:
    """信号后处理器：应用黑名单和 LLM 风险排查结果"""
    
    def __init__(self, blacklist_manager, openclaw_client=None):
        self.bm = blacklist_manager
        self.ocl = openclaw_client
    
    def process_signals(
        self, 
        signal_df: pl.DataFrame, 
        mode: str = "manual",  # manual | auto | skip
        top_n: int = 10,
        min_score: float = 1.5
    ) -> pl.DataFrame:
        """
        处理信号数据：
        1. 过滤黑名单股票
        2. 对 top-N 高分股票进行 LLM 风险排查
        3. 应用信号调整系数
        
        Returns: filtered signal_df
        """
    
    def _auto_screen(self, candidates: List[dict]) -> List[dict]:
        """通过 OpenClaw API 自动排查"""
    
    def _export_task_file(self, candidates: List[dict]) -> str:
        """导出任务文件供手动操作"""
    
    def _import_results(self, result_path: str) -> List[dict]:
        """导入 LLM 分析结果"""
    
    def _apply_results(self, signal_df, results):
        """将风险排查结果应用到信号：降权或剔除"""
```

**信号调整规则：**
```python
def apply_risk_actions(signal_df, results):
    for r in results:
        if r["risk_level"] == "blacklist":
            # 加入黑名单，信号分数置 0
            bm.add(r["vt_symbol"], r["reason"], source="llm_auto")
            signal_df = signal_df.filter(pl.col("vt_symbol") != r["vt_symbol"])
        elif r["risk_level"] == "watch":
            # 降权：signal *= multiplier
            signal_df = signal_df.with_columns(
                pl.when(pl.col("vt_symbol") == r["vt_symbol"])
                .then(pl.col("final_signal") * r["signal_multiplier"])
                .otherwise(pl.col("final_signal"))
                .alias("final_signal")
            )
```

---

### 3.8 手动操作工作流

```
# 1. 训练完成后，运行信号处理（手动模式）
python core/signal_processor.py --mode manual --top-n 10 --min-score 1.5

# 输出:
#   → core/alpha_db/llm_tasks/task_2026-04-24.json
#   → print: "请用 OpenClaw 打开 task_2026-04-24.json 进行分析"

# 2. 用户在 OpenClaw 中：
#   - 打开 task_2026-04-24.json
#   - OpenClaw 按 stocks 列表逐一分析
#   - 使用已安装的 tushare skill 获取公告/财务
#   - LLM 分析并生成 result JSON
#   - 保存到 core/alpha_db/llm_tasks/result_2026-04-24.json

# 3. 导入结果
python core/signal_processor.py --mode import --result result_2026-04-24.json

# 输出:
#   → 更新 blacklist.json
#   → 输出 filtered signals
#   → print 每只股票的处理结果
```

---

## 四、验收标准

1. **黑名单持久化**：新增条目持久化到 `blacklist.json`，重启不丢失
2. **自动过期**：30 天后自动标记为 expired，不参与过滤
3. **训练自动排除**：`python training.py -v9 -t` 时自动跳过黑名单股票
4. **信号过滤**：策略执行前黑名单股票的信号分数为 0
5. **手动工作流**：导出的任务文件能被 OpenClaw 正确解析
6. **结果导入**：OpenClaw 生成的结果 JSON 能被正确导入并应用
7. **向后兼容**：不加参数时，所有新功能不生效，现有流程不受影响

---

## 五、风险与边界

| 风险 | 对策 |
|:---|:---|
| LLM 幻觉（无中生有的风险） | 提示词严格要求引用公告日期和财务数字；confidence<0.6 降级 |
| 过度剔除（blacklist 过多导致空仓） | blacklist 上限为选股池的 10%；超限时只保留最严重的 |
| OpenClaw API 不稳定 | 手动模式作为兜底；API 超时 5 分钟自动降级为 export 模式 |
| 黑名单信息过时 | 默认 30 天过期；手动可提前解除 |
| 回测无法复现（手动操作不可复现） | 黑名单数据持久化，回测时可读 historical blacklist 状态 |
