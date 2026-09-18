# Internationalization and language-aware planning

The locale resolver checks `NMESH_LANG`, `LC_ALL`, `LC_MESSAGES`, `LANG`, and
then `locale.getlocale()`, in that order. It normalizes the primary language
subtag and currently supports English and Japanese. Translation keys have an
English and Japanese entry; missing keys and formatting parameters are
failure-tolerant.

The planner remains pure with respect to the environment. `Policy.lang`
controls the language used to render planner warnings, while
`Policy.languages` contains an explicit, optional model-language preference.
The CLI resolves the locale and populates `Policy.lang`; locale alone never
changes requested model languages. `--lang ja,en` applies a soft score
multiplier of `1.0` when all requested languages are claimed and `0.7`
otherwise, without filtering candidates.

Warnings persisted in `plan.json` are rendered in the language of the process
that produced the plan. Catalog language metadata describes publisher/vendor
claims, not measured language benchmarks. Argparse help remains untranslated.
# nmesh — ローカルLLM自動オーケストレーター 設計書

弱いGPUのノートPCから複数GPUのワークステーションまで、**ハードウェアを自動検出して最適なローカルLLM構成を自動で組み上げる**ソフトウェア。

- パッケージ名: `nmesh` (Python 3.10+)
- CLI: `nmesh`
- 対応OS: Windows / Linux / macOS(Apple Silicon)

## 1. 全体アーキテクチャ

```
nmesh/
  probe/        ハードウェア検出 → HardwareProfile
  catalog/      モデルカタログ (params, layers, kv形状, 品質スコア, 役割タグ)
  planner/      メモリ見積り + 配置計画 → Plan
  backends/     ollama / llama.cpp / vLLM / mlx アダプタ (起動コマンド・設定生成)
  runtime/      プロセススーパーバイザ (起動/停止/ヘルスチェック/スワップ)
  gateway/      OpenAI互換API + ルーター (役割→サービス振り分け)
  bench/        実測ベンチ → プロファイルキャッシュ (次回計画に反映)
  cli.py        doctor / plan / up / down / run / bench / models
```

データフロー:
`probe → HardwareProfile` → `planner(profile, catalog, policy, bench_cache) → Plan` → `runtime.apply(Plan)` → `gateway` が OpenAI 互換で受け付け、ルールに従って各サービスへルーティング。

## 2. HardwareProfile (probe)

```python
@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    vendor: Literal["nvidia", "amd", "intel", "apple", "unknown"]
    total_vram_bytes: int
    free_vram_bytes: int
    compute_capability: tuple[int, int] | None   # NVIDIAのみ
    driving_display: bool                        # 表示出力に使われているか

@dataclass(frozen=True)
class HardwareProfile:
    os: Literal["windows", "linux", "macos"]
    cpu_name: str
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    free_disk_bytes: int
    unified_memory: bool          # Apple Silicon = True
    gpus: list[GPUInfo]
    available_backends: dict[str, str | None]   # backend名 → 検出したバージョン
    tier: Tier                    # 下記の分類器で算出
```

検出手段（すべて失敗時は None / 縮退。例外を投げない）:
- CPU/RAM/ディスク: `psutil`（必須依存）
- NVIDIA: `nvidia-ml-py`（`pynvml` として import する任意依存）→ 失敗時 `nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits`
- AMD: `rocm-smi --showmeminfo vram --json`（Linux）/ Windowsは `wmic`/CIM でVRAM名のみ取得 → vendor="amd", 詳細不明時は保守的に扱う
- Apple: `platform.machine() == "arm64" and os == macos` → `sysctl hw.memsize`、unified_memory=True、GPU VRAM = RAM の 70%（Metal の推奨上限）
- Intel Arc: `xpu-smi` があれば、なければ非対応扱い（CPU扱い）
- backend検出: PATH上の `ollama`, `llama-server`(または`llama-cli`), `vllm`, `python -c "import mlx_lm"` を `--version` 実行で確認（タイムアウト5秒）

`driving_display`: Windowsは表示アダプタ名との一致、Linuxは `DISPLAY`/`WAYLAND_DISPLAY` があり GPU0 のとき True、macOSは常に True。判定不能なら「GPU0 は True」を保守的既定とする。

## 3. Tier 分類器

`effective_vram = 合計VRAM`（同一ベンダの複数GPUは合算、ただし単一モデル配置時は「tensor split 可否」を別途判定）。Apple の unified は `RAM * 0.7` を VRAM とみなす。

| Tier | 条件 | 戦略 |
|---|---|---|
| `T0_CPU` | GPUなし or VRAM < 2 GiB | CPU推論のみ。小型モデル1本、逐次実行 |
| `T1_LOW` | 2 ≤ VRAM < 6 GiB | 部分オフロード。常駐1本、役割はスワップで共有 |
| `T2_MID` | 6 ≤ VRAM < 12 GiB | 中型を全層GPU。tiny(router/embed)を同居 |
| `T3_HIGH` | 12 ≤ VRAM < 24 GiB | 大きめ1本常駐 + tiny 常駐 + 必要時スワップ |
| `T4_WORKSTATION` | 24 ≤ VRAM < 48 GiB | 役割別に複数常駐（chat/code/embed） |
| `T5_SERVER` | VRAM ≥ 48 GiB または GPU 2枚以上 | vLLM優先、テンソル並列、全役割常駐 |

## 4. モデルカタログ

`nmesh/catalog/models.yaml` にビルトインを同梱。ユーザーは `~/.nmesh/models.yaml` で追加/上書き（同一 `id` はユーザー側優先）。

```yaml
- id: qwen2.5-7b-instruct
  family: qwen2.5
  params: 7_620_000_000
  n_layers: 28
  n_heads: 28
  n_kv_heads: 4          # GQA
  head_dim: 128
  hidden_size: 3584
  max_context: 32768
  roles: [chat, code, tool]
  quality: 68            # 0-100 の相対品質スコア（同カタログ内比較用）
  license: apache-2.0
  sources:
    ollama: "qwen2.5:7b-instruct"
    hf_gguf: "Qwen/Qwen2.5-7B-Instruct-GGUF"
    hf: "Qwen/Qwen2.5-7B-Instruct"
    hf_mlx: "mlx-community/Qwen2.5-7B-Instruct-4bit"  # 任意: mlx 用の MLX 形式リポジトリ（無ければ hf を使用）
```

初版に含める役割別ラインナップ（各サイズ帯を1つ以上）:
`qwen2.5-0.5b/1.5b/3b/7b/14b/32b-instruct`, `qwen2.5-coder-1.5b/7b/14b`, `llama3.1-8b-instruct`, `llama3.3-70b-instruct`, `phi-4-14b`, `gemma2-2b/9b`, `mistral-7b-instruct`, 埋め込み `nomic-embed-text-v1.5`, `bge-m3`, ビジョン `qwen2-vl-7b`(任意)。
※ カタログは「事実データ」なので、値が不確かなモデルは含めない。層数・KVヘッド数が確認できないモデルは追加しないこと。

## 5. メモリ見積り（planner の中核・テスト必須）

量子化テーブル（実効 bits per weight、GGUFのメタデータ込みの実測相当値）:

| quant | bpw |
|---|---|
| f16 | 16.0 |
| q8_0 | 8.5 |
| q6_k | 6.6 |
| q5_k_m | 5.7 |
| q4_k_m | 4.85 |
| q4_0 | 4.55 |
| q3_k_m | 3.9 |
| q2_k | 3.35 |

```
weight_bytes      = params * bpw / 8
per_layer_bytes   = weight_bytes / n_layers          # 近似（埋め込み層も按分）
kv_bytes_per_tok  = 2 * n_layers * n_kv_heads * head_dim * kv_elem_bytes
                    # kv_elem_bytes: f16=2, q8_0=1
kv_cache_bytes    = kv_bytes_per_tok * context * parallel_slots
compute_overhead  = 0.06 * weight_bytes + 320 MiB    # 計算バッファ + ランタイム
total_bytes       = weight_bytes + kv_cache_bytes + compute_overhead
```

予算:
```
vram_budget = total_vram * 0.92 - (0.8 GiB if driving_display else 0)
ram_budget  = total_ram * 0.70            # OS/他アプリ用に30%残す
disk_needed = weight_bytes * 1.05         # ダウンロード用
```

部分オフロード（llama.cpp）:
```
n_gpu_layers = clamp(floor((vram_budget - kv_cache_bytes - compute_overhead) / per_layer_bytes), 0, n_layers)
# KVはGPU側に載せる前提。載らない場合は n_gpu_layers=0 にフォールバックし CPU 実行
cpu_bytes = weight_bytes * (n_layers - n_gpu_layers) / n_layers
条件: cpu_bytes <= ram_budget
```

スループット推定（bench 実測が無い場合のヒューリスティック。単位 tok/s）:
```
gpu_bw  = GPU名から推定した帯域(GB/s)  # 不明なら vendor 別既定: nvidia 400, amd 350, apple 200, intel 200
cpu_bw  = 40 GB/s 既定（DDR4/5 デュアルチャネル相当）
eff_bw  = 調和平均: 1 / (gpu_frac/gpu_bw + cpu_frac/cpu_bw)   # frac は重みのバイト比
decode_tps = 0.75 * eff_bw * 1e9 / weight_bytes
```
bench キャッシュ (`~/.nmesh/bench.json`) に `(model_id, quant, backend, gpu_name)` の実測 tok/s があれば必ずそちらを使う。推定値には `estimated: true` を付けて Plan に残す。

## 6. Planner

入力: `HardwareProfile`, カタログ, `Policy`, bench キャッシュ。

```python
@dataclass
class Policy:
    roles: list[str] = ["chat", "code", "embed"]   # 欲しい役割
    min_decode_tps: float = 8.0                    # これを下回る構成は不採用
    max_context: int | None = None                 # None ならモデル上限と予算から自動
    prefer: Literal["quality", "speed", "balanced"] = "balanced"
    allow_download_gb: float = 60.0
```

アルゴリズム（役割ごとに独立に候補生成 → 全体最適化）:
1. 役割 `r` にマッチするカタログモデル × 量子化 × backend の候補を全列挙。
2. 各候補について context を決定: `min(model.max_context, policy.max_context or 8192)` から始め、予算に収まらなければ 8192→4096→2048 と段階的に下げる。2048 でも不可なら候補を捨てる。
3. メモリ見積りで「全層GPU / 部分オフロード / CPUのみ」を判定。`decode_tps < min_decode_tps` は不採用。
4. スコアリング:
   ```
   score = quality_adj * w_q + normalized_tps * w_s
   quality_adj = model.quality - quant_penalty[quant]     # f16:0 q8:0.5 q6:1 q5:2 q4_k_m:3.5 q4_0:5 q3:9 q2:16
   prefer=quality → (w_q, w_s) = (1.0, 0.1)
   prefer=speed   → (0.5, 1.0)
   prefer=balanced→ (1.0, 0.25)
   normalized_tps = min(decode_tps, 30) / 30 * 100
   ```
5. 配置（Tierごと）:
   - `T0/T1`: 全役割を **swap グループ**にまとめる。同時常駐は1本。役割数を減らせる場合（chat と code を汎用1本で兼ねる）は兼務させる。埋め込みは CPU 上の小型モデルで常駐可（<0.5GiB）。
   - `T2/T3`: chat/code の主モデル1本を常駐 + embed(小型) 常駐。残りは swap。
   - `T4/T5`: 役割ごとに常駐。合計が vram_budget を超えないよう、超過分は品質スコアの低い役割から swap グループへ降格。
   - 複数GPU: 単一モデルが1枚に収まるなら GPU 固定割り当て（役割を分散）。収まらないモデルのみ tensor split (`--tensor-split` / vLLM `--tensor-parallel-size`)。
6. backend 選択（利用可能なものの中から）:
   - Linux + NVIDIA + 全層GPU + 非GGUF が使える → `vllm`
   - Apple Silicon + `mlx_lm` あり → `mlx`（生成系のみ。`mlx_lm.server` は `/v1/embeddings` を持たないため embed 役割は警告付き）
   - それ以外で `llama-server` あり → `llamacpp`（部分オフロード可能なのはこれと ollama のみ）
   - `ollama` あり → `ollama`（最も導入が容易。既定のフォールバック）
   - 何も無い → Plan に `install_hints` を出して `runnable: false`

出力:
```python
@dataclass
class PlannedService:
    name: str            # "chat", "code", "embed"
    roles: list[str]
    model_id: str
    quant: str
    backend: str
    context: int
    port: int            # 18010 から連番
    gpu_indices: list[int]
    n_gpu_layers: int | None
    resident: bool       # False なら swap グループ
    memory: MemoryEstimate
    decode_tps: float
    estimated: bool
    launch: LaunchSpec   # argv, env, ヘルスチェックURL
@dataclass
class Plan:
    created_at, nmesh_version, profile, tier, policy
    services: list[PlannedService]
    swap_group: list[str]        # 排他実行するサービス名
    routing: RoutingRules
    warnings: list[str]
    install_hints: list[str]
    total_download_bytes: int
```
Plan は `~/.nmesh/plan.json` に保存。`nmesh plan --explain` で各役割の採用理由と次点候補、メモリ内訳を表形式で出力。

## 7. Runtime（スーパーバイザ）

- `runtime.up(plan)`: resident サービスを順に起動 → ヘルスチェック (`GET /health` or `/v1/models`、最大120秒ポーリング) → 失敗時は 1段階軽い候補（量子化を1段下げる → context 半減 → n_gpu_layers 減）へ自動フォールバックして再試行（最大3回）。フォールバックした事実は Plan に追記して保存。
- swap グループ: gateway が要求時に `ensure_running(service)` を呼ぶ。ロードには排他ロック。既存の swap サービスを停止 → 目的のものを起動。LRU で1本のみ常駐。
- モデル取得: `ollama pull` / `huggingface_hub.hf_hub_download`（進捗表示）。`nmesh up --no-download` で取得をスキップ。
- プロセスは子プロセスとして管理し、`nmesh down` と atexit で確実に終了させる（Windows は `CREATE_NEW_PROCESS_GROUP` + terminate、POSIX は プロセスグループに SIGTERM → 10秒後 SIGKILL）。
- 状態は `~/.nmesh/state.json`（pid, port, service, started_at）。別プロセスの `nmesh status` から参照。

## 8. Gateway（OpenAI互換 + ルーター）

FastAPI + uvicorn、既定 `127.0.0.1:18000`。
- `GET /v1/models` — サービス名とエイリアス（`nmesh-auto`, `nmesh-chat`, `nmesh-code`, `nmesh-embed`）を返す
- `POST /v1/chat/completions` — ストリーミング(SSE)対応。`model` が
  - サービス名/エイリアス → そのサービスへ
  - `nmesh-auto`（既定） → ルーターが決定
- `POST /v1/embeddings` — embed サービスへ
- `GET /health`, `GET /status` — サービス一覧・常駐状況・直近レイテンシ

ルーティング規則（決定的・ヒューリスティック優先、上から順に評価）:
1. `tools` パラメータあり → tool 対応サービス
2. コードらしさ（```/ 拡張子 / `def |class |function |SELECT |import ` などのパターン、または `code` 役割サービスが存在し入力にコードブロックがある） → code サービス
3. 入力トークン概算 > そのサービスの context の 80% → より大きな context を持つサービス
4. それ以外 → chat サービス
5. LLMルーター（任意・`routing.mode: llm`）: tiny モデルに 1 トークン分類を投げる。既定は off（速度優先）。

swap モードでは gateway がリクエストを直列化（`asyncio.Lock`）し、モデル切替中は 503 ではなく待たせる（タイムアウト 300 秒）。

## 9. Bench / Autotune

`nmesh bench [--service chat] [--tokens 128]`:
- prefill: 512トークン相当のプロンプト、decode: 128トークン生成、3回中央値
- 結果を `~/.nmesh/bench.json` に `(model_id, quant, backend, gpu_name, n_gpu_layers)` キーで保存
- `nmesh plan` は次回以降この実測値を推定値の代わりに使う


## 10. CLI

| コマンド | 動作 |
|---|---|
| `nmesh doctor` | ハードウェア検出結果 + backend 有無 + 不足物のインストール手順 |
| `nmesh plan [--prefer quality\|speed\|balanced] [--roles chat,code] [--explain] [--json]` | 計画作成・保存 |
| `nmesh up [--no-download] [--detach]` | 計画に従って起動 + gateway 起動 |
| `nmesh status` / `nmesh down` | 状態表示 / 全停止 |
| `nmesh run "プロンプト"` [--role code] | ワンショット実行（gateway 経由） |
| `nmesh bench` | 実測 |
| `nmesh models [--role code]` | カタログ表示（このPCで動くものに ✓ ） |

出力は `rich` で表組み。`--json` で機械可読。

## 11. テスト方針（pytest、CI必須）

- **メモリ見積りの単体テスト**: 既知の値で回帰固定（例: 7.62Bパラメータ・q4_k_m → weight ≈ 4.62e9 バイト ≈ 4.30 GiB。誤差1%以内）。
- **合成 HardwareProfile フィクスチャ**で planner をテスト:
  1. CPUのみ 8GiB RAM → T0、CPU向け小型モデル、swap 1本、`runnable`
  2. GTX 1650 4GiB / 16GiB RAM → T1、部分オフロード、n_gpu_layers が 0 < x < n_layers
  3. RTX 3060 12GiB → T2/T3 境界、7B q4_k_m 全層GPU
  4. RTX 4090 24GiB → T4、chat+code+embed が常駐で予算内
  5. 2×A100 80GiB Linux → T5、vLLM 選択、tensor parallel
  6. M2 Max 64GiB unified → mlx 選択
  - 各ケースで「予算超過しない」「min_decode_tps を満たす」「役割が全部埋まる（または warning が出る）」を検証。
- probe は subprocess をモックして nvidia-smi / rocm-smi のパース単体テスト（実機GPU非依存）。
- gateway ルーティングの単体テスト（fake サービスで振り分け先だけ検証）。
- 実バックエンド起動はテストしない（CIにGPUなし）。`--dry-run` で launch argv の内容だけ検証。

## 12. 非目標（v1では作らない）

- 学習・ファインチューニング
- 分散マルチノード
- GUI（CLI + OpenAI互換APIのみ）
- 未検証モデルのカタログ自動生成

## 13. 実測フィードバック (telemetry)

- gateway の実トラフィックを `~/.nmesh/telemetry.json` に記録する。キーは `nmesh bench` と同じ `(model_id, quant, backend, gpu_name, n_gpu_layers)`。
- ストリーム要求: 観測した SSE `data:` 行数を completion_tokens とし、最初の行までを `ttft_s`、行数が 16 以上で区間が正のときのみ `decode_tps = (tokens - 1) / (last - first)` を記録する。
- 非ストリーム要求: `total_s` と上流 `usage.completion_tokens` のみ。全体待時間には prefill が含まれるため、decode tok/s は記録しない。embeddings は decode を持たないので記録しない。
- `telemetry.bench_overlay(min_samples=5)` はキーごとに decode 測定値が `min_samples` 以上ある場合の中央値を返す。`nmesh plan` は `{**load_cache(), **bench_overlay()}` を `build_plan` に渡す。overlay はメモリ上だけで、`~/.nmesh/bench.json` には書き込まない。
- `GET /metrics` と `nmesh status` がサービス別のサンプル数・decode tok/s 中央値・TTFT 中央値/p95・全体待時間中央値を返す。

## 14. ルーティングの明示 ID 優先

`/v1/models` が広告する `nmesh-<service>` などの明示 ID はすべてのヒューリスティクより優先される。順序は
明示 ID → `tools` → コード判定 → context 超過 → chat。`nmesh-code` のような役割名は `role_to_service` 経由で解決される。
空文字列・`nmesh-auto`・未知の ID は従来と同じくヒューリスティクにフォールスルーする。

## 15. Free-memory budgets and admission

The persisted `Plan` describes machine capability using total VRAM and RAM by
default. `Policy.budget_source` can be set to `free` for an opt-in plan based
on currently available memory: GPU budgets use `GPUInfo.free_vram_bytes`, and
RAM budgets use `HardwareProfile.available_ram_bytes`, while retaining the
same display reserve and operating-system reserve rules.

`nmesh up` performs a proactive admission check using free memory before
launching services. Resident services are counted individually and swap-group
services are counted only by their largest member because they are mutually
exclusive. If the persisted total-capability plan does not fit, the supervisor
rebuilds it with `budget_source="free"` and preserves the merged benchmark
cache. This keeps saved plans useful as capability descriptions while avoiding
avoidable OOM launches when another application already consumes memory.
If probing fails or replanning produces no services, startup continues with the
existing quantization/context/GPU-layer fallback ladder. Use
`nmesh up --ignore-free-memory` to bypass admission intentionally.

## 16. Gateway reload and swap gate

The gateway keeps a mutable plan state and takes one plan snapshot at the
start of every chat, embedding, and model-list request. A gateway created
without an explicit plan checks `plan.json` mtime and reloads an atomically
replaced plan before handling each such request. `POST /admin/reload` forces
the same reload path and returns whether the plan changed, its service names,
and `created_at`; `nmesh reload` is the CLI wrapper. An explicitly supplied
plan disables mtime auto-reload but still permits the administrative reload
endpoint. Requests already in flight continue using their original snapshot.

Swap-group traffic uses a reader/writer gate. Requests for the currently
loaded service acquire concurrent reader slots. A request for another
swap-group service becomes an exclusive swapper, waits for all readers to
drain, then calls `ensure_running`; no new reader can enter during that drain
or swap. Plan reload invalidates the loaded-service marker so the next
swap-group request revalidates the service.

## 17. Runtime crash recovery

The runtime heartbeat checks services in the active plan and revives managed
processes that have exited. It skips swap-group members that were never loaded,
uses a three-restart budget within a five-minute window, and records services
that exhaust that budget as failed instead of restarting them forever. A
service whose health endpoint already answers can be adopted by a Supervisor
instance that did not launch it; adopted services are marked external and are
never signalled by `down()`.

The gateway server enables a 15-second watchdog by default. It runs the
runtime heartbeat in a worker thread, continues after heartbeat exceptions,
and cancels the task cleanly during shutdown. A request that encounters an
upstream connection failure invokes `ensure_running` once and retries once
  before returning the existing 502 error.

### 17.1 Runtime state ownership and liveness

`~/.nmesh/state.json` is version 2 and is written atomically. It records the
state owner, service PID/port, launch time, PID creation time when available,
health URL, slot count, and whether an entry is shared or external. PID
creation times are checked with a two-second tolerance to prevent PID reuse
from making a stale process appear alive; legacy version-1 files without
creation times remain readable and use PID existence checks.

State writes merge live entries owned by other processes rather than deleting
them. Status probes persisted entries and prunes dead entries, so a stale
state file cannot report a corpse as running. `nmesh down` uses the foreign
mode to terminate live PIDs owned by another process; supervisor cleanup and
startup fallback only stop children owned by the current supervisor.
PID-less shared or external entries are retained only while their persisted
health URL responds, and each service entry carries its owner when state from
multiple supervisors is merged.

The supervisor arms its `atexit` cleanup handler lazily, immediately after it
launches its first child. Constructing a supervisor for `status`, `doctor`, or
`bench` therefore cannot delete another process's state file or stop its
services. A foreground `nmesh up` still owns and cleans up its launched
children on exit.

## 18. Concurrency slots

The planner assigns `parallel_slots` after model placement. Extra slots are
eligible only for fully GPU-resident decode services with nonzero KV bytes per
token and a backend slot cap greater than one. CPU-only and partial-offload
services stay at one slot: batching helps when the GPU is idle during
memory-bound decode, while CPU and partially offloaded workloads are already
limited by memory bandwidth and would only split that bottleneck. Embedding
services also stay at one because their KV bytes per token are zero.

The automatic allocator groups services by their GPU domain or the RAM budget.
It starts with the bytes for one slot, reserves only half of leftover memory
(`SLOT_SPARE_FRACTION = 0.5`) for additional slots, and processes services in
plan order so the primary chat service receives spare capacity first. Services
in a swap group count only the largest member because those services are
mutually exclusive at runtime. A forced `Policy.parallel_slots` value is
clamped to the backend cap and to the memory that actually fits; a warning
reports a requested value that had to be reduced.

llama.cpp receives explicit `--parallel N` and `-c context * slots`. Explicitly
passing `--parallel 1` avoids llama-server's automatic slot selection and
ensures each slot receives the planned context. vLLM receives
`--max-num-seqs N` and `--gpu-memory-utilization`; its fraction is clamped to
`[0.10, 0.95]` and derived from the service's GPU bytes divided by assigned
GPUs' total VRAM. Total VRAM is used for this fraction even when planning uses
free-memory budgets, because vLLM defines the option relative to total VRAM.

### 18.1 Capacity-aware GPU placement

Multi-GPU plans place services with best-fit decreasing: resident services
are considered before swap-group members, and each set is ordered from the
largest GPU footprint to the smallest. A service uses the fitting card with
the least remaining capacity, breaking ties by GPU index. Swap-group members
reserve only the largest footprint assigned to a card because they are
mutually exclusive. If no single card fits, a multi-GPU service falls back to
tensor parallelism across all cards and divides its GPU footprint across them.

After a llama.cpp service is assigned to one card, its GPU-layer count is
re-solved against that card's budget rather than the aggregate VRAM budget.
The memory split and launch arguments are rebuilt together, including `-ngl`;
the tensor-parallel fallback retains its original aggregate layer solution.

## 19. Gateway concurrency limits

The gateway limits chat requests only for llama.cpp and vLLM services, using
the planner's `parallel_slots` value as an asynchronous semaphore. Ollama and
MLX remain unlimited because those runtimes provide their own scheduling, and
embedding requests remain unlimited because embedding services have no decode
slot to protect. A plan reload rebuilds the current semaphore map; requests
that acquired an older semaphore release that same object.

Chat requests acquire a backend slot before entering the swap gate and release
it beside every gate release, including streaming completion and error paths.
They wait up to `NMESH_QUEUE_TIMEOUT` seconds (120 by default); an exhausted
queue returns HTTP 503 with `Retry-After: 1` and the configured slot count.
The metrics endpoint reports each limited service's `limit`, `in_flight`, and
`waiting` counts under `concurrency`.

## 20. GGUF acquisition

For llama.cpp, the planner's model reference is a local destination, not an
assumed Hugging Face filename. Runtime acquisition enumerates the repository's
published `.gguf` files and resolves quantization tokens case-insensitively.
Quantization aliases are matched on token boundaries, and a split model is
downloaded only when every numbered part is present. The selected file and
quantization are recorded in runtime state; a lower-precision substitution is
also surfaced as an acquisition note. A planned quantization is never upgraded
because its memory estimate is already fixed.

## 21. GPU detection provenance and backend capability

Hardware detection uses a strict fallback order. Apple Silicon uses the
existing unified-memory path. Otherwise nmesh tries NVML/`nvidia-smi`, then
`rocm-smi` when NVIDIA detection found no GPUs, and only then invokes generic
platform detection. The generic path must not replace a successful specialized
result. Windows generic detection reads `Win32_VideoController` and the
display-adapter registry class; Linux generic detection reads DRM vendor IDs
and AMD's `mem_info_vram_total`.

Windows `AdapterRAM` is a signed 32-bit field and saturates at approximately
4095 MiB. It must therefore not be treated as the real capacity of an adapter
such as an Intel Arc A770. Generic Windows detection prefers the 64-bit
`HardwareInformation.qwMemorySize` registry value and uses `AdapterRAM` only
when it is below the saturation boundary. Every `GPUInfo` carries the
provenance of its total VRAM (`nvml`, `smi`, `registry`, `sysfs`, or
`unknown`). For registry- and sysfs-derived adapters free VRAM is not
measured; `free = total` is a conservative planning upper bound, while the
display reserve remains applied.

Backend help flags describe an interface, not an implementation capability.
The CPU-only llama.cpp b10734 binary advertises `-ngl` in its help but its
device report is:

```text
Available devices:
  (none)
```

The planner therefore distinguishes a known empty device tuple, `()`, from an
unknown probe result, `None`. For llama.cpp, `()` forces CPU placement,
`n_gpu_layers=0`, zero GPU accounting, and RAM accounting for weights, KV, and
overhead. `None` preserves the prior flag-based behavior because a failed
probe must not falsely degrade a working GPU installation. A detected GPU can
be used by llama.cpp only after installing or building a Vulkan, CUDA, HIP, or
SYCL backend.
