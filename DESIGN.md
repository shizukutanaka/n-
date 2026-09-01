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
- NVIDIA: `pynvml`（任意依存）→ 失敗時 `nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits`
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
   - Apple Silicon + `mlx_lm` あり → `mlx`
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
- `nmesh autotune`: context と n_gpu_layers を数点振って最良 tok/s の設定を Plan に反映

## 10. CLI

| コマンド | 動作 |
|---|---|
| `nmesh doctor` | ハードウェア検出結果 + backend 有無 + 不足物のインストール手順 |
| `nmesh plan [--prefer quality\|speed\|balanced] [--roles chat,code] [--explain] [--json]` | 計画作成・保存 |
| `nmesh up [--no-download] [--detach]` | 計画に従って起動 + gateway 起動 |
| `nmesh status` / `nmesh down` | 状態表示 / 全停止 |
| `nmesh run "プロンプト"` [--role code] | ワンショット実行（gateway 経由） |
| `nmesh bench` / `nmesh autotune` | 実測・自動調整 |
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
