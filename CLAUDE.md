# CLAUDE.md — nmesh 開発ガイド

> このファイルは Claude Code / Devin 等のエージェントが nmesh を開発する際の指示書。
> セッション開始時に必ず参照すること。

---

## プロダクト概要

**nmesh** — ローカル LLM のためのハードウェア自動検出・モデル選択・メモリ見積もり・実行管理ツール。
OpenAI/Anthropic 互換 API ゲートウェイを提供する。

北極星: **「この機械で何が動くか」を証拠付きで推論する。推定値は必ず推定と表示し、予算を静かに超過しない。**

技術スタック: Python 3.10+、FastAPI + Uvicorn（gateway）、argparse + rich（CLI）、psutil、httpx、
llama.cpp / Ollama / vLLM / MLX-LM（バックエンド）

---

## READ ORDER (必須)

```
CLAUDE.md → README.md → nmesh/planner/core.py → 対象モジュール → tests/
```

---

## 絶対不変条件 (変更禁止・設計原則)

| ID | ルール |
|----|--------|
| N1 | **正直見積もり**: 推定値は必ず `~`（estimated）表示。実測なしに「計測値」の体裁を取らない |
| N2 | **予算の静かな超過をしない**: VRAM/RAM 予算を超える配置案・フラグ・環境変数を警告なしに発行しない |
| N3 | **検証不能なフラグを発行しない**: backend_flags 未検出の機能は候補を出さないか警告して省略（`warn.*_unsupported`）|
| N4 | **LAUNCH_REVISION**: launch argv/env の意味が変わる変更は `LAUNCH_REVISION` を +1 する（plan.json に記録され `nmesh up` が古いプランを検出する）|
| N5 | **i18n は en+ja の両方にキーを追加する**（`nmesh/i18n.py`、片方だけ追加すると `test_translation_tables_have_equal_keys` が落ちる）|
| N6 | **スワップメンバーに常駐向け最適化を発行しない**（例: `--mlock`/`--no-warmup`/vLLM sleep は resident/非 resident で出し分ける）|
| N7 | **ユーザが設定した環境変数は尊重する**（OLLAMA_*・NMESH_* を無条件上書きしない）|
| N8 | **テストを緩めて通さない**: テスト修正は最後の手段。新挙動が正しい順位を書き換えた場合のみ、依存テストを比較対象ピン等で追従させる |

---

## モジュール構造 (単方向・上位が下位を使う)

```
nmesh/cli.py            エントリポイント・全サブコマンド
  ├─ nmesh/planner/     ハードウェア profile→配置計画（core.py が中心、LAUNCH_REVISION/見積もり/argv生成）
  ├─ nmesh/runtime/     supervisor(プロセス監視・スワップ・health)・acquisition(取得)・engine・service_unit・logs
  ├─ nmesh/gateway/     FastAPI アプリ・プロキシ・SSE 翻訳・jobs/limit/gate/tokens・server(起動エントリ)
  ├─ nmesh/catalog/     models.yaml（モデルカタログ）とローダ
  ├─ nmesh/bench/       実測ベンチ（runner/cache/epoch/embed/retrieval）
  ├─ nmesh/eval/        決定的マイクロ評価・統計（suite/hard/generated/depth/context/select/stats/runner/cache）
  ├─ nmesh/orchestrate/ リード/ワーカー委譲（measure/protocol/record/aggregate）
  ├─ nmesh/spec/        投機デコード腕比較（measure/engine/record）
  ├─ nmesh/watch/       外部ソース定期ウォッチ（sources/extract/verify/draft/state）
  ├─ nmesh/probe/       HW 検出・tier 分類（detector/models/generic_gpu/caps/serialize）
  ├─ nmesh/artifact.py  GGUF ヘッダ解析・モデル/サービスフィンガープリント
  ├─ nmesh/artifacts.py 取得済みアーティファクトのサイズキャッシュ（artifacts.json）
  ├─ nmesh/evidence.py  証拠述語（refutes）
  ├─ nmesh/evidence_inventory.py  bench/eval/spec 証拠の収集（nmesh evidence）
  ├─ nmesh/inventory.py ローカルモデル走査・重複/バリアント検出（models scan/local）
  ├─ nmesh/telemetry.py リクエストレイテンシ・トークン統計の収集
  ├─ nmesh/paths.py     nmesh_home()（NMESH_HOME or ~/.nmesh）
  └─ nmesh/i18n.py      en/ja メッセージ表
```

- すべての管理状態は **NMESH_HOME 配下**（plan.json・state.json・models・logs）。外に置くと管理単位から外れる。
- `plan.json` の `launch_revision` が現行 `LAUNCH_REVISION` より古いプランは `up` 時にフラグされる。

---

## テスト基準 (CI と同じ)

```bash
pip install -e .[dev,gateway,download]   # extras 欠落で収集自体が失敗する
ruff check .
mypy nmesh                              # nmesh パッケージ全モジュール
pytest -q                               # ~50 秒、760+ 件
```

CI は Python 3.10 / 3.12 マトリクス（`.github/workflows/ci.yml`）。
`typing.Self` 等 3.11+ 専用 API は使えない。

---

## 変更のチェックリスト

- [ ] argv/env 意味変更なら `LAUNCH_REVISION` +1（`nmesh/planner/core.py`）
- [ ] 新規メッセージは `nmesh/i18n.py` に en と ja の両方を追加
- [ ] `CHANGELOG.md` の `[Unreleased]` に `### Added` / `### Fixed` で追記（日本語）
- [ ] 新モデルは `nmesh/catalog/models.yaml` に**公開 config の実値**で追加（層数・heads・kv_heads・head_dim・hidden・vocab・sources）
- [ ] ハイブリッド/MoE モデルは `kv_layers`/`recurrent_state_bytes`/`experts`/`active_params` を必要に応じ設定
- [ ] ユニットテストを追加（新機能は複数ケース、修正は回帰テスト）

## よくあるミスと対処

### backend_flags 未検出なのにフラグ発行

```python
# ❌ 無条件に発行 → 起動失敗 or silent fallback
argv += ["--n-cpu-moe", str(k)]

# ✅ 検出済み or 未検出で警告。backend_flags=None は「未プローブ」なので発行してよい
if not known or "--n-cpu-moe" in flags:
    argv += ["--n-cpu-moe", str(k)]
elif warnings is not None:
    warnings.append(t("warn.moe_cpu_unsupported", language, model=model.id))
```

### llama.cpp の容量引数は「スロット合計」

`-c` は parallel 時 `context * slots`（プール総量）。`-np/--parallel` 未対応ビルドでは
スロット数を増やさず `warn.parallel_unsupported` を出す。

### 共有デーモン（ollama）の env は全サービスに効く

`ollama serve` は単一デーモンなので、サービス A の env が B にも効く。
per-service 値が必要なら env に集約ルールを持たせる（例: 最大値で統一・全 resident のみ `-1`）。

### i18n キー漏れ

`note.*`/`warn.*`/`err.*` を片言語だけ追加すると `test_i18n.py` が即落ちる。両方に入れる。

---

## セキュリティ・クリティカルなコード

以下は変更前に実際の挙動を必ず確認:

- `nmesh/planner/core.py` — メモリ予算・argv/env 生成（誤発行は起動不能・予算超過・silent fallback の温床）
- `nmesh/runtime/supervisor.py` — kill/adopt/spawn（pid 取り違え・ポート奪取・外部プロセス誤殺は禁止）
- `nmesh/gateway/__init__.py` — 上流プロキシ（SSE 翻訳・スロット管理・タイムアウト・5xx 応答）

これらを変えた場合は `nmesh status`/`up --dry-run`/stub サーバでの実挙動確認を推奨。

## コミット規約

- 1 変更 1 コミット（`feat:`/`fix:` 等の prefix 運用は履歴に従う — 現在は日本語要約形）
- PR は `devin/<timestamp>-<slug>` ブランチから main へ。テンプレは `## Summary` + `## Verification`
- 変更ファイルは最小に保つ。他機能の無関係な差分を含めない

---

## よく使うコマンド

```bash
nmesh doctor        # HW・バックエンド検出の可視化
nmesh plan          # 配置計画の提示（--dry-run 的確認に）
nmesh up --detach   # プラン適用・起動
nmesh status        # サービス状態（sleeping 含む）
nmesh bench         # 実測ベンチ（推定値を計測値へ置換）
nmesh eval          # 決定的マイクロ評価
nmesh down          # 停止
```

## 参照文書

| 文書 | 用途 |
|---|---|
| `README.md` | 全仕様（flags/env/モード/証拠ゲート/known limitations） |
| `CHANGELOG.md` | リリース差分・LAUNCH_REVISION 履歴 |
| `nmesh/catalog/models.yaml` | モデルカタログ実値の参照 |
| `tests/` | 挙動の実例（stub サーバ・fixture の作り方） |
