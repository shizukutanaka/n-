# 変更履歴

## 未リリース

- 同梱の CI レシピ（`pip install -e .[dev]` → `pytest -q`）はテストを 1 件も収集できませんでした（`tests/conftest.py` が gateway を、`tests/test_acquisition.py` が `huggingface_hub` を import するため）。`gateway`・`download` extras を追加し、`mypy nmesh` をゲートに加えました。型検査は既知負債の 5 モジュール（`nmesh.cli`・`nmesh.gateway`・`nmesh.planner`・`nmesh.runtime.engine`・`nmesh.runtime.supervisor`）以外の全モジュールで緑です。
- `scripts/e2e.py`（実バックエンド E2E）は `nmesh` 自身が `$NMESH_HOME/engines/llamacpp` に入れたエンジンを探さず、モデルも `~/.nmesh/models` 固定だったため、`nmesh up` が動く機でも常に 77（SKIP）でした。エンジンは `nmesh engine install` の成果物（active 優先）を、モデルは設定済み `NMESH_HOME` を見るようにし、`status`/`down` には自身のゲートウェイポートを渡すようにしました。この機で doctor→plan→up→chat/completions→metrics→bench→status→down→ポート解放まで完走します。
- 型検査で見つかった潜在的な None/型の取り違えを修正しました（委譲記録の数値検証、GGUF の file_type 欠落時の量子化ラベル、Zenn 応答の articles、HF siblings 欠落、spec 記録の制御行、catalog ドラフトの sources、gateway ルート一覧）。

- `nmesh up` が既に稼働中のゲートウェイのポートに対して二重起動し、その死んだ pid を状態に記録していたため、以後の `nmesh down` が本物のゲートウェイを殺せずウォッチドッグがサービスを再起動し続けていました。既に `/health` が応答する場合は実際のリッスン pid を採用し、`down`/`stop_gateway` は記録された pid が死んでいても（または状態にゲートウェイ項目が無くても）ポート番号から孤児ゲートウェイを掃討するようにしました。`down`/`status` に `--port` を追加し、`down` はゲートウェイをサービスより先に停止します。`unload` が所有していないサービスを指す誘導文の存在しない `nmesh down --foreign` 表記も修正しました。
- `nvidia` 任意依存を非推奨化された `pynvml` 配布物から `nvidia-ml-py` に切り替え、CLI 出力に漏れていた FutureWarning を消しました（import 名は `pynvml` のまま）。
- `nmesh evidence` の検索（retrieval）行が、使用可能な記録にも再測定コマンドを常に出していたのを、再測定で判定が変わりうる場合（harness 不一致・digest 陳腐化・コントロール不合格・未測定）にだけ出すようにしました。
- `nmesh evidence` の棚卸しに投機実行（spec）の測定記録を加え、テキスト表では検索（retrieval）の記録も表示するようにしました。どちらもこれまで保存されながら一覧に出ていませんでした。
- 計画したサービス全てに自身の構成の実測合格率がある場合は、「順位付けは未検証のカタログ品質主張を使用します」ではなく測定値そのものを報告するようにしました。
- `nmesh bench --service embed --retrieval` が言う所要時間を固定の「10分」から、キャリブレーション要求と実測エンコード速度で推定した分数に変えました（この機では 34.8 分と見積もり、実測 33.5 分）。
- 埋め込みベンチの上限プローブでバックエンドが明示的に拒否した入力を静かな切り詰めと区別して記録し、測定済みで上限が無い場合は「未検証」ではなく「切り詰めなし」として報告するようにしました。
- Linux の llama.cpp 配布物に含まれる相対シンボリックリンクと実行ビットを保持して展開し、`nmesh up --dry-run` ではエンジンを自動取得しないようにしました。
- Add character-bound gateway auto-chunking for non-whitespace text.
- 既定の役割で候補がない役割は警告付きで計画から除外し、残りの起動可能なサービスで起動します。
- `--roles` を明示した場合は、要求した役割を満たせない計画を従来どおり失敗として扱います。
- 純粋な推定値にもデコード速度の下限を適用し、未確認の実測値だけを再測定待ちとして残すようにしました。
- bench harness を `bench-v2` に更新しました。デコード速度が n/(n-1) 倍に膨らんでいたため、`bench-v1` の記録は比較に使いません。再測定が必要です。
- `nmesh --version` が bench harness（`bench-v2`）と深度プローブ規則の識別子を出力するようになりました。`bench-v1` の記録は比較に使いません。
- 深度プローブの採点規則変更により既存の深度証拠は無効です。`nmesh eval --depth` の再実行が必要です。
- GPU 実機・vLLM・MLX は未検証です。
- CI は未稼働です（ワークフロー未設置）。
- Add pooled retrieval evidence and opt-in gateway auto-chunking.
- `nmesh bench --retrieval` records single-vector retrieval usability; the planner warns rather than clamps when recall degrades and recommends chunking long inputs.
- Retrieval evidence now confirms whether chunking at the measured usable length restores rank-1 retrieval before recommending it.
- `nmesh bench` now rejects embedding-only services and reports benchmark HTTP failures without a traceback.
- The gateway now rejects proven silently truncated single-input embeddings and marks saturated multi-input requests as unverified.
- Embedding-only plans no longer claim decode throughput or use decode speed for planner gating and ranking; `nmesh plan` shows `—`.
- 旧 benchmark harness の保存済み計測は planner の証拠に使わず、`nmesh bench` による再測定を必要とするようにしました。
- `nmesh evidence` で保存済み証拠の使用可否、理由コード、再測定コマンドを棚卸しできるようにしました。
- 埋め込みサービスの実測入力上限とエンコード速度を保存し、計画の文脈を実際の served cap に合わせて短縮するようにしました。Ollama 0.33.2 の bge-m3 がこのホストで `num_ctx` に関係なく 2048 トークンで静かに切り詰めること、llama.cpp が計画した窓全体を提供したことを記録します。