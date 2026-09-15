# 変更履歴

## 未リリース

- `nmesh up` が admission でサービスを落としたとき、その警告が RuntimeStatus に載らずユーザーに伝わりませんでした（実行結果に出ないため「落とした」こと自体が静か）。`RuntimeStatus.warnings` を追加し、`up` の結果表示・`--json` で警告を出すようにしました。
- `nmesh up` が admission（free メモリ圧の再計画）でサービスを落とした場合、その縮退したプランを `plan.json` に保存してしまい、一時的なメモリ状況がユーザーの意図を恒久に縮めていました。保存は「実行するサービス集合がプランと同じ場合」に限定するようにしました（アーティファクトの実測値更新は引き続き保存されます）。
- `nmesh status`（およびゲートウェイ内の `runtime_status`）が、supervisor のメモリ内に1つでもエントリがあると state.json の永続記録を一切マージしないため、プロセスが知らない稼働中サービスを一覧から落としていました。これにより `nmesh unload` が生存中のサービスに「実行中ではありません」と返す原因にもなっていました。メモリ内に無いサービス名の永続エントリも `running` を再計算して併記するようにしました。
- `nmesh unload <service>` が、実際には稼働中のサービスに対して「実行中ではありません」と返すことがありました。ゲートウェイ内の supervisor はプランを初回 heartbeat 時に一度だけ読み、plan.json が後から更新されてもメモリ内のプランを再読み込みしないため、プランに後から追加されたサービスを採用対象として見つけられませんでした（別プロセスから直接 `runtime.unload` すれば成功する、という不一致）。`unload` はメモリ内プランにサービスが見つからない場合、永続化されたプランにも照会して採用を試みるようにしました。
- `nmesh watch` の flag_unknown（未検証フラグ）検出が、記事に登場する任意の `--x` を拾っていたため `git --no-ext-diff` や `docker run --gpus` など nmesh が一切参照しないフラグまで一覧を埋めていました。`caps.json` は nmesh が起動する llama.cpp バイナリのフラグだけを収録しているので、記事内で `llama-server`/`llama-bench` などを実際に起動するコマンド行のフラグのみを比較対象にしました（実記事コーパスで 60 件超のノイズ → 0 件の真の未検証フラグ）。
- `nmesh orchestrate measure` の worker 既定値が、計画の中で lead 以外の最初のサービス（既定計画では埋め込み専用の `embed`）を選んでいたため、`nmesh up` 済みの機で実行すると必ず「生成用ではありません」で失敗していました。既定は生成可能なサービス（chat・code・worker）からのみ選び、候補が無い場合は「lead 以外に生成用サービスがもう1つ必要」と案内するようにしました。
- 型検査（`mypy nmesh`）の既知負債リストを `nmesh.cli`・`nmesh.planner` の 2 つに縮小し、`nmesh.gateway`・`nmesh.runtime.engine`・`nmesh.runtime.supervisor` を検査対象に戻しました（60/62 ファイル）。潜在バグを含めて修正: `subprocess.CREATE_NEW_PROCESS_GROUP` が Windows 以外に存在しない起動分岐、state.json の pid/services が dict 以外だった場合の `int(object)`・反復子エラー、ゲートウェイの `_reap` が KEEP_ALIVE 比較で None を演算し得た点、Prometheus メトリクス出力が非数値を `float()` に通す点、委譲判定の `int(object)`、manifest の `installed_at`/`flags` 未検証。
- `nmesh plan --profile` のシミュレーションが、別マシンを計画しているのにこの機で測ったベンチ・実測ライブ計測・埋め込み入力上限と検索限界をそのまま適用し、「この機で測定」と言い続けていました（ベンチのキーは GPU 名を含むが CPU を含まないため、CPU プロファイルにこの機の実測 tok/s が混入していました）。シミュレートされた機には測定が無いので計画は一切使わず、バナーもその旨を説明するようにしました。モデル品質・使用可能な文脈深度・アーティファクト実測サイズなど機によらない証拠は引き続き使います。
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