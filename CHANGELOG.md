# 変更履歴

## 未リリース

- GitHub 公開用ファイル一式: `install.sh`（POSIX）/`install.ps1`（Windows）インストーラー、`.github/ISSUE_TEMPLATE/`（bug/feature）、`.github/PULL_REQUEST_TEMPLATE.md`、`SECURITY.md`、README の Quickstart にインストーラー経路を追加


- `pip install nmesh`（base）で `nmesh` が起動不能だった — httpx が `gateway` extra に隔離されていたのを core deps へ移動。`serve` は extras 未導入時に raw traceback ではなく `pip install nmesh[gateway]` を案内

- `nmesh down` が停止したサービスを表示する — 以前は常に「実行中のサービスはありません」とだけ返し、何を止めたか確認できなかった（Supervisor.down() が常に空の結果を返していた）

||||||| parent of 18155ad (bench/eval/autotune の無言 exit を修正 — プラン不在でメッセージなしだった)
- `bench`/`eval`/`autotune` がプラン不在で無言 exit 1 だったのを「No active plan」と出力するよう修正

- `nmesh orchestrate measure --worker-url` の案内文は「外部 worker を渡せ」と言うのに、プラン内に2つ目の生成サービスが無いと拒否されていました。`--worker-url` 指定時はプラン内 worker 解決をスキップし、外部エンドポイントで委譲測定が実行できます。

- `nmesh eval --categories` で絞り込んで実行した評価の証拠が「古い採点規則」（grader_digest_mismatch）と誤診断されていました。部分的な実行は `partial_suite`（絞込実行・planner には全スイートが必要）として正しく表示します。

- `nmesh up` の Services テーブルで起動直後のサービスの Port セルが空白になっていました（直後の `status` では正しく表示）。`Supervisor.status()` がプロセス管理下のエントリにポートを含めていなかったのを修正し、計画済みポートを必ず出します。

- `nmesh status` / `nmesh down` / `nmesh up` の非 JSON 出力が `RuntimeStatus(...)` の dataclass repr をそのまま表示していました。サービス一覧をテーブル（Service / State / Port / Model / Backend）で表示するようにし、稼働サービスが無い場合は「no services running」と表示します（en/ja）。

- サービス起動失敗時に、計画ポートが別プロセスで占有されている場合はその旨を明示するようにしました（`port N is still in use by another process`）。これまでは上流の生ログ（`couldn't bind HTTP server socket`）だけが出ていました。判定は connect ではなく bind 試行で行います — connect だと外部リスナーの accept バックログを消費してリトライ時に誤判定するため。
- 実行可能な保存済み `plan.json` がある状態で `nmesh up` に計画系オプション（`--roles`/`--kv-quant`/`--spec*`/`--sleep-idle-seconds`/`--cache-reuse`/`--context-shift`）を渡しても、保存済みプランがそのまま使われオプションが無警告で無視されていました。無視されるオプションを stderr に警告するようにしました。
- `--context-shift`（`plan`/`up`）を追加しました。対応する llama.cpp ビルドの生成系サービスに `--context-shift` を渡し、生成中に出力がコンテキスト窓を超えてもウィンドウをずらして生成を継続します（従来は窓の終端で打ち切り）。**最古のトークンは静かに捨てられる**ため、有効時は計画に警告を必ず出します。なお窓を超える**入力プロンプト自体**は従来どおり llama-server が拒否します（context-shift は生成中のシフトであり、窓を超える入力の受理ではありません）。埋め込みサービスには付けません（埋め込みの静かな切り詰めは回答を破損させるため）。既定はオフ。ビルド非対応時は警告のみ。

- `--cache-reuse N`（`plan`/`up`）を追加しました。対応する llama.cpp ビルドの全サービスに `--cache-reuse N` を渡し、リクエスト間でプロンプトキャッシュを KV シフトで再利用します（固定システムプロンプトを持つ会話の TTFT を改善）。既定は 0（従来どおり無効）。ビルドがフラグに対応しない場合は警告を出してフラグを出力しません。
- `--sleep-idle-seconds N`（`plan`/`up`）を追加しました。対応する llama.cpp ビルドの全サービスに `--sleep-idle-seconds N` を渡し、アイドル N 秒後にモデルと KV キャッシュを RAM から退避させます（次のリクエストで自動復帰）。既定は 0（従来どおり常駐）。ビルドがフラグに対応しない場合は警告を出してフラグを出力しません。
- `nmesh spec measure` が、計測用の一時 Supervisor で `up()` を呼んだ際、内部の `save_plan` が常に実際の `plan.json` に書き込むため、単一サービスの計測用プラン（エフェメラルポート付き）でユーザーのプランを上書きしていました。実際に plan.json が chat 単独・計測用ポートに置き換わるのを確認しました。Supervisor に `plan_path` を追加し、`spec measure` は一時ディレクトリのプランに書き込むようにしました。
- Apple Silicon 実機（M4・macOS 26）で MLX バックエンドを実地検証し、`nmesh up` が起動できない 2 つの不具合を修正しました。取得フェーズは backend 問わず GGUF リポジトリ（`hf_gguf` ソース）を `snapshot_download` していたため、mlx が読めない GGUF を取得した上、取得済みパスの argv 差し込みが最初の `-m` の直後を書き換えるため `python -m mlx_lm.server` の `-m` がモデルフラグと衝突し、`python -m <GGUFスナップショットパス>` となって `ModuleNotFoundError` で即死していました。`-m` の差し込みは llama.cpp 専用に限定し、mlx/vLLM は `--model`/位置引数を使うように修正し、ダウンロード先リポジトリも backend 別に解決するようにしました。さらにカタログに `hf_mlx` ソースキー（mlx-community の MLX 形式リポジトリ）を追加し、mlx がフル精度 HF リポジトリ（7B で約 15GiB）ではなくプランの量子化見積りに一致する 4bit 変換版を供給するようにしました。修正後、実機で `doctor`（Apple Silicon・unified メモリ・mlx=installed を検出）→ `plan`（mlx 選択）→ `up --detach` → ゲートウェイ経由の `/v1/chat/completions` が実応答を返すまで完走しました。なお `mlx_lm.server` は `/v1/embeddings` を持たず bge-m3 等の埋め込みモデルは起動できないため、embed 役割の mlx 計画は従来どおり警告付きのままです。
- `nmesh autostart --install` はランチャーと `gateway.env` だけを書き込み、unit 本文を表示するだけでした。そのため表示される `systemctl --user enable --now ~/.config/systemd/user/nmesh-gateway.service` が参照するファイルが存在せず、Linux で自動起動の設定が完了できませんでした（macOS の plist も同様）。`--install` は unit/plist を実際の保存先（`$XDG_CONFIG_HOME/systemd/user/` または `~/Library/LaunchAgents/`）に書き込み、保存先を `--json` の `unit_path` と表示で明示するようにしました。enable コマンドのパスも XDG_CONFIG_HOME を反映した実パスを出します。
- `nmesh up` が、計画時に解決されたエンジンバイナリ（例: `engines/llamacpp/b10955/.../llama-server`）が `nmesh engine remove` や入替で消えていた場合、素の `FileNotFoundError` で落ちていました。起動前に argv[0] の実在を確認し、無ければインストール済みエンジン（active 優先）へ付け替えて起動し、置換は status の note と計画の永続化で明示するようにしました。
- `nmesh engine install`（タグ未指定）が、アセットがまだ1件も公開されていない最新リリースを選んで即座に失敗していました（実際に b10975 が「published assets: none」で新規インストール不能を確認）。アセットを公開している最新タグを選ぶようにしました。明示 `--version` 指定時は従来どおり正直にエラーにします。
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
- GPU 実機・vLLM は未検証です。MLX は Apple Silicon（M4）上で chat 完走済みですが、`mlx_lm.server` は埋め込みエンドポイントを持たないため embed 役割は提供できません。
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