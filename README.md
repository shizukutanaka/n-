# nmesh

nmesh は、ローカル LLM のためのハードウェア自動検出、モデル選択、メモリ見積もり、
実行管理ツールです。ノート PC から複数 GPU のワークステーションまで構成を自動で計画し、
OpenAI 互換 API を提供します。

## クイックスタート

```text
pip install -e .
nmesh doctor
nmesh plan
nmesh up
```

Gateway は `http://127.0.0.1:18000/v1` の OpenAI 互換エンドポイントで待ち受けます。

## ハードウェア tier

| Tier | 条件 |
|---|---|
| T0_CPU | GPU なし、または VRAM 2 GiB 未満 |
| T1_LOW | 2–6 GiB |
| T2_MID | 6–12 GiB |
| T3_HIGH | 12–24 GiB |
| T4_WORKSTATION | 24–48 GiB |
| T5_SERVER | 48 GiB 以上、または GPU 2 枚以上 |

## バックエンドとモデル

Ollama、llama.cpp、vLLM、MLX-LM をサポートします。未インストールでも
`nmesh plan` は候補と具体的なインストール方法を表示します。
モデルの追加や上書きは `~/.nmesh/models.yaml` に bundled catalog と同じ形式で記述します。
同じ `id` のモデルはユーザー定義が優先されます。

## CI workflow

GitHub の権限がある利用者は `ci/github-workflow-ci.yml` を
`.github/workflows/ci.yml` にコピーして使用してください。
