# Contributing

## 開発環境

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,gateway,download]"
```

## ゲート

コミット前に以下を全て通してください:

```bash
.venv/bin/ruff check .
.venv/bin/mypy nmesh
.venv/bin/pytest -q
```

現状の基準: ruff エラー0、mypy 63ファイル対象エラー0、pytest 全緑。

## 規約

- **正直さ優先**: 推定値・未検証値は必ず `estimated` 等で明示する。測定したふりをしない
- **i18n**: ユーザー向け文字列は `nmesh/i18n.py` の `MESSAGES` に en/ja 両方を追加。`t()` は欠落パラメータを許容する
- **最小差分**: 無関係なリファクタ・依存追加を避ける
- **ライセンス**: MIT（LICENSE 参照）
