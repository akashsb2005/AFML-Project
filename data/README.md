# Data (not committed)

Produced by Member 2's `src/preprocess.py`. See `docs/interfaces.md` for the file formats.

| file | content |
|---|---|
| `tiny_stories_{train,validation,test}.jsonl` | `{"text": ...}` |
| `dolly_{train,validation,test}.jsonl` | `{"prompt","response","category"}` |
| `tiny_stories_stats.json`, `dolly_stats.json` | split sizes and preprocessing counts |

How to obtain: TinyStories (`roneneldan/TinyStories`) and Dolly 15K (`databricks/databricks-dolly-15k`)
from the Hugging Face Hub, then run `python src/preprocess.py --stage all`.
