# Data (not committed)

Produced by Member 2's `src/preprocess.py`. See `docs/interfaces.md` for the file formats.

| file | content |
|---|---|
| `tinystories_{train,val,test}.bin` | uint16 token ids |
| `dolly_{train,val,test}.jsonl` | `{"id","instruction","context","response","category"}` |
| `split_manifest.json` | seeds, sizes, hashes of each split |

How to obtain: TinyStories (`roneneldan/TinyStories`) and Dolly 15K (`databricks/databricks-dolly-15k`)
from the Hugging Face Hub, then run the preprocessing script.
