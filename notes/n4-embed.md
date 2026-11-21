# N4 train embedding cache

| Field | Value |
|---|---|
| timestamp | 2026-11-21T13:55:44+07:00 start, done 15:32+07:00 |
| host | gx10@100.104.163.47, tmux `n4-embed`, log `/tmp/n4-embed.log` |
| model | google/siglip2-base-patch16-384 |
| fps | 1 |
| dtype | float16 |
| manifest | manifests/ucf_train.csv, 1610 rows, paths relinked to the real mirror layout |
| embedded this run | 1550/1550, 417137 frames, 97.1 min, 71.6 frames/s, 0 failures |
| already cached | 60, skipped by `precompute_embeddings.py` |
| check | `scripts/check_train_cache.py` exit 0: `count=1610 shards=1/1 dtype=float16` |
| commit | 5666838 |
| exit | 0 |
