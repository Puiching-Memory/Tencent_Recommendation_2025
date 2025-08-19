BaseLine:
```
Score: 0.0267527
NDCG@10: 0.0206547
HitRate@10: 0.0403256
---
batch_size=128
lr=0.001
maxlen=101
hidden_units=32
num_blocks=1
num_epochs=3
num_heads=1
dropout_rate=0.2
l2_emb=0.0
---
mm_emb_id=['81']
```

BaseLine-0819:
```
Score: 0.0165481
NDCG@10: 0.0125492
HitRate@10: 0.025449
---
batch_size=128
lr=0.001
maxlen=101
hidden_units=64
num_blocks=2
num_epochs=6
num_heads=4
dropout_rate=0.1
l2_emb=0.001
---
mm_emb_id=['81', '82', '83', '84']
```