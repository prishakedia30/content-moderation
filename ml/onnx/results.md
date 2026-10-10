Model: martin-ha/toxic-comment-model | threads: 2 | runs: 300 (latency includes tokenization)

| Backend | Size (MB) | p50 (ms) | p95 (ms) | p99 (ms) | Agrees with PyTorch (%) | Max prob diff |
|---|---|---|---|---|---|---|
| PyTorch FP32 | 267.8 | 35.97 | 51.53 | 146.31 | 100.0 | 0.0 |
| ONNX FP32 | 267.9 | 18.61 | 26.32 | 41.09 | 100.0 | 0.0 |
| ONNX INT8 | 67.4 | 3.26 | 4.86 | 7.12 | 100.0 | 0.0117 |
