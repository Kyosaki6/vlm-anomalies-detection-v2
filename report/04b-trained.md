# Lần chạy đã huấn luyện

Fork này đã được huấn luyện bằng Berkeley RAFT (arXiv:2403.10131): mô hình trích các clip láng giềng rồi buộc phải trích dẫn đúng clip trước khi gọi tên lớp. Lượt holdout không vượt class prior, nên đánh giá 290 video không được chạy.

| Mã | ROC-AUC (%) [95% CI] |
|---|---|
| A4c | 73.00 [64.45, 81.15] |
| A6 | 67.90 [58.39, 77.49] |
| R0 | not measured |
| R1 | not measured |
| R2 | not measured |
| R3 | not run (citation accuracy 2.11%, class prior 26.32%) |
