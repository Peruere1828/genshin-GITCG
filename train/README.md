# train — SoG/ReBeL 训练内核（PLAN.md WS3 / M3）
cvpn_model（bf16 小网络 + deck-id 条件化）、continual_resolving + resolving_agent（公共信念搜索，先实用采样 resolver）、
cvpn_training + sog_pipeline（搜索蒸馏外环 + 门禁）、batched_inference（按作业拉起的 GPU 批推理，无常驻服务）。
