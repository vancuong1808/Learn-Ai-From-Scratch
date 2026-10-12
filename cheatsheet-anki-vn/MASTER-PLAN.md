# MASTER-PLAN — Cheatsheet + Anki tiếng Việt cho AI/Data Zero→Hero

> Repo này xây dựng **hệ thống cheatsheet markdown (tra cứu sâu) + CSV Anki (review hàng ngày)** để đưa một người từ zero lên Junior/Senior AI Engineer. Ngôn ngữ: **tiếng Việt + giữ thuật ngữ tiếng Anh gốc** (gradient descent, backprop, embedding...) để dễ tra paper/docs.

---

## 1. Mục tiêu

Sau khi hoàn thành toàn bộ 6 phase, người học có thể:
- Đọc hiểu code AI do người / AI khác sinh ra
- Tự code AI từ scratch (xử lý data → visualize → ML → DL → GenAI/LLM → Infra)
- Debug được
- Đọc paper
- Thiết kế AI System (RAG, Agent, MLOps, distributed training)
- Build ML / DL / LLM / RAG / Agent
- Deploy production (Docker, FastAPI, vLLM, quantization, distributed)

**Triết lý học (theo thứ tự ưu tiên):**
1. Mental Model → 2. Intuition → 3. Concept → 4. Formula → 5. Architecture →
6. Trade-off → 7. Code Reading → 8. Debugging → 9. Syntax → 10. Production

Syntax là bắt buộc nhưng **không tách rời ngữ cảnh sử dụng** — mỗi cú pháp gắn với tình huống dùng thực tế.

---

## 2. Cấu trúc repo

```
cheatsheet-anki-vn/
├── MASTER-PLAN.md                      ← bạn đang ở đây (index + progress)
├── README.md                            ← hướng dẫn dùng Anki + import CSV
├── _schema/SCHEMA.md                    ← schema 8 cột CSV + tag convention
├── _templates/anki-card-templates.md   ← 4 loại thẻ Anki + mẫu
├── _tools/csv_validator.py              ← validator CSV
├── 01-math/                             ← Phase 1: Toán
│   ├── cheatsheet/                      ← markdown tra cứu sâu
│   └── anki-csv/                        ← CSV import Anki
├── 02-machine-learning/                 ← Phase 2: ML
├── 03-deep-learning/                    ← Phase 3: DL
├── 04-genai-llm/                        ← Phase 4: GenAI/LLM
├── 05-rag-llm-frameworks/              ← Phase 5: RAG/Agent
└── 06-ai-infra-mlops/                   ← Phase 6: AI Infra + MLOps
```

Mỗi phase có 2 thư mục con:
- `cheatsheet/` — markdown tra cứu, viết sâu (công thức + import + API + tham số + pattern + pitfall + tradeoff + liên hệ)
- `anki-csv/` — file `.csv` có 8 cột theo `_schema/SCHEMA.md`, import trực tiếp vào Anki

---

## 3. Quy ược chung

| Quy ước | Giá trị |
|---------|---------|
| Ngôn ngữ | Tiếng Việt + giữ thuật ngữ TA gốc (gradient descent, backprop, embedding, attention...) |
| Cheatsheet format | Markdown, MathJax inline `\(...\)`, block `$$...$$`, code block ```python |
| CSV format | 8 cột, UTF-8, divider `,`, quote `"`, escape `""` (xem `_schema/SCHEMA.md`) |
| Anki deck naming | `AI::01-Math::Linear-Algebra` (hierarchical, parent `AI`) |
| Anki tags | `math::linear-algebra::svd` (hierarchical, lowercase, gạch-nối) |
| Card types | `Basic`, `Cloze`, `TypeIn`, `Reversed` (xem `_templates/anki-card-templates.md`) |
| Số thẻ / topic | 2-4 thẻ (không tách từng tham số thành thẻ riêng) |
| Tổng thẻ V1 | ~1.800-2.850 (không quá 3.000 để review được hết) |
| Validator | `_tools/csv_validator.py` chạy pass trước khi commit |

---

## 4. Pipeline 6 phases + progress board

### Phase 1 — TOÁN (đang build)

| # | Cheatsheet | CSV | Cards | Status | Session |
|---|-----------|-----|-------|--------|---------|
| 01 | `01-linear-algebra.md` | `01-linear-algebra.csv` | ~120 (mẫu 4) | ✅ partial | 1.A |
| 02 | `02-calculus-optimization.md` | `02-calculus-optimization.csv` | 26 | ✅ built | 1.B |
| 03 | `03-probability.md` | `03-probability.csv` | 33 | ✅ built | 1.B |
| 04 | `04-statistics.md` | `04-statistics.csv` | ~100 | 🔲 planned | 1.C |
| 05 | `05-information-theory.md` | `05-information-theory.csv` | ~50 | 🔲 planned | 1.C |
| 06 | `06-formulas-from-scratch.md` | `06-formulas-from-scratch.csv` | ~80 | 🔲 planned | 1.D |

**Phase 1 tổng**: 6 cheatsheet + 6 CSV, **~550 thẻ**

Topics chính:
- **Linear Algebra**: vector, matrix, dot/cross, norm (L0/L1/L2/inf), projection, trace, determinant, inverse/pseudo-inverse, eigendecomp, **SVD**, **PCA**, QR, LU
- **Calculus + Optimization**: derivative, partial, gradient, Jacobian, Hessian, chain rule, **GD/SGD/Mini-batch/Momentum/Adam from scratch**, Lagrangian, convex
- **Probability**: Bayes, conditional/marginal/joint, chain rule prob, distributions (Bernoulli/Binomial/Multinomial/Categorical/Normal/Multivariate Normal/Exp/Poisson/Beta/Gamma), sampling, Monte Carlo
- **Statistics** (tận dụng `ml-math/statistics/`): mean/median/mode, var/std, covariance/correlation, CI, hypothesis testing, p-value, ANOVA, **MLE/MAP**, bootstrap, A/B test
- **Information Theory**: entropy, cross-entropy, KL, JS divergence, mutual info — nền loss LLM
- **Formulas from scratch**: Sigmoid, tanh, ReLU/GELU/SwiGLU, softmax, LinReg normal eq, LogReg, SVM, SVD-code, PCA-code, Naive Bayes, KNN, KMeans

### Phase 2 — MACHINE LEARNING (sau Phase 1)

| Cheatsheet (đề xuất) | Cards ước |
|---------------------|-----------|
| 01-sklearn-fundamentals.md (Pipeline, preprocessing, model_selection, metrics) | ~50 |
| 02-linear-models.md (LinReg, LogReg, Ridge, Lasso, ElasticNet, SGDClassifier) | ~40 |
| 03-trees-ensembles.md (DecisionTree, RandomForest, GBDT, XGBoost, LightGBM, CatBoost) | ~60 |
| 04-svm-knn-naive-bayes.md (SVM, SVC/SVR, kernels, KNN, NB) | ~40 |
| 05-clustering-dimred.md (KMeans, DBSCAN, PCA, t-SNE, UMAP, TruncatedSVD) | ~40 |
| 06-model-evaluation.md (CV, metrics classification/regression, ROC/PR, calibration, hyperparam) | ~40 |
| 07-ml-from-scratch-concept.md (concept + công thức: LinReg, LogReg, SVM, KMeans, NB, KNN, RF, GBDT) | ~40 |

**Phase 2 tổng**: ~250-400 thẻ

### Phase 3 — DEEP LEARNING

| Cheatsheet (đề xuất) | Cards ước |
|---------------------|-----------|
| 01-pytorch-core.md (tensor, autograd, nn.Module, optimizer, dataloader, device) | ~80 |
| 02-tensorflow-keras.md (Tensor, Keras Sequential/Functional, callbacks, tf.data) | ~50 |
| 03-activation-loss.md (Sigmoid/tanh/ReLU/GELU/SwiGLU, CE/focal/MSE/huber/contrastive) | ~50 |
| 04-optimizers.md (SGD/Momentum/Adam/AdamW/RMSProp/LAMB, LR schedule) | ~40 |
| 05-cnn.md (Conv2D, Pooling, BatchNorm, ResNet/EffNet/MobileNet) | ~50 |
| 06-rnn-lstm-gru.md (RNN, LSTM, GRU, bidirectional) | ~40 |
| 07-attention-transformer.md (Self-attention, MHA, transformer block, positional encoding) | ~80 |
| 08-regularization.md (Dropout, weight decay, label smoothing, mixup, data aug) | ~30 |
| 09-dl-from-scratch.md (numpy: NN, CNN, attention, transformer) | ~50 |
| 10-training-patterns.md (init, grad clipping, mixed precision, gradient accumulation, EMA) | ~50 |

**Phase 3 tổng**: ~400-600 thẻ

### Phase 4 — GenAI / LLM

| Cheatsheet (đề xuất) | Cards ước |
|---------------------|-----------|
| 01-huggingface-fundamentals.md (transformers, datasets, tokenizers, accelerate) | ~60 |
| 02-tokenization-embeddings.md (BPE, WordPiece, SentencePiece, Tiktoken, embedding matrix) | ~40 |
| 03-transformer-architectures.md (Encoder-Decoder, Decoder-only, T5, BART, GPT, LLaMA cấu trúc) | ~50 |
| 04-fine-tuning-peft.md (full FT, LoRA, QLoRA, Adapter, Prefix-tuning, IA3) | ~50 |
| 05-rlhf-dpo.md (RLHF, PPO, DPO, ORPO, reward model) | ~30 |
| 06-quantization-inference.md (PTQ, GPTQ, AWQ, GGUF, KV-cache, vLLM, TGI) | ~40 |
| 07-prompt-engineering.md (CoT, ToT, ReAct, few-shot, system prompt, function calling) | ~40 |
| 08-llm-from-scratch-concept.md (concept + công thức attention, RoPE, MoE, GQA) | ~40 |

**Phase 4 tổng**: ~300-500 thẻ

### Phase 5 — RAG / LLM Frameworks / Agent

| Cheatsheet (đề xuất) | Cards ước |
|---------------------|-----------|
| 01-langchain-core.md (LCEL, chains, prompts, memory, retrievers) | ~50 |
| 02-langgraph-agent.md (state graph, nodes, edges, conditional, multi-agent) | ~40 |
| 03-llamaindex.md (indices, query engine, response synth, routers) | ~30 |
| 04-vectordb.md (Chroma, Pinecone, Weaviate, FAISS, Milvus, pgvector) | ~40 |
| 05-rag-fundamentals.md (chunking, embedding models, retrieval, reranking) | ~40 |
| 06-advanced-rag.md (HyDE, RAPTOR, multi-query, parent-doc, graph RAG, self-RAG) | ~50 |

**Phase 5 tổng**: ~250-400 thẻ

### Phase 6 — AI Infra + MLOps

| Cheatsheet (đề xuất) | Cards ước |
|---------------------|-----------|
| 01-docker-mlops.md (Dockerfile, compose, registry) | ~30 |
| 02-fastapi-serving.md (FastAPI, REST, streaming, batch) | ~30 |
| 03-experiment-tracking.md (MLflow, W&B, DVC) | ~30 |
| 04-pipeline-orchestration.md (Airflow, Prefect, Kubeflow) | ~30 |
| 05-distributed-training.md (DDP, FSDP, DeepSpeed, pipeline parallel) | ~40 |
| 06-monitoring-observability.md (drift, latency, Prometheus, Grafana) | ~20 |
| 07-k8s-cloud.md (Kubernetes, KServe, AWS/GCP/Azure basics) | ~30 |

**Phase 6 tổng**: ~200-350 thẻ

**📌 TỔNG V1**: ~1.800-2.850 thẻ, ~40 cheatsheet markdown

---

## 5. Quy trình build mỗi phase

```
1. Đọc curriculum (ml-math/AI_ENGINEERING_MASTER_CURRICULUM_V6.md) + tài liệu sẵn
2. Master topic index cho phase (≤120 topics con)
3. Build cheatsheet markdown sâu — công thức + import + API + tham số + pattern + pitfall + tradeoff + liên hệ
4. Generate CSV Anki (validator pass):
   - 2-4 thẻ/topic con, mix 4 loại (Basic/Cloze/TypeIn/Reversed)
   - Q tình huống KHÔNG hỏi suông, Back đủ 5 phần
5. csv_validator.py parse OK, dedupe Front, field không rỗng, tag quy ước
6. Cập nhật progress.md (đánh dấu ✅ trong bảng trên)
7. Spot-check 10 thẻ ngẫu nhiên (MathJax render, Q tình huống, Back đủ phần)
```

---

## 6. Quyết định đã chốt (không đổi trong V1)

- ✅ Ngôn ngữ: TV + giữ thuật ngữ TA gốc
- ✅ Lưu ở thư mục mới `cheatsheet-anki-vn/` (không đè `ml-math/`)
- ✅ Không sinh scratch code .py (tự code sau khi review card)
- ✅ Cheatsheet markdown + CSV Anki songsong
- ✅ Schema 8 cột ổn định cho toàn V1
- ✅ 4 loại thẻ: Basic / Cloze / TypeIn / Reversed
- ✅ Tổng ~1.800-2.850 thẻ (không quá 3.000)
- ✅ Build theo pipeline, Phase 1 Toán trước

---

## 7. Quy ước commit (nếu cần)

- 1 commit = 1 cheatsheet + 1 CSV + validator pass
- Message format: `add(phase-1): cheatsheet + anki-csv cho <topic> (<n> cards)`
- Ví dụ: `add(phase-1): cheatsheet + anki-csv cho linear-algebra (120 cards)`

---

## 8. Tài liệu tham chiếu tận dụng

- `ml-math/AI_ENGINEERING_MASTER_CURRICULUM_V6.md` — curriculum 436 dòng, topic checklist mỗi phase
- `ml-math/Anki_Prompt_Template_AI_Roadmap.md` — template prompt Anki (mở rộng thành `_templates/`)
- `ml-math/cheatsheets/cheatsheet_t0..t10_*.md` — seed content cho cheatsheet sâu
- `ml-math/statistics/*.md` (15 file) — reuse nguyên cho Phase 1 statistics

---

## 9. Lịch sử update

| Ngày | Nội dung |
|------|---------|
| 2026-07-06 | Khởi tạo repo cheatsheet-anki-vn/, viết MASTER-PLAN, SCHEMA, templates, csv_validator (Session 1.A) |
| 2026-07-06 | Session 1.B: cheatsheet + anki-csv cho calculus-optimization (26 cards) & probability (33 cards) — pass validator ALL CLEAN |

---

## 10. Bước tiếp theo (Session 1.C)

- Cheatsheet `01-math/cheatsheet/04-statistics.md` (tận dụng `ml-math/statistics/` 15 file sẵn)
- CSV `01-math/anki-csv/04-statistics.csv` (~100 thẻ)
- Cheatsheet `01-math/cheatsheet/05-information-theory.md` (entropy, cross-entropy, KL, JS, mutual info)
- CSV `01-math/anki-csv/05-information-theory.csv` (~50 thẻ)
- Chạy `csv_validator.py` pass, đánh ✅ progress board
