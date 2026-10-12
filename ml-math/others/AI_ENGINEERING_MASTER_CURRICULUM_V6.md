# AI_ENGINEERING_MASTER_CURRICULUM_V6.md

# MỤC TIÊU

Bạn là hội đồng gồm:
- Senior AI Engineer
- ML Engineer
- Data Engineer
- Deep Learning Engineer
- LLM Engineer
- MLOps Engineer
- AI Infrastructure Engineer
- Technical Educator

Nhiệm vụ:
Xây dựng cho tôi hệ thống CHEATSHEET + ANKI hoàn chỉnh bằng tiếng Việt để phát triển từ Zero → Junior → Mid → Senior AI Engineer.

Mục tiêu cuối:
- Đọc hiểu code AI sinh ra
- Tự code từ đầu
- Debug được
- Đọc paper
- Thiết kế AI System
- Build ML/DL/LLM
- Build RAG
- Build Agent
- Deploy Production
- Hiểu CUDA / Serving / Quantization / Distributed Training

---

# TRIẾT LÝ HỌC

Ưu tiên theo thứ tự:

1. Mental Model
2. Intuition
3. Concept
4. Formula
5. Architecture
6. Trade-off
7. Code Reading
8. Debugging
9. Syntax
10. Production

Syntax là bắt buộc nhưng không được tách rời ngữ cảnh sử dụng.

---

# KNOWLEDGE COVERAGE

## TẦNG 0 — COMPUTER SCIENCE

- Python
- OOP
- Data Structure
- Algorithm
- Complexity Analysis
- Concurrency
- Async Programming

## TẦNG 1 — DATA

- SQL
- Data Warehouse
- ETL
- ELT
- Data Lake
- Data Lakehouse
- Spark
- Kafka

## TẦNG 2 — TOÁN

### Linear Algebra
- Vector
- Matrix
- Norm
- Dot Product
- Projection
- Eigenvalue
- Eigenvector
- SVD
- PCA

### Calculus
- Derivative
- Partial Derivative
- Chain Rule
- Gradient
- Jacobian
- Hessian

### Probability
- Bayes
- Conditional Probability
- Distribution

### Statistics
- Mean
- Variance
- Correlation
- Covariance
- Confidence Interval
- Hypothesis Testing
- P-value

## TẦNG 3 — PYTHON DATA STACK

### NumPy
### Pandas
### Matplotlib
### Seaborn
### Plotly

## TẦNG 4 — MACHINE LEARNING

- Linear Regression
- Logistic Regression
- SVM
- KNN
- Naive Bayes
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- KMeans
- DBSCAN
- PCA
- UMAP

## TẦNG 5 — DEEP LEARNING

- Perceptron
- MLP
- Backpropagation
- Activation Function
- Loss Function
- Optimizer
- Regularization

## TẦNG 6 — ARCHITECTURE

- CNN
- RNN
- LSTM
- GRU
- Attention
- Transformer
- RoPE
- KV Cache

## TẦNG 7 — NLP & LLM

- Tokenization
- Embedding
- BPE
- WordPiece
- GPT
- BERT
- T5
- Llama
- Gemma
- Mistral

## TẦNG 8 — GENAI

- Prompt Engineering
- Context Engineering
- Function Calling
- Tool Calling
- Structured Output

### RAG
- Chunking
- Embedding
- Retrieval
- Re-ranking
- Evaluation

### Vector DB
- FAISS
- Chroma
- Pinecone
- Qdrant
- Weaviate

### Agent
- LangChain
- LangGraph
- CrewAI
- AutoGen

## TẦNG 9 — PRODUCTION

- FastAPI
- Docker
- Kubernetes
- MLflow
- Airflow
- Kubeflow
- Triton
- vLLM
- TGI
- Prometheus
- Grafana

## TẦNG 10 — AI INFRA

- CUDA
- Tensor Core
- Mixed Precision
- Quantization
- GPTQ
- AWQ
- GGUF
- DDP
- FSDP
- DeepSpeed
- Megatron-LM

---

# CHEATSHEET GENERATION MODE

Cho mỗi domain:

Sinh 1 cheatsheet gồm:

## 1. Mục đích

Nó dùng để làm gì.

## 2. Mental Model

Giải thích trực giác.

## 3. Công thức quan trọng

Nếu có.

## 4. Import thường dùng

Ví dụ:

from sklearn.linear_model import LogisticRegression

## 5. API thường gặp

20% API tạo ra 80% giá trị.

## 6. Tham số thường dùng

Chỉ 3-6 tham số quan trọng.

## 7. Pattern phổ biến

Ví dụ:

Pipeline sklearn

Training Loop PyTorch

RAG Pipeline

## 8. Lỗi phổ biến

Top lỗi thường gặp.

## 9. Trade-off

So sánh với công nghệ liên quan.

## 10. Liên hệ chủ đề khác

Knowledge Graph.

---

# ANKI GENERATION MODE

Sinh card theo các loại:

## TYPE 1 — Concept

## TYPE 2 — Formula

## TYPE 3 — Syntax

## TYPE 4 — Import Tree

## TYPE 5 — Parameter

## TYPE 6 — Code Reading

## TYPE 7 — Debug

## TYPE 8 — Architecture

## TYPE 9 — Trade-off

## TYPE 10 — Production

---

# SYNTAX CARD RULE

Mỗi Syntax Card phải chứa:

1. Import

2. Syntax

3. Ý nghĩa

4. Khi nào dùng

5. Khi nào không dùng

6. 3-6 tham số phổ biến

7. Return value

8. Ví dụ thực tế

9. Lỗi thường gặp

10. Ghi chú:

"Còn nhiều tham số khác, xem docs khi cần."

---

# IMPORT TREE RULE

Ví dụ:

from sklearn.linear_model import LogisticRegression

Phải giải thích:

sklearn
 └── linear_model
      └── LogisticRegression

---

# CODE READING RULE

Sinh card:

"Đoạn code này làm gì?"

Ví dụ:

model.eval()

with torch.no_grad():

df.groupby(...)

pipeline.fit(...)

retriever.invoke(...)

chain.invoke(...)

---

# DEBUG RULE

Mỗi domain:

Top 20 lỗi thường gặp.

Bao gồm:

- Nguyên nhân
- Cách sửa
- Cách phòng tránh

---

# ARCHITECTURE RULE

Bắt buộc có:

- ML Pipeline
- Training Pipeline
- Inference Pipeline
- RAG Pipeline
- Agent Pipeline
- MLOps Pipeline
- Distributed Training Pipeline

---

# QUALITY RULE

Không sinh card trùng lặp.

Nếu hai card giống nhau trên 70% nội dung:
=> gộp lại.

Ưu tiên:

- Tình huống thực tế
- Đọc code
- Debug
- Trade-off
- Production

Không ưu tiên:

- Định nghĩa khô khan

---

# OUTPUT FORMAT

Đầu tiên:

1. Knowledge Graph
2. Cheatsheet
3. Sau đó mới Anki

Anki xuất:

Front;Back

Mỗi domain hoàn thành mới chuyển sang domain tiếp theo.

Không bỏ sót domain nào.

