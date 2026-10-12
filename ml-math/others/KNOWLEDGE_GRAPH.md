# AI Engineering Knowledge Graph

Sơ đồ dưới đây trực quan hóa mối quan hệ, các tầng phụ thuộc từ căn bản đến chuyên sâu của Lộ trình AI Engineering (Tầng 0 đến Tầng 10) kết hợp cùng 3 bộ deck chuyên sâu: **AI-generated Code Reading**, **System Design**, và **Interview Preparation**.

---

## 1. Bản Đồ Mối Quan Hệ (Knowledge Graph)

```mermaid
graph TD
    %% Styling
    classDef foundation fill:#1a237e,stroke:#3f51b5,stroke-width:2px,color:#fff;
    classDef data fill:#004d40,stroke:#009688,stroke-width:2px,color:#fff;
    classDef math fill:#3e2723,stroke:#795548,stroke-width:2px,color:#fff;
    classDef ml fill:#827717,stroke:#c0ca33,stroke-width:2px,color:#fff;
    classDef dl fill:#b71c1c,stroke:#f44336,stroke-width:2px,color:#fff;
    classDef llm fill:#4a148c,stroke:#9c27b0,stroke-width:2px,color:#fff;
    classDef prod fill:#e65100,stroke:#ff9800,stroke-width:2px,color:#fff;
    classDef advanced fill:#1b5e20,stroke:#4caf50,stroke-width:2px,color:#fff;

    %% Nodes
    T0["<b>TẦNG 0: Computer Science</b><br>Python, OOP, Data Structures,<br>Algorithms, Concurrency, Async"]:::foundation
    T1["<b>TẦNG 1: Data</b><br>SQL, Data Warehouse, ETL/ELT,<br>Data Lake, Spark, Kafka"]:::data
    T2["<b>TẦNG 2: Toán cho AI</b><br>Linear Algebra, Calculus,<br>Probability, Statistics"]:::math
    T3["<b>TẦNG 3: Python Data Stack</b><br>NumPy, Pandas, Matplotlib,<br>Seaborn, Plotly"]:::foundation
    T4["<b>TẦNG 4: Machine Learning</b><br>Linear/Logistic, Tree Models,<br>XGBoost, KMeans, PCA"]:::ml
    T5["<b>TẦNG 5: Deep Learning</b><br>MLP, Backpropagation,<br>Activations, Optimizers"]:::dl
    T6["<b>TẦNG 6: Architecture</b><br>CNN, RNN/LSTM, Attention,<br>Transformer, RoPE, KV Cache"]:::dl
    T7["<b>TẦNG 7: NLP & LLM</b><br>Tokenization, Embedding,<br>GPT, BERT, Llama, Gemma"]:::llm
    T8["<b>TẦNG 8: GenAI & Agents</b><br>Prompt/Context Eng, RAG,<br>Vector DB, LangChain, LangGraph"]:::llm
    T9["<b>TẦNG 9: Production</b><br>FastAPI, Docker, K8s,<br>MLflow, Triton, vLLM, Airflow"]:::prod
    T10["<b>TẦNG 10: AI Infra</b><br>CUDA, Tensor Cores, Quantization,<br>DDP/FSDP, DeepSpeed"]:::prod

    %% Advanced Decks
    AD_CODE["<b>AI-generated Code Reading Deck</b><br>LangChain, LangGraph,<br>PyTorch, HuggingFace Code"]:::advanced
    AD_SYS["<b>System Design Deck</b><br>RAG, Agent, Multi-Agent,<br>MLOps, Serving Architectures"]:::advanced
    AD_INT["<b>Interview Deck</b><br>Junior, Mid, Senior<br>AI Engineer Scenarios"]:::advanced

    %% Dependencies
    T0 --> T1
    T0 --> T3
    T2 --> T4
    T3 --> T4
    T4 --> T5
    T2 --> T5
    T5 --> T6
    T6 --> T7
    T7 --> T8
    T1 --> T8
    T8 --> T9
    T9 --> T10
    T0 --> T9

    %% Advanced Deck Mappings
    T5 & T6 & T7 & T8 --> AD_CODE
    T8 & T9 & T10 --> AD_SYS
    T0 & T1 & T2 & T3 & T4 & T5 & T6 & T7 & T8 & T9 & T10 & AD_CODE & AD_SYS --> AD_INT
```

---

## 2. Chi Tiết Lộ Trình Tri Thức (Knowledge Paths)

Lộ trình được thiết kế nhằm đưa một kỹ sư đi từ nền tảng máy tính cơ bản nhất đến mức làm chủ toàn bộ hạ tầng và kiến trúc hệ thống AI hiện đại.

### Con Đường 1: Foundation & Data Flow (Nền Tảng & Luồng Dữ Liệu)
- **Computer Science (Tầng 0)** cung cấp nền tảng tư duy lập trình cấu trúc dữ liệu, thuật toán và bất đồng bộ (Async/Concurrency). Đây là điều kiện tiên quyết để làm việc với dữ liệu quy mô lớn ở **Data (Tầng 1)** và thao tác mảng tính toán hiệu năng ở **Python Data Stack (Tầng 3)**.
- Dữ liệu thô từ **Tầng 1** đi qua **ETL/ELT** sẽ làm đầu vào cho các mô hình học máy **Tầng 4**.

### Con Đường 2: Mathematics to Deep Learning (Toán học tới Học Sâu)
- **Toán cho AI (Tầng 2)** gồm Đại số tuyến tính, Giải tích và Xác suất thống kê tạo thành "xương sống" cho **Machine Learning (Tầng 4)** và **Deep Learning (Tầng 5)**. Không có toán học, việc tối ưu hóa trọng số (Gradient Descent, Backpropagation) ở Tầng 5 chỉ là lập trình vẹt.
- Từ **Deep Learning (Tầng 5)**, người học tiếp tục đi sâu vào các cấu trúc nâng cao tại **Architecture (Tầng 6)** (đặc biệt là cơ chế Attention và Transformer) để làm bước đệm cho **NLP & LLM (Tầng 7)**.

### Con Đường 3: GenAI & Productization (GenAI và Đưa vào Sản Phẩm)
- **NLP & LLM (Tầng 7)** dẫn lối trực tiếp đến kỷ nguyên **GenAI & Agents (Tầng 8)**, nơi lập trình viên tương tác với LLM thông qua Prompt, Context, Tool Calling và xây dựng các hệ thống Agentic (LangChain, LangGraph).
- Các ứng dụng GenAI cần được đưa lên môi trường thực tế thông qua **Production (Tầng 9)** nhờ containerization (Docker, Kubernetes) và tối ưu phục vụ (vLLM, Triton).
- Cuối cùng, để vận hành ở quy mô siêu lớn với chi phí tối ưu, **AI Infra (Tầng 10)** huấn luyện phân tán (DDP, DeepSpeed) và tối ưu hóa phần cứng (CUDA, Quantization) là nấc thang cao nhất của AI Engineer.

---

## 3. Vai Trò Của 3 Advanced Decks

3 Deck chuyên sâu được đề xuất đóng vai trò **cầu nối thực tế** chuyển hóa lý thuyết thành năng lực làm việc chuyên nghiệp:

1. **AI-generated Code Reading Deck**:
   - *Mục tiêu*: Khắc phục điểm yếu "hiểu lý thuyết nhưng không đọc được code thực tế". 
   - *Phạm vi*: Đọc và phân tích các mẫu code thực tế trong PyTorch (Custom training loops, Autograd hooks), HuggingFace (PEFT, Tokenizers), LangChain (LCEL chains), và LangGraph (State management).
   
2. **System Design Deck**:
   - *Mục tiêu*: Nâng tầm tư duy từ lập trình viên đơn thuần lên kiến trúc sư hệ thống AI.
   - *Phạm vi*: Thiết kế hệ thống RAG nâng cao (Reranking, Query Translation), Agentic workflow, Multi-Agent orchestration, hạ tầng MLOps (Experiment tracking, Feature Store) và các mô hình Serving có độ trễ cực thấp.

3. **Interview Deck**:
   - *Mục tiêu*: Hệ thống hóa toàn bộ kiến thức để sẵn sàng cho các vòng tuyển dụng kỹ thuật khắt khe.
   - *Phạm vi*: Các tình huống phỏng vấn thực tế từ mức Junior (thuật toán cơ bản, so sánh mô hình), Mid (xử lý dữ liệu, tinh chỉnh mô hình) đến Senior (thiết kế hạ tầng, phân tích trade-off và tối ưu hóa CUDA).
