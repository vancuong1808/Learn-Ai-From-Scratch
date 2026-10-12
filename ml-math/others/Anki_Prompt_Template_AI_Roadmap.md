# BỘ PROMPT TẠO ANKI — Roadmap AI/Data Zero to Hero
### (Dùng lại 1 template cho từng giai đoạn, không sinh "tất cả" trong 1 lần)

---

## PHẦN A — LỘ TRÌNH TỔNG (để biết thứ tự tạo bộ Anki)

Mỗi giai đoạn = 1 bộ deck Anki riêng. Học xong & review ổn giai đoạn trước rồi mới tạo bộ tiếp theo — đừng tạo nhiều bộ cùng lúc.

| # | Giai đoạn | Nội dung chính | Quy mô ước tính |
|---|---|---|---|
| 1 | **Python nền tảng** | syntax core, data structures, OOP, file/IO, error handling | ~80-120 thẻ |
| 2 | **Numpy/Pandas** | array ops, broadcasting, indexing, groupby, merge | ~100-150 thẻ |
| 3 | **Visualize** | Matplotlib/Seaborn/Plotly cú pháp + khi nào dùng loại chart nào | ~60-90 thẻ |
| 4 | **Toán cho AI** | Linear Algebra → Calculus → Probability (công thức + ý nghĩa hình học) | ~150-200 thẻ |
| 5 | **ML cổ điển** | sklearn: Linear/Logistic Regression, SVM, PCA, SVD, Decision Tree, Clustering | ~120-180 thẻ |
| 6 | **Deep Learning nền tảng** | PyTorch core, activation functions, loss functions, optimizer, backprop | ~100-150 thẻ |
| 7 | **DL kiến trúc** | CNN, RNN/LSTM, Attention, Transformer (từng block) | ~80-120 thẻ |
| 8 | **NLP/LLM** | Tokenization, Embedding, Fine-tuning, Prompt techniques | ~80-120 thẻ |
| 9 | **GenAI/LLM framework** | HuggingFace, LangChain, LangGraph, RAG pipeline | ~100-150 thẻ |
| 10 | **Vector DB & RAG nâng cao** | FAISS/Chroma/Pinecone, chunking, retrieval strategies | ~60-100 thẻ |
| 11 | **MLOps** | Experiment tracking, deployment, Docker cho ML, CI/CD cơ bản | ~60-100 thẻ |
| 12 | **CV (nếu cần)** | OpenCV, torchvision, các kiến trúc CV phổ biến | tùy nhu cầu |
| 13 | **RL (nếu cần)** | Gymnasium, các thuật toán RL cơ bản | tùy nhu cầu |

→ Tổng toàn bộ roadmap rơi vào khoảng **1,200-1,800 thẻ chất lượng cao**, khả thi để thực sự ôn và nhớ — thay vì 10,000+ thẻ không bao giờ review hết.

---

## PHẦN B — TEMPLATE PROMPT (copy & điền vào [GIAI ĐOẠN] mỗi lần)

```
Tôi đang tạo bộ Anki cho giai đoạn: [GIAI ĐOẠN, ví dụ: "Numpy/Pandas"]

Bối cảnh: Tôi là pre-AI Engineering, đang học theo roadmap Zero to Hero AI/Data 
bằng tiếng Việt, mục tiêu sau khi hoàn thành toàn bộ roadmap là code chay được 
và chỉ cần đọc thêm paper/docs để cập nhật kiến thức mới.

YÊU CẦU FORMAT MỖI THẺ ANKI:

Mặt trước (Q): Một TÌNH HUỐNG hoặc CÂU HỎI ỨNG DỤNG THỰC TẾ — không hỏi suông 
"cú pháp của X là gì". Ví dụ đúng: "Bạn có 1 DataFrame và muốn nhóm theo cột 
'category' rồi tính trung bình cột 'price' — dùng hàm nào, viết sao?"

Mặt sau (A), gồm đủ các phần sau (chỉ phần áp dụng được, không ép buộc đủ hết 
nếu chủ đề không có):
1. **Cú pháp/code mẫu** đúng, chạy được
2. **Giải thích ý nghĩa** — hàm này làm gì, tại sao dùng nó ở tình huống này
3. **Tham số chính/phổ biến nhất** (3-6 tham số hay gặp nhất trong thực tế và 
   trong code AI generate ra) — KHÔNG cần liệt kê toàn bộ signature
4. Ghi chú cuối thẻ: "Còn nhiều tham số/biến thể khác, xem docs khi cần" 
   (để tôi biết đây không phải toàn bộ, tránh ngộ nhận đã học hết)
5. **Trường hợp dùng thực tế** — 1 ví dụ ứng dụng đời thường hoặc dự án AI cụ thể
6. Nếu có công thức toán liên quan → viết công thức + 1 câu giải thích trực 
   giác hình học/hình ảnh trước khi vào ký hiệu

ĐỘ DÀI: Mỗi chủ đề nhỏ trong giai đoạn này → 2-4 thẻ là đủ (không tách lẻ 
từng tham số thành thẻ riêng, không nhồi 1 thẻ quá dài làm tôi lười đọc khi 
review hàng ngày).

TRÁNH:
- Không hỏi kiểu "định nghĩa X là gì" khô khan
- Không liệt kê đầy đủ TẤT CẢ tham số có thể có của hàm (làm loãng, sai lệch 
  trọng tâm, và nhiều khi không cần dùng thực tế)
- Không dùng thẻ để dạy tôi gõ lại từng ký tự syntax — thẻ này để tôi NHỚ 
  KHI NÀO DÙNG GÌ, syntax chính xác tôi sẽ luyện bằng cách code thật, không 
  phải học vẹt qua Anki

Đầu ra: xuất theo định dạng CSV (Front;Back) hoặc bảng tôi dễ import vào Anki, 
ghi rõ tên Deck đề xuất.

Chủ đề cụ thể trong giai đoạn này tôi muốn cover:
[LIỆT KÊ CÁC SUB-TOPIC, ví dụ: "groupby, merge/join, pivot_table, apply/map, 
boolean indexing, missing data handling"]
```

---

## PHẦN C — VÍ DỤ ÁP DỤNG (Giai đoạn 1: Python nền tảng)

```
Tôi đang tạo bộ Anki cho giai đoạn: Python nền tảng

[... giữ nguyên phần Bối cảnh + YÊU CẦU FORMAT như template trên ...]

Chủ đề cụ thể trong giai đoạn này tôi muốn cover:
- List/Dict/Set/Tuple: cách tạo, các method hay dùng, khi nào chọn loại nào
- List comprehension vs loop thường — khi nào nên dùng cái nào
- *args, **kwargs — ý nghĩa và tình huống dùng
- Decorator cơ bản
- Try/except — các pattern xử lý lỗi phổ biến
- Context manager (with...)
- Class cơ bản: __init__, self, inheritance
```

---

## GHI CHÚ QUAN TRỌNG

1. **Sau khi AI sinh thẻ, hãy đọc lướt và spot-check vài thẻ** trước khi import 
   hết vào Anki — đặc biệt với công thức toán và tham số, để tránh nhớ sai mà 
   không biết.
2. Syntax chính xác từng ký tự (dấu phẩy, thứ tự tham số) → **học qua việc 
   code thật, không qua Anki**. Anki giúp bạn nhớ "có hàm này, dùng khi nào, 
   tham số chính là gì" — phần còn lại tay sẽ nhớ khi gõ đủ nhiều lần.
3. Khi xong 1 giai đoạn và review ổn (Anki retention >85%), mới tạo bộ tiếp theo.
