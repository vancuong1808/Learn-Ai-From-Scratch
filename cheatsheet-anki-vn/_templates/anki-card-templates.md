# Anki Card Templates — 4 loại thẻ cho toàn bộ `cheatsheet-anki-vn/`

> Kèm ví dụ mẫu cho từng loại. Khi import CSV vào Anki, cột `CardType` quyết định mapping sang note type. Setup mapping 1 lần trong Anki (xem `README.md`).

---

## 1. Tổng quan 4 loại thẻ

| CardType | Khi nào dùng | Anki Note Type mapping | Số card tạo / note |
|----------|-------------|----------------------|---------------------|
| `Basic` | Q/A thông thường (syntax, hàm, concept) | Basic | 1 |
| `Cloze` | Điền khuyết công thức / code pattern | Cloze | N (theo số `{{cN::}}`) |
| `TypeIn` | Gõ chính xác syntax (named fn, import) | Basic (type answer in field) | 1 |
| `Reversed` | Đảo chiều nhận diện concept ↔ thực tế | Basic (and reversed card) | 2 |

**Tỷ lệ mix đề xuất / deck** (xem `_schema/SCHEMA.md` §7):
- Syntax-harness deck (sklearn, PyTorch): 60% Basic / 20% Cloze / 15% TypeIn / 5% Reversed
- Concept deck (Toán, Info Theory): 40% Basic / 30% Cloze / 5% TypeIn / 25% Reversed
- Architecture deck (Transformer, RAG): 50% Basic / 30% Cloze / 0% TypeIn / 20% Reversed

---

## 2. Basic — Q/A thông thường

**Mục đích**: Hỏi tình huống, đáp án gồm cú pháp + giải thích + tham số + ứng dụng + (công thức nếu có).

### Ví dụ 1 — Syntax-harness

**Front**:
> Bạn có 1 DataFrame `df` gồm cột `category` (phân loại) và `price` (số). Muốn nhóm theo `category` rồi tính trung bình `price` — viết code numpy/pandas nào?

**Back** (*đủ 5 phần*):
> <h3>pandas groupby + mean</h3>
> <pre><code>df.groupby('category')['price'].mean()</code></pre>
> <p><b>Ý nghĩa</b>: Gom các dòng cùng <code>category</code> lại thành 1 nhóm, rồi tính trung bình cột <code>price</code> cho từng nhóm. Kết quả là Series indexed by category.</p>
> <p><b>Tham số chính</b>:</p>
> <ul>
> <li><code>by</code>: tên cột / list cột / function — khóa nhóm.</li>
> <li><code>as_index</code> (bool, default True): True trả Series indexed by group key; False giữ dạng DataFrame.</li>
> <li><code>sort</code> (bool, default True): sắp xếp key tăng dần — đặt False để chạy nhanh hơn nếu không cần thứ tự.</li>
> </ul>
> <p><i>Còn nhiều tùy chọn khác (<code>dropna, group_keys, observed</code>...), xem docs pandas.DataFrame.groupby khi cần.</i></p>
> <p><b>Dùng thực tế</b>: Tính giá trung bình nhà theo quận (EDA), tính conversion rate theo cohort (A/B test), nhóm customer theo segment rồi tính LTV.</p>

### Ví dụ 2 — Concept

**Front**:
> Tại sao softmax thường dùng ở lớp output cho bài toán classification nhiều lớp, thay vì sigmoid cho từng lớp độc lập? Nêu 1 lý do + 1 công thức.

**Back**:
> <h3>Softmax vs Sigmoid (multi-class)</h3>
> <p>Softmax <b>chuẩn hóa toàn bộ logits trên 1 mặt</b> sao cho tổng xác suất = 1, tạo quan hệ cạnh tranh giữa các lớp. Sigmoid độc lập không ràng buộc tổng → 2 lớp có thể cùng xác suất cao → không hợp lý khi chỉ 1 nhãn đúng.</p>
> <p><b> Công thức</b>: $$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$ <i>Trực giác</i>: mỗi logit <code>z_i</code>_JO đội lên qua <code>e^</code> (làm dương + phóng to khoảng cách), rồi chia cho tổng → đưa về phân phối xác suất.</p>
> <p><b>Dùng thực tế</b>: Lớp output của ResNet/VGG phân loại ImageNet 1000 lớp; output logits của LLM trước khi sample token tiếp theo (top-k/top-p sampling cũng hoạt động trên phân phối này).</p>
> <p><i>Trường hợp dùng sigmoid đa lớp: multi-label classification (1 ảnh có thể có nhiều nhãn: dog + pet + outdoor).</i></p>

---

## 3. Cloze — Điền khuyết

**Mục đích**: Nhớ công thức, code pattern, thứ tự thành phần. 1 note tạo nhiều card qua `{{c1::...}}`, `{{c2::...}}`...

### Quy ước
- `{{c1::đáp án::gợi ý}}` — gợi ý tùy chọn, hiển thị dạng `[...]` thay vì blank trống
- Mỗi `cN` độc lập tạo 1 card: review card 1 ẩn `c1`, card 2 ẩn `c2`, ...
- Nếu 2 chunk cùng `c1` (cùng số) → ẩn đồng thời trên 1 card (liên kết)

### Ví dụ 1 — Công thức + code

**Front** (chính là nội dung cùng có cloze deletion, sẽ được ẩn):
> SVD phân rã ma trận \(A \in \mathbb{R}^{m \times n}\) thành 3 ma trận:
> $$A = {{c1::U}} \cdot {{c2::\Sigma}} \cdot {{c3::V^T}}$$
> với `U` kích thước `(m, k)`, `\Sigma` kích thước `(k, k)`, `V^T` kích thước `(k, n)`, và `k = {{c4::\min(m, n)}}`.
>
> Numpy:
> <pre><code>U, s, Vt = np.linalg.{{c5::svd}}(A, full_matrices={{c6::False}})</code></pre>
> Chú ý numpy chỉ trả về vector trị suy biến `s`, để có ma trận `\Sigma` thì:
> <pre><code>np.diag(s)</code></pre>

→ Tạo **6 thẻ** Anki từ 1 note. Mỗi lần review ẩn 1 chunk.

### Ví dụ 2 — Pattern phổ biến

**Front**:
> Tuning learning rate phổ biến: warmup dùng để tránh gradient nổ ở đầu huấn luyện. Công thức LR cosine decay với warmup:
> $$LR(t) = \begin{cases} \frac{t}{T_{warmup}} \cdot LR_{max}, & t < T_{warmup} \\[4pt] \frac{1}{2} \left( 1 + \cos\left( \frac{t - T_{warmup}}{T_{total} - T_{warmup}} \pi \right) \right) \cdot LR_{max}, & t \geq T_{warmup} \end{cases}$$
>
> Trong PyTorch, dùng scheduler nào từ `torch.optim.lr_scheduler` cho pattern này:
> <pre><code>sched = {{c1::CosineAnnealingLR}}(opt, T_max=...)</code></pre>
> Tuy nhiên nếu cần warmup trước khi decay, dùng thư viện nào phổ biến:
> <pre><code>from {{c2::transformers}} import get_cosine_schedule_with_warmup</code></pre>

→ Tạo **2 thẻ**, yếu tố học: tên scheduler + thư viện.

---

## 4. TypeIn — Gõ chính xác syntax

**Mục đích**: Cường độ cao cho named function / import line. Anki so khớp chính xác chuỗi nhập với Back.

### Quy ước
- Front: câu hỏi "viết câu lệnh / import / khai báo X"
- Back: **chỉ** chuỗi đáp án chính xác, không kèm giải thích (để so khớp tốt)
- CardType = TypeIn, Anki note type Basic với option "Show answer: typed in" — kiểm tra chính xác

### Ví dụ 1 — Import

**Front**:
> Viết import line cho hàm Adam optimizer từ `torch.optim`.

**Back**:
> `from torch.optim import Adam`

### Ví dụ 2 — Khai báo layer

**Front**:
> Khai báo 1 MultiheadAttention layer trong PyTorch với embed_dim=512 và num_heads=8, gán vào biến `attn`.

**Back**:
> `attn = torch.nn.MultiheadAttention(embed_dim=512, num_heads=8)`

### Ví dụ 3 — sklearn

**Front**:
> Import Random Forest classifier từ sklearn.

**Back**:
> `from sklearn.ensemble import RandomForestClassifier`

**Lưu ý TypeIn**:
- Phân biệt hoa thường: Anki đánh green nếu khớp chính xác, red nếu lệch 1 ký tự → học chính xác
- Khuyến khích dùng TypeIn cho named function / API yang hay quên (sklearn, torch.nn, langchain import paths)

---

## 5. Reversed — Đảo chiều nhận diện

**Mục đích**: Học 2 chiều (concept ↔ symptom / fix ↔ problem / tên ↔ tác dụng). Anki note type "Basic (and reversed card)" tự tạo 2 card.

### Khi nào dùng Reversed:
- Gặp "tình huống X" → nhớ "khái niệm Y" **và** gặp "khái niệm Y" → nhớ "tình huống X"
- Symptom → fix và fix → symptom
- Tên thuật toán ↔ ý tưởng cốt lõi

### Ví dụ 1 — Symptom ↔ Fix

**Front**:
> Tình huống: Training loss giảm đều nhưng validation loss tăng dần sau epoch 30. Đây là triệu chứng gì? Nêu 3 cách khắc phục thường gặp.

**Back** (*enter để review chiều ngược — trở thành "overfitting"`s definition"→ Gặp tình huống này thì đó là gì`):
> Triệu chứng **overfitting**. Mô hình học được pattern + noise của train, không tổng quát được.
> **Cách khắc phục**:
> 1. **Regularization**: Dropout, weight decay (L2), early stopping
> 2. **Data augmentation**: tăng đa dạng training data
> 3. **Giảm capacity**: giảm số layer/channel, dùng model nhỏ hơn

→ Khi import, Anki tạo **2 thẻ**:
- Card 1: Front (tình huống) → Back (overfitting + cách khắc phục)
- Card 2: "Triệu chứng của overfitting là gì? Nêu cách khắc phục" → answer

### Ví dụ 2 — Công thức ↔ Ý nghĩa

**Front**:
> Công thức: $$\mathcal{L} = -\sum_i y_i \log \hat{y}_i$$ Đây là loss function gì? Dùng cho bài toán nào?

**Back**:
> **Cross-entropy loss** (cho multi-class). <br>
> Dùng cho classification khi output đã qua softmax và target là one-hot. <br>
> Đặc biệt trong LLM: token prediction là bài toán classification vocab 50K+ lớp → cross-entropy chuẩn.

→ 2 chiều:
- Thấy công thức → nhớ cross-entropy / dùng cho LLM
- "Loss cho LLM là gì" → nhớ công thức

---

## 6. Anti-patterns — KHÔNG nên làm

| ❌ Sai | ✅ Đúng |
|-------|--------|
| "Định nghĩa X là gì?" hỏi suông | "Bạn gặp Y, đó là X. Nêu cách khắc phục" tình huống |
| Liệt kê TOÀN BỘ signature hàm (10+ tham số) | Chỉ 3-6 tham số phổ biến nhất + ghi chú "còn nhiều khác, xem docs" |
| 1 thẻ 800+ chữ | 1 thẻ max ~200 chữ trên Back để review không bị ngán |
| Tách từng tham số thành 1 thẻ riêng | Gộp 2-4 thẻ / topic, mỗi thẻ cover 1 khía cạnh (syntax / concept / application / pitfall) |
| Single dollar `$...$` cho inline MathJax | `\(...\)` cho inline, `$$...$$` cho block (tránh xung đột tiền tệ) |
| Markdown thô trong `Back` | HTML (`<p>`, `<pre><code>`, `<ul><li>`, `<b>`) — Anki render đẹp hơn |

---

## 7. Quy ước viết `Back` chất lượng cao

Template HTML khuyến nghị (copy paste):

```html
<h3>Tiêu đề ngắn (hàm / concept)</h3>

<h4>Cú pháp</h4>
<pre><code>import numpy as np
# code mẫu chạy được
result = np.linalg.svd(A, full_matrices=False)</code></pre>

<h4>Ý nghĩa</h4>
<p>...</p>

<h4>Tham số chính</h4>
<ul>
<li><code>tên_param</code> (kiểu, default): mô tả.</li>
<li><code>tên_param</code> (kiểu, default): mô tả.</li>
</ul>

<p><i>Còn nhiều tham số/biến thể khác, xem docs khi cần.</i></p>

<h4>Dùng thực tế</h4>
<p>...</p>

<h4>Công thức (nếu có)</h4>
<p>$$...$$</p>
<p><i>Trực giác</i>: ...</p>
```

Sử dụng template giúp:
- Anki hiển thị đẹp nhất trên desktop / mobile
- Code block có monospace
- Tag `<i>` italic cho ghi chú phụ dễ phân biệt

---

## 8. Cấu hình Anki Import (1 lần)

Khi import CSV đầu tiên, Anki hỏi mapping. Save lại để import sau nhanh:

1. Type: **Basic** (cho CardType=Basic và TypeIn)
2. Type: **Cloze** (cho CardType=Cloze) — Anki tự nhận `{{cN::}}` trong field
3. Type: **Basic (and reversed card)** (cho CardType=Reversed)
4. Field mapping:
   - Column 1 (Front) → **Front**
   - Column 2 (Back) → **Back**
   - Column 3 (Tags) → **Tags** (Anki import dialog có ô Tags)
   - Column 4 (Deck) → **Deck** (qua "Deck" import dialog)
   - Column 6 (Source) → thêm field "Source" (tạo custom note type nếu cần)
   - Column 7 (Difficulty) → add tag `difficulty::easy/medium/hard`

Hoặc tạo custom note type **AIVN-Card** với các field: Front, Back, Source, Linked, Difficulty (chỉ cần tạo 1 lần trong Anki — chi tiết `README.md`).

---

## 9. Số thẻ mỗi topic con

Quy ước:
- Topic con phổ biến (vd: SVD): **2-4 thẻ**
  - 1 Basic syntax-harness
  - 1 Cloze công thức hoặc code pattern
  - 0-1 Reversed (concept ↔ ứng dụng)
  - 0-1 Basic pitfall / tradeoff
- Topic con nền (vd: vector norm): **1-2 thẻ** đủ nhớ
- Topic con khó / quan trọng (vd: backprop, attention): **3-5 thẻ** với pitfall + from scratch

Tổng / deck (ước Phase): ~80-120 thẻ / deck, cộng dồn ~550 thẻ Phase 1, ~1.800-2.850 toàn V1.

---

## 10. Quy tắc review (xem `Anki_Prompt_Template_AI_Roadmap.md` gốc)

- Mỗi ngày review tối đa 30-50 thẻ mới (không quá 100) — Anki algorithm ổn định
- Thẻ khó / xx-important ưu tiên đánh "Again" hoặc "Hard" tăng cường review
- Sau 2 tuần nếu thẻ easy vẫn ở interval > 21 ngày → đánh "Easy" để giảm tần suất
- Mỗi phase học xong → đóng deck không mới, review duy trì (Anki tổng hợp)
