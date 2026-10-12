# SCHEMA — CSV Anki cho toàn bộ `cheatsheet-anki-vn/`

> Schema ổn định 8 cột, dùng cho **mọi file CSV** trong mọi phase. Validate bằng `_tools/csv_validator.py` trước khi commit.

---

## 1. Đặc tả file CSV

| Thuộc tính | Giá trị |
|-----------|---------|
| Encoding | UTF-8 (có BOM cũng OK — validator tự bỏ) |
| Delimiter | `,` (comma) |
| Quote char | `"` |
| Escape | `""` (double-quote doubling) |
| Line ending | `\n` (Unix) hoặc `\r\n` (Windows) |
| Header row | **Bắt buộc** (dòng 1) |
| Số cột | 8 (cố định) |
| Số dòng dữ liệu | 1 dòng = 1 thẻ Anki |

**Lý do dùng comma (không phải tab)**: Anki desktop import mặc định nhận comma CSV, hỗ trợ tốt với multiline field do quote. Tab dễ bị dính khi copy từ Excel.

---

## 2. 8 cột (theo thứ tự)

| # | Tên cột | Bắt buộc | Kiểu | Mô tả |
|---|---------|----------|------|-------|
| 1 | `Front` | ✅ | string | Câu hỏi tình huống KHÔNG hỏi suông. Ra đề ở dạng "Bạn có X, muốn Y — làm sao?" |
| 2 | `Back` | ✅ | string (HTML+MathJax) | Đáp án chứa đủ 5 phần (xem §3). Hỗ trợ HTML + MathJax |
| 3 | `Tags` | ✅ | string (space-separated) | Tag Anki hierarchical `math::linear-algebra::svd`. Nhiều tag cách nhau bằng khoảng trắng |
| 4 | `Deck` | ✅ | string | Tên deck hierarchical `AI::01-Math::Linear-Algebra` |
| 5 | `CardType` | ✅ | enum | Một trong: `Basic`, `Cloze`, `TypeIn`, `Reversed` (xem `_templates/anki-card-templates.md`) |
| 6 | `Source` | ⚪ | string | File/line tham chiếu để quay tra cứu. Ví dụ `01-linear-algebra.md#svd` |
| 7 | `Difficulty` | ✅ | enum | Một trong: `easy`, `medium`, `hard` |
| 8 | `Linked` | ⚪ | string | ID/logic thẻ liên quan. Ví dụ `svd -> pca` để build graph liên hệ |

---

## 3. Trường `Back` — 5 phần bắt buộc (nếu áp dụng được)

```
1. **Cú pháp / code mẫu** đúng, chạy được (code block python)
2. **Giải thích ý nghĩa** — hàm/concept làm gì, tại sao dùng ở tình huống này
3. **Tham số chính / phổ biến nhất** (3-6 tham số hay gặp) — KHÔNG liệt kê toàn bộ signature
4. Ghi chú cuối: "Còn nhiều tham số/biến thể, xem docs khi cần"
5. **Trường hợp dùng thực tế** — 1 ví dụ ứng dụng AI cụ thể
6. (Nếu có công thức toán) Công thức MathJax + 1 câu trực giác hình học trước khi vào ký hiệu
```

**KHÔNG ép đủ hết 6 phần** nếu concept không có công thức/code. Thẻ syntax-harness mạnh phần 1-3-5; thẻ concept mạnh phần 2-6; thẻ architecture mạnh 1-2-5.

---

## 4. Quy ước MathJax trong `Back`

Anki desktop có **MathJax add-on** mặc định + hỗ trợ sẵn từ version 2.1+. Render inline và block:

| Loại | Cú pháp | Ví dụ |
|------|---------|------|
| Inline | `\(...\)` | Đạo hàm sigmoid: `\(\sigma'(x) = \sigma(x)(1-\sigma(x))\)` |
| Block (centered) | `$$...$$` | `$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$` |

**KHÔNG dùng** `$...$` (single dollar) — có thể xung đột với ký hiệu tiền tệ trong code. Dùng `\(...\)` cho inline.

**Lưu ý khi viết CSV**: ký tự `\` trong MathJax **phải giữ nguyên** (không escape double trong CSV). Validator chỉ kiểm tra `\\(` → cảnh báo.

---

## 5. Quy ước `Tags` (hierarchical Anki)

- Format: `<phase>::<topic>::<subtopic>` (nhiều cấp độ OK)
- Lowercase, dùng gạch-nối `-` cho từ phức, gạch hai chấm `::` phân cấp
- Nhiều tag cách nhau bằng khoảng trắng: `math::linear-algebra::svd math::linear-algebra::pca`
- Tag đặc biệt (prefixed `xx-`): `xx-important`, `xx-from-scratch`, `xx-pitfall`, `xx-tradeoff` để filter

**Ví dụ**:
- `math::linear-algebra::svd` — SVD trong linear algebra của Phase 1
- `ml::ensembles::xgboost xx-important` — XGBoost trong ensembles, quan trọng
- `dl::transformer::attention xx-from-scratch` — attention từ scratch

---

## 6. Quy ước `Deck` (hierarchical Anki)

- Format: `AI::<phase-number>-<phase-name>::<topic>`
- Số phase 2 chữ số: `01`, `02`, ... `06`
- Topic dùng PascalCase hoặc gạch-nối — thống nhất dùng **gạch-nối** cho topic dài

**Danh sách deck Phase 1**:
- `AI::01-Math::Linear-Algebra`
- `AI::01-Math::Calculus-Optimization`
- `AI::01-Math::Probability`
- `AI::01-Math::Statistics`
- `AI::01-Math::Information-Theory`
- `AI::01-Math::Formulas-From-Scratch`

**Danh sách đề xuất phase 2-6** (xem `MASTER-PLAN.md` bảng progress).

---

## 7. Quy ước `CardType`

| CardType | Khi nào dùng | Đặc điểm Anki |
|----------|-------------|--------------|
| `Basic` | Q/A thông thường (syntax, hàm, concept) | 1 mặt trước / 1 mặt sau |
| `Cloze` | Điền khuyết công thức, code pattern, đa khuyết | Dùng `{{c1::...}}`, `{{c2::...}}` trong Back; tạo nhiều card từ 1 note |
| `TypeIn` | Gõ chính xác syntax (named function, import) | Có ô nhập, Anki so khớp chính xác |
| `Reversed` | Đảo chiều nhận diện (concept ← thực tế / symptom ← fix) | Basic + reverse (2 card) |

**Tỷ lệ mix đề xuất mỗi deck**:
- Syntax-harness deck (sklearn, PyTorch, LangChain): ~60% Basic, 20% Cloze, 15% TypeIn, 5% Reversed
- Concept deck (Toán, Information Theory): ~40% Basic, 30% Cloze, 5% TypeIn, 25% Reversed
- Architecture deck (Transformer, RAG): ~50% Basic, 30% Cloze, 0% TypeIn, 20% Reversed

---

## 8. Quy ước `Difficulty`

| Level | Khi nào | Anki custom study |
|-------|---------|-------------------|
| `easy` | Concept nền, syntax cơ bản, công thức đã quen | Không ưu tiên |
| `medium` | Áp dụng thực tế, hàm thông dụng, pattern phổ biến | Default |
| `hard` | Edge case, từ scratch, công thức phức tạp, tradeoff sâu | Ưu tiên review |

Anki tag tương ứng: `difficulty::easy`, `difficulty::medium`, `difficulty::hard` (tự thêm vào cột Tags khi cần filter).

---

## 9. Quy ước `Source` và `Linked`

- `Source`: `<cheatsheet-file>#<anchor>`. Ví dụ: `01-linear-algebra.md#svd`, `02-optimizers.md#adam`. Dùng anchor tự sinh từ heading.
- `Linked`: ID tự do viết tay. Ví dụ:
  - `svd -> pca` (SVD liên hệ PCA)
  - `grad-desc <- chain-rule` (gradient descent cần chain rule)
  - `attention <- softmax` (attention cần softmax)
- Sẽ có script build network graph từ `Linked` nếu cần (chưa build trong V1).

---

## 10. Ví dụ thẻ CSV mẫu (3 dòng đầy đủ)

```csv
"Front","Back","Tags","Deck","CardType","Source","Difficulty","Linked"
"Bạn có ma trận A kích thước (m, n) với m > n và muốn tìm phân rã A = U·diag(s)·V^T. Dùng hàm numpy nào, kết quả trả về có hình th ái gì?","<h3>SVD bằng numpy</h3><pre><code>import numpy as np\nU, s, Vt = np.linalg.svd(A, full_matrices=False)</code></pre><p><b>Ý nghĩa</b>: Phân rã A thành 3 ma trận — U (m, k), s (k,), Vt (k, n) với k = min(m, n). s là trị suy biến đã sắp xếp giảm dần.</p><p><b>Tham số chính</b>:</p><ul><li><code>full_matrices</code> (bool, default=True): True trả U (m, m), Vt (n, n); False rút gọn thành U (m, k), Vt (k, n) — <b>nên dùng False</b> khi m ≫ n để tiết kiệm RAM.</li><li><code>compute_uv</code> (bool, default=True): False chỉ trả trị suy biến s, bỏ U/Vt.</li><li><code>hermitian</code> (bool, default=False): True nếu A là Hermitian, dùng thuật toán nhanh hơn.</li></ul><p><i>Còn nhiều tùy chọn khác, xem docs numpy.linalg.svd khi cần.</i></p><p><b>Dùng thực tế</b>: PCA (giảm chiều dữ liệu), nén ảnh, recommend system (collaborative filtering), trong LLM cho Low-Rank Adaptation (LoRA dùng A ≈ U·S·V^T để nén weight).</p><p><b>Công thức</b>: $$A = U \Sigma V^T$$ <i>Trực giác</i>: phép quay (U) + co giãn theo từng trục (Σ) + phép quay (V^T) — phân rã phép biến đổi tuyến tính thành 3 bước: xoay, kéo giãn theo trục chính, xoay tiếp.</p>","math::linear-algebra::svd xx-from-scratch xx-important","AI::01-Math::Linear-Algebra","Basic","01-linear-algebra.md#svd","hard","svd -> pca svd -> lora"
"Bạn cần hạng ma trận (rank) của A. Từ SVD, rank(A) bằng gì?","<h3>Rank từ SVD</h3><pre><code>U, s, Vt = np.linalg.svd(A, full_matrices=False)\nrank = np.sum(s > 1e-10)</code></pre><p><b>Rank(A)</b> = số trị suy biến khác 0 (ứng dụng ngưỡng để ổn định số). Trong thực tế dùng ngưỡng <code>tol = max(A.shape) * eps</code> với <code>eps</code> độ chính xác float.</p><p><b>Dùng thực tế</b>: Detect đa cộng tuyến trong LinReg (rank < số feature → ma trận X^T X kỳ dị → dùng pinv thay inv).</p><p><b>Công thức</b>: $$\text{rank}(A) = \#\{i : \sigma_i > 0\}$$ với \(\sigma_i\) là trị suy biến.</p>","math::linear-algebra::svd math::linear-algebra::rank","AI::01-Math::Linear-Algebra","Cloze","01-linear-algebra.md#svd-rank","medium","rank -> pinv"
"Viết câu lệnh numpy tính SVD rút gọn của ma trận X (data), không trả U/Vt.","import numpy as np\n\ns = np.linalg.svd(X, full_matrices=False, compute_uv=False)","math::linear-algebra::svd","AI::01-Math::Linear-Algebra","TypeIn","01-linear-algebra.md#svd-api","easy",""
```

**Lưu ý khi viết CSV**:
- Mọi field có ký tự đặc biệt (`,`, `"`, newline) phải quote bằng `"` và escape `""` bên trong
- Field có newline (như `Back` chứa code block nhiều dòng) **bắt buộc** quote
- HTML trong `Back`: dùng `<h3>`, `<p>`, `<pre><code>`, `<ul><li>`, `<b>`, `<i>`. Anki render đẹp hơn markdown thô
- Code block dùng `<pre><code>...</code></pre>` (escape `<` thành `<`, `>` thành `>` nếu code có tag HTML)

---

## 11. Validate

Chạy:
```powershell
python _tools\csv_validator.py 01-math\anki-csv\*.csv
```

Validator kiểm tra:
1. File CSV parse OK (delimiter `,`, quote `"`, encoding UTF-8)
2. Đúng 8 cột tên header
3. Không field `Front` / `Back` / `Tags` / `Deck` / `CardType` / `Difficulty` rỗng
4. `CardType` thuộc 4 giá trị enum
5. `Difficulty` thuộc 3 giá trị enum
6. `Deck` khớp format `AI::XX-Name::Topic`
7. `Tags` chứa ít nhất 1 tag hierarchical
8. Dedupe `Front` trong cùng deck (cảnh báo nếu trùng)
9. MathJax balance: đếm `\(` / `\)` và `$$` (cảnh báo nếu lẻ)
10. Báo cáo: tổng thẻ / deck / card type mix / difficulty mix

Chi tiết implementation: `_tools/csv_validator.py`.
