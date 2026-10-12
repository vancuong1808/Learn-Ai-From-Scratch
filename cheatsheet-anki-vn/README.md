# README — Hướng dẫn dùng `cheatsheet-anki-vn/`

> Repo xây dựng cheatsheet markdown (tra cứu sâu) + CSV Anki (review hàng ngày) cho AI/Data Zero→Hero bằng tiếng Việt + giữ thuật ngữ TA gốc.

---

## 1. Tìm hiểu cấu trúc

```
cheatsheet-anki-vn/
├── MASTER-PLAN.md              ← BẮT ĐẦU TỪ ĐÂY: lộ trình 6 phase + progress
├── _schema/SCHEMA.md           ← Schema 8 cột CSV (bắt buộc đọc trước khi tạo CSV)
├── _templates/                 ← 4 loại thẻ Anki + ví dụ mẫu
├── _tools/csv_validator.py     ← Validator CSV
├── 01-math/ → 06-ai-infra-mlops/
│   ├── cheatsheet/             ← markdown tra cứu
│   └── anki-csv/               ← CSV import Anki
```

Cách học:
1. Đọc cheatsheet markdown khi cần tra cứu syntax / concept sâu
2. Review Anki CSV hàng ngày để **nhớ KHI NÀO DÙNG GÌ** (không phải nhớ syntax từng ký tự — syntax tra cheatsheet/code thật)

---

## 2. Cài đặt Anki (1 lần)

1. Tải Anki desktop từ https://apps.ankiweb.net/ (free)
2. Cài đặt, sync với AnkiWeb (free account)
3. (Tùy chọn) Cài AnkiDroid / AnkiMobile để review trên điện thoại

**Add-on khuyến nghị** (Tools → Add-ons → Get Add-ons):
- **MathJax** — render công thức `\( ... \)` và `$$ ... $$`. Anki 2.1+ hỗ trợ sẵn, không cần cài.
- **Hierarchical Tags** (code: `314447711`) — hiển thị tag hierarchical gọn trong browser
- **Review Heatmap** (code: `1771074083`) — visual streak review

---

## 3. Tạo custom note type "AIVN-Card" (1 lần)

Để import CSV với 8 cột chuẩn full schema, tạo note type tùy chỉnh:

1. Mở Anki → **Tools → Manage Note Types**
2. **Add** → Clone: **Basic** → tên `AIVN-Card`
3. **Fields...** → thêm/sửa các fields:
   - `Front` (mặc định)
   - `Back` (mặc định)
   - `Source`
   - `Linked`
   - `Difficulty`
4. **Cards...** (cho card templates):
   - Front Template: `{{Front}}`
   - Back Template:
     ```
     {{FrontSide}}<hr id=answer>
     {{Back}}
     <p><small><b>Source:</b> {{Source}}</small><br>
     <small><b>Linked:</b> {{Linked}}</small><br>
     <small><b>Difficulty:</b> {{Difficulty}}</small></p>
     ```
5. **Save**

Làm tương sung cho AIVN-Cloze (clone Cloze + thêm Source/Linked/Difficulty).

**Hoặc dùng note type có sẵn** của Anki nếu không cần field tùy chỉnh:
- `CardType=Basic` hoặc `TypeIn` → note type "Basic"
- `CardType=Cloze` → note type "Cloze"
- `CardType=Reversed` → note type "Basic (and reversed card)"

---

## 4. Tạo deck hierarchy (1 lần)

Trước khi import CSV, tạo sẵn các deck từ `MASTER-PLAN.md`:

Trong Anki main window → **Create Deck**:
- `AI::01-Math::Linear-Algebra`
- `AI::01-Math::Calculus-Optimization`
- `AI::01-Math::Probability`
- ... (xem MASTER-PLAN.md bảng Phase 1)

Anki tự sinh parent deck `AI` và `AI::01-Math`.

---

## 5. Import CSV (mỗi lần build xong)

1. Mở Anki → **File → Import**
2. Chọn file CSV (vd `01-math/anki-csv/01-linear-algebra.csv`)
3. Import dialog:
   - **Type**: `AIVN-Card` (hoặc Basic/Cloze tùy deck)
   - **Deck**: để `AI::01-Math::Linear-Algebra` (theo cột `Deck` trong CSV — có thể dùng "deck from CSV column")
   - **Field mapping**:
     - 1: Front
     - 2: Back
     - 3: Tags (mauto Anki nhận)
     - 4: Deck
     - 5: CardType → vào tag `cardtype::basic` (nếu không có custom note type)
     - 6: Source
     - 7: Difficulty
     - 8: Linked
   - **Tags**: Anki đọc từ cột Tags directly khi import
4. **Import** → kiểm tra log báo "X notes imported"

**Lưu ý**:
- Anki option "Field 1 is the first field" ✓
- Tag separator: **space** (default Anki)
- HTML in fields ✓ (allow HTML)
- Updated existing notes when first field matches ✓ (cho re-import khi sửa)

---

## 6. Validate CSV trước khi import (buộc)

```powershell
cd C:\Users\ADMIN\Documents\code\ai-from-scratch\cheatsheet-anki-vn
python _tools\csv_validator.py 01-math\anki-csv\*.csv
# hoac validate tat ca:
python _tools\csv_validator.py --all
```

Validator sẽ:
- Kiểm tra schema 8 cột đúng
- Field bắt buộc không rỗng
- CardType / Difficulty đúng enum
- Deck format đúng regex `AI::XX-Name::Topic`
- Tags có hierarchical
- Dedupe Front trong cùng deck
- MathJax balance `\(` / `\)` và `$$`
- Báo cáo mix: deck / CardType / Difficulty

Exit 0 = pass, exit 1 = error, exit 2 = usage error.

**Chỉ import vào Anki khi validator PASS.** (warning OK, error không.)

---

## 7. Review Anki hàng ngày

- Tối đa **30-50 thẻ mới/ngày** (quá nhiều → algorithm vỡ, không review kịp)
- Có AI::xx deck → review lúc nào cũng nhanh (mobile-friendly)
- Thẻ khó đánh "Again" hoặc "Hard" tăng cường (Anki sẽ hiện lại sớm)
- Sau 2 tuần, thẻ dễ ở interval > 21 ngày → đánh "Easy" giảm tần suất
- Tạo custom study "By difficulty: hard" 1 tuần trước interview / project

---

## 8. Quy ước viết CSV (cho contributor)

- Doc `_schema/SCHEMA.md` đầy đủ trước khi viết CSV
- Doc `_templates/anki-card-templates.md` cho ví dụ 4 loại thẻ
- MathJax: **inline `\( ... \)`**, **block `$$ ... $$`**, KHÔNG single `$`
- CSV: UTF-8, delimiter `,`, quote `"`, escape `""` bên trong field
- Field có newline (Back có code block nhiều dòng) → **bắt buộc quote `"..."`**
- HTML trong Back: `<p>`, `<pre><code>`, `<ul><li>`, `<b>`, `<i>`, `<h3>` — đẹp hơn markdown thô
- 2-4 thẻ/topic con, max ~200 chữ trên Back
- Tags: `<phase>::<topic>::<subtopic>`, lowercase, gạch-nối, tag đặc biệt `xx-important` / `xx-pitfall` / `xx-tradeoff` / `xx-from-scratch`

---

## 9. Quy trình build mỗi phase (cho maintainer)

```
1. Đọc ml-math/AI_ENGINEERING_MASTER_CURRICULUM_V6.md (topic checklist)
2. Viết cheatsheet/XX-topic.md sâu (công thức + import + API + tham số + pattern + pitfall + tradeoff)
3. Sinh anki-csv/XX-topic.csv (~80-120 thẻ/topic chính)
4. python _tools/csv_validator.py anki-csv/XX-topic.csv → PASS
5. Cập nhật MASTER-PLAN.md progress board (đánh ✅ + số thẻ)
6. (Tùy chọn) Commit: "add(phase-N): cheatsheet + anki-csv cho <topic> (<n> cards)"
7. Import vào Anki, spot-check 10 thẻ ngẫu nhiên
```

---

## 10. Tài liệu tham chiếu

- `MASTER-PLAN.md` — lộ trình + progress
- `_schema/SCHEMA.md` — schema CSV đầy đủ + ví dụ
- `_templates/anki-card-templates.md` — 4 loại thẻ + mẫu
- `_tools/csv_validator.py --help` — usage validator
- `ml-math/AI_ENGINEERING_MASTER_CURRICULUM_V6.md` (repo cũ) — curriculum gốc
- `ml-math/Anki_Prompt_Template_AI_Roadmap.md` (repo cũ) — prompt template tạo Anki

---

## 11. Troubleshooting

| Vấn đề | Fix |
|--------|-----|
| MathJax không render | Anki → Preferences → "Show mathjax" ✓ ; cài add-on MathJax nếu Anki cũ |
| Import báo "field count mismatch" | Chạy validator trước — thường Back có newline nhưng quên quote |
| Tag không gọn trong browser | Cài add-on **Hierarchical Tags** (314447711) |
| CSV mở Excel bị font lạ | Excel mặc định ANSI → save as UTF-8 CSV, hoặc dùng Google Sheets / VSCode |
| Validator báo MathJax imbalance | Kiểm tra `\( ... \)` có chẵn không, `$$` có chẵn không |
| Đánh tag sai (vd `XX-Important`) | Tags **lowercase**, gạch-nối `-`, dùng `xx-important` |
