# Từ điển Dữ liệu Khảo sát Nghề nghiệp (`data/raw/survey.csv`)

Tài liệu này mô tả chi tiết cấu trúc bộ dữ liệu khảo sát định hướng nghề nghiệp thực tế, ý nghĩa **thang đo Likert 1–5**, **mô hình sở thích nghề nghiệp Holland (RIASEC)**, các đặc trưng phân loại và hệ thống nhãn nhóm nghề.

---

## 1. Tổng quan bộ dữ liệu

- **Đường dẫn tệp gốc**: `data/raw/survey.csv`
- **Số lượng mẫu khảo sát**: 190 phản hồi thực tế
- **Tổng số cột**: 19 cột, bao gồm:
  - **16 đặc trưng đầu vào (`FEATURES`)**:
    - 8 đặc trưng năng lực & kỹ năng cá nhân (thang đo Likert 1–5)
    - 6 đặc trưng sở thích nghề nghiệp theo mô hình RIASEC (thang đo Likert 1–5)
    - 2 đặc trưng phân loại định tính (`Đối tượng công việc`, `Môi trường`)
  - **1 cột nhãn mục tiêu (`TARGET`)**: `Nhóm nghề` (`N1`, `N2`, `N3`, `N4`)
  - **2 cột nhãn tham chiếu (`REFERENCE_LABELS`)**: `Nghề cụ thể` và `Tên nhóm nghề` (chỉ dùng để đối chiếu, thống kê và minh họa; không đưa vào làm đặc trưng huấn luyện để tránh rò rỉ nhãn).

---

## 2. Thang đo điểm Likert 1–5

Toàn bộ **14 đặc trưng định lượng** được đánh giá theo thang đo **Likert 5 mức độ** (số nguyên từ `1` đến `5`), chia thành hai nhóm thang đo cụ thể theo bản chất câu hỏi khảo sát:

### 2.1. Nhóm Năng lực / Kỹ năng (8 đặc trưng: `1` Rất kém $\rightarrow$ `5` Giỏi)

Áp dụng cho 8 cột: `Khả năng tư duy logic`, `Năng lực tính toán và phân tích số liệu`, `Công nghệ`, `Sáng tạo`, `Giao tiếp`, `Tổ chức`, `Cẩn thận`, `Đồng cảm`.

| Điểm | Mức năng lực / kỹ năng | Mô tả chi tiết mức độ tự đánh giá | Mức rời rạc hóa (`score_bins`) |
|:---:|:---:|---|:---:|
| **1** | **Rất kém** | Chưa có nền tảng hoặc kỹ năng, gặp nhiều lúng túng và khó khăn khi thực hiện công việc liên quan. | `low` (Thấp: 1–2) |
| **2** | **Kém / Thấp** | Năng lực còn hạn chế, chỉ thực hiện được các thao tác đơn giản khi có người hướng dẫn chi tiết. | `low` (Thấp: 1–2) |
| **3** | **Trung bình** | Đáp ứng được yêu cầu cơ bản trong học tập và công việc thông thường, mức độ tự chủ vừa phải. | `medium` (Trung bình: 3) |
| **4** | **Khá** | Thực hiện tốt, chủ động giải quyết vấn đề và vận dụng kỹ năng hiệu quả trong nhiều tình huống. | `high` (Cao: 4–5) |
| **5** | **Giỏi** | Năng lực nổi trội, thành thạo cao, có thể xử lý tình huống phức tạp và là thế mạnh cốt lõi cá nhân. | `high` (Cao: 4–5) |

### 2.2. Nhóm Sở thích nghề nghiệp RIASEC (6 đặc trưng: `1` Rất không thích $\rightarrow$ `5` Rất thích)

Áp dụng cho 6 cột: `RIASEC_R_Thuc_hanh`, `RIASEC_I_Nghien_cuu`, `RIASEC_A_Nghe_thuat`, `RIASEC_S_Xa_hoi`, `RIASEC_E_Thuyet_phuc`, `RIASEC_C_Quy_cu`.

| Điểm | Mức độ sở thích | Mô tả chi tiết mức độ yêu thích hoạt động nghề nghiệp | Mức rời rạc hóa (`score_bins`) |
|:---:|:---:|---|:---:|
| **1** | **Rất không thích** | Hoàn toàn không có hứng thú, cảm thấy nhàm chán hoặc né tránh các hoạt động thuộc nhóm này. | `low` (Thấp: 1–2) |
| **2** | **Không thích** | Ít hứng thú, chỉ tham gia khi bắt buộc và không muốn gắn bó lâu dài. | `low` (Thấp: 1–2) |
| **3** | **Bình thường (Trung lập)** | Mức độ quan tâm vừa phải, có thể tham gia khi cần nhưng chưa phải ưu tiên hàng đầu. | `medium` (Trung bình: 3) |
| **4** | **Thích** | Có hứng thú rõ rệt, chủ động tìm hiểu và cảm thấy thoải mái khi làm các công việc thuộc nhóm này. | `high` (Cao: 4–5) |
| **5** | **Rất thích** | Đam mê mạnh mẽ, hào hứng khi thực hiện và ưu tiên lựa chọn làm định hướng nghề nghiệp lâu dài. | `high` (Cao: 4–5) |

> **Quy tắc tiền xử lý điểm số**: Trong bước tiền xử lý (`SurveyPreprocessor`), các ô trống (nếu có) được điền bằng trung vị (`median`) của tập huấn luyện, sau đó rời rạc hóa thành 3 mức: `low` ($\le 2$), `medium` ($3$), và `high` ($\ge 4$) để phục vụ thuật toán Cây quyết định ID3 và Categorical Naive Bayes.

---

## 3. Giải thích Mô hình Sở thích Nghề nghiệp RIASEC (Holland Codes)

Mô hình **RIASEC** do nhà tâm lý học **John L. Holland** phát triển, phân loại xu hướng sở thích nghề nghiệp của con người thành **6 nhóm tính cách – công việc** cốt lõi:

| Mã cột trong CSV | Nhóm RIASEC | Tên tiếng Việt & Bản chất | Hoạt động yêu thích đặc trưng | Nhóm nghề liên quan mạnh |
|---|---|---|---|---|
| `RIASEC_R_Thuc_hanh` | **R — Realistic** | **Thực hành / Kỹ thuật**: Thích làm việc trực tiếp với công cụ, máy móc, thiết bị, phần cứng hoặc vận động thực tế. | Lắp ráp, vận hành thiết bị, sửa chữa mạch điện, chế tạo cơ khí, kiểm tra hiện trường. | `N1` (Kỹ sư điện, cơ khí, tự động hóa), `N2` (Logistics), `N4` (Điều dưỡng) |
| `RIASEC_I_Nghien_cuu` | **I — Investigative** | **Nghiên cứu / Phân tích**: Thích quan sát, tìm tòi, suy luận logic, phân tích dữ liệu và giải quyết bài toán phức tạp. | Lập trình thuật toán, phân tích dữ liệu/tài chính, nghiên cứu hành vi tâm lý, kiểm thử chuyên sâu. | `N1` (Lập trình viên, Phân tích dữ liệu), `N2` (Phân tích tài chính), `N4` (Tâm lý học) |
| `RIASEC_A_Nghe_thuat` | **A — Artistic** | **Nghệ thuật / Sáng tạo**: Thích sáng tác, thiết kế hình ảnh, âm thanh, giao diện và biểu đạt ý tưởng độc đáo trong môi trường mở. | Thiết kế UI/UX, thiết kế đồ họa, dựng phim, biên tập video, viết và sáng tạo nội dung số. | `N3` (Thiết kế UI/UX, Đồ họa, Sáng tạo nội dung, Dựng phim) |
| `RIASEC_S_Xa_hoi` | **S — Social** | **Xã hội / Hỗ trợ**: Thích giao tiếp, giảng dạy, tư vấn, chăm sóc sức khỏe, lắng nghe và giúp đỡ cộng đồng. | Giảng dạy, tham vấn tâm lý, công tác xã hội, phát triển nhân sự, chăm sóc bệnh nhân. | `N4` (Sư phạm, Tâm lý học, Công tác xã hội, Điều dưỡng) |
| `RIASEC_E_Thuyet_phuc` | **E — Enterprising** | **Thuyết phục / Quản lý**: Thích dẫn dắt, đàm phán, kinh doanh, truyền thông thương hiệu và hướng tới mục tiêu kết quả. | Bán hàng, phát triển thị trường, quản trị vận hành, chiến dịch Digital Marketing, tuyển dụng nhân sự. | `N2` (Kinh doanh, Vận hành, Logistics), `N3` (Digital Marketing), `N4` (Quản trị nhân lực) |
| `RIASEC_C_Quy_cu` | **C — Conventional** | **Quy củ / Nghiệp vụ**: Thích làm việc có kế hoạch, tuân thủ quy trình, kiểm soát chất lượng, sổ sách, biểu mẫu và số liệu chính xác. | Kế toán chứng từ, kiểm thử phần mềm (QA/QC), phân tích tài chính, chuẩn hóa quy trình vận hành. | `N2` (Kế toán viên, Phân tích tài chính), `N1` (Kiểm thử phần mềm), `N4` (Điều dưỡng) |

---

## 4. Chi tiết 19 cột trong `data/raw/survey.csv`

### 4.1. Nhóm 8 đặc trưng Năng lực & Kỹ năng mềm (Điểm nguyên 1–5)

| STT | Tên cột | Kiểu dữ liệu | Miền giá trị | Mô tả ý nghĩa |
|:---:|---|---|:---:|---|
| 1 | `Khả năng tư duy logic` | `int64` | `1..5` | Khả năng suy luận có hệ thống, phân tách vấn đề, tìm quy luật nguyên nhân – kết quả. |
| 2 | `Năng lực tính toán và phân tích số liệu` | `int64` | `1..5` | Khả năng làm việc với con số, thống kê, bảng biểu, công thức toán học và tài chính. |
| 3 | `Công nghệ` | `int64` | `1..5` | Mức độ nhạy bén và hứng thú sử dụng phần mềm, hệ thống kỹ thuật số, công cụ kỹ thuật mới. |
| 4 | `Sáng tạo` | `int64` | `1..5` | Khả năng đề xuất ý tưởng mới, thẩm mỹ thiết kế, cách tiếp cận linh hoạt và khác biệt. |
| 5 | `Giao tiếp` | `int64` | `1..5` | Kỹ năng trình bày, lắng nghe, thuyết trình, trao đổi thông tin rõ ràng và thuyết phục. |
| 6 | `Tổ chức` | `int64` | `1..5` | Kỹ năng lập kế hoạch, điều phối công việc, quản lý thời gian và sắp xếp nguồn lực. |
| 7 | `Cẩn thận` | `int64` | `1..5` | Sự tỉ mỉ, chú trọng độ chính xác chi tiết, tuân thủ tiêu chuẩn và hạn chế sai sót. |
| 8 | `Đồng cảm` | `int64` | `1..5` | Khả năng thấu hiểu cảm xúc, kiên nhẫn lắng nghe và quan tâm hỗ trợ người khác. |

### 4.2. Nhóm 6 đặc trưng Sở thích RIASEC (Điểm nguyên 1–5)

| STT | Tên cột | Kiểu dữ liệu | Miền giá trị | Mô tả ý nghĩa |
|:---:|---|---|:---:|---|
| 9 | `RIASEC_R_Thuc_hanh` | `int64` | `1..5` | Điểm xu hướng Thực hành – Kỹ thuật (*Realistic*). |
| 10 | `RIASEC_I_Nghien_cuu` | `int64` | `1..5` | Điểm xu hướng Nghiên cứu – Phân tích (*Investigative*). |
| 11 | `RIASEC_A_Nghe_thuat` | `int64` | `1..5` | Điểm xu hướng Nghệ thuật – Sáng tạo (*Artistic*). |
| 12 | `RIASEC_S_Xa_hoi` | `int64` | `1..5` | Điểm xu hướng Xã hội – Hỗ trợ cộng đồng (*Social*). |
| 13 | `RIASEC_E_Thuyet_phuc` | `int64` | `1..5` | Điểm xu hướng Thuyết phục – Quản lý – Kinh doanh (*Enterprising*). |
| 14 | `RIASEC_C_Quy_cu` | `int64` | `1..5` | Điểm xu hướng Quy củ – Nghiệp vụ – Tổ chức dữ liệu (*Conventional*). |

### 4.3. Nhóm 2 đặc trưng Phân loại định tính (`category`)

| STT | Tên cột | Kiểu dữ liệu | Các danh mục hợp lệ |
|:---:|---|---|---|
| 15 | `Đối tượng công việc` | `category` (19 giá trị) | `Phần mềm`, `Dữ liệu`, `Máy móc - thiết bị`, `Khách hàng - thị trường`, `Tài chính - số liệu`, `Sổ sách - chứng từ`, `Quy trình - hoạt động`, `Chuỗi cung ứng - hàng hóa`, `Sản phẩm số - Giao diện`, `Nội dung - Truyền thông`, `Thương hiệu - Quảng cáo`, `Hình ảnh - Ấn phẩm`, `Video - Âm thanh`, `Học sinh - sinh viên`, `Con người - tri thức`, `Tâm lý - hành vi`, `Cộng đồng - yếu thế`, `Nhân sự - tổ chức`, `Bệnh nhân - y tế` |
| 16 | `Môi trường` | `category` (6 giá trị) | `Linh hoạt - sáng tạo`, `Có quy trình`, `Tương tác - hợp tác`, `Cạnh tranh - mục tiêu`, `Phân tích chuyên sâu`, `Ổn định - rõ ràng` |

### 4.4. Nhóm cột Nhãn mục tiêu và Nhãn tham chiếu

| STT | Tên cột | Vai trò | Giá trị hợp lệ / Chi tiết |
|:---:|---|---|---|
| 17 | `Nghề cụ thể` | Nhãn tham chiếu (`SPECIFIC_CAREER_COL`) | 21 nghề cụ thể tương ứng với 4 nhóm nghề (không đưa vào làm đặc trưng đầu vào khi huấn luyện). |
| 18 | `Nhóm nghề` | **Nhãn mục tiêu (`TARGET`)** | 4 mã nhóm nghề: `N1`, `N2`, `N3`, `N4`. |
| 19 | `Tên nhóm nghề` | Nhãn tham chiếu (`CAREER_GROUP_NAME_COL`) | Tên đầy đủ tiếng Việt của 4 nhóm nghề `N1`–`N4`. |

---

## 5. Bảng ánh xạ Nhóm nghề (`N1`–`N4`) và Nghề cụ thể

| Mã nhóm (`Nhóm nghề`) | Tên nhóm nghề (`Tên nhóm nghề`) | Số mẫu | Các nghề cụ thể trực thuộc (`Nghề cụ thể`) |
|:---:|---|:---:|---|
| **`N1`** | Công nghệ - Kỹ thuật - Phân tích | 50 | `Lập trình viên` (10), `Chuyên viên phân tích dữ liệu` (8), `Kiểm thử phần mềm` (8), `Kỹ sư điện` (8), `Kỹ sư tự động hóa` (8), `Kỹ sư cơ khí` (8) |
| **`N2`** | Kinh doanh - Tài chính - Quản lý | 50 | `Chuyên viên kinh doanh` (10), `Chuyên viên phân tích tài chính` (10), `Kế toán viên` (10), `Chuyên viên vận hành` (10), `Chuyên viên logistics` (10) |
| **`N3`** | Sáng tạo - Truyền thông - Thiết kế | 40 | `Thiết kế UI/UX` (8), `Sáng tạo nội dung` (8), `Digital Marketing` (8), `Thiết kế đồ họa` (8), `Dựng phim - Biên tập video` (8) |
| **`N4`** | Giáo dục – Xã hội – Hỗ trợ con người | 50 | `Sư phạm` (10), `Tâm lý học` (10), `Công tác xã hội` (10), `Quản trị nhân lực` (10), `Điều dưỡng` (10) |
