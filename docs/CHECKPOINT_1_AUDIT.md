# Rà soát Checkpoint 1 — FAR-Trans data pipeline

Ngày rà soát: 2026-08-14  
Branch hiện có trong bản checkout: `work` (`1a4d68d`)

## Kết luận ngắn

**Chưa nên merge/push thẳng vào `main` với tuyên bố rằng Checkpoint 1 đã hoàn tất.**

Phần làm sạch, kiểm tra schema, temporal split, point-in-time snapshot và holdings
đã được triển khai khá đầy đủ. Tuy nhiên repository **chưa có code dựng temporal
investor–asset graph**, chưa có loader/batching dành cho graph hoặc baseline, và chưa
có baseline có thể nạp dữ liệu/chạy song song. Vì checkout này không có local branch
`main` và cũng không cấu hình remote, báo cáo không thể thực hiện diff giữa `main` và
branch hiện tại; đây là audit độc lập trên nội dung của commit đang checkout.

## Đối chiếu từng yêu cầu

| Yêu cầu | Trạng thái | Bằng chứng / nhận xét |
|---|---|---|
| Tải 6 file FAR-Trans | **Một phần** | Có đủ 6 CSV nghiệp vụ trong `data/raw/` và 6 CSV sạch trong `data/processed/`. Không có script tải dữ liệu; README yêu cầu tải thủ công. File `questionnaires.csv` là file thứ 7 của bộ phát hành nhưng không thuộc 6 bảng nghiệp vụ đã thống nhất. |
| Làm sạch 6 file | **Đạt có điều kiện** | `data/processed/clean.py` có loader và hàm clean cho cả 6 bảng; các artifact sạch đã được commit. Chính sách xử lý conflict trong cleaner và tài liệu hiện chưa thống nhất (xem mục P0/P1). |
| Snapshot point-in-time | **Đạt** | Customer/asset lấy bản ghi gần nhất có `timestamp <= cutoff`; price chỉ lấy đến cutoff. Primary có snapshot tại train-end và validation-end; rolling có snapshot tại từng cutoff. |
| Dựng đồ thị tương tác investor–asset theo thời gian | **Chưa đạt** | Mới chỉ có `GraphEvent` schema trong tài liệu và transaction CSV được sort. Không có graph builder, edge/event tensor, temporal neighbor index, hay API tạo event stream cho TGN. |
| Tránh information leakage | **Đạt một phần** | Split và snapshot có kiểm tra biên thời gian; holdings/candidate universe có quy tắc PIT. Chưa thể kiểm tra leakage ở graph sampling, negative sampling, feature fitting/encoding hoặc baseline vì các phần đó chưa tồn tại. |
| Pu: data loader/pipeline hiệu quả | **Chưa đạt đầy đủ** | `FARTransLoader` nạp CSV bằng pandas và nạp split; chưa có chunking/cache/batching, graph dataloader, negative sampler, baseline-ready interaction matrix, hoặc API load holdings/ID mapping. |
| Báo cáo Train/Val/Test | **Đạt** | Có `primary_split_summary.csv`, rolling summary và JSON quality report. |
| Dataset sạch + graph dựng được | **Chưa đạt checkpoint** | Dataset artifacts có và quality report hiện báo `ok=true`; graph chưa dựng được vì không có implementation. |
| Nạp dữ liệu cho baseline song song | **Chưa đạt** | Không có code model/baseline hoặc adapter/dataset để Popularity/BPR/LightGCN cùng dùng một protocol. Shared ID mapping mới chỉ là bước chuẩn bị. |

## Vấn đề phải chốt trước khi merge

### P0 — Thiếu deliverable cốt lõi

1. **Không có temporal graph builder.** Cần tối thiểu một API biến transaction đã
   sort thành event stream với `src`, `dst`, `timestamp`, `event_type`, `units`,
   `totalValue`, `marketID`, dùng chung ID mapping và bảo đảm mọi event/feature của
   batch dự đoán không vượt cutoff.
2. **Không có data loader cho mô hình/baseline.** Loader hiện tại chỉ đọc toàn bộ CSV
   vào RAM. Cần chốt baseline nào nằm trong Checkpoint 1 (ít nhất Popularity; hay cả
   BPR/LightGCN), sau đó bổ sung interface nhất quán cho train/validation/test,
   candidate filtering và negative sampling PIT.
3. **Không thể so với `main`.** Checkout chỉ có branch `work`, không có remote refs.
   Cần fetch/add remote hoặc cung cấp commit SHA của `main` trước khi xác nhận branch
   này chỉ chứa đúng thay đổi cần merge.

### P1 — Hành vi hoặc tài liệu chưa nhất quán

1. `docs/data_protocol.md` nói conflicting key phải làm pipeline dừng, nhưng
   `clean.py` đang tự giữ một dòng đối với conflict của `markets`, `close_prices` và
   `transactionID`. Việc tự chọn first/last có thể che lỗi nguồn và làm kết quả phụ
   thuộc thứ tự file. Cần chọn một policy duy nhất: fail-fast, hoặc whitelist có lý do
   và ghi audit log cho từng exception.
2. Quality report ghi transaction lớn nhất là `2022-11-30 00:00:00`, trong khi test
   protocol kết thúc `2022-11-29 23:59:59`. Các event ngày 30/11 vì vậy không thuộc
   Train/Val/Test. Cần xác nhận đây là chủ ý hay sửa `PRIMARY_TEST_END` và tài liệu.
3. `build_id_mapping(tx)` dùng toàn bộ transaction, bao gồm validation/test. Tài liệu
   coi đây là bookkeeping không chứa target statistics, nhưng nó vẫn làm lộ trước
   danh tính customer/asset sẽ xuất hiện trong tương lai và loại bỏ bài toán cold-start.
   Cần chốt rõ evaluation là **transductive** (mapping toàn kỳ, cho phép node tương lai)
   hay **inductive** (mapping/feature fit chỉ từ dữ liệu tại cutoff, xử lý unknown node).
4. `limit_prices.profitability` được nạp nhưng chủ động không dùng vì chứa thống kê
   toàn kỳ. Đây là đúng hướng chống leakage, nhưng nên loại bảng này khỏi model-facing
   loader hoặc gắn cờ rõ để người viết baseline không vô tình dùng nó.
5. README nói dataset không được đưa vào repository vì kích thước lớn, nhưng raw,
   processed và toàn bộ split artifacts hiện đang được Git track. Cần chốt chính sách
   phát hành: giữ artifact để tái lập nhanh, dùng Git LFS/DVC, hay chỉ commit code +
   manifest/checksum/report.

### P2 — Dư thừa / vệ sinh repository

1. Có hai bản giống mục đích của `data_quality_report.json` ở `data/reports/` và
   `reports/`; orchestrator cũng chủ động ghi cả hai. Chỉ nên giữ một canonical path.
2. `data/processed/Report.xlsx` và `data/processed/dashboard.ipynb` không nằm trong
   pipeline được tài liệu hóa và không phải một trong 6 cleaned CSV. Cần xác nhận đây
   là deliverable báo cáo hay artifact cá nhân trước khi merge.
3. `data/raw/.DS_Store` vẫn được Git track dù `.gitignore` đã cấm `.DS_Store`; nên xóa
   khỏi index.
4. README mô tả nhiều thư mục/model chưa tồn tại và lệnh cài đặt dùng
   `requirements.txt`, trong khi repository chỉ có `requirements-data.txt`. Người mới
   clone hiện không thể làm theo hướng dẫn cài đặt nguyên văn.
5. Test là script assertion tự viết thay vì test suite chuẩn; chưa có CI configuration.
   Nên chuyển/bao bằng `pytest` và chạy pipeline trên fixture nhỏ trong CI.

## Những gì đã làm đúng và nên giữ

- Tách trách nhiệm cleaning và dataset building rõ ràng.
- Transaction được sort ổn định theo `(timestamp, transactionID)`.
- Snapshot customer/asset và price đều giới hạn bằng cutoff.
- Primary Train/Validation/Test tách theo thời gian, có thêm rolling windows độc lập.
- Holdings dùng `BUY - SELL`; candidate set loại tài sản đang nắm giữ thay vì loại
  mọi cặp từng tương tác.
- Validator kiểm tra schema, key, referential integrity, giá trị giao dịch và leakage
  ở các artifact split hiện có.
- `profitability` toàn kỳ không được đưa vào split features.

## Các câu hỏi cần chủ dự án xác nhận

1. “Dựng đồ thị” ở Checkpoint 1 cần artifact nào: pandas event table, PyTorch tensors,
   PyTorch Geometric `TemporalData`, hay dataloader riêng của TGN?
2. “Baseline song song” gồm chính xác Popularity, BPR và LightGCN, hay chỉ cần loader
   chung để người khác triển khai model?
3. Evaluation cần transductive hay inductive đối với customer/asset xuất hiện sau
   cutoff?
4. Test end đúng là hết **2022-11-29** hay phải gồm event ngày **2022-11-30**?
5. Có muốn commit dữ liệu/artifact dung lượng lớn vào `main`, hay chuyển sang
   download script + checksum/DVC/Git LFS?
6. `Report.xlsx` và `dashboard.ipynb` có thuộc deliverable chính thức không?

## Điều kiện đề xuất để duyệt Checkpoint 1

- Có graph builder + graph/event loader với test chứng minh không đọc event/feature
  sau cutoff.
- Có baseline-facing loader và ít nhất một smoke test train/evaluate trên fixture.
- Policy duplicate/conflict thống nhất giữa cleaner, validator và protocol.
- Chốt biên cuối dataset và test lại để không rơi event ngoài mọi split ngoài ý muốn.
- Chốt transductive/inductive protocol và sinh ID mapping tương ứng.
- Dọn artifact dư thừa, sửa installation/run instructions, và có CI chạy tests.
- Fetch được `main`, review `git diff main...work`, sau đó mới quyết định danh sách file
  chính xác để merge.
