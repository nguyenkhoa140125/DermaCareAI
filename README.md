# DermaCare AI: khoanh vùng nghi ngờ nám trên ảnh khuôn mặt

> **English summary.** Deep-learning pipeline that segments *suspected melasma* regions on face photos and
> estimates their share of the image area. It trains and compares SegFormer-B0, U-Net (ResNet34) and
> YOLO11n-seg with patient-grouped 5-fold cross-validation on the public MEMI-DS dataset, tests
> out-of-domain on a Roboflow set, and ships a web demo for the DermaCare AI Flask platform.
> Everything runs on a single 4 GB GPU. Results are modest (Dice ≈ 0.66) and the tool is **not** a diagnostic device.

> **Lưu ý y tế.** Đây là công cụ nghiên cứu và tham khảo, **không phải chẩn đoán y khoa** và không thay thế bác sĩ da liễu.
> Mô hình chưa phân biệt được nám với tàn nhang, đồi mồi hay tăng sắc tố sau viêm, và có thể bỏ sót sắc tố nhạt lan tỏa.

## 1. Dự án làm gì

- **Đầu vào:** một ảnh khuôn mặt (chụp thẳng hoặc nghiêng khoảng 45 độ).
- **Đầu ra:** ảnh phủ các vùng nghi ngờ nám (màu đỏ), mặt nạ nhị phân và **phần trăm diện tích khung ảnh** mà các vùng đó chiếm.
- **Không làm:** chẩn đoán, đề xuất điều trị. Chỉ số MASI tự động (mục 7b) là **ước lượng tham khảo**, chưa kiểm chứng với bác sĩ.

Repo gồm hai phần:

| Phần | Nội dung |
|---|---|
| `melasma/` | Mã nghiên cứu: chuẩn bị dữ liệu, huấn luyện 3 mô hình, đánh giá, phân tích, kiểm tra ngoài miền |
| `DermaCareAI/` | Nền tảng DermaCare (Flask) cùng module demo `melasma_seg.py` và trang `melasma.html` |

## 2. Kết quả

Đo trên tập kiểm tra của kiểm định chéo 5-fold, **chia theo bệnh nhân** (130 ảnh của 63 bệnh nhân, mỗi ảnh được kiểm tra đúng một lần
bởi mô hình chưa từng thấy bệnh nhân đó):

| Mô hình | Dice (trung bình ± độ lệch chuẩn giữa các fold) | IoU | Sai số % diện tích (MAE, điểm %) |
|---|---|---|---|
| SegFormer-B0 | 0.661 ± 0.051 | 0.519 | 4.02 |
| U-Net ResNet34 | 0.661 ± 0.053 | 0.519 | **3.35** |
| YOLO11n-seg | 0.635 ± 0.050 | 0.493 | 4.99 |

- SegFormer-B0 và U-Net **không khác nhau có ý nghĩa thống kê** về Dice (chênh +0.002, khoảng tin cậy 95% theo bệnh nhân [-0.014, +0.018]).
  U-Net ước lượng phần trăm diện tích tốt hơn, nên được dùng cho demo.
- Mô hình yếu hơn ở nám dạng chấm (Punctate), ảnh chính diện và vùng nám nhỏ (dưới 3% ảnh: Dice 0.33–0.42).
- **Kiểm tra ngoài miền** trên 819 ảnh Roboflow (chỉ có nhãn cấp ảnh, không có mặt nạ): mô hình phân biệt melasma với da bình thường
  khá tốt (AUC 0.864, lạc quan vì lớp "normal" nhiều ảnh không phải mặt) nhưng **không đặc hiệu cho nám**:
  AUC 0.651 so với `dark spots` và 0.400 so với `hyperpigmentation`. Mô hình khoanh vùng tăng sắc tố nói chung.
- Thí nghiệm thêm ảnh da bình thường làm mẫu âm (`train_unet_neg.py`) giảm mạnh báo nhầm nhưng làm mô hình gần như ngừng phát hiện nám
  ở miền ảnh khác ("đường tắt"), nên **bị loại**. Mã được giữ lại như một kết quả âm tính.

- **Hướng 1** (pipeline hai giai đoạn, U-Net ResNet34 trên 582 ROI ở 512×512): Dice theo ROI 0.486 ± 0.063 (IoU 0.410). Ghép các ROI về ảnh gốc
  và so sánh ghép cặp với U-Net toàn mặt trên cùng 130 ảnh: Dice **0.589 ± 0.058 so với 0.661 ± 0.051**, chênh lệch trung bình −0.075 (khoảng tin cậy 95%
  theo bệnh nhân [−0.102, −0.048]); Hướng 1 chỉ cao hơn ở 44/130 ảnh. **Hướng 1 không cải thiện độ chính xác phân vùng so với đường cơ sở.**
  Huấn luyện ở 768×768 với batch 1 cho Dice theo ROI 0.342 (chưa kết luận được về ảnh hưởng của độ phân giải). Độ bao phủ nám của các ROI
  trung bình 86.5% (chính diện 80.2%); MASI ước lượng chỉ đáng tin ở 17/130 ảnh. Số liệu tóm tắt nằm trong thư mục `results/`.

Nhãn của MEMI-DS khá thô nên Dice bị giới hạn. Xem mục [Hạn chế](#9-hạn-chế).

## 3. Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt            # mã nghiên cứu + demo dòng lệnh
├── requirements-demo.txt       # thêm cho máy chủ DermaCare
├── melasma/                    # mã nghiên cứu (chạy từ thư mục gốc repo)
│   ├── prepare_memi.py         # chuẩn bị dữ liệu, mặt nạ nhị phân, chia theo bệnh nhân + 5-fold
│   ├── memi_common.py          # hàm dùng chung (đọc dữ liệu, augmentation, chỉ số, overlay)
│   ├── train_segformer.py      # SegFormer-B0
│   ├── train_unet.py           # U-Net ResNet34
│   ├── train_yolo_seg.py       # YOLO11n-seg
│   ├── analyze_cv.py           # phân tích lỗi theo nhóm cho một mô hình
│   ├── compare_models.py       # so sánh mô hình, bootstrap theo bệnh nhân
│   ├── tune_threshold.py       # quét ngưỡng và TTA (SegFormer)
│   ├── eval_roboflow.py        # kiểm tra ngoài miền trên bộ Roboflow
│   ├── train_unet_neg.py       # thí nghiệm mẫu âm (kết quả âm tính)
│   ├── neg_utils.py, fp_normals.py
│   ├── predict.py              # dự đoán bằng dòng lệnh (1 mô hình hoặc ghép nhiều mô hình)
│   ├── export_demo_models.py   # chép trọng số vào DermaCare để chạy demo
│   ├── hg1_roi.py              # Hướng 1, giai đoạn 1: YOLO tìm khuôn mặt + chia ROI
│   ├── hg1_masi.py             # Hướng 1: điểm A/D/H và công thức MASI
│   ├── prepare_roi.py          # cắt ROI từ MEMI-DS để train giai đoạn 2
│   ├── pipeline_hg1.py         # chạy cả pipeline: ảnh mặt -> ROI -> phân vùng -> MASI
│   └── test_hg1.py             # kiểm tra logic bằng dữ liệu giả (không cần GPU)
├── DermaCareAI/                # nền tảng DermaCare
│   ├── server.py
│   ├── modules/melasma_seg.py  # module demo (thêm điểm cuối /api/melasma/*)
│   ├── static/melasma.html     # trang demo
│   └── models/melasma/         # trọng số (KHÔNG đưa lên git)
├── data/                       # dữ liệu (KHÔNG đưa lên git)
└── runs/                       # kết quả huấn luyện, trọng số (KHÔNG đưa lên git)
```

## 4. Yêu cầu và cài đặt

Đã thử trên: Windows, GPU **GTX 1050 Ti 4 GB**, Python 3.12.10, PyTorch 2.5.1 (CUDA 12.1), Ultralytics 8.4.212.
Linux và macOS chưa được thử. Chạy bằng CPU được nhưng rất chậm.

```powershell
# 1. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate                     # Linux/macOS: source venv/bin/activate

# 2. Cài PyTorch bản CUDA trước (chọn đúng lệnh cho máy bạn tại pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Cài các thư viện còn lại
pip install -r requirements.txt

# 4. Kiểm tra GPU (phải in True nếu có GPU)
python -c "import torch; print(torch.cuda.is_available())"
```

Cài `ultralytics` hoặc `segmentation-models-pytorch` đôi khi kéo theo bản PyTorch CPU. Nếu bước kiểm tra GPU in `False`
sau khi cài, hãy cài lại PyTorch theo bước 2. GPU đời cũ (GTX 10xx) có thể không chạy được bản PyTorch quá mới; nếu lỗi, thử cu118 hoặc cu121.

## 5. Dữ liệu

Repo **không chứa dữ liệu**. Hãy kiểm tra giấy phép của từng bộ trước khi dùng và không phát hành lại.

**MEMI-DS (bắt buộc)** – ảnh nám có mặt nạ phân vùng:
<https://figshare.com/articles/figure/MEMI-DS_A_Benchmark_Melasma_Image_Dataset_for_Image_Segmentation/29209229>

Tải các tệp zip của bài trên Figshare và giải nén **tất cả vào cùng một thư mục** `data/memi_raw/` sao cho có đủ bốn thư mục:

```
data/memi_raw/Flaky_Image/        data/memi_raw/Flaky_Annotation/
data/memi_raw/Punctate_Image/     data/memi_raw/Punctate_Annotation/
```

Bộ dữ liệu có 130 cặp ảnh và mặt nạ (một mặt nạ `Flaky/18_R` không có ảnh và bị bỏ qua).

**Roboflow "Pigmentation Melasma Dark Spots" (tùy chọn)** – chỉ để kiểm tra ngoài miền. Bộ này ở định dạng *folder* (nhãn theo lớp ảnh, **không có mặt nạ**), giấy phép CC BY 4.0.
Tải từ Roboflow Universe ở định dạng "Folder Structure" và giải nén vào `data/roboflow/` sao cho có `train/`, `valid/`, `test/`, mỗi thư mục chứa các thư mục lớp (`melasma`, `normal`, ...).

## 6. Tái lập thí nghiệm

Mọi lệnh chạy từ **thư mục gốc của repo**.

### Bước 1. Chuẩn bị dữ liệu

```bash
python melasma/prepare_memi.py --raw data/memi_raw --out data/memi_processed --yolo
```

Tạo ảnh 1024×1536, mặt nạ nhị phân, bảng chia `split.csv` (chia **theo bệnh nhân**, gộp hai nhóm Flaky và Punctate vì có mã bệnh nhân trùng),
ảnh kiểm tra trong `qa/` và nhãn định dạng YOLO. Hãy mở `data/memi_processed/qa/` và xem ảnh phủ mặt nạ trước khi huấn luyện.

### Bước 2. Thử nhanh (1–2 phút, không cần tải mô hình)

```bash
python melasma/train_segformer.py --smoke
python melasma/train_unet.py --smoke
python melasma/train_yolo_seg.py --smoke
```

### Bước 3. Huấn luyện kiểm định chéo 5-fold

Thời gian ước tính trên GTX 1050 Ti (theo nhật ký của tác giả): U-Net khoảng 40 phút, YOLO11n-seg khoảng 40 phút, SegFormer-B0 khoảng 1.6 giờ.

```bash
python melasma/train_unet.py      --data data/memi_processed --all-folds --out runs/unet_r34
python melasma/train_segformer.py --data data/memi_processed --all-folds --out runs/segformer_b0
python melasma/train_yolo_seg.py  --data data/memi_processed --all-folds --out runs/yolo11n_seg
```

Lần đầu cần Internet để tải trọng số khởi tạo (ImageNet, `nvidia/mit-b0`, `yolo11n-seg.pt`). Bị ngắt giữa chừng thì chạy lại đúng lệnh cũ và thêm `--resume`.
Mỗi fold lưu nhật ký huấn luyện, trọng số tốt nhất (U-Net: `best.pt`; SegFormer: thư mục `best_model/`; YOLO: `train/weights/best.pt`), `per_image_test.csv` và ảnh phủ mặt nạ trong `test_overlays/`.

### Bước 4. Phân tích và so sánh

```bash
python melasma/analyze_cv.py --runs runs/unet_r34 --split data/memi_processed/split.csv
python melasma/compare_models.py --runs runs/segformer_b0 runs/unet_r34 runs/yolo11n_seg --split data/memi_processed/split.csv
```

`compare_models.py` in bảng so sánh theo nhóm (loại nám, góc chụp, kích thước vùng nám), sai số diện tích và
**khoảng tin cậy 95% của chênh lệch Dice theo bệnh nhân** (bootstrap). Nếu khoảng tin cậy chứa 0 thì chưa thể nói mô hình nào tốt hơn.

### Bước 5. Kiểm tra ngoài miền (tùy chọn)

```bash
python melasma/eval_roboflow.py --data data/roboflow --out runs/rf_eval \
  --ckpt runs/unet_r34/fold0/best.pt runs/unet_r34/fold1/best.pt runs/unet_r34/fold2/best.pt runs/unet_r34/fold3/best.pt runs/unet_r34/fold4/best.pt
```

Kết quả gồm `summary.txt` (phần trăm diện tích dự đoán theo lớp, AUC) và các tệp `montage_*.jpg`. **Hãy xem các ảnh montage**: AUC chỉ cho biết mô hình thấy
"có nhiều sắc tố hay không", không cho biết nó khoanh đúng chỗ.

## 7. Dự đoán bằng dòng lệnh

```bash
# một mô hình
python melasma/predict.py --ckpt runs/unet_r34/fold0/best.pt --image anh.jpg --out ket_qua --tta

# ghép 5 mô hình (khuyên dùng), có thể cho cả thư mục ảnh
python melasma/predict.py --image thu_muc_anh --out ket_qua --tta --ckpt \
  runs/unet_r34/fold0/best.pt runs/unet_r34/fold1/best.pt runs/unet_r34/fold2/best.pt runs/unet_r34/fold3/best.pt runs/unet_r34/fold4/best.pt
```

Ra: `<tên>_overlay.jpg`, `<tên>_mask.png` (cùng kích thước ảnh gốc) và `results.json` (phần trăm diện tích).
Bản ghép 5 mô hình của các fold **không** dùng để báo cáo Dice vì mỗi mô hình đã thấy ảnh kiểm tra của các fold khác.

## 7b. Pipeline Hướng 1 (Coarse-to-Fine): YOLO tìm ROI → phân vùng từng ROI → MASI

```
Ảnh mặt → [YOLO face] → ROI (trán, má phải, má trái, mũi, cằm) → crop ở độ phân giải gốc
        → [U-Net / SegFormer huấn luyện trên ROI] → vùng nám từng ROI → diện tích % + MASI
```

**Bước 0. Kiểm tra logic (không cần GPU, dữ liệu hay mô hình):** `python melasma/test_hg1.py`

**Bước 1. Trọng số nhận diện khuôn mặt.** Cần một file `.pt` của YOLO đã huấn luyện nhận diện khuôn mặt (ví dụ YOLOv8-face),
tự tải về và đặt vào `models/face/`. Không có file này code vẫn chạy bằng Haar Cascade của OpenCV nhưng ROI kém chính xác hơn.
Kiểm tra giấy phép trọng số trước khi dùng (Ultralytics theo AGPL-3.0).

**Bước 2. Xem ROI có cắt đúng không (bắt buộc, vì ROI chia theo tỉ lệ hộp khuôn mặt):**

```bash
python melasma/pipeline_hg1.py --image data/memi_processed/images --out qa_roi --qa --face-weights models/face/yolov8n-face.pt
```

Nếu ROI lệch, chỉnh tỉ lệ trong `hg1_roi.ROI_SPECS` (có riêng cho ảnh F/L/R) hoặc truyền `--roi-json`.
Góc chụp lấy từ đuôi tên file (`1_F.jpg`, `1_L.jpg`, `1_R.jpg`), hoặc ép bằng `--view F|L|R`; không rõ thì dùng F.

**Bước 3. Tạo dữ liệu ROI và train giai đoạn 2** (chia tập vẫn theo bệnh nhân):

```bash
python melasma/prepare_roi.py --data data/memi_processed --out data/memi_roi --face-weights models/face/yolov8n-face.pt
python melasma/train_unet.py --data data/memi_roi --all-folds --out runs/unet_roi --width 512 --height 512
```

Đọc `data/memi_roi/MANIFEST_ROI.txt`: số ảnh không tìm thấy mặt và "độ bao phủ nám" (tỉ lệ nám của ảnh gốc nằm trong các ROI).
Độ bao phủ thấp nghĩa là ROI chưa khớp dữ liệu. SegFormer train tương tự bằng `train_segformer.py`.

**Bước 4. Chạy pipeline:**

```bash
python melasma/pipeline_hg1.py --image anh.jpg --out ket_qua_hg1 --face-weights models/face/yolov8n-face.pt \
  --ckpt runs/unet_roi/fold0/best.pt runs/unet_roi/fold1/best.pt runs/unet_roi/fold2/best.pt runs/unet_roi/fold3/best.pt runs/unet_roi/fold4/best.pt --tta
```

Ra: ảnh phủ ROI + vùng nám, mask, và `results.json` với từng ROI (`area_percent`, A, D, H) cùng MASI (0-48).

**MASI được tính thế nào và giới hạn:**
- `MASI = 0.3·A·(D+H)` cho trán, má phải, má trái và `0.1·A·(D+H)` cho cằm. Mũi được phân vùng nhưng không tính vào MASI.
- **A** (diện tích 0-6) tính trực tiếp từ mask, trên phần da của ROI.
- **D** (độ đậm) và **H** (độ đồng đều) vốn do bác sĩ chấm. Ở đây chỉ là **quy tắc thủ công**: D từ chênh lệch độ sáng L* giữa da bình thường và vùng nám,
  H từ độ đặc của vùng nám. Các ngưỡng (`DARK_BINS`, `HOMO_BINS` trong `hg1_masi.py`) là **giả định, chưa hiệu chỉnh**. Nếu có điểm MASI của bác sĩ, hãy đối chiếu để chỉnh.
- ROI chia theo hộp khuôn mặt, neo vào đáy hộp và chiều rộng (ổn định hơn cạnh trên vì mắt bị che mosaic). Tỉ lệ đo trên 3 ảnh mẫu F/L/R của một bệnh nhân, cần kiểm tra thêm trên nhiều ảnh. Ảnh nghiêng chỉ có má bên nhìn thấy, nên MASI của từng ảnh nghiêng không đầy đủ (cờ `complete=false`) và thấp hơn thật.
- D so vùng nám với **vành da sát bên ngoài vùng nám** (ít bị ảnh hưởng bởi ánh sáng giảm dần từ giữa mặt ra mép); mốc trong ROI hoặc toàn mặt chỉ là dự phòng. Cờ `d_reliable=false` (và `masi.reliable=false`, `unreliable_D`) bật khi vùng nám không tối hơn da xung quanh (`delta_L <= 0`) hoặc gần như không khác da xung quanh (`contrast_dE < 3`): khi đó nghi dương tính giả hoặc nám rất nhạt, và MASI của ảnh đó **không nên dùng**. Các ngưỡng D vẫn là giả định, chưa hiệu chỉnh.
- MEMI-DS không có điểm MASI của bác sĩ nên MASI tự động **chưa được kiểm chứng**.

### Tái lập số liệu Hướng 1

Chạy từ thư mục gốc của repo (cần `data/memi_processed`, `models/face/yolov8n-face.pt` và đã có `runs/unet_r34` của đường cơ sở):

```bash
python melasma/prepare_roi.py --data data/memi_processed --out data/memi_roi --face-weights models/face/yolov8n-face.pt
python melasma/train_unet.py --data data/memi_roi --all-folds --out runs/unet_roi --width 512 --height 512
python melasma/phan_tich.py runs/unet_roi            # Dice theo vùng ROI, theo lượng nám
python melasma/coverage_qa.py                        # độ bao phủ ROI, ảnh QA vào qa_low/
python melasma/eval_hg1_fullmask.py                  # Dice toàn mặt của Hướng 1 (ảnh held-out)
python melasma/so_sanh_ghep_cap.py runs/unet_r34     # so sánh ghép cặp với đường cơ sở
```

Thư mục `results/` chứa nhật ký và bảng số liệu tóm tắt của hai lần chạy (512 và 768) để đối chiếu mà không cần chạy lại; không có ảnh hay trọng số.


## 8. Demo web trong DermaCare AI

Demo gồm ba tab trong trang `melasma.html`:

1. **Phân tích Coarse-to-Fine (Hướng 1)**: YOLO tìm mặt, chia 5 ROI, U-Net phân vùng từng ROI, tính diện tích % và MASI tham khảo (mục 7b). **Chưa được đánh giá định lượng** (xem mục 9).
2. **Phân vùng nhanh toàn mặt (baseline)**: U-Net ResNet34, ghép 5 fold, có thanh trượt so sánh trước/sau.
3. **Khảo sát nguy cơ**: bảng câu hỏi 10 yếu tố. **Đây không phải mô hình học máy.** `melasma_risk.py` là bảng điểm do tác giả tự đặt trọng số dựa trên các yếu tố nêu trong bài báo Hồ Văn Lâm và cs. (2021, 795 mẫu). Trọng số không được huấn luyện hay hiệu chỉnh trên dữ liệu, và con số "xác suất nguy cơ" **không phải xác suất đã kiểm định**; chỉ nên đọc như điểm sàng lọc sơ bộ.

**Điều kiện:** cần đã cài `requirements.txt` và `requirements-demo.txt` (mục 4), và có trọng số (không nằm trong git).

```bash
# 1. Chép trọng số baseline (sau khi đã train U-Net ở mục 6, bước 3)
python melasma/export_demo_models.py --runs runs/unet_r34 --dest DermaCareAI/models/melasma

# 2. (Tùy chọn, cho tab Hướng 1) đặt trọng số YOLO nhận diện khuôn mặt vào models/face/yolov8n-face.pt
#    và train giai đoạn ROI (mục 7b, bước 3) để có runs/unet_roi/fold*/best.pt

# 3. Chạy máy chủ bằng môi trường ảo đã cài PyTorch
cd DermaCareAI
python server.py
```

Mở <http://127.0.0.1:5090/melasma.html>. Chưa có trọng số nào thì trang báo "Chưa có mô hình". Thiếu `models/face/yolov8n-face.pt` thì Hướng 1 dùng bộ dò mặt dự phòng (ROI kém chính xác hơn). Thiếu `runs/unet_roi/` thì tab Hướng 1 không chạy được, hãy dùng tab baseline.

**Các điểm cuối API:**
- `GET  /api/melasma/status`: trạng thái sẵn sàng của Hướng 1, baseline và khảo sát.
- `POST /api/melasma/risk_survey`: gửi JSON khảo sát, nhận điểm nguy cơ và khuyến cáo.
- `POST /api/melasma/segment`: phân tích ảnh (`mode="hg1"` hoặc `mode="baseline"`).
- `POST /api/melasma/pipeline_hg1`: điểm cuối chuyên dụng cho Hướng 1.

**Quyền riêng tư và an toàn:**
- Ảnh được xử lý trong bộ nhớ và không được lưu; nhật ký chỉ ghi kích thước, thời gian và phần trăm.
- Máy chủ **mặc định chỉ lắng nghe `127.0.0.1`**. Muốn mở cho mạng nội bộ, chạy `python server.py --host 0.0.0.0`. Khi đó CORS đang là `*`, nên đừng dùng ảnh thật của người khác.
- Ảnh tối đa 10 MB, được thu nhỏ về cạnh dài 1600 điểm ảnh khi xử lý.

## 9. Hạn chế

- **Pipeline Hướng 1 chưa được đánh giá định lượng** (chưa có Dice trên ROI, chưa so với baseline). MASI chỉ là ước lượng tham khảo: A tính từ mask, còn D và H dùng ngưỡng giả định chưa hiệu chỉnh và chưa đối chiếu với điểm của bác sĩ.
- Nhãn MEMI-DS khá thô (mảng lớn khoanh quanh vùng sẫm màu), có thể không nhất quán; Dice tối đa đạt được bị giới hạn bởi chất lượng nhãn.
- Chỉ 63 bệnh nhân, cùng một thiết bị chụp, nền đen, mắt bị che. Độ chính xác trên ảnh điện thoại **chưa được đo bằng nhãn thật**.
- Chưa có ý kiến của bác sĩ da liễu để xác nhận nhãn hoặc dự đoán.
- Mô hình khoanh vùng tăng sắc tố nói chung, chưa đặc hiệu cho nám, và báo thừa trên ảnh cận cảnh kết cấu da hoặc không phải khuôn mặt.
- Phần trăm diện tích tính trên **khung ảnh**, không phải trên diện tích da, và không phải chỉ số MASI. Sai số trung bình khoảng 3–4 điểm phần trăm
  (đo trên từng mô hình đơn lẻ; bản ghép 5 mô hình chưa được kiểm định riêng).

## 10. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `torch.cuda.is_available()` in `False` | Cài lại PyTorch bản CUDA (mục 4); cập nhật trình điều khiển NVIDIA |
| Hết VRAM khi huấn luyện | SegFormer: `--batch 2 --accum 2`. U-Net: `--encoder resnet18` hoặc `--batch 1 --accum 4`. Hoặc giảm `--width`/`--height` (khi đó kết quả không còn so sánh được) |
| Lỗi DataLoader trên Windows | Thêm `--workers 0` |
| `ModuleNotFoundError: torch` | Chưa kích hoạt môi trường ảo đã cài PyTorch |
| `prepare_memi.py` báo thiếu thư mục | Kiểm tra `--raw` trỏ đúng thư mục chứa 4 thư mục `Flaky_*`, `Punctate_*` |
| Trang demo báo "Máy chủ chưa cài PyTorch" | Chạy `server.py` bằng môi trường ảo có PyTorch (mục 8, bước 3 của khối lệnh) |
| Trang demo báo "Chưa có mô hình" | Chạy bước 1 của mục 8 (`export_demo_models.py`) |
| GPU GTX 10xx không nhanh hơn khi bật AMP | Bình thường: dòng này không có lõi tensor. Thử `--no-amp` và so thời gian mỗi epoch |

## 11. Giấy phép, dữ liệu và ghi công

- Mã trong repo theo giấy phép ghi trong tệp `LICENSE`. Lưu ý `ultralytics` (dùng trong `train_yolo_seg.py`) theo giấy phép AGPL-3.0; hãy đọc kỹ điều kiện nếu bạn phân phối lại mã dùng thư viện này.
- Dữ liệu **MEMI-DS** do nhóm tác giả công bố trên Figshare; vui lòng trích dẫn và tuân thủ giấy phép trên trang đó.
- Dữ liệu **Roboflow "Pigmentation Melasma Dark Spots"** theo CC BY 4.0 (ghi nguồn). Nhiều ảnh gốc có watermark và có nguồn từ web, bản quyền ảnh gốc không rõ; chỉ dùng để đánh giá, không phát hành lại.
- Thư viện: [PyTorch](https://pytorch.org), [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch), [Transformers](https://github.com/huggingface/transformers), [Ultralytics](https://github.com/ultralytics/ultralytics), [Flask](https://flask.palletsprojects.com).
- Tài liệu: U-Net (Ronneberger và cs., 2015), SegFormer (Xie và cs., 2021), ResNet (He và cs., 2016), YOLO11 (Jocher và Qiu, 2024).

Tác giả: Nguyễn Văn Khoa. Thực hiện trong khuôn khổ thực tập tốt nghiệp, Trường Đại học Ngoại ngữ - Tin học TP. Hồ Chí Minh, năm 2026.
