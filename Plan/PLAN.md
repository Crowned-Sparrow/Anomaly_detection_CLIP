Đã scan mã nguồn tại commit **DAPO `290411d`**, **UniSpector `0f8cc5a`**.

**Lưu ý:** UniSpector hiện công bố code nền dựa trên DINOv; README ghi implementation đầy đủ sẽ ra sau. Vì vậy chưa thể lấy nguyên các block SSPE/CPE/PQS của paper từ repo này. [geonuk-kimmm/UniSpector · GitHub](https://github.com/geonuk-kimmm/UniSpector?utm_source=chatgpt.com)

**DAPO** — đường dẫn tính từ gốc repo:

| Tên block | Vị trí |
|---|---|
| Vision Encoder — CLIP ViT | [`open_clip/transformer.py:515`](https://github.com/boschresearch/defect-aware-prompt-optimization/blob/290411d7374cc2182ad3491b826131af1ff73224/open_clip/transformer.py#L515) — `VisionTransformer`; forward tại dòng 695 |
| Patch embedding, CLS và positional embedding | `open_clip/transformer.py:695` — `VisionTransformer.forward()` |
| Chèn prompt tokens và lấy đặc trưng nhiều tầng | `open_clip/transformer.py:364` — `Transformer.forward()` |
| Visual Prompt Tuning | `visual_prompt_tuning.py:8` — `VisualPromptTuning` |
| Defect-Aware Hybrid Prompt | `learnable_prompt.py:5` — `LearnablePrompt`; ghép prompt tại `forward():100` |
| Text Encoder cho learned prompts | `open_clip/model.py:303` — `CLIP.encode_learn_prompts()` |
| EOS pooling, text projection, trung bình prompt ensemble | `open_clip/model.py:303` — bên trong `encode_learn_prompts()` |
| Adapter cho patch tokens từng tầng | `adapter.py:5` — `LinearAdapter`; forward tại dòng 10 |
| Similarity ảnh–văn bản | `train.py:32` — `compute_similarity()` |
| Reshape và nội suy similarity map | `train.py:39` — `compute_similarity_map()` |
| Toàn bộ forward khi train | `train.py:178` — `Trainer.train_epoch()`; nhánh ảnh–chữ bắt đầu dòng 217 |
| Global image classification loss | `train.py:256` — cross-entropy |
| Local segmentation loss | `train.py:261` — Focal đa loại + Dice normal/anomaly |
| Định nghĩa Focal / Binary Dice | `loss.py:10` — `FocalLoss`; `loss.py:92` — `BinaryDiceLoss` |
| Optimizer và danh sách module được cập nhật | `train.py:94` — prompt learner + adapter + visual prompt |
| Khởi tạo các block | `train.py:326` — `create_model_components()` |
| Inference phân đoạn đa loại | `test_multi_defect.py:193` — `test()`; forward chính từ dòng 328 |

**UniSpector** — đường dẫn tính từ gốc repo:

| Tên block | Vị trí |
|---|---|
| Kiến trúc tổng thể | [`unispector/architectures/unispector.py:25`](https://github.com/geonuk-kimmm/UniSpector/blob/0f8cc5afd3ad73c71fb6f0d9a5280d783bd5523f/unispector/architectures/unispector.py#L25) — `UniSpector`; forward tại dòng 310 |
| Vision backbone — Swin | `unispector/backbone/swin.py:843` — `D2SwinTransformer` |
| Pixel decoder, tạo đặc trưng đa tỉ lệ | `unispector/body/encoder/encoder_deform.py:177` — `MaskDINOEncoder`; `forward_features():355` |
| Nối backbone features → pixel decoder → predictor | `unispector/body/general_head.py:78` — `IMaskDINOHead.layers()` |
| Visual prompt extractor — masked cross-attention | `unispector/body/decoder/unispector_decoder.py:365` — `prepare_visual_prompt()`; attention tại dòng 416 |
| Learnable content tokens và prompt attention layers | `unispector/body/decoder/unispector_decoder.py:165` |
| Object query embeddings hiện có | `unispector/body/decoder/unispector_decoder.py:181`; sử dụng tại dòng 898 |
| Transformer decoder | `unispector/body/decoder/utils/dino_decoder.py:19` — `TransformerDecoder` |
| Forward khi train open-set | `unispector/body/decoder/unispector_decoder.py:860` — `forward_train_openset()` |
| Trích prompt features để dùng lại | `unispector/body/decoder/unispector_decoder.py:830` — `inference_openset_get_prompt_content_features()` |
| Inference với prompt features đã trích | `unispector/body/decoder/unispector_decoder.py:959` — `forward_openset_image_with_extracted_content()` |
| Query–prompt similarity và mask prediction | `unispector/body/decoder/unispector_decoder.py:1071` — `forward_prediction_heads()` |
| Hungarian matching | `unispector/modules/matcher.py:76` — `HungarianMatcher` |
| Classification / box / mask losses | `unispector/modules/criterion_visual_openset.py:31` — `SetCriterionVisualOpenSet` |
| **SSPE — radial spectrum + spatial fusion** | **Chưa tìm thấy implementation tương ứng paper** |
| **CPE — angular-margin contrastive loss** | **Chưa tìm thấy implementation tương ứng paper** |
| **PQS — Gumbel–Softmax query selection** | **Chưa tìm thấy implementation tương ứng paper** |
| Spectral gating có trong repo, nhưng **không phải SSPE** | `unispector/backbone/swin.py:510` — `SpectralGatingNetwork`; khởi tạo và lời gọi đang comment tại dòng 650, 818 |

**Các block bổ sung trong sơ đồ của chúng ta**:

| Tên block | Vị trí |
|---|---|
| Local windows → radial spectrum | Chưa có; cần triển khai |
| Spatial–spectral fusion cho CLIP patch tokens | Chưa có; điểm nối sau `LinearAdapter`, trước similarity trong `train.py` |
| GT-masked region pooling | Chưa có; dùng patch features và mask nguồn trong `Trainer.train_epoch()` |
| Region contrastive loss / region–text alignment | Chưa có; thêm định nghĩa loss và tích hợp vào `total_loss` |
| Text-guided PQS | Chưa có; cần thiết kế cùng query-based mask decoder |
