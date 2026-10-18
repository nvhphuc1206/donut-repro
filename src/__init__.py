import os

# Windows không tạo được symlink cho cache HF -> chỉ là cảnh báo (tốn thêm dung lượng), tắt cho gọn log.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# transformers tự mở luồng nền tải bản model.safetensors (từ PR trên Hub) khi repo chỉ có pytorch_model.bin
# -> không cần thiết, hay treo ở 0%.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
