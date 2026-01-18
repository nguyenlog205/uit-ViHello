import unicodedata
import re

def clean_text(text):
    if not isinstance(text, str): return ""
    # Chuyển về chữ thường
    text = text.lower().strip()
    # Loại bỏ dấu tiếng Việt (Normalization Form KD)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Chỉ giữ lại chữ cái, số và khoảng trắng
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Áp dụng cho dữ liệu của bạn
raw_data_cleaned = [(clean_text(q), clean_text(r)) for q, r in raw_data]