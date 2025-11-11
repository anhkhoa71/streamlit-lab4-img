import streamlit as st
import os
from PIL import Image

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Confusion Matrix",
    layout="wide",
)

# ============ CUSTOM GLOBAL CSS ============
st.markdown("""
<style>
    img {
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Model card styling */
    .model-card {
        background: linear-gradient(145deg, #ffffff, #f3f4f6);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    /* Model title */
    .model-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Image container */
    .img-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Section divider */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 3rem 0;
    }
    
    /* Warning box */
    .custom-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============ PAGE HEADER ============
st.markdown('<h1 class="main-title">Confusion Matrix Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detailed Classification Performance for Each Model</p>', unsafe_allow_html=True)

# Divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thư mục confusion matrix tuyệt đối
confusion_dir = os.path.join(BASE_DIR, "assets", "confusion_matrix")

# Định nghĩa các models
models_config = [
    {"name": "VGG19"},
    {"name": "ResNet34"},
    {"name": "ShuffleNet"},
    {"name": "ConvNeXt"},
    {"name": "ViT"},
]

# Layout 2 cột với khoảng cách đẹp
col1, col2 = st.columns(2, gap="large")

# ============ DISPLAY CONFUSION MATRICES ============
for idx, model in enumerate(models_config):
    img_path = os.path.join(confusion_dir, f"{model['name']}_confusion_matrix.png")
    
    # Chọn cột hiển thị
    current_col = col1 if idx % 2 == 0 else col2
    
    with current_col:
        if os.path.exists(img_path):
            # Container cho mỗi model
            with st.container():
                st.markdown(f"""
                <div class="model-card">
                    <div class="model-title">{model['name']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hiển thị ảnh confusion matrix
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
                
                # Thêm khoảng trống
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-warning">
                <strong>Warning:</strong> Confusion matrix not found for model: {model['name']}
            </div>
            """, unsafe_allow_html=True)

# ============ FOOTER ============
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #9ca3af; padding: 2rem 0;">
    <p style="font-size: 0.9rem;">Confusion Matrix Insights</p>
    <p style="font-size: 0.8rem;">Analyze true positives, false positives, true negatives, and false negatives</p>
</div>
""", unsafe_allow_html=True)