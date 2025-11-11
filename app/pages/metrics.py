# pages/metrics.py
import streamlit as st
import os
from PIL import Image

# Page config
st.set_page_config(page_title="Model Metrics", layout="wide")
st.markdown("""
<style>
img {
    max-width: 100% !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# Custom CSS for beautiful styling
st.markdown("""
<style>
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
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f3f4f6);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 2rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric title */
    .metric-title {
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

# Header Section
st.markdown('<h1 class="main-title">Model Evaluation Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comprehensive Performance Analysis on Test Dataset</p>', unsafe_allow_html=True)

# Divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Thư mục chứa ảnh metrics
metrics_dir = r"assets\metrics"

# Định nghĩa metrics với icon và màu sắc
metrics_config = [
    {"name": "accuracy", "icon": "🎯", "title": "Accuracy"},
    {"name": "precision", "icon": "🔍", "title": "Precision"},
    {"name": "recall", "icon": "📍", "title": "Recall"},
    {"name": "f1", "icon": "⚖️", "title": "F1-Score"},
    {"name": "mcc", "icon": "🔗", "title": "Matthews Corr"},
    {"name": "log_loss", "icon": "📉", "title": "Log Loss"}
]

# Layout 2 cột với khoảng cách đẹp
col1, col2 = st.columns(2, gap="large")

# Hiển thị metrics
for idx, metric in enumerate(metrics_config):
    img_path = os.path.join(metrics_dir, f"{metric['name']}_comparison.png")
    
    # Chọn cột hiển thị
    current_col = col1 if idx % 2 == 0 else col2
    
    with current_col:
        if os.path.exists(img_path):
            # Container cho mỗi metric
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{metric['title']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hiển thị ảnh với border radius
                image = Image.open(img_path)
                st.image(image, use_container_width=True)
                
                # Thêm khoảng trống
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="custom-warning">
                <strong>⚠️ Warning:</strong> Metric image not found: {metric['title']}
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #9ca3af; padding: 2rem 0;">
    <p style="font-size: 0.9rem;">📈 Evaluation metrics generated from test dataset</p>
    <p style="font-size: 0.8rem;">Use these metrics to compare and validate model performance</p>
</div>
""", unsafe_allow_html=True)