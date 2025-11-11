import streamlit as st
import sys, os
from PIL import Image
import torch
from time import sleep

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils.inference import load_model_dict, inference_model_dict

@st.cache_resource
def load_models_cached():
    return load_model_dict(MODEL_PATH, DEVICE)

# ----- Page Configuration -----
st.set_page_config(
    page_title="AI Model Playground", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Custom Ultra WOW Styling -----
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main title - Ultra WOW */
    .main-wow {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0 1rem 0;
        letter-spacing: 3px;
        text-shadow: 0 0 80px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 40px rgba(118, 75, 162, 0.8)); }
    }
    
    /* Subtitle with animation */
    .subtitle-wow {
        font-size: 1.25rem;
        text-align: center;
        color: #4b5563;
        margin-bottom: 3rem;
        font-weight: 400;
        line-height: 1.8;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .subtitle-wow .highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 3rem 0 2rem 0;
        text-align: center;
        letter-spacing: 2px;
    }
    
    /* Image container */
    .image-container {
        border-radius: 25px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
        border: 4px solid transparent;
        background: linear-gradient(white, white) padding-box,
                    linear-gradient(135deg, #667eea, #764ba2) border-box;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 30px 80px rgba(102, 126, 234, 0.3);
    }
    
    /* Model prediction cards */
    .model-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid transparent;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #667eea15, #764ba215);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .model-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 25px 60px rgba(102, 126, 234, 0.25);
        border: 2px solid #667eea;
    }
    
    .model-card:hover::before {
        opacity: 1;
    }
    
    .model-name {
        font-size: 1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.2rem;
        text-align: center;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .prediction-label {
        font-size: 1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .prediction-value {
        font-size: 1rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        display: block;
    }
    
    .confidence-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #059669;
        display: block;
        margin-top: 0.5rem;
    }
    
    /* Progress bar custom */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Divider */
    .custom-divider {
        height: 4px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, #f093fb, transparent);
        margin: 3rem 0;
        border-radius: 2px;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 6px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1e40af;
        font-weight: 500;
    }
    
    .info-box strong {
        font-weight: 700;
        color: #1e3a8a;
    }
    
    /* Best model card */
    .best-model-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 3px solid #f59e0b;
        border-radius: 25px;
        padding: 2.5rem;
        margin: 2rem auto;
        max-width: 700px;
        box-shadow: 0 20px 50px rgba(245, 158, 11, 0.3);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .best-model-title {
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #f59e0b 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    .best-model-name {
        font-size: 2rem;
        font-weight: 800;
        color: #92400e;
        text-align: center;
        margin: 1rem 0;
    }
    
    .best-model-stats {
        font-size: 1.3rem;
        font-weight: 600;
        color: #78350f;
        text-align: center;
        line-height: 2;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Animation for cards */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .model-card {
        animation: slideInUp 0.6s ease-out backwards;
    }
    
    .model-card:nth-child(1) { animation-delay: 0.1s; }
    .model-card:nth-child(2) { animation-delay: 0.2s; }
    .model-card:nth-child(3) { animation-delay: 0.3s; }
    .model-card:nth-child(4) { animation-delay: 0.4s; }
    .model-card:nth-child(5) { animation-delay: 0.5s; }
</style>
""", unsafe_allow_html=True)

# ----- Header Section -----
st.markdown('<div class="main-wow">AI Models</div>', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ----- Sidebar -----
with st.sidebar:
    st.markdown("### Control Panel")
    st.markdown("---")
    st.markdown("#### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a landscape image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    st.markdown("---")
    st.markdown("#### Classification Classes")
    st.markdown("""
    - **Building**
    - **Forest**
    - **Glacier**
    - **Mountain**
    - **Sea**
    - **Street**
    """)
    
    st.markdown("---")
    st.markdown("#### AI Models")
    st.markdown("""
    - **VGG19**
    - **ResNet34**
    - **ShuffleNet**
    - **ConvNeXt**
    - **ViT**
    """)

# ----- Inference Settings -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models" if os.path.exists("models") else "../models"
LABELS = ["building", "forest", "glacier", "mountain", "sea", "street"]

with st.spinner("Loading models, please wait..."):
    model_dict = load_models_cached()
st.success("Models loaded successfully!")


# ----- Main Content -----
if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown('<div class="section-header">Selected Image</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Run inference
        with st.spinner('Analyzing image with 5 AI models...'):
            results = inference_model_dict(model_dict, image, LABELS, DEVICE)
        
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Display results in columns
        cols = st.columns(len(results))
        
        # Track best model
        best_model = None
        best_confidence = 0
        
        for idx, (model_name, res) in enumerate(results.items()):
            with cols[idx]:
                pred_label = res["pred_label"]
                pred_acc = res["pred_acc"]
                elapsed_time = res['elapsed_time']
                
                # Update best model
                if pred_acc > best_confidence:
                    best_confidence = pred_acc
                    best_model = model_name
                    best_prediction = pred_label
                
                st.markdown(f'''
                    <div class="model-card">    
                        <div class="model-name">{model_name}</div>
                        <div style="position: relative; z-index: 1;">
                            <div class="prediction-label">Prediction:</div>
                            <span class="prediction-value">{pred_label}</span>
                            <div class="prediction-label">Confidence:</div>
                            <span class="confidence-value">{pred_acc:.2f}%</span>
                            <div class="prediction-label" style="margin-top: 1rem;">Time:</div>
                            <span style="font-size: 1rem; font-weight: 700; color: #6366f1;">{elapsed_time:.3f}s</span>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
        
                st.progress(pred_acc / 100)
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Please try uploading a different image or check the file format.")
else:
    # Empty state
    st.markdown('<div class="section-header">Welcome to AI Playground</div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="info-box">
        <strong>Get Started:</strong>
        <br><br>
        1. Click the <strong>"Browse files"</strong> button in the left sidebar<br>
        2. Choose a landscape image (JPG/PNG format)<br>
        3. Watch 5 AI models analyze and predict results<br>
        4. Compare accuracy and confidence of each model<br>
        <br>
        <strong>Tip:</strong> Try different images to see how models differ in their predictions!
    </div>
    ''', unsafe_allow_html=True)
    
    # Example images section
    st.markdown('<div class="section-header">Classification Classes</div>', unsafe_allow_html=True)
    
    example_cols = st.columns(3)
    example_classes = [
        ("Building", "Buildings and architectural structures"),
        ("Forest", "Forests and green nature"),
        ("Glacier", "Glaciers and ice regions"),
    ]
    
    for idx, (name, desc) in enumerate(example_classes):
        with example_cols[idx]:
            st.markdown(f'''
            <div class="model-card">
                <div class="model-name">{name}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    example_cols2 = st.columns(3)
    example_classes2 = [
        ("Mountain", "Mountains and high terrain"),
        ("Sea", "Seas and oceans"),
        ("Street", "Streets and urban areas"),
    ]
    
    for idx, (name, desc) in enumerate(example_classes2):
        with example_cols2[idx]:
            st.markdown(f'''
            <div class="model-card">
                <div class="model-name">{name}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            