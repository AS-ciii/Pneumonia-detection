import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from util import classify
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PneumoScan AI - Medical Imaging Analysis",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
    }
    h2 {
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=PneumoScan+AI", width=150)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["Scan Analysis", "About Pneumonia", "Statistics", "Help"]
    )
    
    st.markdown("---")
    st.markdown("### Quick Facts")
    st.info("""
        • Pneumonia affects 450 million people annually
        • It's a leading cause of hospitalization
        • Early detection increases survival rate by 40%
        • AI can detect pneumonia with >90% accuracy
    """)
    
    st.markdown("---")
    st.markdown("### Tips for Best Results")
    st.warning("""
        1. Use clear, high-resolution images
        2. Ensure proper image orientation
        3. Avoid blurry or dark images
        4. Follow standard imaging protocols
    """)

# Main content
if page == "Scan Analysis":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🫁 PneumoScan AI")
        st.markdown("### Advanced Pneumonia Detection System")
        
        # Model selection
        model_choice = st.selectbox(
            'Select Imaging Type:',
            ('X-ray Model', 'CT Model'),
            help="Choose the appropriate model based on your image type"
        )
        
        # Load models
        if 'model' not in st.session_state or st.session_state.model_choice != model_choice:
            if model_choice == 'X-ray Model':
                try:
                    st.session_state.model = load_model('./models/xray_classification_model.h5')
                    st.session_state.model_type = 'xray'
                except Exception as e:
                    st.error(f"Error loading X-ray model: {str(e)}")
            else:
                try:
                    st.session_state.model = load_model('./models/CT_classification_model.h5')
                    st.session_state.model_type = 'ct'
                except Exception as e:
                    st.error(f"Error loading CT model: {str(e)}")
            
            st.session_state.model_choice = model_choice
        
        # Load class names
        if 'class_names' not in st.session_state:
            model_type = st.session_state.get('model_type', 'xray')
            with open(f'./models/{model_type}_labels.txt', 'r') as f:
                st.session_state.class_names = [line.strip() for line in f.readlines()]
        
        # File upload
        file = st.file_uploader(
            'Upload Medical Image',
            type=['jpeg', 'jpg', 'png'],
            help="Supported formats: JPEG, JPG, PNG"
        )
        
        if file:
            try:
                image = Image.open(file).convert('RGB')
                st.image(image, use_column_width=True, caption="Uploaded Image")
                
                with st.spinner("Analyzing image..."):
                    class_name, conf_score = classify(image, st.session_state.model, st.session_state.class_names)
                
                # Create result card
                result_container = st.container()
                with result_container:
                    if conf_score > 0.7:  # High confidence
                        status_color = "#28a745" if class_name == "Normal" else "#dc3545"
                    else:  # Low confidence
                        status_color = "#ffc107"
                    
                    st.markdown(f"""
                        <div style="padding:1rem;border-radius:0.5rem;background-color:{status_color};color:white">
                            <h2 style="color:white">Analysis Results</h2>
                            <h3>Diagnosis: {class_name}</h3>
                            <h4>Confidence: {conf_score * 100:.1f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.history.append({
                    'timestamp': datetime.now(),
                    'diagnosis': class_name,
                    'confidence': conf_score,
                    'model_type': model_choice
                })
                st.session_state.total_scans += 1
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        st.markdown("### Recent Analysis History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(
                history_df[['timestamp', 'diagnosis', 'confidence']].tail(),
                hide_index=True
            )

elif page == "About Pneumonia":
    st.title("Understanding Pneumonia")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Symptoms", "Prevention"])
    
    with tab1:
        st.markdown("""
            ### What is Pneumonia?
            Pneumonia is an infection that inflames the air sacs in one or both lungs.
            The air sacs may fill with fluid or pus, causing symptoms such as:
            
            - Cough with phlegm
            - Fever
            - Chills
            - Difficulty breathing
            
            ### Types of Pneumonia
            - Bacterial Pneumonia
            - Viral Pneumonia
            - Fungal Pneumonia
            - Aspiration Pneumonia
        """)
    
    with tab2:
        st.markdown("""
            ### Common Symptoms
            - Chest pain when breathing or coughing
            - Confusion or changes in mental awareness (in adults age 65 and older)
            - Cough, which may produce phlegm
            - Fatigue
            - Fever, sweating and shaking chills
            - Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)
            - Nausea, vomiting or diarrhea
            - Shortness of breath
        """)
    
    with tab3:
        st.markdown("""
            ### Prevention Methods
            1. **Vaccination**
               - Get vaccinated against pneumococcal pneumonia
               - Annual flu shot
            
            2. **Good Hygiene**
               - Regular hand washing
               - Cover your mouth when coughing
            
            3. **Healthy Lifestyle**
               - Don't smoke
               - Regular exercise
               - Adequate rest
               - Balanced diet
            
            4. **Environmental Factors**
               - Good ventilation
               - Avoid air pollution
               - Clean living space
        """)

elif page == "Statistics":
    st.title("Analysis Statistics")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scans", st.session_state.total_scans)
    with col2:
        if st.session_state.history:
            normal_count = sum(1 for h in st.session_state.history if h['diagnosis'] == 'Normal')
            st.metric("Normal Cases", normal_count)
    with col3:
        if st.session_state.history:
            pneumonia_count = sum(1 for h in st.session_state.history if h['diagnosis'] == 'Pneumonia')
            st.metric("Pneumonia Cases", pneumonia_count)
    
    # Add visualizations if there's data
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        # Confidence distribution
        fig1 = px.histogram(
            history_df,
            x='confidence',
            title='Distribution of Confidence Scores',
            labels={'confidence': 'Confidence Score', 'count': 'Number of Cases'}
        )
        st.plotly_chart(fig1)
        
        # Diagnosis distribution
        fig2 = px.pie(
            history_df,
            names='diagnosis',
            title='Distribution of Diagnoses'
        )
        st.plotly_chart(fig2)

else:  # Help page
    st.title("Help & Support")
    
    st.markdown("""
        ### Frequently Asked Questions
        
        #### 1. What types of images can I upload?
        You can upload chest X-ray or CT scan images in JPEG, JPG, or PNG format.
        
        #### 2. How accurate is the system?
        The system's accuracy varies based on image quality and type. Generally, it achieves:
        - 90-95% accuracy for high-quality X-rays
        - 85-90% accuracy for CT scans
        
        #### 3. What should I do if I get an error?
        - Check if the image is in the correct format
        - Ensure the image is clear and properly oriented
        - Try uploading a different image
        
        #### 4. How should I interpret the confidence score?
        - >90%: Very high confidence
        - 70-90%: High confidence
        - <70%: Low confidence, consider recapturing the image
        
        ### Contact Support
        For technical support or questions, please contact:
        - Email: support@pneumoscan.ai
        - Phone: 1-800-PNEUMO-AI
        
        ### Disclaimer
        This tool is meant to assist medical professionals and should not be used as the sole basis for diagnosis. Always consult with a qualified healthcare provider.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>PneumoScan AI - Advanced Medical Imaging Analysis</p>
        <p style='font-size: 0.8em'>© 2024 Medical AI Solutions</p>
    </div>
    """,
    unsafe_allow_html=True
)