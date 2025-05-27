import streamlit as st
import tempfile
import os
from datetime import datetime
from cabd_segmentation.cabd_segmentation_final import analyze_sign_sequence_ui, BiomechanicalSignProcessor, add_captions_to_video
from tts.deepgram import text_to_speech


model_path="/path/to/isl_model.keras"
encoder_path = "/path/to/label_encoder.pkl"



processor = BiomechanicalSignProcessor(model_path, encoder_path)

def load_custom_css():
    st.markdown("""
        <style>
        /* Set Main Page Background to White */
        .stApp {
            background-color: #f5fff7 !important;
        }
        
        /* Modern Green Theme */
        :root {
            --primary: #059669;      /* Green 600 */
            --primary-light: #D1FAE5; /* Green 100 */
            --secondary: #0EA5E9;    /* Sky 500 */
            --accent: #10B981;       /* Emerald 500 */
            --surface: #ECFDF5;      /* Green 50 */
            --success: #059669;      /* Green 600 */
            --warning: #F59E0B;      /* Amber 500 */
            --text-primary: #064E3B; /* Green 900 */
            --text-secondary: #065F46;/* Green 800 */
            --shadow-sm: 0 2px 4px rgba(5, 150, 105, 0.05);
            --shadow-md: 0 4px 6px rgba(5, 150, 105, 0.08);
            --shadow-lg: 0 10px 15px rgba(5, 150, 105, 0.1);
            --glass: rgba(236, 253, 245, 0.8);
        }

        /* Modern Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--primary-light);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }

        /* Fixed: Added dark border to header box */
        .main-header {
            background: linear-gradient(135deg, var(--primary-light), white);
            border-radius: 20px;
            padding: 2.5rem;
            margin: 1rem 0 2rem 0;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-lg);
            border: 2px solid #064E3B; /* Added dark border */
            animation: headerGlow 2s ease-in-out infinite alternate;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-content {
            flex: 2;
        }

        .header-content h5 {
            color: black;
        }

        /* Added specific styling for header image */
        .header-image {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            max-width: 280px;
            margin-left: 20px;
        }
        
        .header-image img {
            max-width: 100%;
            max-height: 180px;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: var(--shadow-md);
        }

        .avatar-container {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            min-width: 200px;
            min-height: 200px;
        }

        @keyframes headerGlow {
            0% { box-shadow: var(--shadow-lg); }
            100% { box-shadow: 0 15px 25px rgba(5, 150, 105, 0.2); }
        }

        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 200%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(5, 150, 105, 0.1),
                transparent
            );
            animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        .main-header h1 {
            font-size: 2.8rem;
            background: linear-gradient(135deg, var(--text-primary), var(--primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            font-weight: 800;
        }

        /* Enhanced Time Display */
        .time-display {
            background: var(--glass);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(5, 150, 105, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeIn 0.5s ease;
        }

        .time-display:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        /* Fixed: Modern Button Styles - Changed hover color from red to green */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.8rem 2rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
        }

        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                120deg,
                transparent,
                rgba(255, 255, 255, 0.3),
                transparent
            );
            transition: 0.5s;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(5, 150, 105, 0.2);
            background: linear-gradient(135deg, #047857, #059669); /* Darker green on hover */
        }

        .stButton > button:hover::before {
            left: 100%;
        }

        /* Fixed: File Upload Area - Darker text and border for better visibility */
        .upload-area {
            background: var(--surface);
            border: 2px dashed var(--primary);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .upload-area:hover {
            border-color: var(--accent);
            transform: scale(1.01);
            box-shadow: var(--shadow-lg);
        }

        /* Make file uploader text darker */
        .stFileUploader > div > label {
            color: #064E3B !important; /* Dark green */
            font-weight: 600 !important;
        }
        
        /* Make help icon darker */
        .stFileUploader span[data-testid="stHelpIcon"] {
            color: #064E3B !important;
            border: 1px solid #064E3B !important;
        }

        /* Make uploaded filename visible with dark color */
        .stFileUploader p {
            color: #064E3B !important;
            font-weight: 600 !important;
        }
        
        /* FIX 4: Make uploaded filename visible */
        .stFileUploader [data-testid="stFileUploaderFileContent"] {
            color: #064E3B !important;
            font-weight: 600 !important;
            background-color: rgba(5, 150, 105, 0.1) !important;
            padding: 8px !important;
            border-radius: 8px !important;
            margin-top: 10px !important;
            border: 1px solid #064E3B !important;
        }

        /* Status Messages */
        .status-message {
            background: var(--glass);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 1.25rem;
            margin: 1rem 0;
            border-left: 4px solid var(--primary);
            animation: slideIn 0.3s ease;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        @keyframes slideIn {
            from { transform: translateX(-10px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* Video Container */
        .video-containerr {
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow-lg);
            transition: transform 0.3s ease;
            border: 1px solid rgba(5, 150, 105, 0.1);
        }

        .video-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 30px rgba(5, 150, 105, 0.15);
        }

        /* Processing Options */
        .processing-options {
            background: var(--glass);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(5, 150, 105, 0.1);
            transition: all 0.3s ease;
        }

        .processing-options:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }

        /* Toggle Switch */
        .stCheckbox > div > label {
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }

        .stCheckbox > div > label:hover {
            background-color: var(--primary-light);
        }

        /* Section Titles */
        .section-title {
            font-size: 1.5rem;
            color: var(--text-primary);
            margin: 1.5rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--primary-light);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            animation: fadeIn 0.5s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Enhanced Icons */
        .icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            background: var(--primary-light);
            border-radius: 6px;
            margin-right: 0.5rem;
            color: var(--primary);
        }

        /* FIX 1: Make sidebar radio buttons text color darker and visible */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: var(--surface);
            border-right: 1px solid rgba(5, 150, 105, 0.1);
        }

        .sidebar-content {
            padding: 1.5rem;
        }
        
        /* Make sidebar text darker */
        [data-testid="stSidebar"] {
            color: #064E3B !important;
        }
        
        /* Make radio button text darker */
        [data-testid="stSidebar"] .st-bf {
            color: #064E3B !important;
            font-weight: 600 !important;
        }
        
        /* Make radio button itself darker */
        [data-testid="stSidebar"] .st-ae {
            border-color: #064E3B !important;
        }
        
        /* Selected radio button fill color */
        [data-testid="stSidebar"] .st-ae:checked {
            background-color: #064E3B !important;
        }
        
        /* NEW FIX: Make radio buttons text more visible with important black color */
        div[role="radiogroup"] label {
            color: #000000 !important;
            font-weight: 700 !important;
        }
        
        .stRadio > div > div > label {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Radio buttons text */
        [data-testid="stRadio"] label {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* FIX 3: Make sidebar close icon darker */
        [data-testid="stSidebarNav"] button {
            color: #064E3B !important;
        }
        
        [data-testid="stSidebarNav"] button:hover {
            background-color: rgba(5, 150, 105, 0.1) !important;
        }
        
        [data-testid="stSidebarNav"] svg {
            fill: #064E3B !important;
        }
        
        /* FIX 2: Add border and improved styling to expander content */
        [data-testid="stExpander"] {
            border: 1px solid #064E3B !important;
            border-radius: 10px !important;
            margin-bottom: 15px !important;
        }
        
        [data-testid="stExpander"] details {
            background-color: rgba(5, 150, 105, 0.05) !important;
            padding: 10px !important;
            border-radius: 10px !important;
        }
        
        [data-testid="stExpander"] summary {
            color: #064E3B !important;
            font-weight: 600 !important;
            padding: 8px !important;
        }
        
        [data-testid="stExpander"] div {
            color: #064E3B !important;
        }

        /* Radio Buttons */
        .stRadio > div {
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }

        .stRadio > div:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        /* Static Avatar Container */
        .static-avatar-container {
            width: 200px;
            height: 200px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            position: relative;
        }

        .static-avatar-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        /* FIX 5: Add progress bar styling */
        .progress-container {
            background-color: #f0f9f4 !important;
            border-radius: 10px !important;
            margin: 15px 0 !important;
            padding: 15px !important;
            border: 1px solid #064E3B !important;
        }
        
        .st-bq {
            background-color: var(--primary) !important;
        }
        
        .stProgress .st-bp {
            background-color: var(--primary) !important;
        }
        
        .stProgress > div > div {
            background-color: var(--primary) !important;
        }
        
        /* Processing status text */
        .processing-status {
            color: #064E3B !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
        }
        
        /* FIX 6: Make result text darker */
        .stMarkdown h3 {
            color: #064E3B !important;
            font-weight: 700 !important;
            margin-top: 25px !important;
            margin-bottom: 15px !important;
            border-bottom: 2px solid rgba(5, 150, 105, 0.2) !important;
            padding-bottom: 8px !important;
        }
        
        .stMarkdown p {
            color: #064E3B !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
            background-color: rgba(5, 150, 105, 0.05) !important;
            padding: 10px 15px !important;
            border-radius: 8px !important;
            border-left: 3px solid var(--primary) !important;
        }
        
        .result-text {
            color: #064E3B !important;
            font-weight: 600 !important;
            background-color: rgba(5, 150, 105, 0.05) !important;
            padding: 12px !important;
            border-radius: 8px !important;
            border-left: 4px solid var(--primary) !important;
            margin: 10px 0 !important;
        }
        
        .stSuccess {
            color: #064E3B !important;
            font-weight: 600 !important;
            background-color: #D1FAE5 !important; /* Lighter green background */
            border: 1px solid var(--primary) !important;
        }

        .stSuccess p {
            color: #064E3B !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        }

        .stSelectbox > div > div {
            background-color: white !important;
            border: 2px solid #064E3B !important;
            border-radius: 10px !important;
            color: #064E3B !important;
            font-weight: 600 !important;
        }

        /* Make dropdown text darker and bolder */
        .stSelectbox label {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        /* Style the dropdown options */
        .stSelectbox [data-baseweb="select"] {
            color: #064E3B !important;
            font-weight: 600 !important;
        }

        /* Make dropdown placeholder text visible */
        .stSelectbox [data-baseweb="select"] [data-testid="stMarkdown"] p {
            color: #064E3B !important;
            font-weight: 600 !important;
        }

        [data-testid="stSidebar"] .stSelectbox > div > div > div {
    color: #064E3B !important;
    font-weight: 600 !important;
}

        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="ISL Recognition",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    with st.sidebar:
        
        st.markdown("""
            <div class="section-title" style="color: #064E3B; font-weight: 700;">
                Mode Selection
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
           
            """, unsafe_allow_html=True)

        mode = st.selectbox(
            "Select Mode",
            ["📤 Upload Video", "🎥 Live Camera"],
            index=0,
            help="Choose between uploading a video or using live camera feed"
        )

        with st.expander("ℹ️ About"):
            st.markdown("""
                <div style='color: #064E3B;'>
                    <h4 style='color: #064E3B; margin-bottom: 1rem;'>
                        <i class="fas fa-info-circle"></i> ISL Recognition AI
                    </h4>
                    <p style='color: #064E3B; font-weight: 500;'>Next-generation sign language interpretation powered by advanced AI.</p>
                    <p style='color: #064E3B; margin-top: 1rem; font-weight: 500;'>Version 2.1.0</p>
                </div>
            """, unsafe_allow_html=True)

        with st.expander("📚 How to Use"):
            st.markdown("""
                <div style='color: #064E3B;'>
                    <h4 style='color: #064E3B; margin-bottom: 1rem;'>Quick Guide</h4>
                    <ol style='color: #064E3B; font-weight: 500;'>
                        <li><strong>Video Upload</strong>
                            <ul>
                                <li>Drop your video file</li>
                                <li>Configure options</li>
                                <li>Process and view results</li>
                            </ul>
                        </li>
                        <li><strong>Live Camera</strong>
                            <ul>
                                <li>Enable camera access</li>
                                <li>Position yourself clearly</li>
                                <li>View real-time results</li>
                            </ul>
                        </li>
                    </ol>
                    <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--primary-light);'>
                        <h5 style='color: #064E3B;'>💡 Pro Tips</h5>
                        <ul style='color: #064E3B; font-weight: 500;'>
                            <li>Ensure good lighting</li>
                            <li>Keep hands and upper body visible</li>
                            <li>Make steady movements</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)


    st.markdown(f"""
        <div class="main-header">
            <div class="header-content">
                <h1>Indian Sign Language Recognition</h1>
                <h5>
                    Breaking communication barriers through real-time Sign Language interpretation
                </h5>
            </div>
            <div class="header-image">
                <img src="https://www.signlanguagenyc.com/wp-content/uploads/2016/04/asl-interpreter-services-appreciation-01.png" alt="ISL Recognition">
            </div>
        </div>
    """, unsafe_allow_html=True)


    if mode == "📤 Upload Video":
        st.markdown("""
            <div class="section-title">
                 Video Upload Mode
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov"],
            help="Supported formats: MP4, AVI, MOV"
        )

        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="video-container">', unsafe_allow_html=True)
                temp_dir = tempfile.mkdtemp()
                temp_video_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                st.video(temp_video_path)
                st.markdown('</div>', unsafe_allow_html=True)
                
                isl_text_placeholder = st.empty()
                eng_text_placeholder = st.empty()
                
                st.markdown("""
                    <div style="margin-top: 20px; padding: 10px;">
                        <h3 style="color: #064E3B; font-weight: 700;">Live Recognition Results</h3>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                    <h4 style='color: #064E3B; margin-bottom: 1rem;'>
                        <i class="fas fa-cog"></i> The video is ready to be processed
                    </h4>
                """, unsafe_allow_html=True)
                
                process_button = st.button("🔍 Process Video", use_container_width=True)
                
                if process_button:
                    with st.spinner("Processing video..."):
                        def update_ui(isl_sentence, eng_sentence, frame=None):
                            isl_text_placeholder.markdown(
                                f'<div class="result-text" style="background-color: rgba(5, 150, 105, 0.05); padding: 15px; '
                                f'border-radius: 8px; border-left: 4px solid #059669; margin-bottom: 10px;">'
                                f'<span style="color: #064E3B; font-weight: 700;">📝 ISL Sentence:</span> '
                                f'<span style="color: #064E3B; font-weight: 600;">{isl_sentence}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            eng_text_placeholder.markdown(
                                f'<div class="result-text" style="background-color: rgba(5, 150, 105, 0.05); padding: 15px; '
                                f'border-radius: 8px; border-left: 4px solid #059669;">'
                                f'<span style="color: #064E3B; font-weight: 700;">📖 English Translation:</span> '
                                f'<span style="color: #064E3B; font-weight: 600;">{eng_sentence}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                        headless_mode = True
                        
                        results_container = st.container()
                        
                        results = analyze_sign_sequence_ui(
                            temp_video_path, 
                            processor, 
                            ui_update_callback=update_ui, 
                            headless=headless_mode
                        )

                        text_to_speech(results['eng_sentence'])

                        frames = results["segments"]
                        predictions = [pred["prediction"] for pred in results["predictions"]]

                        with results_container:
                            st.markdown('<h3 style="color: #064E3B; font-weight: 700;">📖 Final ISL Sentence</h3>', unsafe_allow_html=True)
                            st.markdown(
                                f'<p style="color: #064E3B; font-weight: 600; background-color: rgba(5, 150, 105, 0.05); '
                                f'padding: 15px; border-radius: 8px; border-left: 4px solid #059669;">{results["sentence"]}</p>', 
                                unsafe_allow_html=True
                            )

                            st.markdown('<h3 style="color: #064E3B; font-weight: 700;">📝 Final English Translation</h3>', unsafe_allow_html=True)
                            st.markdown(
                                f'<p style="color: #064E3B; font-weight: 600; background-color: rgba(5, 150, 105, 0.05); '
                                f'padding: 15px; border-radius: 8px; border-left: 4px solid #059669;">{results["eng_sentence"]}</p>', 
                                unsafe_allow_html=True
                            )

                            output_video_path = add_captions_to_video(temp_video_path, frames, predictions)
                            
                            st.success("Video processing completed!")

                            with open(output_video_path, "rb") as file:
                                btn = st.download_button(
                                    label="📥 Download Captioned Video",
                                    data=file,
                                    file_name="captioned_output.mp4",
                                    mime="video/mp4"
                                )

    else:  
        st.markdown("""
            <div class="section-title">
                <span class="icon">🎥</span> Live Recognition Mode
            </div>
        """, unsafe_allow_html=True)
        
        live_isl_text = st.empty()
        live_eng_text = st.empty()
        status_message = st.empty()
        
        status_message.markdown("""
            <div class="status-message">
                <i class="fas fa-info-circle"></i>
                <span style='color: #064E3B; font-weight: 600;'>Position yourself clearly in front of the camera for best results</span>
            </div>
        """, unsafe_allow_html=True)
        
        start_button = st.button("▶️ Start Live Recognition", use_container_width=True)
        
        if start_button:
            status_message.markdown("""
                <div class="status-message success">
                    <i class="fas fa-check-circle"></i>
                    <span style='color: #064E3B; font-weight: 600;'>Opening live recognition in a separate window. Please check your taskbar.</span>
                </div>
            """, unsafe_allow_html=True)
            

            results_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
            
            import subprocess
            import sys
            
            cmd = [
                sys.executable, 
                "segmentation/live_video_helper.py", 
                model_path,
                encoder_path,
                results_file
            ]
            
            def update_ui_with_results(isl_sentence, eng_sentence):
                live_isl_text.markdown(
                    f'<div class="result-text" style="background-color: rgba(5, 150, 105, 0.05); padding: 15px; '
                    f'border-radius: 8px; border-left: 4px solid #059669; margin-bottom: 10px;">'
                    f'<span style="color: #064E3B; font-weight: 700;">📝 ISL Sentence:</span> '
                    f'<span style="color: #064E3B; font-weight: 600;">{isl_sentence}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                live_eng_text.markdown(
                    f'<div class="result-text" style="background-color: rgba(5, 150, 105, 0.05); padding: 15px; '
                    f'border-radius: 8px; border-left: 4px solid #059669;">'
                    f'<span style="color: #064E3B; font-weight: 700;">📖 English Translation:</span> '
                    f'<span style="color: #064E3B; font-weight: 600;">{eng_sentence}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            process = subprocess.Popen(cmd)
            
            status_message.markdown("""
                <div class="status-message info">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span style='color: #064E3B; font-weight: 600;'>Processing video in a separate window. Results will appear here when finished.</span>
                </div>
            """, unsafe_allow_html=True)
            
            process.wait()
            
            try:
                import json
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                status_message.markdown("""
                    <div class="status-message success">
                        <i class="fas fa-check-circle"></i>
                        <span style='color: #064E3B; font-weight: 600;'>Processing complete! Results shown below.</span>
                    </div>
                """, unsafe_allow_html=True)
                
                update_ui_with_results(results['sentence'], results['eng_sentence'])
                text_to_speech(results['eng_sentence'])
                
            except Exception as e:
                status_message.markdown(f"""
                    <div class="status-message error">
                        <i class="fas fa-exclamation-circle"></i>
                        <span style='color: #7f1d1d; font-weight: 600;'>Could not retrieve results. The window may have been closed prematurely.</span>
                    </div>
                """, unsafe_allow_html=True)
                st.error(f"Error: {str(e)}")
            

            try:
                os.unlink(results_file)
            except:
                pass

if __name__ == "__main__":
    main()