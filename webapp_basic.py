import streamlit as st
import tempfile
import os
import subprocess
import sys
from cabd_segmentation.cabd_segmentation_final import analyze_sign_sequence_ui, BiomechanicalSignProcessor, add_captions_to_video


model_path="/path/to/isl_model.keras"
encoder_path = "/path/to/label_encoder.pkl"


processor = BiomechanicalSignProcessor(model_path, encoder_path)

st.set_page_config(page_title="ISL Recognition App", layout="wide")

st.title("🤖 Indian Sign Language (ISL) Recognition Web App")
st.write("Upload a video or start a live session to recognize Indian Sign Language.")

mode = st.sidebar.selectbox("Choose Mode:", ["Upload a Video", "Live Video Processing"])

if mode == "Upload a Video":
    st.header("📤 Upload a Video for ISL Recognition")
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_video_path)

        if st.button("🔍 Process Video"):
            st.info("Processing the video... This may take some time ⏳")
            
            isl_text_placeholder = st.empty()  
            eng_text_placeholder = st.empty()  
            def update_ui(isl_sentence, eng_sentence):
                isl_text_placeholder.text(f"📝 ISL Sentence: {isl_sentence}")
                eng_text_placeholder.text(f"📖 English Translation: {eng_sentence}")

            results = analyze_sign_sequence_ui(temp_video_path, processor, ui_update_callback=update_ui)

            frames = results["segments"]  

            predictions = [pred["prediction"] for pred in results["predictions"]]  

            st.subheader("📖 Final ISL Sentence")
            st.write(results['sentence'])

            st.subheader("📝 Final English Translation")
            st.write(results['eng_sentence'])

            output_video_path = add_captions_to_video(temp_video_path, frames, predictions)
            
            st.success("✅ Video Processing Complete!")
            # st.video(output_video_path)

            with open(output_video_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Captioned Video",
                    data=file,
                    file_name="captioned_output.mp4",
                    mime="video/mp4"
                )


elif mode == "Live Video Processing":
    st.header("🎥 Live Video ISL Recognition")

    st.write("Click the button below to start the live video recognition in a separate window.")
    
    if st.button("▶ Start Live Processing"):
        st.info("The results will be displayed here once you close the video window.")
        
        status = st.empty()
        status.text("Processing video in separate window...")
        
        cmd = [
            sys.executable, 
            "segmentation/live_video_helper.py", 
            model_path,
            encoder_path,
            "live_results.json"
        ]
        
        process = subprocess.Popen(cmd)
        process.wait()  
        
        try:
            import json
            with open('live_results.json', 'r') as f:
                results = json.load(f)
                
            status.text("Processing complete!")
            
            st.subheader("📖 Final ISL Sentence")
            st.write(results['sentence'])

            st.subheader("📝 Final English Translation")
            st.write(results['eng_sentence'])
        except:
            status.text("Could not retrieve results. The window may have been closed prematurely.")