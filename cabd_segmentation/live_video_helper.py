import cv2
import numpy as np
import sys
import json
from cabd_segmentation_final import process_live_video, BiomechanicalSignProcessor

def run_live_video_processing(model_path, encoder_path, output_file='live_results.json'):
    """
    Run live video processing with OpenCV window and save results to a JSON file.
    
    Args:
        model_path (str): Path to the model file
        encoder_path (str): Path to the encoder file
        output_file (str): Path to save results
    """
    # Initialize processor
    processor = BiomechanicalSignProcessor(model_path, encoder_path)
    
    # Run live video processing
    print("Starting live video processing...")
    print("Press 'q' in the video window to exit")
    
    # Define a simple callback that just prints updates (won't affect Streamlit)
    def console_callback(isl_sentence, eng_sentence, frame=None):
        print(f"ISL: {isl_sentence}")
        print(f"ENG: {eng_sentence}")
    
    results = process_live_video(processor, ui_update_callback=console_callback)
    
    # Save results to file that Streamlit can read
    with open(output_file, 'w') as f:
        json.dump({
            'sentence': results['sentence'],
            'eng_sentence': results['eng_sentence']
        }, f)
    
    print("Processing complete. Results saved.")
    return results

if __name__ == "__main__":
    # This part runs when the script is called directly
    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        encoder_path = sys.argv[2]
        
        output_file = 'live_results.json'
        if len(sys.argv) >= 4:
            output_file = sys.argv[3]
            
        run_live_video_processing(model_path, encoder_path, output_file)
    else:
        print("Usage: python live_video_helper.py <model_path> <encoder_path> [output_file]")