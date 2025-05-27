import cv2

def extract_frame(video_path, frame_number, output_image_path):
    
    """
    Extracts a specific frame from a video and saves it as an image.

    Args:
        video_path (str): The path to the video file.
        frame_number (int): The frame index to extract.
        output_image_path (str): The path where the extracted frame should be saved.

    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    success, frame = cap.read()
    
    if success:
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
    else:
        print(f"Error: Could not read frame {frame_number}")
    
    cap.release()



video_path = '/path/to/video.mp4' 
frame_number = 0 #Required frame number
output_image_path = "/path/to/output_image.jpg"  

extract_frame(video_path, frame_number, output_image_path)

