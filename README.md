# Continuous-ISL-Recognition-from-Live-Video-Input

This repository contains the implementation of a real-time Indian Sign Language (ISL) recognition system capable of translating continuous ISL signing into coherent English sentences and speech. The system integrates deep learning, computer vision, and NLP techniques to bridge communication gaps between the deaf community and non-signers. It supports both recorded and live video input and features modules for word recognition, sentence segmentation, translation, and text-to-speech conversion. The project also includes a Streamlit-based web application.

a. URL / Source for Dataset
  Dataset: (https://zenodo.org/records/4010759) (Used for training the word recognition model)
  Custom Dataset: A manually prepared ISL video dataset consisting of 15-20 sentences (Used for sentence-level analysis)


b. Software and Hardware Requirements

Hardware Requirements:
    The project was developed using the following hardware: an Apple M2 chip with an 8-core
    CPU and a 10-core GPU, 8 GB of unified memory, and a 512 GB SSD. Since the
    Augmented INCLUDE dataset exceeds 1 TB, an external hard disk is necessary for
    data storage and management.

Software Requirements:
    This project utilizes Python 3.10 with key libraries such as TensorFlow, Keras,
    NumPy, OpenCV, and MediaPipe for deep learning-based sign language
    recognition. Streamlit provides an interactive UI, while SciPy and scikit-learn
    assist with signal processing and machine learning tasks. Concurrent processing is 
    handled using ThreadPoolExecutor to improve performance. Additionally, Deepgram, gTTS, 
    and pygame are integrated for text-to-speech synthesis, while LangChain supports 
    AI-driven conversation.



c. How to run the code:

Note:The source code contains several file paths. Ensure all paths are correctly set before execution.

1. Data Augmentation: Navigate to the data_augmentation directory and run video_augmentation.py to augment the dataset.

2. Model Training: Go to the model_training directory and execute the desired script based on the model needed.

3. Model Testing: Inside the model_training directory, run the model testing script to evaluate the word recognition model by providing isl word video as input.

4. Fine-tuning : To fine-tune the model by adding more samples or training on unseen words, execute finetune.py inside the finetuning directory.

5. Continuous Sentence Recognition (Choose any one):
     (a) Run hybrid_segmentation.py in the main directory to perform ISL recognition on video input .
     (b) Run cabd_segmentation_main.py in the main directory, selecting either live input or recorded video for ISL recognition.

6. Web Application:
   Skip Step 4 if using the web app.
   To launch a basic web app, run webapp_basic.py.
   For an enhanced, user-friendly version, execute webapp_final.py.
   The web application allows users to upload a recorded video or test the system with live video input.

7. utils-dir : Contains necessary utility function that was needed for development.

8. evals-dir : Cotains all the code snippets required for evaluation of the system.
