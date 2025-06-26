# 🎭 Face Sentiment Analysis (3-Class Emotion Detection)

A deep learning-based face sentiment analysis system using Convolutional Neural Networks (CNN) that classifies facial expressions into **Happy**, **Sad**, or **Angry** emotions. This project includes custom training, live webcam detection, and image-based emotion predictions.

---

## 📌 Project Features

- ✅ Trains a CNN on FER-2013 dataset filtered to 3 emotions.
- 🎥 Real-time facial emotion detection using OpenCV and Haar cascades.
- 🧠 Uses Keras + TensorFlow backend for deep learning.
- 🖼️ Accepts static image files for emotion prediction.
- 💾 Saves predictions with bounding boxes and labels.

---

## 📁 Folder Structure

face-sentiment-analysis/

├── models/
|   ── emotion_model.h5 # Trained CNN model
│   ── haarcascade_frontalface_default.xml # Face detection model
│
├── dataset/ # NOT pushed to GitHub
│ ├── training_set/
│ │ ├── happy/
│ │ ├── sad/
│ │ └── angry/
│ └── test_set/
│ ├── happy/
│ ├── sad/
│ └── angry/
│
├── train_model.py # Trains the CNN model
├── predict_sentiment.py # Predicts emotion from an input image
├── test_images/ # Sample images for testing
├── requirements.txt # Python dependencies
└── README.md # Project description

---

## 🚀 How to Use

### 📌 1. Install Requirements
pip install -r requirements.txt

📌 2. Train the Model
Make sure you’ve placed the dataset correctly. Then:
python train_model.py
This will train the model and save it as models/emotion_model.h5.

📌 3. Run Prediction
To analyze emotion from a test image:
python predict_sentiment.py test_images/child_face.jpg
Press Q or wait for the window to close automatically.

📚 Dataset
This project uses the FER-2013 dataset trimmed to 3 emotions: happy, sad, and angry.

You can download the dataset manually from [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), and extract the following folders:

- dataset/training_set/happy
- dataset/training_set/sad
- dataset/training_set/angry
- dataset/test_set/happy
- dataset/test_set/sad
- dataset/test_set/angry
  
⚠️ Dataset is excluded from GitHub due to size limits.

  
🔧 Requirements

 - Python 3.8+

 - TensorFlow / Keras

 - OpenCV

 - NumPy

Install via:
pip install tensorflow keras opencv-python numpy

💡 Credits

- Model trained by Abeer.

- FER-2013 dataset from Kaggle.

- Face detection via OpenCV Haar Cascades.

📬 License
This project is open-source and free to use for non-commercial purposes.
