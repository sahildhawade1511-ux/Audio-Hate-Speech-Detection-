🎧 Audio Hate Speech Detection

📌 Overview

This project focuses on detecting harmful or abusive speech from audio data.
It converts voice signals into Mel-Spectrogram images and uses a deep learning model (MobileNetV2) to classify whether the audio contains hate speech.

---

🚀 Features

- 🎙️ Audio preprocessing using Librosa
- 📊 Conversion of audio into Mel-Spectrograms
- 🤖 Deep learning model using MobileNetV2
- ⚡ Fast and efficient prediction pipeline
- 🖥️ Simple interface for testing audio inputs

---

🧠 Model Details

- Model: MobileNetV2 (Transfer Learning)
- Input: Mel-Spectrogram Images
- Framework: TensorFlow / Keras
- Output: Classification of Hate / Non-Hate Speech

---

📂 Project Structure

Audio-Hate-Speech-Detection/
│── mobilenet_audio_model.keras     # Trained model
│── train_and_gui_mobilenet_audio_fixed.py   # Training + GUI script
│── README.md

---

⚙️ Installation

1. Clone the repository:

git clone https://github.com/sahildhawade1511-ux/Audio-Hate-Speech-Detection-.git
cd Audio-Hate-Speech-Detection-

2. Install dependencies:

pip install -r requirements.txt

---

▶️ Usage

Run the script:

python train_and_gui_mobilenet_audio_fixed.py

- Upload or record audio
- Model predicts whether it contains hate speech

---

📊 Results

- Achieved high accuracy in detecting harmful audio content
- Efficient and lightweight due to MobileNet architecture

(Tip: You can add exact accuracy here if you have it)

---

🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Matplotlib

---

💡 Future Improvements

- Deploy as a web app (Streamlit/Flask)
- Improve dataset diversity
- Real-time audio moderation system
- Add multilingual support

---

📌 Use Cases

- Social media moderation
- Voice-based content filtering
- Online gaming voice chat monitoring

---

👨‍💻 Author

Sahil Dhawade
GitHub: https://github.com/sahildhawade1511-ux

---

⭐ If you like this project

Give it a ⭐ on GitHub!
