# 😊 Facial Emotion Recognition Using Deep Learning

## 🧠 Project Domain
**Computer Vision / Deep Learning / Human-Computer Interaction**

---

## 📖 Introduction

Facial expressions are powerful indicators of a person's emotional state and play a vital role in non-verbal communication. Recognizing emotions automatically from facial expressions has numerous applications, including mental health analysis, human-computer interaction, virtual assistants, and surveillance systems.

This project aims to **classify human facial emotions into seven categories** using **a Convolutional Neural Network (CNN)** trained on the **FER-2013 dataset**. The model is capable of detecting emotions like **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral** from grayscale facial images.

---

## 🧩 Functional Requirements

- 🖼️ **Dataset Loading:** Loaded FER-2013 dataset from Kaggle.
- 🧹 **Data Preprocessing:**
- Reshaped and normalized pixel values.
- One-hot encoded the target emotion labels.
- Augmented data for generalization.
- 🧠 **Model Architecture:** Built a custom CNN using Keras and TensorFlow.
- 📉 **Training Strategy:**
- Used callbacks for early stopping and learning rate scheduling.
- Tuned the learning rate and batch size.
- 📈 **Model Evaluation:** Evaluated model on:
- Accuracy
- Loss
- Confusion Matrix
- 📷 **Real-time Prediction:** Integrated with OpenCV to recognize facial emotions via webcam.


---

## 📂 Dataset

- Source: FER-2013 Kaggle Dataset
- Records: 35,887 labeled grayscale images (48x48 pixels)
- Categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Format: JPG image files



---

## ⚙️ Tools and Libraries Used

- [Python 3.x](https://www.python.org/)
- [Jupyter Notebook](https://jupyter.org/) / Spyder (Anaconda)
- [[TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- `NumPy`, `Pandas` for Data manipulation
- `Matplotlib`, `Seaborn` for Data visualization
- `OpenCV`: for real-time webcam input and face detection

---

## 📊 Model Results

| Metric                | Value     |
|-----------------------|-----------|
| Final Accuracy        | 72.36%    |
| Validation Accuracy   | 66.33%    |
| Final Loss            | 1.0065    |
| Validation Loss       | 1.2288    |


### ✅ Final Model Optimizations**

- Reduced learning rate progressively (0.00015 → 0.000075 → 0.0000375).
- Achieved stable training and improved generalization on validation data.

---

## 🧠 Emotion Classes

| Label                 | Emotion    |
|-----------------------|------------|
| 0                     | Angry      |
| 1                     | Disgust    |
| 2                     | Fear       |
| 3                     | Happy      |
| 4                     | Sad        |
| 5                     | Surprise   |
| 6                     | Neutral    |

---
## 🎯 Real-Time Emotion Recognition
- Implemented using `OpenCV` with live webcam feed
- Detected faces using Haar cascades
- Passed cropped face image through the trained model
- Displayed predicted emotion label in real-time



---
## 📌 Conclusion

This project demonstrates how **deep learning models can classify facial emotions** with a decent level of accuracy using CNNs. With a final validation accuracy of **~67%**, the model performs well on challenging real-world expressions from the FER-2013 dataset.
Such systems can be integrated into:
- **Virtual Assistants**
- **Therapeutic Apps**
- **Gaming and AR/VR**
- **Smart Surveillance Systems**


---

