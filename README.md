# 📌 Deep Learning–Based Facial Recognition Attendance System

---

## 📖 Overview
This project is an **AI-powered attendance system** that uses **facial recognition and deep learning (CNNs)** to automatically mark attendance in real time.  
It eliminates manual processes and ensures **accuracy, security, and efficiency**.

---

## 🚀 Features
- 📷 Real-time face detection using OpenCV  
- 🧠 CNN-based facial recognition model  
- 🗂️ Automatic attendance marking with timestamps  
- 🛡️ Prevents proxy attendance  
- 💾 SQLite database integration  
- 🖥️ User-friendly GUI (KivyMD)  
- ⚡ Fast detection (~0.2 sec/frame)  

---

## 🧠 Technologies Used
- **Python 3.x**  
- **TensorFlow & Keras**  
- **OpenCV**  
- **SQLite**  
- **KivyMD (GUI Framework)**  

---

## 🏗️ System Architecture
The system consists of the following modules:

1. **Dataset Collection**  
2. **Data Preprocessing & Augmentation**  
3. **Model Training (CNN)**  
4. **Face Detection & Recognition**  
5. **Attendance Management**  
6. **Database Handling**  
7. **Graphical User Interface**  

---

## 🔄 Workflow
1. Capture user images via webcam  
2. Preprocess and store dataset  
3. Train CNN model  
4. Detect and recognize faces in real-time  
5. Mark attendance automatically  
6. Store records in database  

---

## 📊 Results
- ✅ Training Accuracy: **99.09%**  
- ✅ Validation Accuracy: **96.36%**  
- ⚡ Detection Speed: **~0.2 sec/frame**  
- 💯 Attendance Logging: **100% success**  

---

## ⚠️ Challenges
- Limited dataset size  
- Performance drop in low lighting  
- Multi-face detection complexity  
- Hardware limitations during training  

---

## 🔮 Future Enhancements
- Integrate **FaceNet / VGGFace**  
- Add **mask detection & emotion recognition**  
- Cloud-based dashboard  
- Mobile app version  
- SMS/Email alerts for attendance  

---

## 👥 Contributors
- **Ahmad Ali Sultan**  
- **Farhan Khan**  

---

## 🎓 Supervisor
- **Mr. Umar Numan**  

---

## 📚 References
- FaceNet (Google Research)  
- DeepFace (Facebook AI)  
- VGGFace Research  

---

## 📷 Screenshots (Optional)
_Add your GUI screenshots here_

---

## 🛠️ Installation
```bash
git clone https://github.com/your-username/Deep-Learning-Facial-Recognition-Attendance-System.git
cd Deep-Learning-Facial-Recognition-Attendance-System
pip install -r requirements.txt
python main.py
