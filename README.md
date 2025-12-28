# 🧠 Image Recognition System using OpenCV

## 📌 About the Project

This project was developed by me as part of my **AI Internship at CodeClause (December 2025)**.
The main goal of this project was to **understand how basic computer vision techniques work practically**, not just theoretically.

Instead of using heavy deep learning models, I focused on **classical OpenCV methods** such as **Haar Cascade classifiers** and **contour-based shape detection** to clearly understand the core concepts.

---

## 👩‍💻 Author

**Siddhi Mishra**
AI Intern – CodeClause
December 2025

---

## 🎯 Why I Chose This Project

During my internship, I wanted to build something that:

* Works in real time
* Uses real images instead of datasets only
* Helps me understand how detection actually happens frame by frame

This project helped me understand:

* Why preprocessing is important
* How detection accuracy changes with parameters
* What problems occur in real-world images

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy

---

## ✨ What I Implemented

### 🔍 Face Detection

I implemented face detection using **Haar Cascade classifiers** provided by OpenCV.
I learned that:

* Detection works better on frontal faces
* Lighting conditions affect accuracy
* `scaleFactor` and `minNeighbors` are very important for tuning

---

### 👀 Eye Detection

Eye detection is performed **only inside the detected face region**, which improves performance and reduces false detections.

---

### 🔺 Shape Detection

To understand contour detection, I implemented shape detection using:

* Grayscale conversion
* Gaussian blur
* Binary thresholding
* Contour approximation

Based on the number of vertices, shapes are classified as rectangles, circles, or others.

---

### 📹 Webcam Detection

I also added real-time webcam detection to understand how:

* Frame-by-frame processing works
* Detection speed affects performance
* User interaction can be handled using keyboard inputs

---

## 🧪 Testing & Experiments

I tested the project using:

* Synthetic images created using OpenCV
* Multiple downloaded test images
* Group images to observe detection differences

This helped me understand the limitations of classical detection methods.

---

## 📂 Project Structure

```
image_recognition_project/
├── image_recognition_project.py
├── project_demo.py
├── test_detection.py
├── requirements.txt
├── README.md
├── test_images/
└── output/
```

---

## 📈 What I Learned from This Project

* Practical understanding of OpenCV
* Image preprocessing techniques
* How classical CV differs from deep learning
* Debugging detection errors
* Writing structured and readable code

---

## 🔮 Future Improvements

If I extend this project further, I would like to:

* Try deep learning-based face detection
* Improve accuracy using better datasets
* Create a simple web interface
* Add emotion or mask detection

---

## ✅ Internship Outcome

This project helped me strengthen my basics of **Computer Vision** and gave me confidence to work on more advanced AI projects in the future.


bhi karwa dunga.
Bas bolo ❤️
