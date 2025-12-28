# Image Recognition System using OpenCV  
**AI Internship Project – CodeClause**

## 👩‍💻 Author  
**Siddhi Mishra**  
AI Intern, CodeClause  
December 2025

---

## 📌 Introduction  
This project was developed by me during my **AI Internship at CodeClause**.  
The goal was to learn and apply **basic image recognition concepts** using OpenCV — not just copy code from the internet.

I focused on techniques that help understand how images are processed, how detection works step by step, and how parameters affect results.

---

## 🛠️ Technologies Used  
- Python  
- OpenCV  
- NumPy

---

## 🔍 Project Features  

### ➤ Face Detection  
I used **Haar Cascade classifiers** from OpenCV to detect human faces.  
This works best with clear and frontal face images. I manually tuned the detection parameters so that results improved on different test images.

---

### ➤ Eye Detection  
Within every detected face, the system also detects eyes.  
This helped me understand *region-based detection* and how to limit detection to smaller areas for better performance.

---

### ➤ Shape Detection  
I wrote code to detect basic shapes using contour detection.  
Steps include:
- Converting image to grayscale  
- Applying Gaussian blur  
- Thresholding  
- Finding contours and classifying shapes (rectangles, circles, others)

---

### ➤ Real-Time Webcam Detection  
Face detection also works on live webcam feed.  
You can save frames by pressing the **‘s’** key.  
This part helped me learn how video streams are processed frame by frame.

---

## 🧪 Testing Tools  
I added a separate testing script to:
- Process shapes  
- Detect faces in real images
- Auto-process all images in a folder
- Save results in an organized way

This improved my understanding of automation and script structuring.

---

## 📂 Project Structure  
image_recognition_project/
├── image_recognition_project.py
├── project_demo.py
├── test_detection.py
├── requirements.txt
├── README.md
├── test_images/
└── output/

---

## 📈 What I Learned  
- How basic OpenCV detection works  
- Importance of image preprocessing  
- How to debug and tune detection parameters  
- Writing modular and readable Python code  
- Understanding real-time video processing

---

## ⚠️ Limitations  
- Haar cascade works best for frontal faces  
- Accuracy decreases in low light or angled faces  
- Not suited for advanced recognition tasks yet

These limitations motivated me to learn more about deep learning-based detection.

---

## 🔮 Future Improvements  
If I expand this project, I would like to:
- Add deep learning-based detection
- Implement face recognition (identity matching)
- Build a simple web/app interface
- Add emotion detection

---

## 📝 Final Notes  
All code and scripts were written and tested by me during this internship.  
I studied OpenCV documentation, experimented with multiple images, and created test cases to verify results.

This project helped me understand computer vision fundamentals and prepared me for more advanced AI applications.
