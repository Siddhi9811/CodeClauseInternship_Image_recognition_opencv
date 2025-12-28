# Image Recognition System with OpenCV
# Project: Building a Simple Object Detection and Facial Recognition System
# Author: Siddhi Mishra
# Date: December 2025
# 
# This project demonstrates basic image processing techniques using OpenCV
# including object detection using Haar cascades and simple shape detection

import cv2
import numpy as np
import os
from datetime import datetime

class ImageRecognitionSystem:
    """
    A simple image recognition system that can detect faces, eyes, and basic shapes
    This class implements various computer vision techniques learned during the internship
    """
    
    def __init__(self):
        """Initialize the recognition system with required classifiers"""
        # Load pre-trained Haar cascade classifiers
        # These are XML files that come with OpenCV installation
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("✓ Haar cascades loaded successfully")
        except Exception as e:
            print(f"Error loading cascades: {e}")
            
        # Initialize variables for tracking
        self.detection_count = 0
        self.processed_images = []
    
    def detect_faces_and_eyes(self, image_path):
        """
        Detect faces and eyes in an image using Haar cascades
        Returns the processed image with detection rectangles
        """
        print(f"\n--- Processing: {os.path.basename(image_path)} ---")
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return None
            
        # Convert to grayscale for better detection performance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using the classifier
        # scaleFactor: how much the image size is reduced at each scale
        # minNeighbors: how many neighbors each candidate rectangle should retain
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        face_count = len(faces)
        print(f"Found {face_count} face(s)")
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            # Draw face rectangle in blue
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of Interest (ROI) for eye detection within face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            # Draw rectangles around detected eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        self.detection_count += face_count
        self.processed_images.append(os.path.basename(image_path))
        
        return img
    
    def detect_shapes(self, image_path):
        """
        Detect basic geometric shapes (circles, rectangles) in an image
        This demonstrates understanding of contour detection and shape analysis
        """
        print(f"\n--- Shape Detection: {os.path.basename(image_path)} ---")
        
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shape_count = {'rectangles': 0, 'circles': 0, 'other': 0}
        
        for contour in contours:
            # Filter out very small contours
            if cv2.contourArea(contour) < 1000:
                continue
                
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Classify shapes based on number of vertices
            if len(approx) == 4:
                # Rectangle
                cv2.drawContours(img, [approx], -1, (0, 255, 255), 3)
                shape_count['rectangles'] += 1
            elif len(approx) > 6:
                # Circle (or ellipse)
                cv2.drawContours(img, [contour], -1, (255, 0, 255), 3)
                shape_count['circles'] += 1
            else:
                # Other shapes
                cv2.drawContours(img, [contour], -1, (128, 128, 128), 2)
                shape_count['other'] += 1
        
        print(f"Detected shapes: {shape_count}")
        return img
    
    def process_webcam_feed(self):
        """
        Real-time face detection using webcam
        Demonstrates live video processing capabilities
        """
        print("\n--- Starting Webcam Face Detection ---")
        print("Press 'q' to quit, 's' to save current frame")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
        
        frame_count = 0
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Display frame count and face count
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Real-time Face Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as: {filename}")
            
            frame_count += 1
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"Webcam session ended. Processed {frame_count} frames.")
    
    def generate_report(self):
        """Generate a summary report of all processing activities"""
        print("\n" + "="*50)
        print("IMAGE RECOGNITION PROJECT REPORT")
        print("="*50)
        print(f"Project Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Detections: {self.detection_count}")
        print(f"Images Processed: {len(self.processed_images)}")
        print(f"Processed Files: {', '.join(self.processed_images) if self.processed_images else 'None'}")
        print("="*50)

def demonstrate_image_processing():
    """
    Main demonstration function that shows various image processing techniques
    This function orchestrates the entire recognition system
    """
    print("🔍 OPENCV IMAGE RECOGNITION SYSTEM")
    print("Developed for AI Internship Project")
    print("-" * 40)
    
    # Create instance of our recognition system
    recognition_system = ImageRecognitionSystem()
    
    # Create a sample test image programmatically if no images available
    print("\n📷 Creating test images for demonstration...")
    
    # Create a simple test image with geometric shapes
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    test_img.fill(255)  # White background
    
    # Draw some shapes for testing
    cv2.rectangle(test_img, (50, 50), (150, 150), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(test_img, (300, 100), 50, (255, 0, 0), -1)  # Blue circle
    cv2.rectangle(test_img, (450, 50), (550, 150), (0, 0, 255), -1)  # Red rectangle
    
    # Save test image
    cv2.imwrite('/home/user/test_shapes.jpg', test_img)
    print("✓ Test image created: test_shapes.jpg")
    
    # Process the test image for shape detection
    result_shapes = recognition_system.detect_shapes('/home/user/test_shapes.jpg')
    if result_shapes is not None:
        cv2.imwrite('/home/user/detected_shapes.jpg', result_shapes)
        print("✓ Shape detection completed")
    
    # Demonstrate webcam functionality (commented out for safety)
    print("\n📹 Webcam Detection Available")
    print("Uncomment the following line to test webcam detection:")
    print("# recognition_system.process_webcam_feed()")
    
    # Generate final report
    recognition_system.generate_report()
    
    print("\n✅ Project demonstration completed successfully!")
    print("Key learning outcomes achieved:")
    print("• Image loading and preprocessing")
    print("• Haar cascade classifier implementation")
    print("• Contour detection and shape analysis") 
    print("• Real-time video processing concepts")
    print("• Object-oriented programming in CV applications")

if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_image_processing()
    
    # Additional utility functions for extended functionality
    
    def batch_process_images(folder_path):
        """Process multiple images in a folder - useful for large datasets"""
        recognition_system = ImageRecognitionSystem()
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        processed_count = 0
        
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(folder_path, filename)
                result = recognition_system.detect_faces_and_eyes(image_path)
                
                if result is not None:
                    output_path = f"processed_{filename}"
                    cv2.imwrite(output_path, result)
                    processed_count += 1
        
        print(f"Batch processing completed: {processed_count} images processed")
    
    def compare_detection_methods():
        """Compare different detection methods for educational purposes"""
        print("\n🔬 DETECTION METHODS COMPARISON")
        print("This function demonstrates understanding of different CV approaches")
        
        # This would typically compare Haar cascades vs HOG vs DNN methods
        # Implementation would depend on specific requirements
        
        methods = {
            'Haar Cascades': 'Fast, good for frontal faces, less accurate for variations',
            'HOG + SVM': 'Better for object detection, slower than Haar',
            'Deep Learning': 'Most accurate, requires more computational power'
        }
        
        for method, description in methods.items():
            print(f"• {method}: {description}")