#ADDITIONAL TEST SCRIPT-FACE AND SHAPE DETECTION DEMO
import cv2
import numpy as np
import os
from image_recognition_project import ImageRecognitionSystem

def create_test_face_image():
    """Create a simple synthetic face-like image for testing detection"""
    # Create a white background
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw a simple face-like structure
    # Face outline (oval)
    cv2.ellipse(img, (200, 200), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    
    # Eyes
    cv2.circle(img, (175, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (225, 180), 15, (50, 50, 50), -1)
    
    # Nose
    cv2.circle(img, (200, 210), 8, (180, 160, 140), -1)
    
    # Mouth
    cv2.ellipse(img, (200, 240), (20, 10), 0, 0, 180, (100, 50, 50), 3)
    
    return img

def test_your_downloaded_images():
    """Test specifically with your downloaded images"""
    print("\n🎯 TESTING YOUR DOWNLOADED IMAGES")
    print("="*50)
    
    recognizer = ImageRecognitionSystem()
    
    # downloaded images
    your_images = [
        'test_images/face1.jpg.jpg',      #  face1 image
        'test_images/face2.jpg.jpg',      #  face2 image
        'test_images/group_face.jpg.jpg', #  group face image
        'test_images/shapes1.jpg.jpg',    # shapes1 image
        'test_images/shapes2.jpg.jpg'     # shapes2 image
    ]
    
    # Output folder created
    os.makedirs("output", exist_ok=True)
    
    success_count = 0
    
    for img_path in your_images:
        if os.path.exists(img_path):
            print(f"\n🔍 Processing: {os.path.basename(img_path)}")
            
            try:
                # Face detection करें
                if 'face' in img_path.lower():
                    print("   → Running FACE DETECTION...")
                    result = recognizer.detect_faces_and_eyes(img_path)
                    if result is not None:
                        output_name = f"detected_faces_{os.path.basename(img_path)}"
                        cv2.imwrite(f"output/{output_name}", result)
                        print(f"   ✅ Faces detected! Saved: output/{output_name}")
                        success_count += 1
                    else:
                        print("   ⚠️  No faces detected in this image")
                
                # Shape detection करें
                if 'shape' in img_path.lower():
                    print("   → Running SHAPE DETECTION...")
                    result = recognizer.detect_shapes(img_path)
                    if result is not None:
                        output_name = f"detected_shapes_{os.path.basename(img_path)}"
                        cv2.imwrite(f"output/{output_name}", result)
                        print(f"   ✅ Shapes detected! Saved: output/{output_name}")
                        success_count += 1
                    else:
                        print("   ⚠️  No shapes detected in this image")
                
                # Group images detection 
                if 'group' in img_path.lower():
                    print("   → Running BOTH face and shape detection...")
                    
                    # Face detection
                    face_result = recognizer.detect_faces_and_eyes(img_path)
                    if face_result is not None:
                        face_output = f"faces_in_{os.path.basename(img_path)}"
                        cv2.imwrite(f"output/{face_output}", face_result)
                        print(f"   ✅ Faces detected! Saved: output/{face_output}")
                    
                    # Shape detection
                    shape_result = recognizer.detect_shapes(img_path)
                    if shape_result is not None:
                        shape_output = f"shapes_in_{os.path.basename(img_path)}"
                        cv2.imwrite(f"output/{shape_output}", shape_result)
                        print(f"   ✅ Shapes detected! Saved: output/{shape_output}")
                    
                    if face_result is not None or shape_result is not None:
                        success_count += 1
                        
            except Exception as e:
                print(f"   ❌ Error processing {img_path}: {str(e)}")
        else:
            print(f"❌ File not found: {img_path}")
    
    return success_count

def test_auto_detect_all():
    """Automatically detect and process all images in test_images folder"""
    print("\n🤖 AUTO-DETECTION: Processing all images...")
    
    recognizer = ImageRecognitionSystem()
    test_folder = "test_images"
    
    if not os.path.exists(test_folder):
        print(f"❌ Folder '{test_folder}' not found!")
        return 0
    
    # image files find 
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(test_folder):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(os.path.join(test_folder, file))
    
    if not image_files:
        print(f"❌ No images found in '{test_folder}'!")
        return 0
    
    print(f"✅ Found {len(image_files)} images:")
    for img in image_files:
        print(f"   • {os.path.basename(img)}")
    
    # Output folder created
    os.makedirs("output", exist_ok=True)
    
    success_count = 0
    for img_path in image_files:
        print(f"\n🔍 Auto-processing: {os.path.basename(img_path)}")
        
        try:
            # Face detection for all
            face_result = recognizer.detect_faces_and_eyes(img_path)
            if face_result is not None:
                face_output = f"auto_faces_{os.path.basename(img_path)}"
                cv2.imwrite(f"output/{face_output}", face_result)
                print(f"   ✅ Face detection saved: output/{face_output}")
            
            # Shape detection for all
            shape_result = recognizer.detect_shapes(img_path)
            if shape_result is not None:
                shape_output = f"auto_shapes_{os.path.basename(img_path)}"
                cv2.imwrite(f"output/{shape_output}", shape_result)
                print(f"   ✅ Shape detection saved: output/{shape_output}")
            
            if face_result is not None or shape_result is not None:
                success_count += 1
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    return success_count

def run_complete_test():
    """Complete comprehensive testing"""
    print("🧪 COMPREHENSIVE IMAGE RECOGNITION TEST")
    print("🎯 Testing Your Downloaded Images")
    print("="*60)
    
    # Initialize system
    recognizer = ImageRecognitionSystem()
    
    print("\n📝 Step 1: Creating synthetic test image...")
    synthetic_face = create_test_face_image()
    cv2.imwrite('synthetic_test_face.jpg', synthetic_face)
    
    result = recognizer.detect_faces_and_eyes('synthetic_test_face.jpg')
    if result is not None:
        cv2.imwrite('output/detected_synthetic_face.jpg', result)
        print("✅ Synthetic face test completed")
    
    print("\n📝 Step 2: Testing your specific downloaded images...")
    specific_success = test_your_downloaded_images()
    
    print("\n📝 Step 3: Auto-detection on all images...")
    auto_success = test_auto_detect_all()
    
    # Generate final report
    print("\n📊 FINAL TEST RESULTS")
    print("="*40)
    print(f"✅ Specific image tests: {specific_success} successful")
    print(f"✅ Auto-detection tests: {auto_success} successful")
    
    if specific_success > 0 or auto_success > 0:
        print("\n🎉 SUCCESS! Your images were processed!")
        print("\n📁 Check these locations for results:")
        print("   • output/detected_faces_*.jpg - Face detection results")
        print("   • output/detected_shapes_*.jpg - Shape detection results")
        print("   • output/auto_*.jpg - Auto-detection results")
        
        print("\n🚀 Next steps:")
        print("1. Open the 'output' folder to see all results")
        print("2. Compare original vs detected images")
        print("3. Run main project: python image_recognition_project.py")
        print("4. Use webcam mode for real-time detection!")
    else:
        print("\n⚠️  No detections found. This could mean:")
        print("   • Images might be too small or unclear")
        print("   • Try using clearer face images")
        print("   • Check image file formats")
    
    # Show available files
    if os.path.exists("output"):
        output_files = os.listdir("output")
        if output_files:
            print(f"\n📋 Generated {len(output_files)} result files:")
            for file in sorted(output_files):
                print(f"   • {file}")

if __name__ == "__main__":
    run_complete_test()