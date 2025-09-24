#!/usr/bin/env python3
"""
Test YOLO integration for pedestrian and vehicle detection.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys

def test_yolo_basic():
    """Test basic YOLO functionality."""
    print("🤖 Testing YOLO Integration...")
    
    try:
        # Initialize YOLO model
        print("📥 Loading YOLOv8n model...")
        model = YOLO("yolov8n.pt")  # This will download if not present
        print("✅ YOLO model loaded successfully!")
        
        # Test with a simple image (create a test image)
        print("🖼️ Creating test image...")
        test_image = create_test_image()
        
        # Run detection
        print("🔍 Running object detection...")
        results = model(test_image, conf=0.5)
        
        # Process results
        print("📊 Detection Results:")
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    print(f"  Detection {i+1}: {class_name} (confidence: {confidence:.2f})")
                    print(f"    Bounding box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        
        print("✅ YOLO test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
        return False

def create_test_image():
    """Create a simple test image with shapes."""
    # Create a white image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw some shapes that might be detected
    cv2.rectangle(img, (100, 100), (200, 300), (0, 0, 0), -1)  # Rectangle (car-like)
    cv2.circle(img, (400, 200), 50, (0, 0, 0), -1)  # Circle (person-like)
    cv2.rectangle(img, (300, 150), (350, 250), (128, 128, 128), -1)  # Another shape
    
    return img

def test_yolo_with_webcam():
    """Test YOLO with webcam (if available)."""
    print("📹 Testing YOLO with webcam...")
    
    try:
        # Initialize YOLO
        model = YOLO("yolov8n.pt")
        
        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Webcam not available")
            return False
        
        print("✅ Webcam opened. Press 'q' to quit, 's' to save image")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection every 10 frames
            if frame_count % 10 == 0:
                results = model(frame, conf=0.5)
                
                # Draw detections
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[class_id]
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('YOLO Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('test_detection.jpg', frame)
                print("💾 Image saved as 'test_detection.jpg'")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam test completed")
        return True
        
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")
        return False

def test_our_detector():
    """Test our custom PedestrianDetector class."""
    print("🔧 Testing our PedestrianDetector class...")
    
    try:
        # Import our detector
        sys.path.append('src/cv')
        from pedestrian_detector import PedestrianDetector
        
        # Initialize detector
        detector = PedestrianDetector()
        print("✅ PedestrianDetector initialized")
        
        # Create test image
        test_image = create_test_image()
        
        # Run detection
        detections = detector.detect_objects(test_image)
        
        print("📊 Our Detector Results:")
        print(f"  Pedestrians detected: {len(detections['pedestrians'])}")
        print(f"  Vehicles detected: {len(detections['vehicles'])}")
        print(f"  Total detections: {len(detections['all_detections'])}")
        
        # Test privacy protection
        protected_image = detector.apply_privacy_protection(test_image, detections)
        print("✅ Privacy protection applied")
        
        print("✅ Our detector test completed")
        return True
        
    except Exception as e:
        print(f"❌ Our detector test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚦 YOLO Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Basic YOLO
    success1 = test_yolo_basic()
    
    # Test 2: Our detector class
    success2 = test_our_detector()
    
    # Test 3: Webcam (optional)
    print("\n📹 Webcam test (optional - press 'q' to skip)")
    try:
        success3 = test_yolo_with_webcam()
    except KeyboardInterrupt:
        print("⏭️ Webcam test skipped")
        success3 = True
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  Basic YOLO: {'✅' if success1 else '❌'}")
    print(f"  Our Detector: {'✅' if success2 else '❌'}")
    print(f"  Webcam Test: {'✅' if success3 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 YOLO integration is working! Ready for SUMO integration.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")