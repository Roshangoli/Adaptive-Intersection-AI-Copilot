#!/usr/bin/env python3
"""
Test YOLO detection on real video data.
"""

import cv2
import os
import sys
from ultralytics import YOLO

def test_video_detection():
    """Test YOLO detection on video file."""
    print("🎥 Testing YOLO on Video Data")
    print("=" * 40)
    
    # Check if video exists
    video_path = "data/videos/traffic_sample.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return False
    
    print(f"📹 Loading video: {video_path}")
    
    # Initialize YOLO
    print("🤖 Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("✅ YOLO model ready")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"📊 Video info:")
    print(f"  FPS: {fps}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f} seconds")
    
    # Process video
    frame_num = 0
    detections_count = 0
    total_vehicles = 0
    total_pedestrians = 0
    
    print(f"\n🔍 Processing video (showing every 30th frame)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Process every 30th frame to speed up
        if frame_num % 30 == 0:
            # Run YOLO detection
            results = model(frame, conf=0.5)
            
            # Count detections
            frame_vehicles = 0
            frame_pedestrians = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        
                        if class_id == 0:  # person
                            frame_pedestrians += 1
                        elif class_id in [1, 2, 3, 5, 7]:  # vehicles
                            frame_vehicles += 1
            
            total_vehicles += frame_vehicles
            total_pedestrians += frame_pedestrians
            detections_count += 1
            
            print(f"  Frame {frame_num}: {frame_vehicles} vehicles, {frame_pedestrians} pedestrians")
            
            # Draw detections on frame
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        
                        # Draw bounding box
                        color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                  (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save sample frame
            if frame_num == 30:  # Save first processed frame
                cv2.imwrite("sample_detection.jpg", frame)
                print(f"💾 Saved sample detection: sample_detection.jpg")
    
    cap.release()
    
    # Print summary
    print(f"\n📊 Detection Summary:")
    print(f"  Frames processed: {detections_count}")
    print(f"  Total vehicles detected: {total_vehicles}")
    print(f"  Total pedestrians detected: {total_pedestrians}")
    print(f"  Average vehicles per frame: {total_vehicles/detections_count:.1f}")
    print(f"  Average pedestrians per frame: {total_pedestrians/detections_count:.1f}")
    
    print(f"\n✅ Video detection test completed!")
    return True

def test_our_detector_on_video():
    """Test our custom detector on video."""
    print("\n🔧 Testing Our PedestrianDetector on Video")
    print("=" * 40)
    
    try:
        # Import our detector
        sys.path.append('src/cv')
        from pedestrian_detector import PedestrianDetector
        
        # Initialize detector
        detector = PedestrianDetector()
        print("✅ PedestrianDetector initialized")
        
        # Test on video
        video_path = "data/videos/traffic_sample.mp4"
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("❌ Could not open video")
            return False
        
        frame_count = 0
        total_detections = 0
        
        print("🔍 Processing video with our detector...")
        
        while frame_count < 100:  # Process first 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame
            if frame_count % 10 == 0:
                # Run detection
                detections = detector.detect_objects(frame)
                
                # Count detections
                vehicles = len(detections['vehicles'])
                pedestrians = len(detections['pedestrians'])
                total_detections += vehicles + pedestrians
                
                print(f"  Frame {frame_count}: {vehicles} vehicles, {pedestrians} pedestrians")
                
                # Test privacy protection
                protected_frame = detector.apply_privacy_protection(frame, detections)
                
                # Save sample
                if frame_count == 10:
                    cv2.imwrite("sample_protected.jpg", protected_frame)
                    print(f"💾 Saved privacy-protected sample: sample_protected.jpg")
        
        cap.release()
        
        print(f"\n📊 Our Detector Summary:")
        print(f"  Frames processed: {frame_count//10}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average detections per frame: {total_detections/(frame_count//10):.1f}")
        
        print(f"\n✅ Our detector test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Our detector test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚦 Video Detection Test Suite")
    print("=" * 50)
    
    # Test 1: Basic YOLO on video
    success1 = test_video_detection()
    
    # Test 2: Our detector on video
    success2 = test_our_detector_on_video()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"  YOLO on video: {'✅' if success1 else '❌'}")
    print(f"  Our detector on video: {'✅' if success2 else '❌'}")
    
    if success1 and success2:
        print("\n🎉 Video detection is working! Ready for real data.")
        print("\nNext steps:")
        print("1. Record your own intersection videos")
        print("2. Test on campus intersections")
        print("3. Collect real traffic data")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")