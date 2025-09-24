#!/usr/bin/env python3
"""
Script to help you collect real traffic data for the competition.
"""

import os
import cv2
import time
from datetime import datetime
import pandas as pd

def record_intersection_video():
    """Record video of a real intersection."""
    print("📹 Recording Real Intersection Video")
    print("=" * 50)
    
    # Create data directory
    os.makedirs("data/real_videos", exist_ok=True)
    
    print("Instructions for recording:")
    print("1. Go to a busy intersection (campus, downtown, etc.)")
    print("2. Position yourself safely where you can see:")
    print("   - Pedestrians waiting to cross")
    print("   - Vehicles at the intersection")
    print("   - Traffic lights")
    print("3. Record for 5-10 minutes")
    print("4. Save the video in data/real_videos/")
    
    # Check if camera is available
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("\n📱 Camera detected! You can record directly.")
        
        # Get video properties
        fps = 30
        width = 640
        height = 480
        
        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/real_videos/intersection_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        print(f"🎥 Recording to: {filename}")
        print("Press 'q' to stop recording, 's' to save current frame")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (width, height))
            
            # Add timestamp
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add recording indicator
            cv2.putText(frame, "RECORDING", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add frame count
            cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Show frame
            cv2.imshow('Recording Intersection Video', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                frame_filename = f"data/real_videos/frame_{frame_count}_{timestamp}.jpg"
                cv2.imwrite(frame_filename, frame)
                print(f"💾 Saved frame: {frame_filename}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Calculate recording stats
        duration = time.time() - start_time
        print(f"\n📊 Recording Summary:")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Frames: {frame_count}")
        print(f"  FPS: {frame_count/duration:.1f}")
        print(f"  File: {filename}")
        
        return filename
    else:
        print("❌ Camera not available. Please record manually.")
        print("\n📱 Manual Recording Instructions:")
        print("1. Use your phone to record intersection video")
        print("2. Record for 5-10 minutes")
        print("3. Save as MP4 file")
        print("4. Place in data/real_videos/ folder")
        print("5. Name it: intersection_[date].mp4")
        return None

def collect_traffic_observations():
    """Collect manual traffic observations."""
    print("\n📊 Collecting Manual Traffic Observations")
    print("=" * 50)
    
    print("Instructions:")
    print("1. Go to a busy intersection")
    print("2. Observe for 15 minutes")
    print("3. Count pedestrians and vehicles")
    print("4. Note traffic light timing")
    print("5. Record wait times")
    
    # Create observation template
    observations = []
    
    print("\n📝 Traffic Observation Form:")
    print("Enter observations (press Enter to skip):")
    
    # Get basic info
    location = input("📍 Location (e.g., 'Campus Main St & University Ave'): ").strip()
    date = input("📅 Date (YYYY-MM-DD): ").strip()
    time_start = input("⏰ Start time (HH:MM): ").strip()
    weather = input("🌤️ Weather (sunny/cloudy/rainy): ").strip()
    
    print(f"\n📊 Observing traffic at {location}")
    print("Count vehicles and pedestrians for each 5-minute period:")
    
    # Collect observations for 3 periods
    for period in range(1, 4):
        print(f"\n--- Period {period} (5 minutes) ---")
        
        vehicles_ns = input(f"🚗 Vehicles North-South: ").strip()
        vehicles_ew = input(f"🚗 Vehicles East-West: ").strip()
        pedestrians_ns = input(f"🚶 Pedestrians North-South: ").strip()
        pedestrians_ew = input(f"🚶 Pedestrians East-West: ").strip()
        avg_wait_time = input(f"⏱️ Average wait time (seconds): ").strip()
        
        if vehicles_ns or vehicles_ew or pedestrians_ns or pedestrians_ew:
            observations.append({
                'location': location,
                'date': date,
                'time_start': time_start,
                'period': period,
                'weather': weather,
                'vehicles_ns': int(vehicles_ns) if vehicles_ns else 0,
                'vehicles_ew': int(vehicles_ew) if vehicles_ew else 0,
                'pedestrians_ns': int(pedestrians_ns) if pedestrians_ns else 0,
                'pedestrians_ew': int(pedestrians_ew) if pedestrians_ew else 0,
                'avg_wait_time': float(avg_wait_time) if avg_wait_time else 0
            })
    
    # Save observations
    if observations:
        df = pd.DataFrame(observations)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/real_traffic_observations_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n💾 Observations saved to: {filename}")
        
        # Print summary
        print(f"\n📊 Observation Summary:")
        print(f"  Location: {location}")
        print(f"  Total vehicles: {df['vehicles_ns'].sum() + df['vehicles_ew'].sum()}")
        print(f"  Total pedestrians: {df['pedestrians_ns'].sum() + df['pedestrians_ew'].sum()}")
        print(f"  Average wait time: {df['avg_wait_time'].mean():.1f}s")
        
        return filename
    else:
        print("❌ No observations collected")
        return None

def create_data_collection_guide():
    """Create a guide for data collection."""
    print("\n📋 Creating Data Collection Guide")
    print("=" * 50)
    
    guide_content = """# Real Data Collection Guide

## 🎯 What Data You Need for Competition

### 1. Video Data
- **Real intersection videos** (5-10 minutes each)
- **Pedestrians waiting and crossing**
- **Vehicles at intersection**
- **Traffic light changes**

### 2. Traffic Observations
- **Pedestrian counts** per direction
- **Vehicle counts** per direction
- **Wait times** for pedestrians
- **Traffic light timing**

### 3. Performance Data
- **Current intersection performance**
- **Wait time comparisons**
- **Safety incidents** (near-misses)

## 📍 Where to Collect Data

### Campus Locations (Recommended)
- Main campus intersections
- Student crossing areas
- Busy pedestrian zones
- Near dining halls/classrooms

### Public Locations
- Downtown intersections
- Shopping areas
- Near schools
- Bus stops

## 📱 How to Collect Data

### Video Recording
1. **Use phone camera** (1080p or higher)
2. **Record for 5-10 minutes**
3. **Include traffic lights** in frame
4. **Capture pedestrian crossings**
5. **Save as MP4** format

### Manual Observations
1. **Count for 5-minute periods**
2. **Record wait times**
3. **Note traffic light timing**
4. **Include weather conditions**
5. **Record time and location**

## 🔒 Privacy Considerations

### What to Avoid
- ❌ Don't record faces clearly
- ❌ Don't capture license plates
- ❌ Don't record personal conversations
- ❌ Don't invade privacy

### What's OK
- ✅ Public intersection views
- ✅ General traffic patterns
- ✅ Anonymized counts
- ✅ Traffic light timing

## 📊 Data Format

### Video Files
- Format: MP4
- Duration: 5-10 minutes
- Resolution: 1080p or higher
- Location: data/real_videos/

### Observation Files
- Format: CSV
- Columns: location, date, time, vehicles, pedestrians, wait_times
- Location: data/real_observations/

## 🎯 Competition Presentation

### What Judges Want to See
1. **Real problem** (long wait times)
2. **Your solution** (AI copilot)
3. **Real data** (not just simulation)
4. **Measurable impact** (improved wait times)
5. **Privacy protection** (blurred faces)

### Key Metrics to Highlight
- **Pedestrian wait time reduction**
- **Vehicle throughput maintenance**
- **Safety improvement**
- **Fairness across user groups**
- **Privacy protection**

## 🚀 Next Steps

1. **Record 2-3 intersection videos**
2. **Collect manual observations**
3. **Test YOLO on real videos**
4. **Compare with simulation results**
5. **Prepare competition presentation**
"""
    
    with open("data/DATA_COLLECTION_GUIDE.md", "w") as f:
        f.write(guide_content)
    
    print("✅ Data collection guide created: data/DATA_COLLECTION_GUIDE.md")

def main():
    """Main function for data collection."""
    print("🚦 Real Data Collection for Competition")
    print("=" * 60)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Create guide
    create_data_collection_guide()
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Record intersection video")
    print("2. Collect manual observations")
    print("3. Just show the guide")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        video_file = record_intersection_video()
        if video_file:
            print(f"\n🎉 Video recorded: {video_file}")
            print("Next: Test YOLO detection on this video!")
    
    elif choice == "2":
        obs_file = collect_traffic_observations()
        if obs_file:
            print(f"\n🎉 Observations collected: {obs_file}")
            print("Next: Use this data in your simulation!")
    
    elif choice == "3":
        print("\n📋 Guide created. Check data/DATA_COLLECTION_GUIDE.md")
    
    else:
        print("❌ Invalid choice")
    
    print("\n🎯 Summary:")
    print("You now have everything you need to collect real data!")
    print("This will make your competition presentation much stronger.")

if __name__ == "__main__":
    main()