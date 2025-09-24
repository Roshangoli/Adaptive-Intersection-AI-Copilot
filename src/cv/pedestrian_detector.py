#!/usr/bin/env python3
"""
Pedestrian and Vehicle Detection using YOLO for Adaptive Intersection AI Copilot.
This module provides real-time counting of pedestrians and vehicles at intersections.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PedestrianDetector:
    """YOLO-based pedestrian and vehicle detector with privacy protection."""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Class IDs for pedestrians and vehicles
        self.pedestrian_classes = [0]  # person
        self.vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
        
        # Privacy protection
        self.blur_faces = True
        self.blur_plates = True
        
    def detect_objects(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect pedestrians and vehicles in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold)
            
            detections = {
                'pedestrians': [],
                'vehicles': [],
                'all_detections': []
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.model.names[class_id]
                        }
                        
                        detections['all_detections'].append(detection)
                        
                        # Categorize detections
                        if class_id in self.pedestrian_classes:
                            detections['pedestrians'].append(detection)
                        elif class_id in self.vehicle_classes:
                            detections['vehicles'].append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {'pedestrians': [], 'vehicles': [], 'all_detections': []}
    
    def count_in_zones(self, detections: Dict[str, List], zones: Dict[str, List]) -> Dict[str, int]:
        """
        Count objects in specific zones.
        
        Args:
            detections: Detection results
            zones: Dictionary of zone definitions
            
        Returns:
            Count of objects per zone
        """
        counts = {}
        
        for zone_name, zone_polygon in zones.items():
            counts[zone_name] = 0
            
            # Check pedestrians
            for detection in detections['pedestrians']:
                if self._point_in_polygon(detection['bbox'], zone_polygon):
                    counts[zone_name] += 1
            
            # Check vehicles
            for detection in detections['vehicles']:
                if self._point_in_polygon(detection['bbox'], zone_polygon):
                    counts[zone_name] += 1
        
        return counts
    
    def _point_in_polygon(self, bbox: List[int], polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if bounding box center is inside polygon.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            polygon: List of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Simple point-in-polygon test
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if center_y > min(p1y, p2y):
                if center_y <= max(p1y, p2y):
                    if center_x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (center_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or center_x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def apply_privacy_protection(self, frame: np.ndarray, detections: Dict[str, List]) -> np.ndarray:
        """
        Apply privacy protection by blurring faces and license plates.
        
        Args:
            frame: Input frame
            detections: Detection results
            
        Returns:
            Frame with privacy protection applied
        """
        protected_frame = frame.copy()
        
        if self.blur_faces:
            # Blur pedestrian faces (simple approach - blur entire pedestrian)
            for detection in detections['pedestrians']:
                x1, y1, x2, y2 = detection['bbox']
                face_region = protected_frame[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(face_region, (15, 15), 0)
                protected_frame[y1:y2, x1:x2] = blurred
        
        if self.blur_plates:
            # Blur vehicle license plates (simple approach - blur entire vehicle)
            for detection in detections['vehicles']:
                x1, y1, x2, y2 = detection['bbox']
                plate_region = protected_frame[y1:y2, x1:x2]
                blurred = cv2.GaussianBlur(plate_region, (15, 15), 0)
                protected_frame[y1:y2, x1:x2] = blurred
        
        return protected_frame
    
    def get_waiting_counts(self, detections: Dict[str, List], 
                          waiting_zones: Dict[str, List]) -> Dict[str, int]:
        """
        Count pedestrians and vehicles waiting at intersection.
        
        Args:
            detections: Detection results
            waiting_zones: Dictionary of waiting zone definitions
            
        Returns:
            Count of waiting objects per zone
        """
        return self.count_in_zones(detections, waiting_zones)
    
    def get_crossing_counts(self, detections: Dict[str, List], 
                           crossing_zones: Dict[str, List]) -> Dict[str, int]:
        """
        Count pedestrians and vehicles crossing intersection.
        
        Args:
            detections: Detection results
            crossing_zones: Dictionary of crossing zone definitions
            
        Returns:
            Count of crossing objects per zone
        """
        return self.count_in_zones(detections, crossing_zones)

class IntersectionAnalyzer:
    """Analyzer for intersection traffic patterns."""
    
    def __init__(self, detector: PedestrianDetector):
        """
        Initialize analyzer.
        
        Args:
            detector: PedestrianDetector instance
        """
        self.detector = detector
        self.history = []
        self.max_history = 100  # Keep last 100 frames
        
    def analyze_frame(self, frame: np.ndarray, 
                     waiting_zones: Dict[str, List],
                     crossing_zones: Dict[str, List]) -> Dict:
        """
        Analyze a single frame for traffic patterns.
        
        Args:
            frame: Input frame
            waiting_zones: Waiting zone definitions
            crossing_zones: Crossing zone definitions
            
        Returns:
            Analysis results
        """
        # Detect objects
        detections = self.detector.detect_objects(frame)
        
        # Count in zones
        waiting_counts = self.detector.get_waiting_counts(detections, waiting_zones)
        crossing_counts = self.detector.get_crossing_counts(detections, crossing_zones)
        
        # Apply privacy protection
        protected_frame = self.detector.apply_privacy_protection(frame, detections)
        
        # Store in history
        analysis = {
            'timestamp': len(self.history),
            'waiting_counts': waiting_counts,
            'crossing_counts': crossing_counts,
            'total_pedestrians': len(detections['pedestrians']),
            'total_vehicles': len(detections['vehicles']),
            'protected_frame': protected_frame
        }
        
        self.history.append(analysis)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return analysis
    
    def get_traffic_summary(self) -> Dict:
        """
        Get summary of recent traffic patterns.
        
        Returns:
            Traffic summary
        """
        if not self.history:
            return {}
        
        recent_data = self.history[-10:]  # Last 10 frames
        
        summary = {
            'avg_pedestrians': np.mean([d['total_pedestrians'] for d in recent_data]),
            'avg_vehicles': np.mean([d['total_vehicles'] for d in recent_data]),
            'peak_pedestrians': max([d['total_pedestrians'] for d in recent_data]),
            'peak_vehicles': max([d['total_vehicles'] for d in recent_data]),
            'total_frames': len(self.history)
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = PedestrianDetector()
    
    # Define zones (example)
    waiting_zones = {
        'north_waiting': [(100, 50), (200, 50), (200, 100), (100, 100)],
        'south_waiting': [(100, 150), (200, 150), (200, 200), (100, 200)],
        'east_waiting': [(50, 100), (100, 100), (100, 150), (50, 150)],
        'west_waiting': [(150, 100), (200, 100), (200, 150), (150, 150)]
    }
    
    crossing_zones = {
        'north_south': [(100, 100), (200, 100), (200, 150), (100, 150)],
        'east_west': [(100, 100), (150, 100), (150, 200), (100, 200)]
    }
    
    # Initialize analyzer
    analyzer = IntersectionAnalyzer(detector)
    
    print("PedestrianDetector initialized successfully!")
    print("Ready for real-time traffic analysis.")