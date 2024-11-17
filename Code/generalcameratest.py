import cv2
from ultralytics import YOLO
import time
import numpy as np
import platform
import threading
from queue import Queue

# Operating system specific sound implementations
if platform.system() == 'Windows':
    import winsound
    def beep():
        """Generate a beep sound using Windows native sound."""
        winsound.Beep(1000, 100)  # 1000Hz frequency, 100ms duration
else:
    try:
        from beepy import beep as make_beep
        def beep():
            """Generate a beep sound using beepy library."""
            make_beep(sound=1)  # Sound type 1 = 'ping'
    except ImportError:
        print("Installing beepy for sound...")
        import os
        os.system('pip install beepy')
        from beepy import beep as make_beep
        def beep():
            make_beep(sound=1)

class YOLOUltraFast:
    """
    A high-performance real-time object detection class using YOLOv8.
    Optimized for speed with minimal processing and optional sound alerts.
    """
    
    def __init__(self):
        """Initialize the YOLO detector with minimal settings for maximum speed."""
        print("Loading minimal YOLO model...")
        try:
            # Load the smallest YOLO model for maximum speed
            self.model = YOLO('yolov8n-cls.pt')
            self.model.fuse()  # Fuse model layers for faster inference
            print("Model loaded!")
        except Exception as e:
            print(f"Error: {e}")
            raise
        
        # Core settings
        self.capture = None  # Camera capture object
        self.frame_size = (240, 320)  # Minimal resolution for speed
        self.confidence_threshold = 0.25  # Minimum confidence for detections
        self.last_inference_time = 0  # Track last inference time
        self.inference_interval = 0.1  # Run inference every 100ms
        
        # Object detection settings
        self.target_object = None  # Object to search for
        self.last_beep_time = 0  # Track last beep time
        self.beep_cooldown = 1.0  # Minimum seconds between beeps
        
        # Initialize sound queue and thread
        self.sound_queue = Queue()
        self.sound_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self.sound_thread.start()
    
    def _sound_worker(self):
        """
        Background worker thread for handling sound alerts.
        Prevents sound playback from blocking video processing.
        """
        while True:
            _ = self.sound_queue.get()  # Wait for sound trigger
            current_time = time.time()
            if current_time - self.last_beep_time >= self.beep_cooldown:
                beep()
                self.last_beep_time = current_time
            self.sound_queue.task_done()
    
    def initialize_camera(self):
        """
        Initialize the camera with optimized settings for speed.
        Returns:
            bool: True if camera initialized successfully
        """
        if self.capture is not None:
            self.capture.release()
        
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError("Camera error")
        
        # Set minimal camera properties for maximum speed
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[1])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[0])
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer size
        
        return True

    def process_frame(self, frame, current_fps, class_name=None):
        """
        Process a single frame with minimal overhead.
        
        Args:
            frame: Current video frame
            current_fps: Current frames per second
            class_name: Name of detected object (if any)
        
        Returns:
            processed_frame: Frame with overlaid information
        """
        # Add FPS counter
        cv2.putText(frame, 
                   f"FPS:{current_fps:.1f}", 
                   (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (0, 255, 0), 
                   1)
        
        # Add detection results if available
        if class_name:
            # Check if detected object matches target
            if self.target_object and self.target_object.lower() in class_name.lower():
                color = (0, 0, 255)  # Red for target object
                self.sound_queue.put(1)  # Trigger beep
            else:
                color = (255, 255, 255)  # White for other objects
            
            # Add object name to frame
            cv2.putText(frame, 
                      class_name, 
                      (5, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      color, 
                      1)
        
        return frame

    def run_detection(self):
        """
        Main detection loop. Captures frames, runs inference, and displays results.
        Handles user input and maintains performance optimizations.
        """
        try:
            if self.capture is None:
                self.initialize_camera()
            
            # Get user preferences
            print("\nDo you want to:")
            print("1. Search for specific object (with beep alert)")
            print("2. Show all detected objects")
            mode = input("Choose mode (1 or 2): ").strip()
            
            if mode == "1":
                self.target_object = input("\nWhat object would you like to find? ").lower().strip()
                print(f"\nLooking for {self.target_object}...")
                print("Will beep when object is detected!")
            
            print("\nPress 'q' to quit, 'n' for new search")
            
            # Performance tracking
            frame_times = []
            last_prediction = None
            
            while True:
                # Capture frame
                ret, frame = self.capture.read()
                if not ret:
                    continue
                
                # Track FPS
                current_time = time.time()
                frame_times.append(current_time)
                frame_times = [t for t in frame_times if current_time - t <= 1.0]
                current_fps = len(frame_times)
                
                # Run inference at specified intervals
                if current_time - self.last_inference_time >= self.inference_interval:
                    try:
                        results = self.model(frame, verbose=False)
                        last_prediction = results[0].probs.top1
                        self.last_inference_time = current_time
                    except Exception as e:
                        print(f"Inference error: {e}")
                        continue
                
                # Get class name if available
                class_name = None
                if last_prediction is not None:
                    class_name = self.model.names[int(last_prediction)]
                
                # Process and display frame
                frame = self.process_frame(frame, current_fps, class_name)
                cv2.imshow('Ultra Fast', frame)
                
                # Handle user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    self.target_object = input("\nWhat object would you like to find? ").lower().strip()
                    print(f"\nLooking for {self.target_object}...")
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release all resources and close windows."""
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point of the application."""
    try:
        detector = YOLOUltraFast()
        detector.run_detection()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()