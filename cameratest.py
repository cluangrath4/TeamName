import cv2
from ultralytics import YOLO
import time

class YOLOLiveDetection:
    def __init__(self, model_path='yolov8n.pt', camera_id=0):
        """Initialize YOLO model and camera."""
        print("Initializing YOLO model...")
        self.model = YOLO(model_path)
        self.camera_id = camera_id
        self.capture = None
        
    def initialize_camera(self):
        """Initialize the camera capture with proper macOS settings."""
        print("Initializing camera...")
        
        # Explicitly use AVFoundation backend for macOS
        self.capture = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
        
        if not self.capture.isOpened():
            raise RuntimeError("Error: Could not open camera.")
        
        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Wait for camera to initialize
        print("Waiting for camera to warm up...")
        time.sleep(2)  # Give camera time to warm up
        
        # Test camera read
        for _ in range(5):  # Try reading a few frames to ensure stable connection
            ret, frame = self.capture.read()
            if ret and frame is not None:
                print("Camera initialized successfully!")
                return
            time.sleep(0.1)
        
        raise RuntimeError("Failed to read initial frames from camera")
        
    def run_detection(self):
        """Run the live detection loop."""
        if self.capture is None:
            self.initialize_camera()
            
        print("Starting detection loop...")
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                # Read frame with retry mechanism
                for _ in range(3):  # Try up to 3 times to read a frame
                    ret, frame = self.capture.read()
                    if ret and frame is not None:
                        break
                    time.sleep(0.1)  # Short delay between retries
                
                if not ret or frame is None:
                    print("Error: Failed to read frame after retries.")
                    continue
                
                # Run YOLO inference
                results = self.model(frame, conf=0.25)
                
                # Get detections
                detections = results[0]
                
                # Draw the detections
                annotated_frame = results[0].plot()
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    current_time = time.time()
                    fps = 30 / (current_time - fps_start_time)
                    fps_start_time = current_time
                    
                    # Draw FPS
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Print detection info
                    if len(detections.boxes) > 0:
                        print(f"Detected {len(detections.boxes)} objects")
                
                # Display the frame
                cv2.imshow('YOLOv8 Live Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    # Save frame
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f'detection_{timestamp}.jpg'
                    cv2.imwrite(filename, annotated_frame)
                    print(f"Saved frame as {filename}")
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources and close windows."""
        print("Cleaning up...")
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Create and run the detector
        detector = YOLOLiveDetection(model_path='yolov8n.pt', camera_id=0)
        detector.run_detection()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()