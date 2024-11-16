import cv2
from ultralytics import YOLO
import time
import numpy as np

class YOLOObjectFinder:
    def __init__(self, model_path='yolov8n.pt', camera_id=0):
        print("Initializing YOLO model...")
        self.model = YOLO(model_path)
        self.camera_id = camera_id
        self.capture = None
        
        # Detection settings
        self.confidence_threshold = 0.4
        self.skip_frames = 2
        self.frame_counter = 0
        self.target_object = None
        
        # Get available class names
        self.class_names = self.model.names
        print("\nAvailable objects to detect:", ", ".join(self.class_names.values()))
        
    def initialize_camera(self):
        """Initialize camera with stable settings."""
        print("Initializing camera...")
        
        if self.capture is not None:
            self.capture.release()
        
        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            raise RuntimeError("Error: Could not open camera.")
        
        # Set lower resolution for stability
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify settings
        actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized at {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Test frame capture
        ret, frame = self.capture.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture test frame")
        
        return True

    def draw_target_alert(self, frame, box, confidence):
        """Draw attention-grabbing alert when target object is found."""
        x1, y1, x2, y2 = box
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        
        pulse = (np.sin(time.time() * 8) + 1) / 2  # Oscillates between 0 and 1
        alpha = 0.3 * pulse
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        text = f"{self.target_object}: {confidence:.2f}"
        cv2.putText(frame, text, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(frame, "TARGET FOUND!", 
                   (frame.shape[1]//2 - 100, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def run_detection(self):
        """Run object detection loop focusing on target object."""
        try:
            if self.capture is None:
                self.initialize_camera()
            
            # Get target object from user
            print("\nWhat object would you like to find?")
            print("Available objects:", ", ".join(self.class_names.values()))
            self.target_object = input("Enter object name: ").lower().strip()
            
            # Verify target object is in our model's classes
            target_found = False
            for class_id, name in self.class_names.items():
                if name.lower() == self.target_object:
                    target_found = True
                    self.target_class_id = class_id
                    break
            
            if not target_found:
                print(f"Sorry, '{self.target_object}' is not in the list of detectable objects.")
                print("Available objects:", ", ".join(self.class_names.values()))
                return
            
            print(f"\nLooking for {self.target_object}...")
            print("Press 'q' to quit, 's' to save a photo")
            
            last_frame_time = time.time()
            consecutive_failures = 0
            
            while True:
                try:
                    # Frame rate control
                    if time.time() - last_frame_time < 0.033:
                        time.sleep(0.01)
                        continue
                    
                    # Read frame
                    ret, frame = self.capture.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            print("Camera connection lost, attempting to reconnect...")
                            self.initialize_camera()
                            consecutive_failures = 0
                        continue
                    
                    consecutive_failures = 0
                    last_frame_time = time.time()
                    self.frame_counter += 1
                    
                    # Process only every nth frame
                    if self.frame_counter % self.skip_frames != 0:
                        continue
                    
                    # Run YOLO inference
                    results = self.model(frame, conf=self.confidence_threshold)
                    
                    # Check for target object
                    target_found = False
                    for detection in results[0].boxes:
                        class_id = int(detection.cls[0])
                        confidence = float(detection.conf[0])
                        
                        if self.class_names[class_id].lower() == self.target_object:
                            target_found = True
                            # Draw attention-grabbing alert
                            self.draw_target_alert(frame, detection.xyxy[0], confidence)
                    
                    # Draw status text
                    if not target_found:
                        cv2.putText(frame, f"Searching for {self.target_object}...", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow('Object Finder', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quitting...")
                        break
                    elif key == ord('s'):
                        filename = f'found_{self.target_object}_{time.strftime("%Y%m%d-%H%M%S")}.jpg'
                        cv2.imwrite(filename, frame)
                        print(f"Saved image as {filename}")
                    
                except Exception as e:
                    print(f"Error in detection loop: {e}")
                    time.sleep(0.1)
                    
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
        finder = YOLOObjectFinder(model_path='yolov8n.pt', camera_id=0)
        finder.run_detection()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()