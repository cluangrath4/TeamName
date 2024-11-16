import cv2
from google.cloud import vision
import time
import numpy as np
import io
from PIL import Image
import os

class GoogleVisionObjectFinder:
    def __init__(self, camera_id=0):
        """Initialize Google Vision client and camera."""
        print("Initializing Google Vision API...")
        self.client = vision.ImageAnnotatorClient()
        self.camera_id = camera_id
        self.capture = None
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.skip_frames = 2
        self.frame_counter = 0
        self.target_object = None
        
        # Cache of common household objects for quick lookup
        self.common_objects = {
            'keys': ['key', 'keys', 'car key', 'house key'],
            'remote': ['remote', 'remote control', 'tv remote', 'controller'],
            'phone': ['mobile phone', 'cell phone', 'smartphone', 'iphone', 'android phone'],
            'wallet': ['wallet', 'purse', 'billfold'],
            'glasses': ['glasses', 'eyeglasses', 'sunglasses', 'spectacles'],
            'watch': ['watch', 'wristwatch', 'smartwatch'],
            'charger': ['charger', 'power adapter', 'cable'],
            'headphones': ['headphones', 'earbuds', 'airpods'],
            'pen': ['pen', 'pencil', 'marker'],
            'notebook': ['notebook', 'notepad', 'journal']
        }
        
    def initialize_camera(self):
        """Initialize camera with stable settings."""
        print("Initializing camera...")
        
        if self.capture is not None:
            self.capture.release()
        
        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            raise RuntimeError("Error: Could not open camera.")
        
        # Set higher resolution for better object detection
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify settings
        actual_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized at {actual_width}x{actual_height} @ {actual_fps}fps")
        
        return True

    def detect_objects(self, image):
        """Detect objects in image using Google Vision API."""
        # Convert the image to bytes
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            return []
            
        content = encoded_image.tobytes()
        image = vision.Image(content=content)
        
        # Perform object detection
        objects = self.client.object_localization(image=image).localized_object_annotations
        
        return objects

    def is_target_object(self, detected_name):
        """Check if detected object matches target, including variations."""
        detected_name = detected_name.lower()
        
        # Check direct match
        if detected_name == self.target_object.lower():
            return True
            
        # Check variations from common objects dictionary
        for key, variations in self.common_objects.items():
            if self.target_object.lower() in variations:
                if detected_name in variations:
                    return True
                
        return False

    def draw_target_alert(self, frame, object_info):
        """Draw attention-grabbing alert when target object is found."""
        # Get bounding box vertices
        vertices = [(vertex.x * frame.shape[1], vertex.y * frame.shape[0]) 
                   for vertex in object_info.bounding_poly.normalized_vertices]
        
        # Convert to integers
        vertices = [(int(x), int(y)) for x, y in vertices]
        
        # Draw polygon
        for i in range(len(vertices)):
            cv2.line(frame, vertices[i], vertices[(i+1)%len(vertices)], (0, 0, 255), 3)
        
        # Add pulsing effect
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(vertices)], (0, 0, 255))
        pulse = (np.sin(time.time() * 8) + 1) / 2
        cv2.addWeighted(overlay, 0.3 * pulse, frame, 1 - 0.3 * pulse, 0, frame)
        
        # Draw confidence
        confidence = f"{object_info.score * 100:.1f}%"
        cv2.putText(frame, f"{object_info.name}: {confidence}", 
                   (vertices[0][0], vertices[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw "TARGET FOUND" text
        cv2.putText(frame, "TARGET FOUND!", 
                   (frame.shape[1]//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    def run_detection(self):
        """Run object detection loop."""
        try:
            if self.capture is None:
                self.initialize_camera()
            
            # Print available common objects
            print("\nCommon household objects you can search for:")
            for obj, variations in self.common_objects.items():
                print(f"- {obj} ({', '.join(variations)})")
            
            # Get target object from user
            self.target_object = input("\nWhat object would you like to find? ").lower().strip()
            
            print(f"\nLooking for {self.target_object}...")
            print("Press 'q' to quit, 's' to save a photo")
            
            last_frame_time = time.time()
            last_api_call_time = 0
            api_call_interval = 1.0  # Minimum seconds between API calls
            
            while True:
                try:
                    # Frame rate control
                    if time.time() - last_frame_time < 0.033:
                        time.sleep(0.01)
                        continue
                    
                    ret, frame = self.capture.read()
                    if not ret or frame is None:
                        print("Failed to capture frame.")
                        continue
                    
                    last_frame_time = time.time()
                    self.frame_counter += 1
                    
                    # Only call API periodically to avoid rate limits
                    current_time = time.time()
                    if current_time - last_api_call_time >= api_call_interval:
                        # Detect objects using Google Vision API
                        objects = self.detect_objects(frame)
                        last_api_call_time = current_time
                        
                        # Process detections
                        target_found = False
                        for obj in objects:
                            if self.is_target_object(obj.name):
                                target_found = True
                                self.draw_target_alert(frame, obj)
                        
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
        finder = GoogleVisionObjectFinder()
        finder.run_detection()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()