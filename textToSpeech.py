import cv2
import easyocr
import pyttsx3
import threading
from queue import Queue

def tts_worker(queue, engine):
    """ Thread worker for handling TTS requests sequentially. """
    while True:
        text = queue.get()
        if text is None:  # Exit signal
            break
        engine.say(text)
        engine.runAndWait()
        queue.task_done()

def main():
    # Initialize OCR reader and TTS engine
    reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speech rate

    # Create a thread-safe queue for TTS requests
    tts_queue = Queue()
    tts_thread = threading.Thread(target=tts_worker, args=(tts_queue, engine), daemon=True)
    tts_thread.start()

    # Start capturing video
    cap = cv2.VideoCapture(0)  # 0 for default camera

    frame_count = 0
    n_frames = 5  # Process every 5th frame
    scale_percent = 50  # percent of original size to scale

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % n_frames != 0:
                continue  # Skip frame

            # Resize frame to reduce resolution and improve processing speed
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            # Convert frame to grayscale to improve OCR accuracy
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

            # Extract text from frame
            results = reader.readtext(gray)
            detected_texts = []

            # Process results
            for (bbox, text, prob) in results:
                if prob >= 0.5:  # Filter out low confidence results
                    detected_texts.append(text)

            if detected_texts:
                combined_text = ' '.join(detected_texts)
                print(f"Detected text: {combined_text}")
                tts_queue.put(combined_text)  # Add text to the TTS queue

            # Display the resized frame (optional)
            cv2.imshow('Video', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts_queue.put(None)  # Send exit signal to the TTS thread
        tts_thread.join()  # Wait for the TTS thread to finish

if __name__ == "__main__":
    main()
