import cv2
from ultralytics import YOLO
import threading
import requests
import os

# Load the pretrained YOLOv8 model
# Change this to your weights file
model_path = 'C:\\Users\\pramo\\Desktop\\Honor Project\\Best1.pt'

# Telegram Bot details
bot_token ='7797792955:AAHT94rhTBbi2A_uGzBWH8KmBgACoQfzsAE' # Replace with your actual bot token
chat_ids = ['1220992657']   #['859923743', '1389002274','2135485533']   Replace with your chat ID and your friend's chat ID
alert_threshold = 0.4  # Confidence threshold to trigger an alert
alert_classes = ['Knife', 'Rifle','Handgun','Axe','Shotgun']  # Classes to send alerts for

# Function to send an alert message and image to Telegram
def send_telegram_alert(message, image_path):
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    for chat_id in chat_ids:
        with open(image_path, 'rb') as photo:
            data = {'chat_id': chat_id, 'caption': message}
            try:
                response = requests.post(url, data=data, files={'photo': photo})
                if response.status_code == 200:
                    print(f"Alert sent to chat_id {chat_id} successfully.")
                else:
                    print(f"Failed to send alert to chat_id {chat_id}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error sending Telegram alert to chat_id {chat_id}: {e}")
# Attempt to load the YOLOv8 model
try:
    model = YOLO(model_path)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# List of video sources (0 for first webcam, 1 for second webcam, or provide file paths)
video_sources = [0]  # Change to other sources if needed

# Function to capture, process, and send alerts for each video source
def process_video_source(source_id):
    cap = cv2.VideoCapture(source_id)

    if not cap.isOpened():
        print(f"Error: Could not access video source {source_id}.")
        return

    print(f"Starting video capture on source {source_id}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to grab frame from source {source_id}.")
            break

        try:
            # Perform object detection
            results = model(frame)

            # Annotate the frame with bounding boxes and labels
            annotated_frame = results[0].plot()

            # Check if any objects of interest are detected
            for detection in results[0].boxes:
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])

                # Get the class name of the detected object
                class_name = model.names[class_id]

                # Send alert if detected object is in the alert_classes and confidence is high enough
                if class_name in alert_classes and confidence > alert_threshold:
                    alert_message = f"Alert! {class_name.capitalize()} detected with confidence {confidence:.2f}"

                    # Save the annotated frame to a file
                    image_path = "annotated_frame.jpg"
                    cv2.imwrite(image_path, annotated_frame)

                    # Send alert with image to Telegram
                    send_telegram_alert(alert_message, image_path)

        except Exception as e:
            print(f"Error during inference or annotation on source {source_id}: {e}")
            break

        # Display the annotated frame with the source ID in the window name
        window_name = f'YOLOv8 Source {source_id}'
        cv2.imshow(window_name, annotated_frame)

        # Exit loop on 'q' key press for any window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Exiting video capture on source {source_id}...")
            break

    # Release the video capture object and close window for this source
    cap.release()
    cv2.destroyWindow(window_name)
    print(f"Released video capture and closed window for source {source_id}.")


# Create a thread for each video source
threads = []
for source in video_sources:
    t = threading.Thread(target=process_video_source, args=(source,))
    threads.append(t)
    t.start()

# Join the threads to wait for all of them to finish
for t in threads:
    t.join()

# Cleanup: Remove the saved image file if it exists
if os.path.exists("annotated_frame.jpg"):
    os.remove("annotated_frame.jpg")

print("All video sources released and all windows closed.")
