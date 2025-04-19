import cv2
import os
import time

# Define the base directory
base_dir = "dataset-dir"
cam_dir = os.path.join(base_dir, "cam0")

# Create the directory structure if it doesn't exist
os.makedirs(cam_dir, exist_ok=True)

# Initialize the camera (0 is usually the default webcam)
cap = cv2.VideoCapture(2)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
# Set camera parameters
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Set codec to MJPEG
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)  # Set brightness (default)
cap.set(cv2.CAP_PROP_CONTRAST, 100)  # Set contrast (default)

cap.set(cv2.CAP_PROP_SATURATION, -1)  # Set saturation (default)

cap.set(cv2.CAP_PROP_SHARPNESS, 100)  # Set sharpness (default)

cap.set(cv2.CAP_PROP_GAIN, -1)  # Set gain (default)
cap.set(cv2.CAP_PROP_AUTO_WB, -1)  # Enable auto white balance
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)  # Enable auto exposure
cap.set(cv2.CAP_PROP_EXPOSURE, 100)  # Set exposure
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
cap.set(cv2.CAP_PROP_FOCUS, -1)  # Set focus (default)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set image width to 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set image height to 1080
# Verify the settings
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {width}x{height}")
# Capture images every second and display them
try:
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Get the current timestamp in nanoseconds
        timestamp = int(time.time() * 1e9)  # Convert to nanoseconds

        # Define the filename using the timestamp
        filename = os.path.join(cam_dir, f"{timestamp}.png")

        # Save the captured frame as an image
        cv2.imwrite(filename, frame)
        print(f"Saved image: {filename}")

        # Display the frame in a window
        cv2.imshow("Camera Feed", frame)

        # Wait for 1 second before capturing the next image
        # Also check for user input to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Press 'q' to quit
            print("User requested to stop.")
            break

except KeyboardInterrupt:
    print("Image capture stopped by user.")

finally:
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and script ended.")
