import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Initialize MTCNN detector
mtcnn = MTCNN()

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Release the webcam
cap.release()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not read frame.")
    exit()

# Convert the image from BGR to RGB (OpenCV uses BGR)
image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Detect faces and landmarks
result = mtcnn.detect_faces(image_rgb, threshold_onet=0.85)

# Plot the results
plt.imshow(image_rgb)
plt.axis('off')  # Turn off axis numbers and ticks
plt.title("Detected Faces")
for face in result:
    x, y, width, height = face['box']
    plt.gca().add_patch(plt.Rectangle((x, y), width, height, color='green', fill=False, linewidth=2))
plt.show()
