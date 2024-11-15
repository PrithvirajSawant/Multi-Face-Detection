import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

mtcnn = MTCNN()

cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Here I'm Initializing a flag to control the loop
running = True

def on_key(event):
    global running
    if event.key == 'q':
        running = False

# Connecting the key event to the matplotlib figure
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)

while running:
    # Capture a single frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Heare I am detecting faces and landmarks
    result = mtcnn.detect_faces(image_rgb)

    ax.clear()

    # Here I'm drawing the frame
    ax.imshow(image_rgb)
    ax.axis('off')
    ax.set_title("Live Face Detection")

    # Drawing bounding boxes and landmarks on the image
    for face in result:
        x, y, width, height = face['box']
        
        # Drawing rectangle for face bounding box
        ax.add_patch(plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='green', facecolor='none'))
        
        # Draw facial landmarks
        keypoints = face['keypoints']
        for key, point in keypoints.items():
            ax.plot(point[0], point[1], 'bo', markersize=5)  # Plot landmarks with blue dots

    # Updating the matplotlib plot
    plt.draw()
    plt.pause(0.001)  # Pause to allow for plot update

cap.release()
plt.close(fig) 
cv2.destroyAllWindows()
