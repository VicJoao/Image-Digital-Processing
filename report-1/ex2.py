import cv2
import numpy as np

boundaries = {
    'black': [[0, 0, 0], [180, 255, 30]],  # Black
    'white': [[0, 0, 231], [180, 18, 255]],  # White
    'red': [[0, 50, 70], [9, 255, 255]],  # Red
    'green': [[36, 50, 70], [89, 255, 255]],  # Green
    'blue': [[90, 50, 70], [128, 255, 255]],  # Blue
    'yellow': [[25, 50, 70], [35, 255, 255]],  # Yellow
    'purple': [[129, 50, 70], [158, 255, 255]],  # Purple
    'orange': [[10, 50, 70], [24, 255, 255]],  # Orange
    'gray': [[0, 0, 40], [180, 18, 230]]  # Gray
}

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Prompt user to input color name
while True:
    color_name = input(
        'Enter the color name to track (black, white, red, green, blue, yellow, purple, orange, gray): ')
    if color_name in boundaries:
        break
    else:
        print(
            'Invalid color name. Please enter one of the following: black, white, red1, red2, green, blue, yellow, purple, orange, gray')

# Define range for blue color detection in HSV
lower_blue = np.array(boundaries[color_name][0])
upper_blue = np.array(boundaries[color_name][1])

# Initialize variables to store previous point and path image
prev_point = None
path_img = np.zeros((480, 640, 3), dtype=np.uint8)
counter = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    counter += 1

    if counter % 10 == 0:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours
        for contour in contours:
            # Get the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                # Draw the path on the path image
                if prev_point is not None:
                    cv2.line(path_img, prev_point, (cx, cy), (255, 255, 255), 2)
                prev_point = (cx, cy)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the path image
cv2.imwrite('path_image.png', path_img)

# Release the capture
cap.release()
cv2.destroyAllWindows()
