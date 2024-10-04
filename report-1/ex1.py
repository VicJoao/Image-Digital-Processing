import cv2
import numpy as np

boundaries = {
    'green': [[60, 50, 50], [80, 255, 255], [79, 255, 255]],  # Hue: 60-80, Saturation: 50-255, Value: 50-255
    'yellow': [[20, 100, 100], [30, 255, 255], [29, 255, 255]],  # Hue: 20-30, Saturation: 100-255, Value: 100-255
    'red': [[0, 100, 20], [10, 255, 255], [9, 255, 255]],  # Hue: 160-180, Saturation: 100-255, Value: 100-255
    'red2': [[160, 100, 20], [179, 255, 255], [178, 255, 255]],  # Hue: 160-180, Saturation: 100-255, Value: 100-255
    'blue': [[101, 50, 38], [112, 255, 255], [111, 255, 255]],  # Hue: 101-110, Saturation: 50-255, Value: 38-255
    'black': [[0, 0, 0], [179, 255, 100], [178, 255, 100]],  # Hue: 0-179, Saturation: 0-50, Value: 0-50
    'gray': [[0, 0, 100], [179, 60, 220], [179, 60, 220]],  # Hue: 0-179, Saturation: 0-50, Value: 100-200
}


def get_processed_hsv_img(original_img):
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

    # Define a mask for white pixels
    white_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([250, 170, 255]))

    # Iterate over each white pixel
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            if white_mask[i, j] == 255:  # Check if the pixel is white
                left_color = None
                right_color = None

                # Find the color on the left
                k = j - 1
                while k >= 0 and left_color is None:
                    if white_mask[i, k] != 255:  # Check if not white
                        for color in boundaries:
                            if np.all(hsv[i, k] >= boundaries[color][0]) and np.all(hsv[i, k] <= boundaries[color][1]):
                                left_color = color
                                break
                    k -= 1

                # Find the color on the right
                k = j + 1
                while k < hsv.shape[1] and right_color is None:
                    if white_mask[i, k] != 255:  # Check if not white
                        for color in boundaries:
                            if np.all(hsv[i, k] >= boundaries[color][0]) and np.all(hsv[i, k] <= boundaries[color][1]):
                                right_color = color
                                break
                    k += 1

                # Replace color if left and right colors match
                if left_color is not None and right_color is not None and left_color == right_color:
                    hsv[i, j] = boundaries[left_color][2]  # Replace with the right color

    # Convert the modified HSV image back to BGR color space
    modified_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return modified_img


def replace_colour(original_img):
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

    # Create a mask for each color range
    masks = {color: cv2.inRange(hsv, np.array(boundaries[color][0]), np.array(boundaries[color][1])) for color in
             boundaries}

    # Replace colors based on the masks
    for color in masks:
        if color == 'black':
            hsv[np.where(masks[color] != 0)] = [0, 0, 0]
        elif color == 'red':
            hsv[np.where(masks[color] != 0)] = boundaries['red2'][1]
        else:
            hsv[np.where(masks[color] != 0)] = boundaries[color][1]

    # Convert the modified HSV image back to BGR color space
    modified_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return modified_img


# Example usage:
original_image = cv2.imread('halteres.jpg')
processed_image = get_processed_hsv_img(original_image)
cv2.imshow('Processed Image', replace_colour(processed_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
