import numpy as np
import cv2

cap = cv2.VideoCapture(0)

WIDTH, HEIGHT = (640, 480)
GREEN_PIXEL = [0, 255, 0]
ORANGE_RANGE = (np.array([8, 40, 140]), np.array([23, 220, 255]))

while not (cv2.waitKey(1) & 0xFF == ord('q')):
    # Capture frame-by-frame
    ret, frame = cap.read()

    delimiter_left = int(WIDTH / 3)
    delimiter_right = int(WIDTH / 1.5)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get mask for colors
    lower_range, upper_range = ORANGE_RANGE
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Apply filters
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                            np.ones((3, 3), np.uint8))

    points = cv2.findNonZero(mask)

    if points is not None:
        avg = np.mean(points, axis=0)
        width = avg[0][0]

        # First region
        if width <= delimiter_left:
            print("first")

        # Second region
        elif width <= delimiter_right:
            print("second")

        # Third region
        else:
            print("third")

    # Paint the delimiters on the video
    for i in range(HEIGHT):
        frame[i][delimiter_left] = GREEN_PIXEL
        frame[i][delimiter_right] = GREEN_PIXEL

    # Display the resulting frame
    cv2.imshow('frame', frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
