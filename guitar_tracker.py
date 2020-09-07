import numpy as np
import cv2
from pygame import mixer


cap = cv2.VideoCapture(0)

WIDTH, HEIGHT = (640, 480)

DELIMITER_LEFT = int(WIDTH / 3)
DELIMITER_RIGHT = int(WIDTH / 1.5)

GREEN_PIXEL = [0, 255, 0]
RED_RANGE = (np.array([0, 120, 135]), np.array([10, 255, 255]))
# ORANGE_RANGE = (np.array([8, 40, 140]), np.array([23, 220, 255]))

# Initialize PyGame Mixer to play WAV files
mixer.init()
F5 = mixer.Sound("F.wav")
G5 = mixer.Sound("G.wav")
A5 = mixer.Sound("Am.wav")

CHANNEL = mixer.Channel(0)


def play_chord(width):
    # First region
    if width <= DELIMITER_LEFT:
        if CHANNEL.get_busy():
            CHANNEL.queue(F5)
        else:
            CHANNEL.play(F5)

    # Second region
    elif width <= DELIMITER_RIGHT:
        if CHANNEL.get_busy():
            CHANNEL.queue(G5)
        else:
            CHANNEL.play(G5)

    # Third region
    else:
        if CHANNEL.get_busy():
            CHANNEL.queue(A5)
        else:
            CHANNEL.play(A5)


def main():
    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        # Capture frame-by-frame
        _, frame = cap.read()

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get mask for colors
        lower_range, upper_range = RED_RANGE
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # Apply filters
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,
                                np.ones((3, 3), np.uint8))

        points = cv2.findNonZero(mask)

        if points is not None:
            avg = np.mean(points, axis=0)[0]
            width, _ = avg
            play_chord(width)

        # Paint the delimiters on the video
        for i in range(HEIGHT):
            frame[i][DELIMITER_LEFT] = GREEN_PIXEL
            frame[i][DELIMITER_RIGHT] = GREEN_PIXEL

        # Display the resulting frame
        cv2.imshow('Guitar Tracker', frame)


if __name__ == "__main__":
    main()

    # Stop sound
    CHANNEL.stop()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
