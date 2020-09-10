import numpy as np
import cv2
from pygame import mixer


WIDTH, HEIGHT = (640, 480)

DELIMITER_LEFT = int(WIDTH / 3)
DELIMITER_RIGHT = int(WIDTH / 1.5)

GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
RED_RANGE = (np.array([0, 120, 150]), np.array([10, 255, 255]))

# Initialize PyGame Mixer to play WAV files
mixer.init()
F5 = mixer.Sound("F.wav")
G5 = mixer.Sound("G.wav")
A5 = mixer.Sound("Am.wav")

REGION_POSITIONS = {
    F5: ((0, 0), (DELIMITER_LEFT, HEIGHT)),
    G5: ((DELIMITER_LEFT, 0), (DELIMITER_RIGHT, HEIGHT)),
    A5: ((DELIMITER_RIGHT, 0), (WIDTH, HEIGHT))
}


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


def paint_region(frame, sound):
    region_start, region_end = REGION_POSITIONS[sound]
    overlay = frame.copy()

    overlay = cv2.rectangle(overlay, region_start, region_end, BLUE, -1)

    return cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def main():
    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        # Capture frame-by-frame
        _, frame = capture.read()

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

        sound = CHANNEL.get_sound()

        if sound:
            frame = paint_region(frame, sound)

        # Paint the delimiters on the video
        cv2.line(frame, (DELIMITER_LEFT, 0),
                 (DELIMITER_LEFT, HEIGHT), GREEN, 2)
        cv2.line(frame, (DELIMITER_RIGHT, 0),
                 (DELIMITER_RIGHT, HEIGHT), GREEN, 2)

        # Display the resulting frame
        cv2.imshow('Guitar Tracker', frame)


if __name__ == "__main__":
    # Initialize video capture
    capture = cv2.VideoCapture(0)

    # Initialize audio channel
    CHANNEL = mixer.Channel(0)

    main()

    # Stop sound
    CHANNEL.stop()

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()
