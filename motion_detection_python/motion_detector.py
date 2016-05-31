import cv2
import imutils as imutils

camera = None
MIN_AREA = 25


def run_capture():
    first_frame = None
    while True:
        (grabbed, frame) = camera.read()
        if not grabbed:
            break  # end of video

        # fix the image size, convert to greyscale, and blur
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if first_frame is None:
            first_frame = gray
            # we now have something to compare to. Yay progress!
            continue

        frame_delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded values
        threshold = cv2.dilate(thresh, None, iterations=2)
        (contours, _) = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = "Occupied"

        cv2.imshow("Feed", frame)
        key = cv2.waitKey(1) & 0xFF


if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    run_capture()
