# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
upper_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    upper=upper_cascade.detectMultiScale(gray, 1.3, 4)
    for (ux,uy,uw,uh) in upper:
          cv2.rectangle(frame, (ux, uy), (ux+uw, uy+uh), (0, 0, 255), 2)
          roi_gray = gray[uy:uy+uh, ux:ux+uw]
          roi_color = frame[uy:uy+uh, ux:ux+uw]
          faces = face_cascade.detectMultiScale(roi_gray, 1.3, 5)
          for (x, y, w, h) in faces:
              cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
              eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
              for (ex, ey, ew, eh) in eyes:
                  cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()