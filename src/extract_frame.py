import cv2

video = cv2.VideoCapture("videos/test.mp4")

ret, frame = video.read()

if ret:
    cv2.imwrite("outputs/frame1.jpg", frame)
    print("Frame saved!")

video.release()