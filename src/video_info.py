import cv2

video = cv2.VideoCapture("videos/test.mp4")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print(f"Width  : {width}")
print(f"Height : {height}")
print(f"FPS    : {fps}")

video.release()