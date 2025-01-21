import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()

if ret:
    cv2.imshow('Captured Image', frame)

    cv2.waitKey(0)
else:
    print("Failed to capture image")

cap.release()
cv2.destroyAllWindows()
