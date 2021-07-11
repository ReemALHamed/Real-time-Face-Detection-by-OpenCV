import cv2

faceCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_FRONTALFACE_DEFAULT.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_EYE.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_SMILE.xml')

cap = cv2.VideoCapture(0)  # write 0 to use the pc embedded camera
cap.set(3,640) # set Width
cap.set(4,380) # set Height
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    for face in faces:
        # (x,y,w,h)
        top, right, bottom, left = face
        # draw a blue rectangle around the face
        cv2.rectangle(img, (top, right), (top + bottom, right + left), (255, 0, 0), 2)
        roi_gray = gray[right:right + left, top:top + bottom]
        roi_color = img[right:right + left, top:top + bottom]
        eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        # draw a green rectangle around the eyes
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        msg = 'press ESC to quit'
        # put a text above the top - right corner of the face rectangle
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

    for (xx, yy, ww, hh) in smile:
        cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        cv2.putText(img, msg, (top + 5, right - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.imshow('Face Detection - Reem', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
