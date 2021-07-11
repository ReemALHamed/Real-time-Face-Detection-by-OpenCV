# Real-time-Face-Detection-by-OpenCV

This project is done using Pycharm IDE and Python, it is a Real-Time Face recognition using OpenCV while performing object detection using Haar feature-based cascade classifiers for detecting face, eyes, and smile.

## 1. Import and initialize
Start by importing OpenCV and create a directory (ex: Cascades) to gather all Haar classifiers files that you want to use in you project, then use their path to load them into your project.
##### Note: you can find Haar cascade classifiers files here (https://github.com/opencv/opencv/tree/master/data/haarcascades)
```
import cv2
faceCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_FRONTALFACE_DEFAULT.xml')
eye_cascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_EYE.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascades/HAARCASCADE_SMILE.xml')
```

## 2. Setting up your camera
To start we need to capture the face and to do so we are using the PC embedded camera which we are referring to it using (0) & setting the window size to specific measures in the following code lines:
```
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
```
## 3. Call the classifier function
We will set our camera and inside the loop, load our input video in grayscale mode then we must call our classifier function, passing it some very important parameters, as scale factor, number of neighbors and minimum size of the detected face.
```
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
```

## 4. Detecting Faces
The function will detect faces on the image. Next, we must "mark" the faces in the image, using, for example, a blue rectangle. 
If faces are found, it returns the positions of detected faces as a rectangle with the left up corner (x,y) and having "w" as its Width and "h" as its Height ==> (x,y,w,h). 
This is done with this portion of the code:
```
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
```

## 5. Detecting Eyes
The function will detect eyes on the image which are marked as a green rectangle. 
If eyes are found, it returns the positions of detected eyes as a rectangle with the left up corner (ex,ey) and having "ew" as its Width and "eh" as its Height ==> (ex,ey,ew,eh). This is done with this portion of the code:
```
    for (ex, ey, ew, eh) in eyes:
        # draw a green rectangle around the eyes
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
```

## 6. Detecting Smiles
The function will detect Smiles on the image which are marked as a green rectangle. 
If Smiles are found, it returns the positions of detected Smiles as a rectangle with the left up corner (xx,yy) and having "ww" as its Width and "hh" as its Height ==> (xx, yy, ww, hh). This is done with this portion of the code:
```
    smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )
    for (xx, yy, ww, hh) in smile:
        cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
```

## 7. Final Touches
If the user wants to quit the program, the button ESC is set to terminate the program in the following code lines, and to alert the user there is a small message shown on the top-right corner of the detected face boarders.
```
k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break
    msg = 'press ESC to quit'
    # put a text above the top - right corner of the face rectangle
    cv2.putText(img, msg, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    cv2.imshow('Face Detection - Reem', img)
```
## 8. The Result !!
To test the program i used a photo of Mark Zuckerberg, cuz why not :)


https://user-images.githubusercontent.com/85634099/125203413-e627ee80-e280-11eb-9ab8-294f829b4398.mp4


Reference: https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826
