import numpy as np
import cv2
import os

path = "ImageSample"
orb = cv2.ORB_create(nfeatures=1000)

# Import Images
images = []
classNames = []
myList = os.listdir(path)
print(myList) #List of all files in ImageQuery
print('Total Classes Detected: ', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])  # Removes file extension in name
print(classNames)

def findDes(images):  # finding descriptors
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

def findID(img, desList, thres=5):
    kp2, des2 = orb.detectAndCompute(img, None)  # find descriptor of new image
    bf = cv2.BFMatcher()
    # loop through descriptors of each file in ImageQuery
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:  # 2 values because k=2
                if m.distance < 0.75 * n.distance:  # Whenever distance is low it is a good match
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    # print(matchList)
    if len(matchList) != 0:  # check if matchList is empty
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal

desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture('sample.mp4')
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (1080, 1920))
detections = []

while True:
    ret, img2 = cap.read()
    if not ret:
        break

    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findID(img2, desList)  # Shows the number of good matches for each ImageQuery
    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # scale, color, thickness

        # Find contours and draw bounding boxes
        contours, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            print('Box', x, y, w, h)

            # Save the detection information in a list
            detections.append((classNames[id], (x, y, x + w, y + h)))

            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output.write(imgOriginal)

    # cv2.imshow('img2', imgOriginal)
    cv2.imshow('output', imgOriginal)

    key = cv2.waitKey(1)
    if key == 27:  # press esc to quit window
        break

cap.release()
output.release()  # Release the output video writer
cv2.destroyAllWindows()

# Print the detection information
print("Detected frames and their coordinates:")
for i, (className, coordinates) in enumerate(detections):
    print(f"Frame {i+1}: Class '{className}', Coordinates: {coordinates}")
