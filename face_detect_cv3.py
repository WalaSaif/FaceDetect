from os import listdir
from os.path import isfile, join
import numpy
import cv2
import sys
from pathlib import Path

# New image destination folders
facesPath = "Faces"
noFacesPath = "NoFaces"

# Get user supplied values
folderPath = sys.argv[1]
imagefiles = [ f for f in listdir(folderPath) if isfile(join(folderPath,f)) ]
images = numpy.empty(len(imagefiles), dtype=object)
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
for n in range(0, len(imagefiles)):
    if not imagefiles[n].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')): continue
    images[n] = cv2.imread( join(folderPath,imagefiles[n]) )
    gray = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} faces!".format(len(faces)))

    # Modify image path
    if len(faces) != 0: Path(folderPath + "/" + imagefiles[n]).rename(facesPath + "/" + imagefiles[n])
    else: Path(folderPath + "/" + imagefiles[n]).rename(noFacesPath + "/" + imagefiles[n])

    # Draw a rectangle around the faces and display image window
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(images[n], (x, y), (x+w, y+h), (0, 255, 0), 2)

    # cv2.imshow("Faces found", images[n])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)