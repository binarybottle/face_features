# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
#
# This code is made available to the students of
# the online course titled "Computer Vision for Faces"
# by Satya Mallick for personal non-commercial use.
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC.
#
# For licensing and other inquiries, please email
# spmallick@bigvisionllc.com
#
#
# Modified by Arno Klein, 2019-02-05
#
# Use CLandmark (http://cmp.felk.cvut.cz/~uricamic/clandmark/index.php?page=installation)
# and OpenCV (https://www.learnopencv.com/install-opencv-docker-image-ubuntu-macos-windows/)
#

commands = """
# Enter the docker container:
docker run -v /Users/arno/GitHub/face_features:/root/demo/opendock -e DISPLAY=$DISPLAY:0 -p 5000:5000 -it spmallick/opencv-docker:opencv /bin/bash
workon OpenCV-master-py3
ipython

# Once you are in the iPython prompt, import OpenCV and exit:
import cv2
cv2.__version__
exit()

# Run the facial feature extraction code from within a shared directory shared between Docker and local host:
cd /root/demo/opendock
source activate OpenCV-master-py3
python extract_features.py
"""

import cv2
import dlib
import os

import argparse

parser = argparse.ArgumentParser(description="""
                                 $ python extract_features.py IMAGE_DIR
                                 where IMAGE_DIR is the directory of facial image files""",
                                 formatter_class=lambda prog:
                                 argparse.HelpFormatter(prog,
                                                        max_help_position=40))
# "positional arguments":
parser.add_argument("IMAGE_DIR", help=("directory containing facial image files"))
args = parser.parse_args()
IMAGE_DIR = args.IMAGE_DIR
IMAGE_LIST = os.listdir(IMAGE_DIR)
# Check to make sure each file has an extension -- TO DO: confirm file is an image
IMAGE_LIST = [x for x in IMAGE_LIST if len(os.path.splitext(x)) > 1]

OUT_DIR = "output"
cmd = 'mkdir {0}'.format(OUT_DIR)
print(cmd)
os.system(cmd)

FEATURES_DIR = os.path.join(OUT_DIR, "features")
cmd = 'mkdir {0}'.format(FEATURES_DIR)
print(cmd)
os.system(cmd)

NO_FACES_DIR = os.path.join(OUT_DIR, "no_faces")
cmd = 'mkdir {0}'.format(NO_FACES_DIR)
print(cmd)
os.system(cmd)

errorsFileName = os.path.join(OUT_DIR, "no_faces.txt")


def writeErrorsToFile(error_string, errorsFileName):
    with open(errorsFileName, 'a') as f:
        f.write(error_string + "\n")


def writeLandmarksToFile(landmarks, landmarksFileName):
    with open(landmarksFileName, 'w') as f:
        for p in landmarks.parts():
            f.write("%s %s\n" % (int(p.x), int(p.y)))
    f.close()


def drawLandmarks(iface, im, landmarks):
    for i, part in enumerate(landmarks.parts()):
        px = int(part.x)
        py = int(part.y)
        cv2.circle(im, (px, py), 1, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(im, str(iface) + ": " + str(i + 1), (px, py), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0), 1)


for IMAGE_FILE in IMAGE_LIST:

    IMAGE_PATH = os.path.join(IMAGE_DIR, IMAGE_FILE)
    IMAGE_STEM = os.path.splitext(IMAGE_FILE)[0]
    outputFileName = os.path.join(FEATURES_DIR, IMAGE_STEM + ".jpg")

    if not os.path.isfile(outputFileName):

        # Landmark model location
        PREDICTOR_PATH = "../../common/shape_predictor_68_face_landmarks.dat"

        # Get the face detector
        faceDetector = dlib.get_frontal_face_detector()

        # The landmark detector is implemented in the shape_predictor class
        landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

        try:

            # Read image
            im = cv2.imread(IMAGE_PATH)
            imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            faceRects = faceDetector(imDlib, 0)
            print("Number of faces detected: {0}".format(len(faceRects)))

            # List to store landmarks of all detected faces
            landmarksAll = []

            # Write to errors file when no face is detected, and copy image to NO_FACES_DIR
            nfaces = len(faceRects)
            if nfaces == 0:
                print("NO faces: {0}".format(IMAGE_FILE))
                writeErrorsToFile(IMAGE_FILE, errorsFileName)
                copyFileName = os.path.join(NO_FACES_DIR, IMAGE_FILE)
                cmd = 'cp "{0}" "{1}"'.format(IMAGE_PATH, copyFileName)
                print(cmd)
                os.system(cmd)

            else:

                if nfaces > 1:
                    print("{0} faces: {1}".format(str(nfaces), IMAGE_FILE))

                # Loop over all detected face rectangles
                for i in range(0, nfaces):
                    newRect = dlib.rectangle(int(faceRects[i].left()), int(faceRects[i].top()),
                                             int(faceRects[i].right()), int(faceRects[i].bottom()))

                    # For every face rectangle, run landmarkDetector
                    landmarks = landmarkDetector(imDlib, newRect)
                    # Print number of landmarks
                    # if i==0:
                    #  print("Number of landmarks",len(landmarks.parts()))

                    # Store landmarks for current face
                    landmarksAll.append(landmarks)
                    # Draw landmarks on face
                    drawLandmarks(i, im, landmarks)

                    landmarksFileName = os.path.join(FEATURES_DIR, IMAGE_STEM + "_" + str(i) + ".txt")
                    print("Saving landmarks to {0}".format(landmarksFileName))
                    # Write landmarks to disk
                    writeLandmarksToFile(landmarks, landmarksFileName)

                print("Saving output image to {0}".format(outputFileName))
                cv2.imwrite(outputFileName, im)

                # cv2.imshow("Facial Landmark detector", im)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        except:

            print("{0} failed".format(IMAGE_FILE))
