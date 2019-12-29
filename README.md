# face_features
Extract 68 landmarks from a facial image (python, using opencv and clandmark).

# Replace PATH_TO_FOLDER_WITH_CODE_AND_IMAGE_FOLDERS and IMAGE_FOLDER with appropriate folder path and name:
docker run -v PATH_TO_FOLDER_WITH_CODE_AND_IMAGE_FOLDERS:/root/demo/opendock -e DISPLAY=$DISPLAY:0 -p 5000:5000 -it spmallick/opencv-docker:opencv /bin/bash

workon OpenCV-master-py3
ipython
import cv2
cv2.__version__
exit()
cd /root/demo/opendock
source activate OpenCV-master-py3

cd PATH_TO_FOLDER_WITH_CODE_AND_IMAGE_FOLDERS
python ./face_features/extract_features.py /root/demo/opendock/IMAGE_FOLDER
