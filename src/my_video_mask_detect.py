import imutils
import numpy as np
from imutils.video import VideoStream
from tensorflow import keras
import cv2
import time

#model_to_use = "mask_detector.model"
model_to_use = "model_NN.h5"
#model_to_use = "model_CNN.h5"
model = keras.models.load_model(model_to_use)


# Create a VideoCapture object and read from input file
cap = VideoStream(src=0).start()
time.sleep(2.0)

# Read until video is completed
while True:

    # Capture frame-by-frame
    frame = cap.read()
    frame = imutils.resize(frame, width=400)

    image_to_detect = frame
    size = 256

    #img = cv2.imread(image_to_detect, cv2.IMREAD_GRAYSCALE)
    img = frame
    img = cv2.resize(img, (size, size))
    img = img / 255
    img = np.array(img).reshape(-1, size, size, 1)

    prediction = model.predict(img)
    print(prediction)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# When everything done, release the video capture object
cap.stop()
# Closes all the frames
cv2.destroyAllWindows()