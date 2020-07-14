import cv2
import numpy as np
from tensorflow.keras.models import load_model

#model = load_model("../models/model_NN.h5")
model = load_model("../models/model_CNN.h5")

pred_label = {
    0: 'Mask detected',
    1: 'Mask not detected',
    2: 'Unknown'
}

pred_color = {
    0: (0, 255, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
}


def prepare_image(pth):
    return cv2.resize(pth, (256, 256)).reshape(-1, 256, 256, 1) / 255.0


classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    faces = classifier.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 2)

    for face in faces:
        slicedImg = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
        prediction = model.predict(prepare_image(img))
        prediction = np.argmax(prediction)
        cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), pred_color[prediction], 2)
        cv2.putText(img, pred_label[prediction], (face[0], face[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('FaceMask Detection', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()