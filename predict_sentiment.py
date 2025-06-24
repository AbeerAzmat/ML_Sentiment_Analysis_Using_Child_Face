from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import sys
 
# path for loading data and images
emotion_model_path = 'models/emotion_model.h5'
detection_model_path =  'models/haarcascade_frontalface_default.xml' #path to haar cascade

#emotion labels
EMOTIONS = ["angry", "happy", "sad"]

 
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

 
#getting image from argument
img_path = sys.argv[1]
orig_frame = cv2.imread(img_path)
frame = cv2.imread(img_path,0) # greyscale

faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
 
if len(faces) > 0:
    faces = sorted(faces, reverse=True,key=lambda x: x[2] * x[3])
    (fX, fY, fW, fH) = faces [0]
 
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
 
    preds = emotion_classifier.predict(roi)[0]
    label = EMOTIONS[preds.argmax()]
 
    cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(255, 0, 0), 2)
 
cv2.imshow('Emotion Detection', orig_frame)
cv2.waitkey(2000)
cv2.destroyAllWindows()


output_path = 'test_output/' + img_path.split('/')[-1]
cv2.imwrite(output_path, orig_frame)
