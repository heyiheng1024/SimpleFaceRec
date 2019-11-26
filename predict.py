#coding utf-8
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('./model/my_face_model.h5')

detector = cv2.CascadeClassifier('/Users/heyiheng/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
#分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
k = 0
d = {0: 'me',1:'second'}

#判断 遍历人脸信息
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_face = img[y: y+h, x:x+w]#找到的人脸
        # # cv2.imwrite('train/1/%s.jpg'%(str(k)), img_face)
        # k+=1
        # img_face = cv2.imread('train/0/0.jpg')
        img_face = cv2.resize(img_face, (64,64))#resize成model输入的尺寸
        img_face = img_face.astype("float") / 255.0
        img_face = img_to_array(img_face)
        pred = model.predict(np.expand_dims(img_face, axis=0))[0]#预测
        # 概率
        with_pred = np.round(pred*100,2)
        print(with_pred)
        if max(pred) < 0.9:
            text = 'others'
        else:
            pred_label = np.argmax(pred)
            print(pred_label)
            text = d[pred_label]
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3, cv2.LINE_AA)  # 写字
        cv2.putText(img,str(max(with_pred)),(x-50,y+50),cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', img)
    # cv2.waitKey()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()