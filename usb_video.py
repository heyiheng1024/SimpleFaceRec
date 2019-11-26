#coding utf-8

import cv2
import sys





#视频流来源(摄像头or本地视频)
cap = cv2.VideoCapture("迪丽热巴.jpg")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#人脸分类器
classfire = cv2.CascadeClassifier("Classifier/haarcascade_frontalface_alt2.xml")

#颜色BGR
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)


while cap.isOpened():
    ok , frame  =cap.read()#当前一帧数据
    if not ok:
        break

    #当前帧灰度处理，减少数据量
    gery = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #检测人脸
    faceRects= classfire.detectMultiScale(gery,scaleFactor=1.2,minNeighbors=4,minSize=(20,20))

    if len(faceRects)>0:
        for face in faceRects :
            x,y,w,h = face
            print(x,y,w,h)

            cv2.rectangle(frame,(x,y),(x+w,y+h),green,thickness=2)

            face_img = '1.jpg'
            cv2.imwrite(face_img, frame)



    cv2.imshow('frame',frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
#关闭退出程序
cap.release()
cv2.destroyAllWindows()

