#coding utf-8

import cv2
import sys



def CatchPicWithVideo(window_name,camera_id,catch_pic_num,path_name):
    """

    :param window_name: 窗口名
    :param camera_id: 数据源
    :param catch_pic_num: 当前保存到第几个pic
    :param path_name: 用户保存文件名
    """
    cv2.namedWindow(window_name)

    #视频流来源(摄像头or本地视频)
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    #人脸分类器
    classfire = cv2.CascadeClassifier("/Users/heyiheng/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")

    #颜色BGR
    blue = (255,0,0)
    green = (0,255,0)
    red = (0,0,255)


    #获取1000张人脸照片
    num = 0
    while cap.isOpened():
        ok , frame  =cap.read()#当前一帧数据
        if not ok:
            break

        #当前帧灰度处理，减少数据量
        gery = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #检测人脸
        faceRects= classfire.detectMultiScale(gery,scaleFactor=1.2,minNeighbors=4,minSize=(20,20))
        #检测到人脸，截图
        if len(faceRects)>0:
            for face in faceRects :
                x,y,w,h = face

                face_img_name = '%s/%d.jpg'%(path_name,num)
                image = frame[y-10:y+h+10,x-10:x+w+10]
                cv2.imwrite(face_img_name,image)

                num+=1
                if num>catch_pic_num:
                    break
                #标记框
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),green,thickness=2)
                #字体
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d'%num,(x+30,y+30),font,1,red,4)

        if num >catch_pic_num:
            break

        cv2.imshow(window_name,frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    #关闭退出程序
    cap.release()
    cv2.destroyAllWindows()


'''
shell 中运行 python catch_usb_video.py 0 1000 data/(数据集文件夹)
'''
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPicWithVideo("Collecting face data", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])