import numpy as np
import cv2

img_path='C:/Users/kgt/OneDrive/Desktop/coding/virtualintern/object_detect/street.jpg'
prototxt_path='C:/Users/kgt/OneDrive/Desktop/coding/virtualintern/object_detect/deploy.prototxt'
model_path='C:/Users/kgt/OneDrive/Desktop/coding/virtualintern/object_detect/mobilenet_iter_73000.caffemodel'
min_conf=0.2
classes=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chai","cow","table","dog","horse","motorbike","person","pottedplant",
         "sheep","sofa","train","T.Vmonitor"]
np.random.seed(534210)
colors=np.random.uniform(0,255,size=(len(classes),3))
net=cv2.dnn.readNetFromCaffe(prototxt_path,model_path) #loading pre-trained model

#img=cv2.imread(img_path)

cap=cv2.VideoCapture(0)
while True:
    _,img=cap.read()

    height,width=img.shape[0],img.shape[1]
    blob=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),0.007,(300,300),130)
    net.setInput(blob)
    detected_obj=net.forward()
    #print(detected_obj[0][0][0])
    for i in range(detected_obj.shape[2]):
        conf=detected_obj[0][0][i][2]
        if conf>min_conf:
            class_index=int(detected_obj[0][0][i][1])
            upper_left_x=int(detected_obj[0][0][i][3]*width)
            upper_left_y=int(detected_obj[0][0][i][4]*height)
            lower_right_x=int(detected_obj[0][0][i][5]*width)
            lower_right_y=int(detected_obj[0][0][i][6]*height)
            prediction_text=f"{classes[class_index]}: {conf:.2f}%"
            cv2.rectangle(img,(upper_left_x,upper_left_y),(lower_right_x,lower_right_y),colors[class_index],3)
            cv2.putText(img,prediction_text,(upper_left_x,upper_left_y-15
                        if upper_left_y>30 else upper_left_y+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,colors[class_index],2)
    cv2.imshow("Detected objects",img)
    cv2.waitKey(5)
    cv2.destroyAllWindows()
    cap.release()
        
