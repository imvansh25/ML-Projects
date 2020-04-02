import cv2
import numpy as np

#VideoCapture
cap = cv2.VideoCapture(0)
harscade_file = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_data = []
name = input("Enter name of the person:-")
skip = 0
data_path="./data/"

while True:
    ret,frame = cap.read()
    if(ret== False):
        continue
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = harscade_file.detectMultiScale(frame,1.3,5)
    faces = sorted(faces,key = lambda f:f[2]*f[3])

    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,255,255),2)
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        skip+=1
        if(skip%10==0):
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame",frame)
    cv2.imshow("gray_frame",gray_frame)
    key_pressed = cv2.waitKey(1) & 0xFF
    if(key_pressed==ord('q')):
        break;
face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print (face_data)
np.save(data_path+name+".npy",face_data)
print("data successfully saved at"+data_path+name+".npy")

cap.release()
cv2.destroyAllWindows()
