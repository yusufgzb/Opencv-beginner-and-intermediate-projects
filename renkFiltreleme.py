import cv2
import numpy as np
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dusukMavi=np.array([45,30,30])
    ustMavi=np.array([190,255,255])

    mask=cv2.inRange(hsv,dusukMavi,ustMavi)
    sonHali=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("orjinal",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("sonHali",sonHali)

    if cv2.waitKey(25)& 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
