import cv2
import numpy as np

img1=cv2.imread("imgs/img1.jpg")
img2=cv2.imread("imgs/img2.jpg")
img3=cv2.imread("imgs/img3.jpg")
img4=cv2.imread("imgs/QR_kodu.jpeg")
img1=cv2.resize(img1,(640,480))
img2=cv2.resize(img2,(640,480))


def EdgeDetection(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny=cv2.Canny(imgGray,1,2)
    cv2.imshow("canny",canny)

def PhotoSketching(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBitNot=cv2.bitwise_not(imgGray)
    imgGaussian=cv2.GaussianBlur(imgBitNot,(21,21),0,0)

    final=cv2.divide(imgGray,imgGaussian)
    cv2.imshow("imgGray",imgGray)
    cv2.imshow("imgBitNot",imgBitNot)
    cv2.imshow("final",final)

#PhotoSketching(img2)

def DetectingContours(img):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # RETR_CCOMP iç ve dış kontür buldurur
    #CHAIN_APPROX_SIMPLE yatay dikey bölgeleri sıkıştırıp sadece çapraz bölgeleri almamızı sağlıyor
    img,contours,hierarch=cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    # iç ve dış kontür tutmak için matrix oluşturalım
    external=np.zeros(img.shape)
    internal=np.zeros(img.shape)

    for i in range(len(contours)):
        if hierarch[0][i][3]==-1:
            cv2.drawContours(external,contours,i,255,-1)
        else:
            cv2.drawContours(internal,contours,i,255,-1)
    
    cv2.imshow("external",external)
    cv2.imshow("internal",internal)
#DetectingContours(img3)


def CollageMosaicGenerator(img1,img2):
    img=cv2.addWeighted(img1,0.7,img2,0.3,1)
    cv2.imshow("img",img)
#CollageMosaicGenerator(img1,img2)

def PanoramaStitching(img1,img2):
    yatay=np.hstack((img1,img2))
    dikey=np.vstack((img1,img2))
    cv2.imshow("yatay",yatay)
    cv2.imshow("dikey",dikey)
#PanoramaStitching(img1,img2)

def QRCodeScanner(img):

    detector=cv2.QRCodeDetector()
    data,bbox,straight_qrcode=detector.detectAndDecode(img)

    if bbox is not None:
        n_line=len(bbox)
        for i in range(n_line):
            point1=tuple(bbox[i][0])
            point2=tuple(bbox[(i+1)%n_line][0])
            cv2.line(img,point1,point2,(255,0,0),thickness=2)
    cv2.imshow("img",img)
QRCodeScanner(img4)



cv2.waitKey(0)
cv2.destroyAllWindows()
