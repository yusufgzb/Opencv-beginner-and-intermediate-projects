import cv2
#Temel Takip Algoritmaları
OPENCV_OBJECT_TRACKERS = {"csrt"      : cv2.TrackerCSRT_create,
		                  "kcf"       : cv2.TrackerKCF_create,
		                  "boosting"  : cv2.TrackerBoosting_create,
		                  "mil"       : cv2.TrackerMIL_create,
		                  "tld"       : cv2.TrackerTLD_create,
		                  "medianflow": cv2.TrackerMedianFlow_create,
		                  "mosse"     : cv2.TrackerMOSSE_create}

tracker_name = "mil"

trackers = cv2.MultiTracker_create()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # Yüksekliği ve Genişiği alma
    (H, W) = frame.shape[:2]
    frame = cv2.resize(frame, dsize = (960, 540))
    
    (success , boxes) = trackers.update(frame)# trackers her adımda update edilecek
    """
    Tracker ismi ve
    Temel sonuçalr başarılı yada başarısız
    """
    info = [("Tracker", tracker_name),
        	("Success", "Yes" if success else "No")]
    
    string_text = ""
    
    #
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        string_text = string_text + text + " "
    
    cv2.putText(frame, string_text, (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    
    for box in boxes:#box takip sonucuna denk gelir
        (x, y, w, h) = [int(v) for v in box]#boxes değerleri int dönüştürdük
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("t"):
        #takip edeceğimiz bölge 
        box = cv2.selectROI("Frame", frame, fromCenter=False)
    
        tracker =cv2.TrackerMIL_create()
        trackers.add(tracker, frame, box)
    elif key == ord("q"):break

    f = f + 1
    
cap.release()
cv2.destroyAllWindows() 

