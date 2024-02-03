import cv2
import utils
import numpy as np


cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)


haar_cascade = cv2.CascadeClassifier('ComputerVision/haarcascade_frontalface_default.xml')

def ilkYakalama(roi):
    faces_rect = haar_cascade.detectMultiScale(roi,1.3,4)
   
    for (x_pos, y_pos, w, h) in faces_rect:
        cv2.rectangle(roi,(x_pos,y_pos),(x_pos+w,y_pos+h),(0,255,0),1)

    if len(faces_rect)>0:
        return True
    else:
        return False


def ikinciYakalame(frame,i,j,cords):
    roiCord = utils.etraftaki_indeksleri_bul(cords,i,j,1) #!

    # print(i,j)
    # print(roiCord)
    roi = frame[cords[roiCord[0][0]][roiCord[0][1]][0][1]:cords[roiCord[-1][0]][roiCord[-1][1]][1][1],
                cords[roiCord[0][0]][roiCord[0][1]][0][0]:cords[roiCord[-1][0]][roiCord[-1][1]][1][0]]
    
    cv2.imshow("constRoi",roi)
    faces_rect = haar_cascade.detectMultiScale(roi,1.3,4)
   
    for (x_pos, y_pos, w, h) in faces_rect:
        cv2.rectangle(roi,(x_pos,y_pos),(x_pos+w,y_pos+h),(255,0,0),1)



#-------------------------------------------------------------
#                 AYARRRRR
ret, frame = cap.read()
cords = utils.alanBol(frame,4,3) #!
#---------------------------------------------------------------
a = np.shape(cords)[1]
b = np.shape(cords)[0]
sayac = 0
i,j = 0, 0
constI,constJ = 0,0
# cap.set(cv2.CAP_PROP_FPS,10)
cv2.namedWindow("roi")
teyit = False
teyit2= True
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    j+=1
    if j >= a:
        i+=1
        j=0
        # print(i)
    
    if i >= b:
        i = 0
 
    
    cv2.rectangle(frame,cords[i][j][0],cords[i][j][1],(0,0,255),1)

    #cords[i][j][0][1] -> ilk noktanın y
    #cords[i][j][0][0] -> ilk noktanın x
    #cords[i][j][1][1] -> ikinci noktanın y
    #cords[i][j][1][0] -> ikinci noktanın x
    
    roi = frame[cords[i][j][0][1]:cords[i][j][1][1],cords[i][j][0][0]:cords[i][j][1][0]]
    cv2.imshow("roi",roi)
    
    # constRoi = frame[cords[constI][constJ][0][1]:cords[constI][constJ][1][1],cords[constI][constJ][0][0]:cords[constI][constJ][1][0]]
    # ikinciYakalame(frame,constI,constJ,cords)

    teyit = ilkYakalama(roi=roi)
    constRoi = frame[cords[constI][constJ][0][1]:cords[constI][constJ][1][1],cords[constI][constJ][0][0]:cords[constI][constJ][1][0]]
    
    if(teyit and teyit2):
        constI,constJ = i,j
        teyit2 = False
    if not teyit2:
        print("takip çalıştırıldı")
        ikinciYakalame(frame,constI,constJ,cords)
    if teyit:
        sayac+=1
    if sayac > 200:
        teyit2 = True
        sayac = 0
    teyit = False


    # if sayac % 10 < 1 & teyit:
    #     teyit2 = True
    #     # teyit = False
    # if teyit2 and teyit:
    #     constI,constJ = i,j
    #     sayac+=1
    #     teyit2 = False
    # elif not teyit :
    #     teyit = ilkYakalama(roi=roi)
    #     teyit2 = True
    #     sayac = 0
    # else:
    #     constRoi = frame[cords[constI][constJ][0][1]:cords[constI][constJ][1][1],cords[constI][constJ][0][0]:cords[constI][constJ][1][0]]
    #     ikinciYakalame(frame,constI,constJ,cords)
    # print(teyit, teyit2)
    # print(sayac)
    
    
    
    
    
    
    if ret:
        cv2.imshow("a",frame)
        

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    sayac +=1  
cv2.destroyAllWindows()
cap.release()
