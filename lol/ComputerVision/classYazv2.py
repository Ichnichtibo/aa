import utils
import cv2
import numpy as np
import torch

import threading
import time


class HurturkTakip:
    teyit1:bool = False
    constI:int = 0 
    constJ:int = 0
    roiCord = []
    x= 720
    y= 1280

    def __init__(self,model_name,camIndex = 0):
        self.haar_cascade = cv2.CascadeClassifier('ComputerVision/haarcascade_frontalface_default.xml')
        self.model = torch.load_model(f"models/{model_name}")
        self.model_names = self.model.names

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # ! camera ayarları
        self.cap = cv2.VideoCapture(camIndex)
        self.cap.set(3,1280)
        self.cap.set(4,720)
        self.cords = utils.alanBol(self.cap.read()[1],3,3)
        self.rowNum,self.coluomNum = np.shape(self.cords)[0:2]
        print(self.rowNum,self.coluomNum)

    
    def load_model(self):
        try:
            model = torch.hub.load('ultralytics/yolov5','custom',path=model_name,force_reload=True)
            return model
        except Exception as e:
            print("Path Yanlış.....")

    def dedect_models(self,frame):
        frame = [frame]
        results = self.model(frame)
        self.model.

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels,cord
    def class_to_label(self, x):
        """
        classlarımızı labela dönüştürüyoruz.
        """
        return self.classes[int(x)]






# *********************************************////////////////////////////////////
    def ilkYakalamav2(self,roi):
        results = self.dedect_models(roi)
        labels,cord = results
        n = len(labels)
     
        for (x_pos, y_pos, w, h) in faces_rect:
            cv2.rectangle(roi,(x_pos,y_pos),(x_pos+w,y_pos+h),(0,255,0),1)

        if len(faces_rect)>0:
            self.teyit1 = True
    
    def ilkYakalama(self,roi):
        faces_rect = self.haar_cascade.detectMultiScale(roi,1.3,4)
    
        for (x_pos, y_pos, w, h) in faces_rect:
            cv2.rectangle(roi,(x_pos,y_pos),(x_pos+w,y_pos+h),(0,255,0),1)

        if len(faces_rect)>0:
            self.teyit1 = True
    

    def roideArama(self,frame):
        for i,j in self.roiCord:
            roi = frame[self.cords[i][j][0][1]:self.cords[i][j][1][1],
                        self.cords[i][j][0][0]:self.cords[i][j][1][0]]
            faces_rect = self.haar_cascade.detectMultiScale(roi,1.3,4)
            
            if len(faces_rect)>0:
                self.constI = i
                self.constJ = j
                break
    
    def ortaKaredenCiktimi(self,frame):
        roi = frame[self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][1]+109:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][1]-109,
                    self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][0]+60:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][0]-60]
        cv2.imshow("b",roi)

        faces_rect = self.haar_cascade.detectMultiScale(roi,1.3,4)

        if len(faces_rect) == 0:
            print("ekranı değiştir")
            self.roideArama(frame=frame)
            
    def ikinciYakalama(self,frame,kareSayisi):
        self.roiCord = utils.etraftaki_indeksleri_bul(self.cords,self.constI,self.constJ,kareSayisi) #!
        print(self.roiCord)

        roi = frame[self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][1]:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][1],
                    self.cords[self.roiCord[0][0]][self.roiCord[0][1]][0][0]:self.cords[self.roiCord[-1][0]][self.roiCord[-1][1]][1][0]]
        
        self.ortaKaredenCiktimi(frame=frame)
            

        cv2.imshow("constRoi",roi)
        faces_rect = self.haar_cascade.detectMultiScale(roi,1.3,4)
    
        for (x_pos, y_pos, w, h) in faces_rect:
            cv2.rectangle(roi,(x_pos,y_pos),(x_pos+w,y_pos+h),(255,0,0),1)
   
    def __call__(self):
        i = j = 0
        
        sayac = 240
        while True:
            self.teyit1 = False 
            ret, frame = self.cap.read()
            frame = cv2.flip(frame ,1)

            j+=1
            if j >= self.coluomNum:
                i +=1
                j =0
            if i >= self.rowNum:
                i = 0 

            #cords[i][j][0][1] -> ilk noktanın y
            #cords[i][j][0][0] -> ilk noktanın x
            #cords[i][j][1][1] -> ikinci noktanın y
            #cords[i][j][1][0] -> ikinci noktanın x
                
                
            cv2.rectangle(frame,self.cords[i][j][0],self.cords[i][j][1],(0,0,255),1)
            cv2.rectangle(frame,self.cords[self.rowNum-i-1][self.coluomNum-j-1][0],self.cords[self.rowNum-i-1][self.coluomNum-j-1][1],(0,0,255),1)
            self.ilkYakalama(roi=frame[self.cords[i][j][0][1]:self.cords[i][j][1][1],
                                       self.cords[i][j][0][0]:self.cords[i][j][1][0]])
            self.ilkYakalama(roi=frame[self.cords[self.rowNum-i-1][self.coluomNum-j-1][0][1]:self.cords[self.rowNum-i-1][self.coluomNum-j-1][1][1],
                                       self.cords[self.rowNum-i-1][self.coluomNum-j-1][0][0]:self.cords[self.rowNum-i-1][self.coluomNum-j-1][1][0]])
            
          
            if self.teyit1 and sayac >= 200: 
                self.constJ = j
                self.constI = i   
                self.teyit2 = False
                sayac = 0
            if sayac <= 200:
                # print("Takip çalıştırıldı")
                self.ikinciYakalama(frame,1)
                # sayac += 1.4
            if ret:
                cv2.imshow("a",frame)            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("silindi")


a = HurturkTakip()
a()