import pafy
import cv2 
import os
import pandas as pd
import numpy as np

path= '' #put the root folder in HERE

# view = {'1B' #as default
    #     '1F': 'Front View',
    #     '1B': 'Back View',
    #     '1L': 'Left View',
    #     '1R': 'Right View',
    #     '2': ' Bicycle 
    #     '5H': 'High-Density Human Crowd',
    #     '5L': 'Low-Density Human Crowd',
    #     '0': 'irrelevant TV }



length = [12550, 105072, 102525, 80818, 200570, 228546, 199242]

urls = ['https://www.youtube.com/watch?v=5AJlYeZ8Ilc',
        'https://www.youtube.com/watch?v=QJBihoz38Qc',
        'https://www.youtube.com/watch?v=w1CE1ykYa3I',
        'https://www.youtube.com/watch?v=HcRcstqHnR4',
        'https://www.youtube.com/watch?v=70-1-hFnpjM',
        'https://www.youtube.com/watch?v=bsLe0awGvzg',
        'https://www.youtube.com/watch?v=kYPYD_RxHWU']

vid = [1, 4, 5, 6, 7, 8, 9]

annot= [None]*7

annot0 = pd.read_csv(path+'\datasets\giro_data\annot\giro1_1.txt', header = None ,sep = ' ' )
annot1 = pd.read_csv(path+'\datasets\giro_data\annot\giro4.txt', header = None ,sep = ' ' )
annot2 = pd.read_csv(path+'\datasets\giro_data\annot\giro5.txt', header = None ,sep = ' ' )
annot3 = pd.read_csv(path+'\datasets\giro_data\annot\giro6.txt', header = None ,sep = ' ' )
annot4 = pd.read_csv(path+'\datasets\giro_data\annot\giro7.txt', header = None ,sep = ' ' )
annot5 = pd.read_csv(path+'\datasets\giro_data\annot\giro8.txt', header = None ,sep = ' ' )
annot6 = pd.read_csv(path+'\datasets\giro_data\annot\giro9.txt', header = None ,sep = ' ' )

annot=[annot0,annot1,annot2,annot3,annot4,annot5,annot6]

for k in range(len(annot)):
    annot[k] = annot[k].rename(columns={0: "dr1", 1: "frame", 2: "ID", 3: "X1", 4: "Y1", 5: "X2", 6: "Y2", 7: "dr2", 8: "object", 9: "dr3"})
    annot[k].drop(columns = ["dr1", "dr2", "dr3"], inplace = True)
    annot[k]["vid"] = vid[k]
    annot[k]["vid"] = annot[k]["vid"].astype(int)

file_name = ["1","4","5","6","7","8","9"]



for j in  range(len(length)):
    vPafy = pafy.new(urls[j])
    play = vPafy.getbest(preftype="mp4")
    cap = cv2.VideoCapture(play.url)
    
    af=list(annot[j].frame.unique())
    #af = list(annot[j][annot[j]["object"] == view].frame.unique())
    i=0
    while (True and i <= length[j]):   
        ret,frame = cap.read()
        if ret == False:
            break
        if i in af:
           cv2.imwrite('giro'+str(file_name[j])+'_'+str(i)+'.jpg', frame)
        i+=1
cap.release()
cv2.destroyAllWindows() 
