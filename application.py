import pygame,sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

windowsizeX = 640
windowsizeY = 480
boundry=5
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)

predict = True

image_cnt=1

Imagesave=False
Model = load_model("C:/Users/ekans/otherMLmodels/hand new/bestmodel.h5")

Label ={0:"ZERO",1:"ONE",
        2:"TWO",3:"THREE",
        4:"FOUR",5:"FIVE",
        6:"SIX",7:"SEVEN",
        8:"EIGHT",9:"NINE"}
pygame.init()
iswriting =False
number_xcord = []
number_ycord = []

Font = pygame.font.SysFont("Segoe UI",18)
DISPLAYSURF=pygame.display.set_mode((640,480))

pygame.display.set_caption("Digit Board")
while True:
    for event in pygame.event.get():
        if event.type ==QUIT:
            pygame.quit()
            sys.exit()
        if event.type== MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF,white,(xcord,ycord), 4,0 )

            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type==MOUSEBUTTONUP:
            iswriting=False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x,rect_max_x  = max(number_xcord[0]-boundry,0),min(windowsizeX,number_xcord[-1]+boundry)
            rect_min_y, rect_max_y = max(number_ycord[0] - boundry, 0), min(number_ycord[-1] + boundry,windowsizeX)

            number_xcord =[]
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)

            if Imagesave:
                cv2.imwrite("image.png")
                image_cnt+=1

            if predict:

                image = cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10), 'constant', constant_values=0)
                image = cv2.resize(image,(28,28))/255

                label = str(Label[np.argmax(Model.predict(image.reshape(1,28,28,1)))])

                textSurface = Font.render(label,True,white,red)
                textRecObj = textSurface.get_rect()
                textRecObj.left , textRecObj.bottom = rect_min_x,rect_max_y

                DISPLAYSURF.blit(textSurface,textRecObj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURF.fill(black)

        pygame.display.update()