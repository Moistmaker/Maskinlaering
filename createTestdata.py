import cv2
from skimage import io
import os
import random
import numpy as np

#Funskjon for å legge til støy på bildene
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

#Antall er hvor mange av hvert bilde som skal bli lagd
#ikkeOK_ for bilder som er ødelagt, og OK_ for bilder som er godkjent
#sp_noise tar inn et bilde og en variabel som legger til mer støy jo høyere variabelen er
def bilde_behandling(image_in, antall=33, counter = 0):
    for i in range(1, antall+1):
        noise_img = sp_noise(image_in,0.02)
        gray_image = cv2.cvtColor(noise_img, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (356, 150), cv2.INTER_AREA)
        blur = cv2.GaussianBlur(resized_image,(5,5),0.7)
        canny_image = cv2.Canny(blur,120,240)
        io.imsave("png/DTCtest2/OK_" + str(counter) + ".png" , canny_image)
        counter += 1
        if i % antall == 0:
            return counter

#n er hvor mange bilder fra mappen som skal brukes 
n = 2
counter = 1

for x in range(1, n):
    #image_in = cv2.imread('png/Lecabilder/Lecatest' + str(x) + '.png',1)
    image_in = cv2.imread('png/Lecabilder/Lecatest.png',1)
    counter = bilde_behandling(image_in, counter = counter)