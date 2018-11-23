'''
NOTA: es posible que este ejercicio falle al no encontrar el metodo 
en ese caso ejecutar la siguiente linea: pip install opencv-contrib-python --user
'''
import numpy as np
import cv2 
cap = cv2.VideoCapture('cochesPasando.mp4') # Abre el video
fgbg = cv2.createBackgroundSubtractorMOG2() #Usa el extractor de fondo por defecto de OpenCV
while(cap.isOpened()):
    _, frame = cap.read() # Devuelve un frame, la _ indica que se rechaza el segundo parametro
    gray_mask = fgbg.apply(frame) # Aplica al frame la extraccion del fondo
    masked_frame = np.copy(frame) # Se copia el frame a otra matriz que sera de resultado
    masked_frame = cv2.bitwise_and(frame,frame, mask=gray_mask) # Se aplica la mascara usando una operacion aritmetica
    cv2.imshow('Mascara',gray_mask) # Muestra la mascara
    cv2.imshow('final',masked_frame) # Se muestra en la ventana con nombre final, y la imagen enmascarada
    if cv2.waitKey(1) & 0xFF == ord('q'): # Si se pulsa la tecla q se sale del bucle
        break
cap.release()
cv2.destroyAllWindows()