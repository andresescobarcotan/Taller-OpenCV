import numpy as np
import cv2

CANAL_SELECCIONADO = 0
KERNEL = (11, 11) # KERNEL del gausiano esta a 11 
mk = np.ones((51,51),np.uint8) # Elemento morfologico
cap = cv2.VideoCapture('cochesPasando.mp4') # Abre el video
background_frame = 0
_, frame = cap.read() # Devuelve un frame, la _ indica que se rechaza el segundo parametro
background_frame = np.copy(frame) # Copia el frame en una matriz nueva
#background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY) # Se convierte a blanco y negro 
background_frame = np.copy(frame[:,:,CANAL_SELECCIONADO])
background_frame = cv2.equalizeHist(background_frame)
background_frame = cv2.GaussianBlur(background_frame,KERNEL, 0) # Se aplica un gausiano para eliminar el ruido
cv2.imshow('fondo', background_frame)     # Se muestra el fondo

while(cap.isOpened()): # Mientras haya frames
    _, frame = cap.read() # Se lee el frame
    gray_frame = np.copy(frame[:,:,CANAL_SELECCIONADO])
    #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Se copia en blanco y negro (por valor) el frame
    gray_frame = cv2.equalizeHist(gray_frame)
    gray_frame =  cv2.GaussianBlur(gray_frame,KERNEL, 0) # Se aplica un gaussiano
    diff = cv2.absdiff(background_frame, gray_frame) # IMPORTANTE se resta el fondo, con el frame actual
    canny = cv2.Canny(diff, 10, 20, 3) # Se hallan los bordes, usando el metodo de Canny
    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Se generan vectores matematicos que simulan los contornos
    # Creacion de un array, que contendra el area de esos contornos
    hull = []
    # Por cada contorno encontrado
    for i in range(len(contours)):
        # Se calcula su area convexa
        hull.append(cv2.convexHull(contours[i], False))

    #Se crea una imagen negra
    drawing = np.zeros((diff.shape[0], diff.shape[1], 1), np.uint8)
    
    # Se rellenan las areas convexas descubiertas de color de blanco,
    # obteniendo una mascara
    drawing = cv2.fillPoly(drawing, pts=hull, color=255)
    # Se aplican elementos morfologicos de cierre
    drawing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, mk)

    masked_frame = cv2.bitwise_and(frame, frame, mask=drawing) # Se aplica la mascara usando una operacion aritmetica
    
    cv2.imshow('Canny', canny)
    cv2.imshow('mascara',drawing)
    cv2.imshow('final',masked_frame) # Se muestra en la ventana con nombre final, y la imagen enmascarada
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Si se pulsa la tecla q se sale del bucle
        break

cap.release()  # Se borra la memoria
cv2.destroyAllWindows() # Y se destruye las ventanas
