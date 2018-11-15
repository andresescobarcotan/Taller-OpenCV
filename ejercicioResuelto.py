import numpy as np
import cv2

KERNEL = (11, 11) # KERNEL del gausiano esta a 11 
mk = np.ones((21,21),np.uint8)
cap = cv2.VideoCapture('cochesPasando.mp4') # Abre el video

background_frame = 0
_, frame = cap.read() # Devuelve un frame, la _ indica que se rechaza el segundo parametro
background_frame = np.copy(frame) # Copia el frame en una matriz nueva
background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY) # Se convierte a blanco y negro 
background_frame = cv2.GaussianBlur(background_frame,KERNEL, 0) # Se aplica un gausiano para eliminar el ruido
cv2.imshow('fondo', background_frame)     # Se muestra el fondo

while(cap.isOpened()): # Mientras haya frames
    _, frame = cap.read() # Se lee el frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Se copia en blanco y negro (por valor) el frame
    gray_frame =  cv2.GaussianBlur(gray_frame,KERNEL, 0) # Se aplica un gaussiano
    diff = cv2.absdiff(background_frame, gray_frame) # IMPORTANTE se resta el fondo, con el frame actual
    canny = cv2.Canny(diff, 10, 20, 3)
    #canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, mk)
    _, gray_mask = cv2.threshold(diff, 10, 200,cv2.THRESH_BINARY) # Se crea una mascara usando el umbralizado
    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    drawing = np.zeros((gray_mask.shape[0], gray_mask.shape[1], 1), np.uint8)
    '''
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    '''
    drawing = cv2.fillPoly(drawing, pts=hull, color=255)
    drawing = cv2.morphologyEx(drawing, cv2.MORPH_CLOSE, mk)
    drawing = cv2.dilate(drawing,  (3,3), iterations=1)
    gray_mask = cv2.blur(gray_mask, KERNEL)  # Se suaviza usando un KERNEL la mascara 
    masked_frame = np.copy(frame) # Se copia el frame a otra matriz que sera de resultado
    masked_frame = cv2.bitwise_and(frame,frame, mask=drawing) # Se aplica la mascara usando una operacion aritmetica
    cv2.imshow('Canny', canny)
    cv2.imshow('MASCARA2', drawing)
    cv2.imshow('mascara',gray_mask)
    cv2.imshow('final',masked_frame) # Se muestra en la ventana con nombre final, y la imagen enmascarada
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # Si se pulsa la tecla q se sale del bucle
        break

cap.release()  # Se borra la memoria
cv2.destroyAllWindows() # Y se destruye las ventanas