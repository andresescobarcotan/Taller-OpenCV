import numpy as np
import scipy.stats
import cv2

'''
FUNCIONES UTILES DE OPENCV
KERNEL : (numeroImparHorizontal, numeroImparVertical)
suavizado gaussiano: resultado = cv2.GaussianBlur(origen,KERNEL, 0)
suavizado general: resultado =  cv2.blur(origen, KERNEL)
conversion a BYN : resultado = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
umbralizado: resultado =  cv2.threshold(origen, min, max,cv2.THRESH_BINARY)
aplicar mascara: resultado = cv2.bitwise_and(origen, origen, mask=mask)
resta: imagen_restada = cv2.absdiff(frame_1, frame_2)
copiar: resultado = np.copy(origen)
mostrar imagen : cv2.imshow('NOMBRE_VENTANA', imagen)



'''

cap = cv2.VideoCapture('cochesPasando.mp4') # Abre el video

_, frame = cap.read() # Devuelve un frame, la _ indica que se rechaza el segundo parametro


while(cap.isOpened()): # Mientras haya frames
    _, frame = cap.read() # Se lee el frame
    # Extraer el color de la crrretera
    carretera = np.copy(frame)
    # BGR -> OPENCV 0 azul, 1 verde, 2 rojo
    _, color_b = scipy.stats.mode(carretera[:,:,0])
    _, color_g = scipy.stats.mode(carretera[:,:,1])
    _, color_r = scipy.stats.mode(carretera[:,:,2])
    print(color_b.shape)
    #carretera[:, :, 0] = carretera[:,:,0] - color_b[0,:]
    #carretera[:, :, 1] = carretera[:,:,1] - color_g[0,:]
    #carretera[:, :, 2] = carretera[:,:,2] - color_r[0,:]
    carretera[:] = carretera[:] - np.transpose([color_b[0,:], color_g[0,:], color_r[0,:]])
    cv2.imshow('final',carretera) # Se muestra en la ventana con nombre final, y la matriz frame
     
    if cv2.waitKey(1) & 0xFF == ord('q'): # Si se pulsa la tecla q se sale del bucle
        break

cap.release() # Se borra la memoria
cv2.destroyAllWindows() # Y se destruye las ventanas
