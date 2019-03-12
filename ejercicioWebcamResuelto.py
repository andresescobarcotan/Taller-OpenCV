import numpy as np
import cv2
TIMER_ELAPSE = 5
DIM_KERNEL = 11 # KERNEL del gausiano esta a 11 
KERNEL = (DIM_KERNEL, DIM_KERNEL) 
IMAGEN = 'camaraSeguridad.mp4'

if __name__ == '__main__':
    cap = cv2.VideoCapture(0) # Abre el video
    fgbg = cv2.createBackgroundSubtractorKNN() #Usa el extractor de fondo por defecto de OpenCV
    img_pointer = cv2.VideoCapture(IMAGEN)
    _, frame = cap.read() # Se lee el frame
    height, width = frame.shape[:2]
    
    while(cap.isOpened()): # Mientras haya frames
        if(img_pointer.isOpened()):
            _,img = img_pointer.read()
            img = cv2.resize(img, (width, height))
        _, frame = cap.read() # Se lee el frame
        gray_mask = fgbg.apply(frame,learningRate=0) # Aplica al frame la extraccion del fondo
        masked_frame = np.copy(frame) # Se copia el frame a otra matriz que sera de resultado
        masked_frame = cv2.bitwise_and(frame,frame, mask=gray_mask) # Se aplica la mascara usando una operacion aritmetica
        mascara_fondo = np.abs(255 -gray_mask)
        mascara_fondo = np.where(mascara_fondo < 255, 0, mascara_fondo)
        img_masked = cv2.bitwise_and(img, img, mask=mascara_fondo)
        masked_frame = cv2.bitwise_or(img_masked, masked_frame)
        cv2.imshow('mascara',gray_mask)
        cv2.imshow('final',masked_frame) # Se muestra en la ventana con nombre final, y la imagen enmascarada
        if cv2.waitKey(1) & 0xFF == ord('q'): # Si se pulsa la tecla q se sale del bucle
            break

    cap.release()  # Se borra la memoria
    cv2.destroyAllWindows() # Y se destruye las ventanas
