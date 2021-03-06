import numpy as np
import cv2

KERNEL = (11, 11) # KERNEL del gausiano esta a 11 
VERDE = (0,255,100)
AZUL = (255,100,0)
ROJO = (100,0,255)
VIDEO = 'cochesPasando.mp4'
ancho = 0
alto = 0
punto_medio = 0

def comprobar_colision(x,w):
    # Esta funcion comprueba la colision entre la linea y el bounding box
    punto_inicio = x
    punto_final = x + w
    color = VERDE
    if((punto_inicio <= punto_medio) and (punto_final >= punto_medio)):
        color = ROJO
    return color

def pintar_linea(masked_frame):
    masked_frame[:alto, punto_medio] = AZUL  # Pinta una linea azul que servira para contar
    return masked_frame

def crear_boundingbox(masked_frame, gray_mask):
    mk = np.ones((51,51),np.uint8) # Elemento morfologico, lo agrego para eliminar los bounding box internos
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, mk)
    contours,hierarchy = cv2.findContours(gray_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    idx =0
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        color = comprobar_colision(x,w)
        masked_frame[y:y+h,x] = color 
        masked_frame[y+h-1,x:x+w] = color 
        masked_frame[y:y+h,x+w-1] = color 
        masked_frame[y,x:x+w] = color
        
    
    return masked_frame

if __name__ == '__main__':
    cap = cv2.VideoCapture(VIDEO) # Abre el video
    background_frame = 0
    _, frame = cap.read() # Devuelve un frame, la _ indica que se rechaza el segundo parametro
    background_frame = np.copy(frame) # Copia el frame en una matriz nueva
    background_frame = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY) # Se convierte a blanco y negro 
    ancho, alto = background_frame.shape
    punto_medio = int(alto/2)
    print(punto_medio)
    background_frame = cv2.GaussianBlur(background_frame,KERNEL, 0) # Se aplica un gausiano para eliminar el ruido
    cv2.imshow('fondo', background_frame)     # Se muestra el fondo

    while(cap.isOpened()): # Mientras haya frames
        _, frame = cap.read() # Se lee el frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Se copia en blanco y negro (por valor) el frame
        gray_frame =  cv2.GaussianBlur(gray_frame,KERNEL, 0) # Se aplica un gaussiano
        diff = cv2.absdiff(background_frame, gray_frame) # IMPORTANTE se resta el fondo, con el frame actual
        _, gray_mask = cv2.threshold(diff, 20, 200,cv2.THRESH_BINARY) # Se crea una mascara usando el umbralizado
        gray_mask = cv2.blur(gray_mask, KERNEL)  # Se suaviza usando un KERNEL la mascara    
        masked_frame = np.copy(frame) # Se copia el frame a otra matriz que sera de resultado
        masked_frame = cv2.bitwise_and(frame,frame, mask=gray_mask) # Se aplica la mascara usando una operacion aritmetica
        
        
        masked_frame = pintar_linea(masked_frame)
        masked_frame = crear_boundingbox(masked_frame, gray_mask)
            
        cv2.imshow('mascara',gray_mask)
        cv2.imshow('final',masked_frame) # Se muestra en la ventana con nombre final, y la imagen enmascarada
        
        if cv2.waitKey(1) & 0xFF == ord('q'): # Si se pulsa la tecla q se sale del bucle
            break
  
    cap.release()  # Se borra la memoria
    cv2.destroyAllWindows() # Y se destruye las ventanas
