from ultralytics import YOLO
import os

def main():
   #Modelo base Nano (yolov8n.pt) é o mais leve e rápido
    model = YOLO("yolov8n.pt")

    # Parametros de treino
    EPOCAS = 100          # Número de iterações de treino 
    BATCH_SIZE = 16      # Quantidade de imagens processadas de cada vez 
    TAMANHO_IMG = 640    # Tamanho da imagem definido no Roboflow

    print("A iniciar o treino local do YOLOv8 para peças de Lego...")
    
    #caminho para o data.yaml e parametros de treino
    results = model.train(
        data="datasets/data.yaml", 
        epochs=EPOCAS, 
        imgsz=TAMANHO_IMG, 
        batch=BATCH_SIZE,
        device="0",      # device="0" para GPU NVIDIA e usa device="cpu" para processador
        workers=2        # Número de subprocessos para carregar dados
    )
    
    print("Treino concluído com sucesso!")

if __name__ == '__main__':
    main()