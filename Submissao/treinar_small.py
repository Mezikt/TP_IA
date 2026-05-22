from ultralytics import YOLO
import os

def main():
    # modelo base SMALL (yolov8s.pt)
    model = YOLO("yolov8s.pt")

    # Hiperparâmetros idênticos para manter a comparação justa
    EPOCAS = 100         # Número de iterações de treino 
    BATCH_SIZE = 16      # Quantidade de imagens processadas de cada vez 
    TAMANHO_IMG = 640    # Tamanho da imagem definido no Roboflow 

    print("A iniciar o treino local do YOLOv8 Small para peças de Lego...")
    
    # 2. Executar o treino apontando para o mesmo data.yaml
    results = model.train(
        data="datasets/data.yaml",   
        epochs=EPOCAS, 
        imgsz=TAMANHO_IMG, 
        batch=BATCH_SIZE,
        device="0",                  
        workers=2,
        name="yolov8_small"          #Cria uma pasta isolada para o Small!
    )
    
    print("Treino do modelo Small concluído com sucesso!")

if __name__ == '__main__':
    main()