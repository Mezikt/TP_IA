# Model Card: YOLOv8 Lego Detector

## 1. Descrição do Modelo
* **Desenvolvido por:** Nuno Mesquita (8160208), Cristiana Macedo(8220404), Guilherme Barbosa (8221008)
* **Tipo de Modelo:** Detetor de Objetos (Object Detection) baseado nas arquiteturas YOLOv8 Nano e YOLOv8 Small.
* **Caso de Uso:** Identificação e contagem automática de peças e geometrias de blocos de Lego em tempo real via Webcam ou Upload de imagem.

## 2. Hardware e Ambiente de Treino
O treino dos dois modelos foi realizado localmente utilizando hardware dedicado para computação de IA:
* **GPU:** NVIDIA GeForce RTX 4060 (8 GB VRAM, 3072 núcleos CUDA)
* **CPU:** Intel Core i5-12600KF
* **Sistema Operativo:** Windows 10
* **Ambiente Técnico:** Python 3.11.0, PyTorch 2.5.1+cu121, Ultralytics 8.4.53

## 3. Hiperparâmetros de Treino
Para garantir uma comparação científica justa entre as variantes, ambos os modelos partilharam as mesmas configurações:
* **Épocas (Epochs):** 100
* **Tamanho do Lote (Batch Size):** 8 (YOLOv8 Nano) / 16 (YOLOv8 Small)
* **Resolução da Imagem (imgsz):** 640x640 píxeis
* **Subprocessos de Dados (Workers):** 2
* **Otimizador:** Auto (AdamW/SGD selecionado dinamicamente pela Ultralytics)

## 4. Dataset Utilizado
* **Origem:** Dataset exportado via Roboflow.
* **Classes de Geometria:** *(Exemplo: bloco_2x2, bloco_2x4, placa_4x4)*
