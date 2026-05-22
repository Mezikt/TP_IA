# Model Card — LEGO Detection YOLOv8s

## Informação geral

| Campo | Valor |
|---|---|
| **Nome do modelo** | lego_detection_yolov8s |
| **Versão** | 1.0 |
| **Arquitetura** | YOLOv8s (small) |
| **Tarefa** | Deteção de objetos (Object Detection) |
| **Framework** | Ultralytics YOLOv8 |
| **UC** | Inteligência Artificial — ESTG 2025/2026 |

---

## Descrição

Este modelo deteta e classifica 40 tipos de peças LEGO em imagens estáticas ou em tempo real via webcam. Foi desenvolvido no âmbito do Trabalho Prático de Inteligência Artificial da Licenciatura em Engenharia Informática da ESTG.

O sistema é capaz de:
- Detetar a presença e localização de peças LEGO numa imagem (bounding boxes)
- Classificar cada peça em uma das 40 classes definidas
- Integrar com deteção de cor via OpenCV para identificar a cor de cada peça
- Recomendar construções possíveis com base no inventário detetado

---

## Caso de uso

**Cenário-alvo:** Inventariação automática de peças LEGO a partir de uma fotografia ou câmara em tempo real, com o objetivo de recomendar construções possíveis (Crocodilo, Flor, Fantasma, Moinho).

**Motivação industrial:** Sistemas de inspeção e inventariação de componentes em ambiente de produção ou logística.

---

## Dataset

| Propriedade | Valor |
|---|---|
| **Nome** | Lego Detection v3 |
| **Plataforma** | Roboflow |
| **Total de imagens** | 2129 |
| **Split treino** | ~70% |
| **Split validação** | ~20% |
| **Split teste** | ~10% |
| **Formato** | YOLOv8 (YOLO txt labels) |
| **Resolução** | 640×640 px |

### Augmentations aplicadas (Roboflow)
- Flip horizontal (50% probabilidade)
- Rotação aleatória ±15°
- Crop aleatório 0–20%
- Ajuste de brilho ±15%
- Ajuste de exposição ±10%

### Augmentations em tempo real (Ultralytics)
- HSV jitter (H: ±1.5%, S: ±70%, V: ±40%)
- Mosaic (4 imagens combinadas)
- Scale ±50%
- Flip horizontal 50%

---

## Classes (40 total)

### Bricks
`angular_brick_1x1` · `brick_1x2` · `brick_1x4` · `brick_2x2` · `brick_2x4` · `brick_arch_2x3` · `brick_bow_1x4` · `brick_cross_2x2` · `brick_knob_1x1` · `brick_pin_2x2` · `brick_slope_curved_1x2` · `transparent_brick_1x2`

### Plates
`plate_1x1x2` · `plate_1x3` · `plate_1x4` · `plate_2x3` · `plate_2x4` · `plate_2x6` · `plate_2x8` · `plate_4x4` · `plate_8x16` · `plate_cross_3x3` · `plate_knob_2x2`

### Roof Tiles
`inverted_roof_tile_2x3` · `roof_2x2` · `roof_tile_1x1x2` · `roof_tile_1x2` · `roof_tile_1x2x2` · `roof_tile_1x3` · `roof_tile_2x1x2` · `roof_tile_2x2` · `roof_tile_2x4`

### Peças Especiais
`door` · `eye_piece` · `nose_cone_1x1` · `propeller_4blades` · `round_brick_1x1` · `round_plate_1x1` · `wall_window_1x2x2` · `window_frame_1x2x2`

---

## Configuração de treino

| Hiperparâmetro | Valor |
|---|---|
| **Modelo base** | yolov8s.pt (pré-treinado COCO) |
| **Epochs** | 100 |
| **Batch size** | 16 |
| **Image size** | 640×640 |
| **Optimizer** | Auto (SGD/Adam) |
| **Learning rate (lr0)** | 0.01 |
| **Learning rate (lrf)** | 0.01 |
| **Momentum** | 0.937 |
| **Weight decay** | 0.0005 |
| **Warmup epochs** | 3 |
| **Patience (early stop)** | 50 |

---

## Métricas de avaliação

*(Preencher após o treino)*

| Métrica | Valor |
|---|---|
| **mAP@0.5** | — |
| **mAP@0.5:0.95** | — |
| **Precision** | — |
| **Recall** | — |

---

## Suposições e restrições

### O modelo funciona bem quando:
- As peças estão sobre um fundo de cor uniforme (branco, cinzento, preto)
- A iluminação é razoavelmente uniforme e sem reflexos fortes
- A câmara está a uma distância de 20–60 cm das peças
- As peças têm pelo menos 40% da sua área visível (não fortemente ocluídas)

### Limitações conhecidas:
- **Peças muito pequenas:** peças 1x1 em imagens tiradas de longe podem ser difíceis de detetar
- **Oclusão forte:** peças sobrepostas mais de 60% podem ser detetadas incorretamente
- **Cores similares em fundo similar:** peças brancas em fundo branco reduzem a precisão
- **Iluminação:** luz direta intensa ou muito baixa degrada o desempenho
- **Rotação extrema:** peças em ângulos acima de 45° podem ser confundidas com outras classes
- **Peças transparentes:** mais difíceis de detetar por falta de textura

### Modos de falha comuns:
- Confusão entre `brick_2x2` e `brick_2x4` em ângulos oblíquos
- `round_plate_1x1` e `round_brick_1x1` podem ser confundidos
- Roof tiles invertidas podem ser confundidas com roof tiles normais

---

## Deteção de cor

A cor de cada peça é identificada por pós-processamento com OpenCV:

1. Recorte da região da bounding box
2. Erosão para remover bordas/fundo
3. Conversão para espaço de cor HSV
4. Comparação com ranges HSV definidos por cor
5. A cor com maior cobertura de pixels é selecionada (mínimo 10%)

**Cores suportadas:** vermelho · verde · amarelo · azul · laranja · rosa · castanho · preto · branco · transparente · cinzento

---

## Inferência

```bash
# Linha de comandos
python infer.py --model modelos/lego_v3_s/weights/best.pt --image foto.jpg

# Aplicação web
cd app && streamlit run app.py
```

**Threshold de confiança recomendado:** 0.25–0.35  
**Threshold IoU recomendado:** 0.60–0.75  
**Velocidade média:** ~30–80ms por imagem (dependendo do hardware)

---

## Autores

Trabalho Prático — Licenciatura em Engenharia Informática  
Unidade Curricular: Inteligência Artificial  
ESTG — Escola Superior de Tecnologia e Gestão  
Ano letivo: 2025/2026
