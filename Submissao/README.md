# 🧱 TP IA — Sistema de Deteção de Peças LEGO

**UC:** Inteligência Artificial | **Ano letivo:** 2025/2026  
**Modelo:** YOLOv8s | **Dataset:** Lego Detection v3 (40 classes, 2129 imagens)  
**Construções:** Crocodilo · Flor · Fantasma · Moinho

---

## 📁 Estrutura do projeto

```
TP_IA_LEGO/
├── dataset_raw/           ← Fotos originais tiradas pelo grupo
├── dataset_yolo/          ← Coloca aqui o ZIP do Roboflow (extraído automaticamente)
│   └── lego_v3/
│       ├── train/images/ & train/labels/
│       ├── valid/images/ & valid/labels/
│       ├── test/images/  & test/labels/
│       └── data.yaml
├── annotations/           ← Ficheiros de anotação e política de etiquetagem
├── modelos/               ← Pesos treinados (gerado automaticamente pelo treino)
│   └── lego_v3_s/
│       └── weights/
│           └── best.pt
├── app/
│   ├── app.py             ← App Streamlit de demonstração
│   └── README.md
├── docs/                  ← Model card, manifesto JSON, dataset.md
├── notebooks/
│   └── 01_treino.ipynb    ← Notebook principal (treino + avaliação + inferência)
├── results/               ← Gráficos e métricas gerados pelo YOLO
├── input/                 ← Imagens de teste para o infer.py
├── output/                ← Resultados de inferência (imagens anotadas + JSON)
├── infer.py               ← Script de inferência (linha de comandos)
└── requirements.txt
```

---

## 🚀 Instalação

### 1. Criar ambiente virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 📦 Preparar o dataset

1. Copia o ficheiro `Lego_detection_v3-test-model-22-05_yolov8.zip` para a pasta `dataset_yolo/`
2. O notebook extrai-o automaticamente na célula 2 (Opção B)

---

## 🏋️ Treinar o modelo

Abre o notebook e corre célula a célula:

```bash
jupyter notebook notebooks/01_treino.ipynb
```

O treino guarda automaticamente em `modelos/lego_v3_s/weights/best.pt`.

### ⚡ Dicas de hardware

| Hardware | Tempo estimado (100 epochs) |
|---|---|
| CPU | 14–25 horas ⚠️ (usa 30 epochs para testar) |
| Apple M-series | 4–7 horas |
| GPU NVIDIA RTX | 1–3 horas ✅ |

> **Só tens CPU?** No notebook, muda `EPOCHS = 100` para `EPOCHS = 30` para um primeiro teste rápido.

---

## 🔍 Inferência pela linha de comandos

```bash
# Uma imagem
python infer.py --model modelos/lego_v3_s/weights/best.pt --image input/foto.jpg

# Pasta de imagens
python infer.py --model modelos/lego_v3_s/weights/best.pt --folder input/

# Com confiança personalizada e visualização
python infer.py --model modelos/lego_v3_s/weights/best.pt --image foto.jpg --conf 0.4 --show
```

Resultados guardados em `output/` (imagem anotada + JSON por imagem).

---

## 🖥️ Aplicação de demonstração (Streamlit)

```bash
cd app
streamlit run app.py
```

Abre automaticamente em http://localhost:8501

### Funcionalidades da app:
- ✅ Upload de imagem ou webcam
- ✅ Seleção de modelo (se houver múltiplos)
- ✅ Slider de confiança ajustável
- ✅ Bounding boxes com labels e confiança
- ✅ Tabela de inventário de peças detetadas
- ✅ Recomendação de construções com score de compatibilidade
- ✅ Exportação de resultados em JSON
- ✅ Histórico das últimas 10 inferências

---

## 📊 TensorBoard

Durante ou após o treino, visualiza as curvas:

```bash
tensorboard --logdir modelos
```

Abre http://localhost:6006

---

## 🏗️ Classes detetadas (40 classes)

| Categoria | Classes |
|---|---|
| Bricks | angular_brick_1x1, brick_1x2, brick_1x4, brick_2x2, brick_2x4, brick_arch_2x3, brick_bow_1x4, brick_cross_2x2, brick_knob_1x1, brick_pin_2x2, brick_slope_curved_1x2, transparent_brick_1x2 |
| Plates | plate_1x1x2, plate_1x3, plate_1x4, plate_2x3, plate_2x4, plate_2x6, plate_2x8, plate_4x4, plate_8x16, plate_cross_3x3, plate_knob_2x2 |
| Roof Tiles | inverted_roof_tile_2x3, roof_2x2, roof_tile_1x1x2, roof_tile_1x2, roof_tile_1x2x2, roof_tile_1x3, roof_tile_2x1x2, roof_tile_2x2, roof_tile_2x4 |
| Especiais | door, eye_piece, nose_cone_1x1, propeller_4blades, round_brick_1x1, round_plate_1x1, wall_window_1x2x2, window_frame_1x2x2 |

---

## 📋 Entregáveis gerados

| Ficheiro | Entregável |
|---|---|
| `modelos/lego_v3_s/weights/best.pt` | D3 — pesos do modelo |
| `modelos/lego_v3_s_eval.json` | D3 — métricas de avaliação |
| `infer.py` | D3 — script de inferência |
| `app/app.py` | D4 — aplicação de demonstração |
| `dataset_yolo/lego_v3/` | D2 — dataset exportado |
| `docs/` | D3 — model card + manifesto JSON |
