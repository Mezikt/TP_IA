# Aplicação de Demonstração — LEGO Detector

Aplicação web desenvolvida em Streamlit para demonstração do sistema de deteção de peças LEGO.  
Deteta peças em imagens ou via webcam, identifica cores, conta o inventário e recomenda construções possíveis.

---

## Pré-requisitos

Antes de correr a app, garante que:

1. **O modelo está treinado** — o ficheiro `best.pt` tem de existir em:
   ```
   ../modelos/lego_v3_s/weights/best.pt
   ```
   Se ainda não treinaste, corre primeiro o notebook `../notebooks/01_treino.ipynb`.

2. **As dependências estão instaladas** — na pasta raiz do projeto:
   ```bash
   pip install -r ../requirements.txt
   ```
   Ou instala apenas o necessário para a app:
   ```bash
   pip install streamlit ultralytics opencv-python numpy
   ```

---

## Como correr

```bash
# A partir da pasta raiz do projeto
cd app
streamlit run app.py
```

A app abre automaticamente no browser em:
```
http://localhost:8501
```

Se não abrir automaticamente, copia o URL do terminal e cola no browser.

---

## Funcionalidades

### Obrigatórias
| Funcionalidade | Descrição |
|---|---|
| Upload de imagem | Suporta JPG, PNG, BMP, WEBP |
| Webcam | Captura foto diretamente pelo browser |
| Seleção de modelo | Dropdown com todos os modelos em `../modelos/` |
| Bounding boxes | Visualização com label e confiança em cada peça |
| Vista estruturada | Tabela de inventário por tipo de peça |

### Bónus implementados
| Funcionalidade | Descrição |
|---|---|
| Deteção de cor | Identifica a cor de cada peça via OpenCV (HSV) |
| Slider de confiança | Ajusta o limiar de deteção em tempo real (0.05–0.95) |
| Comparação de modelos | Compara dois modelos lado a lado na mesma imagem |
| Export JSON | Descarrega todas as deteções em formato JSON |
| Export CSV | Descarrega o inventário de peças em formato CSV |
| Export imagem | Descarrega a imagem anotada com bounding boxes |
| Histórico | Regista as últimas 10 inferências da sessão |

---

## Parâmetros configuráveis (sidebar)

| Parâmetro | Valor padrão | Descrição |
|---|---|---|
| Confiança mínima | 0.25 | Deteções abaixo deste valor são ignoradas |
| Limiar IoU (NMS) | 0.70 | Controla a sobreposição de bounding boxes |
| Detetar cor | Ativo | Analisa HSV de cada bounding box para identificar a cor |
| Comparar dois modelos | Inativo | Ativa o modo de comparação side-by-side |

---

## Deteção de cor (OpenCV)

A cor de cada peça é identificada automaticamente pelo módulo `color_detector.py`:

1. Recorta a região da bounding box na imagem original
2. Aplica erosão morfológica para remover bordas e fundo
3. Converte para espaço de cor HSV
4. Compara com ranges HSV definidos para 11 cores
5. A cor com maior cobertura de pixels é selecionada (mínimo 10%)

**Cores suportadas:** vermelho · verde · amarelo · azul · laranja · rosa · castanho · preto · branco · transparente · cinzento

---

## Construções suportadas

O motor de recomendação compara o inventário detetado com os requisitos de cada construção:

| Construção | Nº de tipos de peças |
|---|---|
| 🐊 Crocodilo | 8 tipos |
| 🌸 Flor | 6 tipos |
| 👻 Fantasma | 7 tipos |
| 🏠 Moinho | 11 tipos |

O score de compatibilidade é calculado como:
```
score = peças_detetadas_que_tens / peças_necessárias × 100%
```

---

## Estrutura de ficheiros

```
app/
├── app.py              ← Aplicação principal Streamlit
├── color_detector.py   ← Módulo de deteção de cor (OpenCV/HSV)
└── README.md           ← Este ficheiro
```

---

## Resolução de problemas

**"Nenhum modelo encontrado"**  
→ Treina o modelo primeiro. O ficheiro `best.pt` tem de estar em `../modelos/lego_v3_s/weights/`.

**Erro ao importar `color_detector`**  
→ Garante que estás a correr o streamlit a partir da pasta `app/` e não de outra pasta.

**Webcam não funciona**  
→ O Streamlit usa a câmara do browser. Aceita a permissão de câmara quando o browser pedir.

**App muito lenta**  
→ Reduz o slider de confiança ou corre o modelo numa máquina com GPU.

---

## Exemplo de output JSON

```json
{
  "model": "../modelos/lego_v3_s/weights/best.pt",
  "conf_threshold": 0.25,
  "total_detections": 5,
  "counts_by_class": {
    "brick_2x4": 2,
    "eye_piece": 2,
    "roof_tile_1x2": 1
  },
  "detections": [
    {
      "class_id": 4,
      "class_name": "brick_2x4",
      "confidence": 0.8921,
      "color": "vermelho",
      "bbox": { "x1": 120.5, "y1": 88.3, "x2": 245.1, "y2": 178.6 }
    }
  ],
  "recommendations": [
    { "construction": "👻 Fantasma", "score": 92, "can_build": false }
  ]
}
```
