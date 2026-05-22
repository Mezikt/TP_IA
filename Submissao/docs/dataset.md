# Dataset — Lego Detection v3

## Visão geral

| Propriedade | Valor |
|---|---|
| **Nome** | Lego Detection v3 |
| **Plataforma** | Roboflow |
| **Formato** | YOLOv8 |
| **Total de imagens** | 2129 |
| **Número de classes** | 40 |
| **Resolução** | 640×640 px |
| **Data de exportação** | 22 de maio de 2026 |

---

## Estrutura de diretórios

```
dataset_yolo/lego_v3/
├── train/
│   ├── images/     ← imagens de treino (~70%)
│   └── labels/     ← labels YOLO (.txt) de treino
├── valid/
│   ├── images/     ← imagens de validação (~20%)
│   └── labels/     ← labels YOLO (.txt) de validação
├── test/
│   ├── images/     ← imagens de teste (~10%)
│   └── labels/     ← labels YOLO (.txt) de teste
└── data.yaml       ← configuração do dataset
```

---

## Classes e nomenclatura

O formato dos nomes segue o padrão `tipo_LarguraxAltura` para peças standard, ou nome descritivo para peças especiais.

### Bricks (tijolo standard)

| Classe | Descrição |
|---|---|
| `angular_brick_1x1` | Brick 1×1 angular (canto) |
| `brick_1x2` | Brick 1×2 |
| `brick_1x4` | Brick 1×4 |
| `brick_2x2` | Brick 2×2 |
| `brick_2x4` | Brick 2×4 (tijolo mais comum) |
| `brick_arch_2x3` | Brick com arco 2×3 |
| `brick_bow_1x4` | Brick curvo 1×4 |
| `brick_cross_2x2` | Brick com encaixe cruzado 2×2 |
| `brick_knob_1x1` | Brick com knob 1×1 |
| `brick_pin_2x2` | Brick com pino 2×2 |
| `brick_slope_curved_1x2` | Brick com rampa curva 1×2 |
| `transparent_brick_1x2` | Brick transparente 1×2 |

### Plates (placa fina)

| Classe | Descrição |
|---|---|
| `plate_1x1x2` | Plate 1×1×2 (alta) |
| `plate_1x3` | Plate 1×3 |
| `plate_1x4` | Plate 1×4 |
| `plate_2x3` | Plate 2×3 |
| `plate_2x4` | Plate 2×4 |
| `plate_2x6` | Plate 2×6 |
| `plate_2x8` | Plate 2×8 |
| `plate_4x4` | Plate 4×4 |
| `plate_8x16` | Plate 8×16 (base grande) |
| `plate_cross_3x3` | Plate com encaixe cruzado 3×3 |
| `plate_knob_2x2` | Plate com knob 2×2 |

### Roof Tiles (telhas/rampas)

| Classe | Descrição |
|---|---|
| `inverted_roof_tile_2x3` | Telha invertida 2×3 |
| `roof_2x2` | Rampa 2×2 |
| `roof_tile_1x1x2` | Telha 1×1×2 |
| `roof_tile_1x2` | Telha 1×2 |
| `roof_tile_1x2x2` | Telha 1×2×2 |
| `roof_tile_1x3` | Telha 1×3 |
| `roof_tile_2x1x2` | Telha 2×1×2 |
| `roof_tile_2x2` | Telha 2×2 |
| `roof_tile_2x4` | Telha 2×4 |

### Peças Especiais

| Classe | Descrição |
|---|---|
| `door` | Porta LEGO |
| `eye_piece` | Peça olho (para construções com cara) |
| `nose_cone_1x1` | Cone de nariz 1×1 |
| `propeller_4blades` | Hélice de 4 pás |
| `round_brick_1x1` | Brick redondo 1×1 |
| `round_plate_1x1` | Plate redonda 1×1 |
| `wall_window_1x2x2` | Janela de parede 1×2×2 |
| `window_frame_1x2x2` | Caixilho de janela 1×2×2 |

---

## Política de etiquetagem

### Regras gerais
- Cada peça LEGO visível foi anotada com **uma bounding box** e **uma label**
- Foram etiquetadas peças com **pelo menos 40% da sua área visível** na imagem
- Peças completamente escondidas por outras não foram anotadas
- A bounding box inclui toda a peça, incluindo os studs (pinos superiores)

### Casos de oclusão
- Peças sobrepostas parcialmente: etiquetadas se ≥ 40% visível
- Peças em pilha: cada peça visível foi anotada individualmente
- Peças nos bordos da imagem: anotadas se a forma é identificável

### Casos ambíguos
- `round_plate_1x1` vs `round_brick_1x1`: distinguidos pela altura (brick é mais alto)
- `roof_tile_*` vs `inverted_roof_tile_*`: distinguidos pela orientação da rampa
- `brick_2x2` vs `brick_2x4`: anotado conforme a dimensão real, mesmo em ângulos difíceis
- Peças transparentes: anotadas normalmente mas com menor confiança esperada

### Problemas conhecidos
- Algumas imagens com iluminação muito baixa podem ter labels imprecisas
- Peças 1×1 em imagens tiradas de longe podem ter bounding boxes muito pequenas
- Cores similares ao fundo podem dificultar a identificação em alguns casos

---

## Recolha de imagens

- **Dispositivos:** Smartphones (câmara principal)
- **Condições:** Interior, iluminação artificial e natural
- **Fundos:** Variados (mesa branca, cinzenta, madeira, alcatifa)
- **Distâncias:** 20–80 cm das peças
- **Ângulos:** Frontal, lateral, 45°, superior
- **Tipos de cena:**
  - Peças isoladas (uma ou poucas peças)
  - Peças misturadas (várias peças juntas)
  - Construções completas (Crocodilo, Flor, Fantasma, Moinho)

---

## Augmentation

### Roboflow (aplicada ao criar versão)
- Flip horizontal com 50% de probabilidade
- Rotação aleatória entre -15° e +15°
- Crop aleatório entre 0% e 20%
- Ajuste de brilho entre -15% e +15%
- Ajuste de exposição entre -10% e +10%
- Fator de multiplicação: ×2 (cada imagem original gera 2 versões augmented)

### Ultralytics em tempo real (durante treino)
- HSV jitter: matiz ±1.5%, saturação ±70%, valor ±40%
- Flip horizontal: 50%
- Mosaic: combinação de 4 imagens (ativado)
- Scale: zoom aleatório ±50%
- Translate: translação ±10%

---

## Verificações de qualidade

- Verificação manual de labels em amostras aleatórias de cada split
- Confirmação de que não existem imagens duplicadas entre train/valid/test
- Distribuição de classes verificada — classes raras (ex: `propeller_4blades`, `door`) têm menos amostras
- Todas as labels estão no formato YOLO normalizado (valores entre 0 e 1)
- Imagens sem anotações (sem peças visíveis) foram excluídas do dataset

---

## Formato das labels (YOLOv8)

Cada ficheiro `.txt` contém uma linha por objeto:

```
<class_id> <x_center> <y_center> <width> <height>
```

Todos os valores normalizados entre 0 e 1 relativamente às dimensões da imagem.

**Exemplo:**
```
4 0.512 0.387 0.124 0.089
```
→ Classe 4 (`brick_2x4`), centrada em (51.2%, 38.7%), com largura 12.4% e altura 8.9% da imagem.

---

## data.yaml

```yaml
train: /caminho/absoluto/train/images
val:   /caminho/absoluto/valid/images
test:  /caminho/absoluto/test/images

nc: 40
names: [angular_brick_1x1, brick_1x2, ..., window_frame_1x2x2]
```

> **Nota:** Os paths no `data.yaml` devem ser absolutos. O notebook `01_treino.ipynb` corrige isso automaticamente na célula 3.
