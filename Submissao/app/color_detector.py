"""
color_detector.py — Deteção de cor de peças LEGO via OpenCV (HSV)
TP Inteligência Artificial 2025/2026

Recebe um crop (região BGR de uma bounding box) e devolve
o nome da cor dominante da peça LEGO.
"""

import cv2
import numpy as np

# ─── Definição dos ranges HSV por cor ────────────────────────────────────────
# Formato: (H_min, S_min, V_min), (H_max, S_max, V_max)
# O OpenCV usa H: 0-179, S: 0-255, V: 0-255

COLOR_RANGES = {
    "vermelho": [
        ((0,   120, 70),  (10,  255, 255)),   # vermelho baixo
        ((170, 120, 70),  (179, 255, 255)),    # vermelho alto (wrap-around)
    ],
    "verde": [
        ((35,  60,  40),  (85,  255, 255)),
    ],
    "amarelo": [
        ((20,  100, 100), (35,  255, 255)),
    ],
    "azul": [
        ((100, 80,  50),  (130, 255, 255)),
    ],
    "laranja": [
        ((10,  120, 100), (20,  255, 255)),
    ],
    "rosa": [
        ((140, 50,  100), (170, 255, 255)),
    ],
    "castanho": [
        ((5,   50,  30),  (20,  200, 150)),
    ],
    "preto": [
        ((0,   0,   0),   (179, 255, 50)),
    ],
    "branco": [
        ((0,   0,   180), (179, 60,  255)),
    ],
    "transparente": [
        ((0,   0,   200), (179, 30,  255)),
    ],
    "cinzento": [
        ((0,   0,   60),  (179, 40,  180)),
    ],
}

# Emoji por cor para a UI
COLOR_EMOJI = {
    "vermelho":     "🔴",
    "verde":        "🟢",
    "amarelo":      "🟡",
    "azul":         "🔵",
    "laranja":      "🟠",
    "rosa":         "🩷",
    "castanho":     "🟫",
    "preto":        "⚫",
    "branco":       "⚪",
    "transparente": "🔳",
    "cinzento":     "🩶",
    "desconhecida": "❓",
}

# Cor CSS aproximada por nome
COLOR_HEX = {
    "vermelho":     "#e74c3c",
    "verde":        "#2ecc71",
    "amarelo":      "#f1c40f",
    "azul":         "#3498db",
    "laranja":      "#e67e22",
    "rosa":         "#ff69b4",
    "castanho":     "#8B4513",
    "preto":        "#2c2c2c",
    "branco":       "#f5f5f5",
    "transparente": "#b0c4de",
    "cinzento":     "#95a5a6",
    "desconhecida": "#cccccc",
}


def detect_color(crop_bgr: np.ndarray) -> str:
    """
    Recebe um crop BGR (região da bounding box) e devolve
    o nome da cor dominante da peça LEGO.

    Estratégia:
    1. Converte para HSV
    2. Remove pixels de fundo (bordas da bbox) usando erosão
    3. Conta pixels que correspondem a cada range de cor
    4. Devolve a cor com mais pixels
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "desconhecida"

    h, w = crop_bgr.shape[:2]
    if h < 5 or w < 5:
        return "desconhecida"

    # Erosão para remover bordas/fundo
    kernel = np.ones((max(1, h // 6), max(1, w // 6)), np.uint8)
    crop_eroded = cv2.erode(crop_bgr, kernel, iterations=1)

    # Converter para HSV
    hsv = cv2.cvtColor(crop_eroded, cv2.COLOR_BGR2HSV)

    # Contar pixels por cor
    scores = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
        for (lo, hi) in ranges:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo_arr, hi_arr))
        scores[color_name] = int(np.sum(mask > 0))

    if not scores or max(scores.values()) == 0:
        return "desconhecida"

    # A cor com mais pixels ganha
    best_color = max(scores, key=scores.get)

    # Só aceita se cobrir pelo menos 10% dos pixels
    if scores[best_color] < total_pixels * 0.10:
        return "desconhecida"

    return best_color


def detect_colors_batch(image_bgr: np.ndarray, detections: list) -> list:
    """
    Recebe a imagem completa e a lista de deteções (com bbox),
    e devolve a lista atualizada com o campo 'color' em cada deteção.
    """
    updated = []
    for det in detections:
        bbox = det.get("bbox", {})
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", 0))
        y2 = int(bbox.get("y2", 0))

        # Garantir limites válidos
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_bgr.shape[1], x2)
        y2 = min(image_bgr.shape[0], y2)

        crop = image_bgr[y1:y2, x1:x2]
        color = detect_color(crop)

        updated.append({**det, "color": color})

    return updated


def get_color_emoji(color_name: str) -> str:
    return COLOR_EMOJI.get(color_name, "❓")


def get_color_hex(color_name: str) -> str:
    return COLOR_HEX.get(color_name, "#cccccc")
