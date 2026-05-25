"""
app.py — Aplicação de Demonstração LEGO Detection
TP Inteligência Artificial 2025/2026

Corre com:
    streamlit run app.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from color_detector import detect_colors_batch, get_color_emoji, get_color_hex

# ─── Inventário das construções ───────────────────────────────────────────────

CONSTRUCTIONS = {
    "🐊 Crocodilo": {
        "angular_brick_1x1":      2,
        "brick_2x2":              1,
        "eye_piece":              2,
        "plate_2x3":              1,
        "plate_2x6":              1,
        "plate_2x8":              1,
        "roof_tile_1x1x2":        4,
        "roof_tile_1x2":          4,
        "roof_tile_1x3":          2,
        "round_plate_1x1":        6,
    },
    "🌸 Flor": {
        "brick_1x2":              4,
        "brick_slope_curved_1x2": 4,
        "nose_cone_1x1":          1,
        "plate_4x4":              1,
        "plate_cross_3x3":        1,
        "plate_knob_2x2":         1,
        "roof_tile_1x2":          4,
        "round_plate_1x1":        1,
   
    },
    "👻 Fantasma": {
        "brick_1x2":              8,
        "brick_arch_2x3":         1,
        "brick_knob_1x1":         2,
        "eye_piece":              2,
        "inverted_roof_tile_2x3": 2,
        "plate_2x4":              2,
        "plate_2x6":              1,
        "roof_tile_1x2x2":        2,
        "roof_tile_2x2":          5,
        "transparent_brick_1x2":  1,
    },
    "🏠 Moinho": {
        "brick_1x2":              13,
        "brick_2x2":              2,
        "brick_2x4":              2, 
        "brick_arch_2x3":         2,
        "brick_cross_2x2":        5,
        "door":                   1,
        "plate_1x3":              4,
        "plate_1x4":              1,
        "plate_8x16":             1,
        "propeller_4blades":      1,
        "roof_tile_1x2":          5,
        "roof_tile_2x1x2":        4,
        "roof_tile_2x4":          1,
        "round_brick_1x1":        2,
        "transparent_brick_1x2":  1,
        "wall_window_1x2x2":      1,
        "window_frame_1x2x2":     1,   
    },
}

# ─── Mapa de PDFs das construções ──────────────────────────────────────────────

CONSTRUCTIONS_PDFS = {
    "👻 Fantasma": "construcoes/fantasma_lego.pdf",
    "🏠 Moinho":   "construcoes/moinho_lego.pdf",
    "🌸 Flor":     "construcoes/manual_lego.pdf",
    "🐊 Crocodilo": "construcoes/manual_lego.pdf",
    
}

# ─── Página ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LEGO Detector",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Fonte e fundo geral */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header hero */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        color: white;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero p  { font-size: 1rem; opacity: 0.75; margin: 6px 0 0; }

    /* Cards de métricas */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-card .val  { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .stat-card .lbl  { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }

    /* Badge de cor */
    .color-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin: 2px;
    }

    /* Card de construção */
    .build-card {
        border-radius: 14px;
        padding: 20px;
        margin: 6px 0;
        border: 1.5px solid #e8ecf0;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }
    .build-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
    .build-card.can-build {
        border-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    }
    .build-card.partial { border-color: #f59e0b; background: #fffbeb; }
    .build-card.missing { border-color: #ef4444; background: #fef2f2; }

    .build-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
    .build-score { font-size: 2.5rem; font-weight: 800; line-height: 1; }

    /* Progress bar */
    .progress-wrap {
        background: #e5e7eb;
        border-radius: 999px;
        height: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }

    /* Tabela de inventário */
    .inv-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 3px 0;
        background: #f9fafb;
        border: 1px solid #f0f0f0;
    }
    .inv-row:hover { background: #f0f4ff; }
    .inv-piece { font-size: 0.85rem; font-weight: 500; color: #374151; }
    .inv-qty {
        background: #1a1a2e;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSlider > div > div {
        background: #334155 !important;
    }

    /* Separador */
    hr { border-color: #e8ecf0; margin: 20px 0; }

    /* Tag de info rápida */
    .tag {
        display: inline-block;
        background: #f0f4ff;
        color: #3b5bdb;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Funções ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


def get_pdf_data(pdf_path: str):
    """Carrega o PDF para ser servido pelo Streamlit."""
    full_path = Path("..") / pdf_path
    if full_path.exists():
        with open(full_path, "rb") as f:
            return f.read()
    return None


def find_models(base_dir: str = "..") -> dict:
    models = {}
    modelos_dir = Path(base_dir) / "modelos"
    if modelos_dir.exists():
        for pt in sorted(modelos_dir.rglob("*.pt")):
            label = f"{pt.parent.parent.name}/{pt.name}" if pt.name == "best.pt" else pt.name
            models[label] = str(pt)
    return models


def run_detection(model, image_bgr, conf, iou):
    results = model.predict(source=image_bgr, conf=conf, iou=iou, imgsz=640, verbose=False)
    result = results[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id":   int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {"x1": round(x1,2), "y1": round(y1,2),
                     "x2": round(x2,2), "y2": round(y2,2)},
        })
    return result, detections


def calculate_recommendation(counts: dict) -> list:
    results = []
    for name, required in CONSTRUCTIONS.items():
        have = need = 0
        missing = {}
        for piece, qty_needed in required.items():
            qty_have = counts.get(piece, 0)
            need += qty_needed
            have += min(qty_have, qty_needed)
            if qty_have < qty_needed:
                missing[piece] = qty_needed - qty_have
        score = int(100 * have / need) if need > 0 else 0
        results.append({"name": name, "score": score, "have": have,
                         "need": need, "missing": missing, "can_build": score == 100})
    return sorted(results, key=lambda x: x["score"], reverse=True)


def score_color(score):
    if score == 100: return "#10b981"
    if score >= 60:  return "#f59e0b"
    return "#ef4444"


def progress_bar(score, color):
    return f"""
    <div class="progress-wrap">
        <div class="progress-fill" style="width:{score}%; background:{color};"></div>
    </div>"""


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧱 LEGO Detector")
    st.markdown("<small>TP Inteligência Artificial 2025/2026</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🔧 Modelo")
    available_models = find_models()

    if not available_models:
        st.warning("Nenhum modelo encontrado.\nTreina primeiro no notebook.")
        model_path = None
    else:
        model_label = st.selectbox("", list(available_models.keys()), label_visibility="collapsed")
        model_path  = available_models[model_label]
        st.caption(f"`{Path(model_path).name}`")

    st.markdown("---")
    st.markdown("### ⚙️ Deteção")
    conf_threshold = st.slider("Confiança mínima", 0.05, 0.95, 0.25, 0.05,
                                help="Deteções abaixo deste valor são descartadas")
    iou_threshold  = st.slider("Limiar IoU (NMS)", 0.10, 0.90, 0.70, 0.05,
                                help="Controla sobreposição de bounding boxes")

    st.markdown("---")
    st.markdown("### 🎨 Cor")
    use_color = st.toggle("Detetar cor (OpenCV)", value=True,
                           help="Analisa o HSV de cada bounding box para identificar a cor da peça")

    st.markdown("---")
    st.markdown("### 📥 Entrada")
    input_mode = st.radio("", ["📁 Upload de imagem", "📷 Webcam"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🔍 Comparação")
    compare_mode = st.toggle("Comparar dois modelos", value=False)
    if compare_mode and len(available_models) >= 2:
        model_label_b = st.selectbox("Segundo modelo:", list(available_models.keys()),
                                      index=1, label_visibility="collapsed")
        model_path_b = available_models[model_label_b]
    else:
        model_path_b = None


# ─── HERO ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🧱 Sistema de Deteção de Peças LEGO</h1>
    <p>Deteta peças · Identifica cores · Conta inventário · Recomenda construções</p>
</div>
""", unsafe_allow_html=True)

if model_path is None:
    st.error("⚠️ Nenhum modelo encontrado. Treina o modelo no notebook `01_treino.ipynb`.")
    st.stop()

# Carregar modelo(s)
with st.spinner("A carregar modelo..."):
    model = load_model(model_path)
    model_b = load_model(model_path_b) if (compare_mode and model_path_b) else None

# ─── INPUT ────────────────────────────────────────────────────────────────────

image_bgr = None

if input_mode == "📁 Upload de imagem":
    uploaded = st.file_uploader("Carrega uma imagem com peças LEGO",
                                  type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
else:
    cam = st.camera_input("Captura uma foto com a webcam")
    if cam:
        arr = np.frombuffer(cam.read(), np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ─── PROCESSAMENTO ────────────────────────────────────────────────────────────

if image_bgr is not None:

    # ── Deteção modelo A ──────────────────────────────────────────────
    t0 = time.perf_counter()
    result_a, detections_a = run_detection(model, image_bgr, conf_threshold, iou_threshold)
    ms_a = (time.perf_counter() - t0) * 1000

    if use_color:
        detections_a = detect_colors_batch(image_bgr, detections_a)

    annotated_a = cv2.cvtColor(result_a.plot(conf=True, labels=True, line_width=2), cv2.COLOR_BGR2RGB)

    # ── Deteção modelo B (comparação) ─────────────────────────────────
    if compare_mode and model_b:
        t0 = time.perf_counter()
        result_b, detections_b = run_detection(model_b, image_bgr, conf_threshold, iou_threshold)
        ms_b = (time.perf_counter() - t0) * 1000
        if use_color:
            detections_b = detect_colors_batch(image_bgr, detections_b)
        annotated_b = cv2.cvtColor(result_b.plot(conf=True, labels=True, line_width=2), cv2.COLOR_BGR2RGB)
    else:
        detections_b = None

    # ── Contagens ─────────────────────────────────────────────────────
    counts_a = Counter(d["class_name"] for d in detections_a)

    # ── Stats rápidas ─────────────────────────────────────────────────
    n_classes = len(counts_a)
    n_colors  = len(set(d.get("color","?") for d in detections_a)) if use_color else "—"
    avg_conf  = round(np.mean([d["confidence"] for d in detections_a])*100, 1) if detections_a else 0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, len(detections_a), "Peças detetadas"),
        (c2, n_classes,         "Tipos únicos"),
        (c3, n_colors,          "Cores distintas"),
        (c4, f"{ms_a:.0f} ms",  "Tempo inferência"),
    ]:
        col.markdown(f"""
        <div class="stat-card">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Imagem(ns) ────────────────────────────────────────────────────
    if compare_mode and model_b:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Modelo A** — `{Path(model_path).name}`")
            st.image(annotated_a, use_container_width=True)
            st.caption(f"{len(detections_a)} deteções · {ms_a:.0f} ms")
        with col_b:
            st.markdown(f"**Modelo B** — `{Path(model_path_b).name}`")
            st.image(annotated_b, use_container_width=True)
            st.caption(f"{len(detections_b)} deteções · {ms_b:.0f} ms")
    else:
        col_img, col_inv = st.columns([3, 2])

        with col_img:
            st.markdown("#### 🖼️ Resultado")
            st.image(annotated_a, use_container_width=True)
            st.caption(f"conf ≥ {conf_threshold} · IoU {iou_threshold} · {ms_a:.0f} ms")

        with col_inv:
            st.markdown(f"#### 📦 Inventário ({n_classes} tipos)")
            if not counts_a:
                st.info("Nenhuma peça detetada. Tenta reduzir a confiança mínima.")
            else:
                # Agrupar cores por peça
                color_map = {}
                if use_color:
                    for d in detections_a:
                        name = d["class_name"]
                        color = d.get("color", "desconhecida")
                        color_map.setdefault(name, Counter())[color] += 1

                for piece, qty in sorted(counts_a.items(), key=lambda x: -x[1]):
                    colors_html = ""
                    if use_color and piece in color_map:
                        for cor, cnt in color_map[piece].most_common(3):
                            hex_c = get_color_hex(cor)
                            emoji = get_color_emoji(cor)
                            colors_html += f'<span class="color-badge" style="background:{hex_c}">{emoji} {cor}</span>'

                    st.markdown(f"""
                    <div class="inv-row">
                        <div>
                            <div class="inv-piece">{piece}</div>
                            <div style="margin-top:3px">{colors_html}</div>
                        </div>
                        <span class="inv-qty">{qty}×</span>
                    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Construções ───────────────────────────────────────────────────
    st.markdown("#### 🏗️ Construções Possíveis")
    recommendations = calculate_recommendation(dict(counts_a))

    cols_build = st.columns(len(CONSTRUCTIONS))
    for i, rec in enumerate(recommendations):
        score  = rec["score"]
        color  = score_color(score)
        status = "can-build" if score == 100 else ("partial" if score >= 50 else "missing")
        badge  = "✅ Pronto!" if score == 100 else (f"🔧 Faltam {len(rec['missing'])} tipo(s)" if rec["missing"] else "")

        with cols_build[i]:
            missing_html = ""
            if rec["missing"]:
                for piece, qty in list(rec["missing"].items())[:4]:
                    missing_html += f"<div style='font-size:0.75rem;color:#6b7280;margin:2px 0'>• {piece}: <b>+{qty}</b></div>"
                if len(rec["missing"]) > 4:
                    missing_html += f"<div style='font-size:0.72rem;color:#9ca3af'>...e mais {len(rec['missing'])-4}</div>"

            st.markdown(f"""
            <div class="build-card {status}">
                <div class="build-title">{rec['name']}</div>
                <div class="build-score" style="color:{color}">{score}%</div>
                {progress_bar(score, color)}
                <div style="font-size:0.8rem;color:#6b7280">{rec['have']}/{rec['need']} peças</div>
                <div style="font-size:0.82rem;font-weight:600;margin-top:6px;color:{color}">{badge}</div>
                {missing_html}
            </div>""", unsafe_allow_html=True)
            
            # Botão para abrir o PDF (se existir)
            if rec["name"] in CONSTRUCTIONS_PDFS:
                pdf_path = CONSTRUCTIONS_PDFS[rec["name"]]
                pdf_data = get_pdf_data(pdf_path)
                if pdf_data:
                    st.download_button(
                        label="📥 Abrir Instruções (PDF)",
                        data=pdf_data,
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                        key=f"pdf_{rec['name']}"
                    )

    st.markdown("---")

    # ── Exportações ───────────────────────────────────────────────────
    st.markdown("#### ⬇️ Exportar")
    col_j, col_i, col_csv, _ = st.columns([1, 1, 1, 2])

    export_data = {
        "model": model_path,
        "conf_threshold": conf_threshold,
        "iou_threshold":  iou_threshold,
        "inference_ms":   round(ms_a, 2),
        "total_detections": len(detections_a),
        "counts_by_class":  dict(counts_a),
        "detections": detections_a,
        "recommendations": [
            {"construction": r["name"], "score": r["score"],
             "can_build": r["can_build"], "missing": r["missing"]}
            for r in recommendations
        ],
    }

    with col_j:
        st.download_button("📄 JSON", data=json.dumps(export_data, indent=2),
                            file_name="lego_result.json", mime="application/json")

    with col_i:
        _, img_enc = cv2.imencode(".jpg", result_a.plot(conf=True, labels=True, line_width=2))
        st.download_button("🖼️ Imagem", data=img_enc.tobytes(),
                            file_name="lego_annotated.jpg", mime="image/jpeg")

    with col_csv:
        csv_lines = ["peça,quantidade,cor_dominante"]
        color_map2 = {}
        if use_color:
            for d in detections_a:
                color_map2.setdefault(d["class_name"], Counter())[d.get("color","?")] += 1
        for piece, qty in sorted(counts_a.items()):
            dom_color = color_map2[piece].most_common(1)[0][0] if piece in color_map2 else "—"
            csv_lines.append(f"{piece},{qty},{dom_color}")
        st.download_button("📊 CSV", data="\n".join(csv_lines),
                            file_name="lego_inventario.csv", mime="text/csv")

    st.markdown("---")

    # ── Histórico ─────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history = st.session_state.history[-9:]
    st.session_state.history.append({
        "n": len(detections_a),
        "tipos": n_classes,
        "top": recommendations[0]["name"] if recommendations else "—",
        "score": recommendations[0]["score"] if recommendations else 0,
        "ms": round(ms_a, 1),
    })

    with st.expander(f"🕓 Histórico de inferências ({len(st.session_state.history)})"):
        for i, h in enumerate(reversed(st.session_state.history)):
            rank = len(st.session_state.history) - i
            c = score_color(h["score"])
            st.markdown(
                f"**#{rank}** &nbsp;|&nbsp; {h['n']} peças &nbsp;|&nbsp; "
                f"{h['tipos']} tipos &nbsp;|&nbsp; "
                f"<span style='color:{c};font-weight:700'>{h['top']} {h['score']}%</span>"
                f" &nbsp;|&nbsp; {h['ms']} ms",
                unsafe_allow_html=True
            )

# ─── ESTADO INICIAL (sem imagem) ─────────────────────────────────────────────

else:
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px; color:#6b7280;">
        <div style="font-size:4rem">📷</div>
        <div style="font-size:1.1rem; margin-top:8px">
            Carrega uma imagem ou usa a webcam para começar
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏗️ Construções suportadas")
    cols = st.columns(len(CONSTRUCTIONS))
    for i, (name, parts) in enumerate(CONSTRUCTIONS.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="build-card partial" style="min-height:180px">
                <div class="build-title">{name}</div>
                <div style="font-size:0.8rem;color:#6b7280;margin-top:8px">
                    {len(parts)} tipos de peças necessários
                </div>
                <div style="margin-top:10px">
                    {''.join(f'<div style="font-size:0.78rem;color:#374151;margin:2px 0">• {p}: {q}×</div>' for p,q in list(parts.items())[:5])}
                    {'<div style="font-size:0.72rem;color:#9ca3af">...</div>' if len(parts)>5 else ''}
                </div>
            </div>""", unsafe_allow_html=True)
            
            # Botão para abrir o PDF (se existir)
            if name in CONSTRUCTIONS_PDFS:
                pdf_path = CONSTRUCTIONS_PDFS[name]
                pdf_data = get_pdf_data(pdf_path)
                if pdf_data:
                    st.download_button(
                        label="📥 Abrir Instruções (PDF)",
                        data=pdf_data,
                        file_name=Path(pdf_path).name,
                        mime="application/pdf",
                        key=f"pdf_initial_{name}"
                    )

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center">
        <span class="tag">🎯 YOLOv8s</span>
        <span class="tag">40 classes</span>
        <span class="tag">🎨 Deteção de cor HSV</span>
        <span class="tag">🏗️ 4 construções</span>
        <span class="tag">📊 Export JSON / CSV</span>
    </div>
    """, unsafe_allow_html=True)
