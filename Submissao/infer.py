"""
infer.py — Script de inferência YOLOv8 para Deteção de Peças LEGO
TP Inteligência Artificial 2025/2026

Uso:
    python infer.py --model modelos/lego_v3_s/weights/best.pt --image caminho/para/imagem.jpg
    python infer.py --model modelos/lego_v3_s/weights/best.pt --folder caminho/para/pasta/
    python infer.py --model modelos/lego_v3_s/weights/best.pt --image img.jpg --conf 0.4 --show
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ─── Argparse ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferência YOLOv8 — Deteção de Peças LEGO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",  required=True,
                        help="Caminho para o ficheiro .pt do modelo treinado")
    parser.add_argument("--image",  default=None,
                        help="Caminho para uma imagem de entrada")
    parser.add_argument("--folder", default=None,
                        help="Pasta com imagens de entrada (processa todas)")
    parser.add_argument("--output", default="output",
                        help="Pasta de saída (default: output/)")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Limiar de confiança (default: 0.25)")
    parser.add_argument("--iou",    type=float, default=0.7,
                        help="Limiar IoU para NMS (default: 0.7)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="Tamanho da imagem de inferência (default: 640)")
    parser.add_argument("--show",   action="store_true",
                        help="Mostrar resultado em janela (requer display)")
    parser.add_argument("--no-json", action="store_true",
                        help="Não guardar ficheiros JSON de resultados")
    return parser.parse_args()


# ─── Inferência ───────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def run_inference(model, image_path: Path, output_dir: Path,
                  conf: float, iou: float, imgsz: int,
                  show: bool, save_json: bool):

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    result = results[0]

    # ── Construir lista de deteções ──────────────────────────────────
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id":   int(box.cls),
            "class_name": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width":  round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            },
        })

    # ── Guardar imagem anotada ───────────────────────────────────────
    annotated = result.plot(conf=True, labels=True, boxes=True, line_width=2)
    out_img = output_dir / image_path.name
    cv2.imwrite(str(out_img), annotated)

    # ── Guardar JSON ─────────────────────────────────────────────────
    if save_json:
        json_path = output_dir / f"{image_path.stem}.json"
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)

    # ── Mostrar na janela ────────────────────────────────────────────
    if show:
        cv2.imshow(f"LEGO Detection — {image_path.name}", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── Resumo no terminal ───────────────────────────────────────────
    print(f"  [{image_path.name}] → {len(detections)} deteção(ões)")
    if detections:
        # Contagem por classe
        from collections import Counter
        counts = Counter(d["class_name"] for d in detections)
        for cls, cnt in sorted(counts.items()):
            print(f"    • {cls}: {cnt}x")

    return detections


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Verificar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        sys.exit(1)

    print(f"🔧 A carregar modelo: {model_path}")
    model = YOLO(str(model_path))
    print(f"✅ Modelo carregado | {len(model.names)} classes")

    # Pasta de saída
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recolher imagens
    if args.image:
        images = [Path(args.image)]
    elif args.folder:
        folder = Path(args.folder)
        images = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
        images.sort()
    else:
        print("❌ Indica --image ou --folder")
        sys.exit(1)

    if not images:
        print("⚠️  Nenhuma imagem encontrada.")
        sys.exit(0)

    print(f"\n🔍 A processar {len(images)} imagem(ns)...\n")

    all_detections = {}
    for img_path in images:
        if not img_path.exists():
            print(f"  ⚠️  Ficheiro não encontrado: {img_path}")
            continue
        dets = run_inference(
            model=model,
            image_path=img_path,
            output_dir=output_dir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            show=args.show,
            save_json=not args.no_json,
        )
        all_detections[img_path.name] = dets

    # Resumo final
    total = sum(len(v) for v in all_detections.values())
    print(f"\n{'─'*50}")
    print(f"✅ Concluído! {len(all_detections)} imagem(ns) | {total} deteção(ões) total")
    print(f"📂 Resultados em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
