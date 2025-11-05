<!-- @import "design/style.css" -->
Autores: Francesco Faustino Greco - Bianca Cocci  
**GRUPO 05**

# **VISIÓN POR COMPUTADOR - PRÁCTICAS (TARGA_lab)**

## Índice

- [Introducción](#introducción)
- [Bloques de código (explicados)](#bloques-de-código-explicados)
- [Pipeline YOLOv11n + EasyOCR](#pipeline-yolov11n--easyocr)
- [Creación de dataset y YAML](#creación-de-dataset-y-yaml)
- [Fuentes y Documentación](#fuentes-y-documentación)

---

## Introducción

Este repositorio contiene el notebook `TARGA_lab.ipynb` cuyo objetivo es **detectar vehículos y leer matrículas** en vídeo mediante **YOLOv11n** (modelo usado en el notebook) y **EasyOCR**. El README siguiente documenta, sección por sección, los bloques de código originales y su función.

---

## Bloques de código (explicados)

A continuación incluyo **cada bloque de código** tal como aparece en el notebook, seguido de una explicación breve.

---

### Célula 1 - Guarda los resultados de detección y OCR en un archivo CSV  
Esta célula define la estructura básica de carpetas del proyecto creando directorios para los datos, modelos y salidas. También inicializa las rutas para el video de entrada, el archivo de salida anotado y el CSV donde se guardarán las detecciones. Sirve como configuración inicial para asegurar que todo el entorno de trabajo esté correctamente preparado antes de ejecutar la detección.

```python
from pathlib import Path

BASE = Path(".").resolve()
DATA = BASE / "data"
MODELS = BASE / "models"
OUT = BASE / "outputs"

OUT.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

VIDEO_IN  = DATA / "input.mp4"
VIDEO_OUT = OUT / "video_annotato.mp4"
CSV_OUT   = OUT / "log_rilevazioni.csv"

print("BASE:", BASE)
print("DATA esiste?", DATA.exists(), "| MODELS?", MODELS.exists(), "| OUTPUTS?", OUT.exists())
print("Video atteso:", VIDEO_IN, "| esiste?", VIDEO_IN.exists())
```

- **Función:** código de apoyo (importaciones, utilidades, manejo de rutas o transformaciones).

---

### Célula 2 - Carga o configura el modelo YOLOv11n para detección de vehículos  
En esta célula se verifica la instalación de las librerías necesarias para el proyecto como Ultralytics, OpenCV y EasyOCR. Si alguna no está instalada, se descarga automáticamente mediante pip. De esta forma se garantiza que el entorno contenga todo lo necesario para ejecutar el modelo YOLOv11n y el sistema de reconocimiento óptico de caracteres.

```python
import sys, subprocess

def ensure(pkg, spec=None):
    try:
        __import__(pkg)
        print(f"✓ {pkg} già presente")
    except Exception:
        to_install = spec or pkg
        print(f"→ installo {to_install} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", to_install])
        __import__(pkg)
        print(f"✓ {pkg} installato")

ensure("ultralytics", "ultralytics==8.3.0")  # YOLO
ensure("cv2", "opencv-python")               # OpenCV
ensure("easyocr", "easyocr")                 # OCR

print("Librerie pronte ✅")
```

- **Función:** carga y/o configura el modelo YOLO (en este notebook: **YOLOv11n**).
- **Función:** inicializa el lector OCR (EasyOCR) para leer matrículas.

---

### Célula 3 - Apertura del video con OpenCV y verificación del archivo de entrada  
Aquí se abre el video de entrada utilizando OpenCV y se comprueba que el archivo exista y pueda ser leído correctamente. Se extrae un fotograma de prueba para confirmar que el flujo de lectura funciona y se imprimen sus dimensiones. Es una etapa de verificación antes de aplicar la detección.

```python
import cv2

cap = cv2.VideoCapture(str(VIDEO_IN))
if not cap.isOpened():
    raise RuntimeError("Non riesco ad aprire data/input.mp4 (controlla il file).")

ok, frame = cap.read()
cap.release()

print("Frame letto?", ok)
if ok:
    print("Dimensioni frame:", frame.shape)
```

- **Función:** abre/lee el vídeo de entrada con OpenCV.

---

### Célula 4 - Carga o configura el modelo YOLOv11n para detección de vehículos  

Esta célula carga el modelo YOLOv11n, define las clases de interés (vehículos y personas) y configura el umbral de confianza. A continuación, procesa el video cuadro a cuadro aplicando el modelo para detectar objetos y dibujar sus bounding boxes. Los resultados se guardan tanto en un video anotado como en un archivo CSV con la información de clase, confianza y coordenadas.

```python
from ultralytics import YOLO
import csv, time
import cv2

# classi COCO che ci interessano: person=0, car=2, motorcycle=3, bus=5, truck=7
TRACKED_CLASSES = [0, 2, 3, 5, 7]
CONF_OBJ = 0.25
COCO_NAMES = {0:"person", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}

print("Carico YOLO… (se serve scarica i pesi)")
obj_model = YOLO("yolo11n.pt")  # auto-download alla prima esecuzione

cap = cv2.VideoCapture(str(VIDEO_IN))
if not cap.isOpened():
    raise RuntimeError("Video non apribile")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(VIDEO_OUT), fourcc, fps, (w, h))

# CSV base (senza targhe per ora)
with open(CSV_OUT, "w", newline="", encoding="utf-8") as csv_f:
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame","cat","conf","track_id","x1","y1","x2","y2"])

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            res = obj_model.track(
                frame, persist=True, conf=CONF_OBJ, classes=TRACKED_CLASSES, verbose=False
            )

            if not res:
                writer.write(frame)
                continue

            r = res[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                writer.write(frame)
                continue

            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int)[0]
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                cls_id = int(b.cls.cpu().numpy()[0])
                conf   = float(b.conf.cpu().numpy()[0])
                tid    = int(b.id.cpu().numpy()[0]) if (b.id is not None) else -1
                cls_nm = COCO_NAMES.get(cls_id, f"c{cls_id}")

                # disegna riquadro e label
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                label = f"{cls_nm}#{tid if tid!=-1 else '-'} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(15,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                # salva riga CSV
                csv_w.writerow([frame_idx, cls_nm, f"{conf:.4f}", tid, x1,y1,x2,y2])

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

print("Creati:", VIDEO_OUT, "e", CSV_OUT)
```

- **Función:** carga y/o configura el modelo YOLO (en este notebook: **YOLOv11n**).
- **Función:** abre/lee el vídeo de entrada con OpenCV.
- **Función:** prepara la escritura del vídeo anotado de salida.

---

### Célula 5 - Guarda los resultados de detección y OCR en un archivo CSV  
Se permite al usuario elegir qué video utilizar para la detección estableciendo nuevas rutas de entrada y salida. Se renombran los archivos de salida para evitar sobrescribir resultados anteriores y se preparan los nombres de los ficheros donde se guardarán las detecciones y el video anotado con y sin OCR.

```python
# --- SCEGLI IL VIDEO DA USARE ORA ---
from pathlib import Path

# scrivi qui il nome preciso del file che hai messo in data/
VIDEO_FILE = "input2.mp4"   # <--- cambia qui se usi un altro nome

# NON toccare sotto
VIDEO_IN = DATA / VIDEO_FILE
assert VIDEO_IN.exists(), f"Non trovo {VIDEO_IN}. Controlla il nome in data/."

# creo nomi output che NON sovrascrivono quelli di prima
VIDEO_TAG = Path(VIDEO_FILE).stem  # es. 'whatsapp01'
VIDEO_OUT = OUT / f"{VIDEO_TAG}_annotato.mp4"
CSV_OUT   = OUT / f"{VIDEO_TAG}_detections.csv"

# (per la pipeline con OCR useremo questi)
VIDEO_OUT_LP = OUT / f"{VIDEO_TAG}_annotato_lp.mp4"
CSV_OUT_LP   = OUT / f"{VIDEO_TAG}_detections_lp.csv"

print("Userò questo input:", VIDEO_IN)
print("Salverò:", VIDEO_OUT)
print("        ", CSV_OUT)
print("        ", VIDEO_OUT_LP)
print("        ", CSV_OUT_LP)
```

- **Función:** código de apoyo (importaciones, utilidades, manejo de rutas o transformaciones).

---

### Célula 6 - Inicializa el lector OCR EasyOCR para leer matrículas  
En esta célula se inicializa el lector OCR EasyOCR probando primero con GPU y, si no está disponible, usando CPU. Esto permite que el sistema pueda leer caracteres alfanuméricos de las matrículas detectadas posteriormente en el video.

```python
import sys, subprocess

def _ensure(pkg, spec=None):
    try:
        __import__(pkg)
        print(f"✓ {pkg} ok")
    except Exception:
        to_install = spec or pkg
        print(f"→ installo {to_install} ... (attendi)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", to_install])
        __import__(pkg)
        print(f"✓ {pkg} installato")

_ensure("easyocr", "easyocr")
import easyocr

try:
    ocr = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR pronto (GPU)")
except Exception:
    ocr = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR pronto (CPU)")
```

- **Función:** inicializa el lector OCR (EasyOCR) para leer matrículas.

---

### Célula 7 - Mejora la región de la matrícula para reconocimiento aplicando CLAHE y binarización  
Contiene funciones auxiliares para mejorar la calidad de la región de la matrícula antes de aplicarle OCR. Se implementan procesos de realce como afilado, ecualización adaptativa (CLAHE) y umbralización adaptativa. Además, se define una función que busca posibles rectángulos con proporciones similares a una placa dentro del área del vehículo detectado.

```python
import cv2
import numpy as np

def normalize_plate_text(txt: str) -> str:
    keep = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(ch for ch in txt.upper() if ch in keep)

def enhance_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    # sharpen → grigio → CLAHE → soglia adattiva
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((10,10), dtype=np.uint8)
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sharp = cv2.filter2D(img_bgr, -1, k)
    gray  = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    eq    = clahe.apply(gray)
    th    = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    return th

def find_plate_candidate_rect(veh_roi_bgr: np.ndarray):
    # cerca rettangolo tipo targa nella ROI del veicolo
    if veh_roi_bgr is None or veh_roi_bgr.size == 0:
        return None
    gry = cv2.cvtColor(veh_roi_bgr, cv2.COLOR_BGR2GRAY)
    gry = cv2.bilateralFilter(gry, 7, 75, 75)
    th  = cv2.adaptiveThreshold(gry, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 7)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h == 0: 
            continue
        ratio = w / float(h)
        if 2.0 < ratio < 6.5 and w > 40 and h > 12:
            return (x, y, x+w, y+h)
    return None
```

- **Función:** preprocesamiento de la región de la matrícula para mejorar OCR (CLAHE, umbralización, etc.).
- **Función:** localizar candidato(s) de placa dentro del bounding box del vehículo.

---

### Célula 8 - Carga o configura el modelo YOLOv11n para detección de vehículos  
Esta célula combina detección y reconocimiento. Usa YOLOv11n para identificar los vehículos en cada frame y luego, para cada uno, aplica las funciones anteriores para localizar la matrícula, mejorarla y pasarla al lector OCR. El texto detectado se normaliza y se muestra sobre el video junto con su nivel de confianza. Finalmente, se guardan todos los datos en un nuevo CSV y un video anotado que incluye tanto los vehículos como las matrículas reconocidas.

```python
from ultralytics import YOLO
import csv, time, cv2
from pathlib import Path

# se stai usando un secondo video, assicurati che VIDEO_IN/VIDEO_OUT siano già settati
try:
    VIDEO_OUT_LP, CSV_OUT_LP
except NameError:
    VIDEO_TAG = Path(str(VIDEO_IN)).stem
    VIDEO_OUT_LP = OUT / f"{VIDEO_TAG}_annotato_lp.mp4"
    CSV_OUT_LP   = OUT / f"{VIDEO_TAG}_detections_lp.csv"

# riusa il modello generale; se non esiste in memoria, ricarico
try:
    obj_model
except NameError:
    obj_model = YOLO("yolo11n.pt")

TRACKED_CLASSES = [0, 2, 3, 5, 7]      # person/car/moto/bus/truck
COCO_NAMES = {0:"person", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
CONF_OBJ = 0.25

cap = cv2.VideoCapture(str(VIDEO_IN))
if not cap.isOpened():
    raise RuntimeError("Video non apribile")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(str(VIDEO_OUT_LP), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

with open(CSV_OUT_LP, "w", newline="", encoding="utf-8") as csv_f:
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "frame","cat","conf","track_id","x1","y1","x2","y2",
        "plate_text","plate_score","px1","py1","px2","py2"
    ])

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1

            res = obj_model.track(frame, persist=True, conf=CONF_OBJ, classes=TRACKED_CLASSES, verbose=False)
            if not res:
                writer.write(frame); continue

            r = res[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                writer.write(frame); continue

            for b in boxes:
                xyxy = b.xyxy.cpu().numpy().astype(int)[0]
                x1,y1,x2,y2 = [int(v) for v in xyxy]
                cls_id = int(b.cls.cpu().numpy()[0])
                conf   = float(b.conf.cpu().numpy()[0])
                tid    = int(b.id.cpu().numpy()[0]) if (b.id is not None) else -1
                cls_nm = COCO_NAMES.get(cls_id, f"c{cls_id}")

                # disegna oggetto
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{cls_nm}#{tid if tid!=-1 else '-'} {conf:.2f}",
                            (x1, max(15,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,0), 1, cv2.LINE_AA)

                # default valori targa vuoti
                plate_txt, plate_scr = "", 0.0
                px1=py1=px2=py2=None

                # OCR solo sui veicoli
                if cls_id in (2,3,5,7):
                    vx1, vy1 = max(0,x1), max(0,y1)
                    vx2, vy2 = min(w-1,x2), min(h-1,y2)
                    veh_roi = frame[vy1:vy2, vx1:vx2]

                    cand = find_plate_candidate_rect(veh_roi)
                    if cand:
                        cx1,cy1,cx2,cy2 = cand
                        cx1 = max(0, min(cx1, veh_roi.shape[1]-1))
                        cx2 = max(0, min(cx2, veh_roi.shape[1]-1))
                        cy1 = max(0, min(cy1, veh_roi.shape[0]-1))
                        cy2 = max(0, min(cy2, veh_roi.shape[0]-1))

                        plate_roi = veh_roi[cy1:cy2, cx1:cx2].copy()

                        # ingrandisco se piccola (aiuta OCR)
                        if plate_roi.size:
                            ph, pw = plate_roi.shape[:2]
                            scale = 3.0 if min(ph,pw) < 60 else 2.0
                            plate_roi = cv2.resize(plate_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                        prep = enhance_for_ocr(plate_roi)
                        ocr_out = ocr.readtext(prep, detail=1, paragraph=False,
                                               allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                        if ocr_out:
                            best_text, best_score = max(((t[1], t[2]) for t in ocr_out), key=lambda x: x[1])
                            plate_txt = normalize_plate_text(best_text)
                            plate_scr = float(best_score)

                            px1, py1 = vx1 + cx1, vy1 + cy1
                            px2, py2 = vx1 + cx2, vy1 + cy2
                            cv2.rectangle(frame, (px1,py1), (px2,py2), (0,255,255), 2)
                            if plate_txt:
                                cv2.putText(frame, f"{plate_txt} {plate_scr:.2f}",
                                            (px1, max(15,py1-7)), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.55, (0,255,255), 2, cv2.LINE_AA)

                csv_w.writerow([
                    frame_idx, cls_nm, f"{conf:.4f}", tid, x1,y1,x2,y2,
                    plate_txt, f"{plate_scr:.4f}",
                    "" if px1 is None else px1,
                    "" if py1 is None else py1,
                    "" if px2 is None else px2,
                    "" if py2 is None else py2
                ])

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

print("Creati:", VIDEO_OUT_LP, "e", CSV_OUT_LP)
```

- **Función:** carga y/o configura el modelo YOLO (en este notebook: **YOLOv11n**).
- **Función:** abre/lee el vídeo de entrada con OpenCV.
- **Función:** prepara la escritura del vídeo anotado de salida.
- **Función:** aplica OCR sobre la imagen de la matrícula procesada.
- **Función:** preprocesamiento de la región de la matrícula para mejorar OCR (CLAHE, umbralización, etc.).
- **Función:** localizar candidato(s) de placa dentro del bounding box del vehículo.

---

### Célula 9 - Descarga del dataset desde Google Drive  
Aquí se descarga automáticamente un dataset de entrenamiento de matrículas desde Google Drive utilizando gdown. Se crean las carpetas necesarias para las imágenes y etiquetas, y se almacenan localmente para su uso posterior en el entrenamiento o evaluación del modelo.

```python
import sys, subprocess
from pathlib import Path

# 1) aggiorna gdown per evitare bug vecchi
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"])

BASE = Path(".").resolve()
RAW_DIR = BASE / "data_lp_raw"
(RAW_DIR / "images").mkdir(parents=True, exist_ok=True)
(RAW_DIR / "labels").mkdir(parents=True, exist_ok=True)

IMAGES_ID = "1iIGD1MT5DEYIu49VBqp1ABf3NzCRkm5H"   # <-- ID cartella images (dal tuo log)
LABELS_ID = "1xMtoReO7tiBE2LA8fRBXvvnX4jPawjkp"   # <-- ID cartella labels (dal tuo log)

# 2) scarica solo images e labels (usiamo --remaining-ok per ignorare il limite)
print(">> Scarico IMAGES ...")
subprocess.check_call(["gdown", "--folder", "--id", IMAGES_ID, "-O", str(RAW_DIR / "images"), "--remaining-ok"])

print(">> Scarico LABELS ...")
subprocess.check_call(["gdown", "--folder", "--id", LABELS_ID, "-O", str(RAW_DIR / "labels"), "--remaining-ok"])

print(">> Download sottocartelle completato in:", RAW_DIR)
```

- **Función:** descarga dataset desde Google Drive (gdown).

---

### Célula 10 - Código auxiliar configuración o imports  
Esta célula verifica que el dataset descargado esté completo comparando las imágenes con sus archivos de etiquetas. Detecta posibles faltantes, muestra estadísticas del contenido y genera una lista de pares imagen–etiqueta válidos para los siguientes pasos de procesamiento.

```python
from pathlib import Path

RAW_IMG = Path("data_lp_raw/images")
RAW_LBL = Path("data_lp_raw/labels")

img_exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
imgs = sorted([p for p in RAW_IMG.rglob("*") if p.suffix.lower() in img_exts])
lbls = sorted([p for p in RAW_LBL.rglob("*.txt")])

print(f"Immagini trovate: {len(imgs)}")
print(f"Label trovate:    {len(lbls)}")

img_names = {p.stem for p in imgs}
lbl_names = {p.stem for p in lbls}

missing_lbl = sorted(img_names - lbl_names)
missing_img = sorted(lbl_names - img_names)

print(f"Immagini SENZA label: {len(missing_lbl)}")
print(f"Label SENZA immagine: {len(missing_img)}")

# useremo solo coppie immagine+label
paired = sorted(list(img_names & lbl_names))
print(f"Coppie immagine+label utilizzabili: {len(paired)}")
print("Esempi senza label:", missing_lbl[:5])
```

- **Función:** código de apoyo (importaciones, utilidades, manejo de rutas o transformaciones).

---

### Célula 11 - Carga o configura el modelo YOLOv11n para detección de vehículos  
En esta etapa se construye la estructura de carpetas del dataset en formato compatible con YOLO (train, val, test). Se divide el conjunto total de imágenes en proporciones adecuadas y se copian las correspondientes imágenes y etiquetas en cada subcarpeta. Esto asegura que el modelo pueda ser entrenado correctamente con los datos organizados.

```python
import shutil, random, math
from pathlib import Path

BASE = Path(".").resolve()
RAW_IMG = BASE / "data_lp_raw" / "images"
RAW_LBL = BASE / "data_lp_raw" / "labels"
DST = BASE / "data_lp"

# Crea struttura YOLO
for p in ["train/images","train/labels","val/images","val/labels","test/images","test/labels"]:
    (DST / p).mkdir(parents=True, exist_ok=True)

# Raccogli coppie immagine+label
img_exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
all_imgs = {p.stem: p for p in RAW_IMG.rglob("*") if p.suffix.lower() in img_exts}
all_lbls = {p.stem: p for p in RAW_LBL.rglob("*.txt")}

paired_stems = sorted(set(all_imgs) & set(all_lbls))
n = len(paired_stems)
print("Coppie utilizzabili:", n)
assert n >= 3, "Servono almeno 3 coppie per avere train/val/test non vuoti."

random.seed(42)
random.shuffle(paired_stems)

# 1) TEST ~40%, ma garantisci che restino almeno 2 per train+val
n_test = int(round(0.40 * n))
n_test = max(1, min(n-2, n_test))

# 2) Restanti per train+val
n_rest = n - n_test

# 3) VAL ~10% dei restanti, ma minimo 1
n_val = int(round(0.10 * n_rest))
n_val = max(1, min(n_rest-1, n_val))  # lascia almeno 1 per train

# 4) TRAIN = tutto il resto
n_train = n - n_test - n_val
assert n_train >= 1

# Assegna gli insiemi
train_stems = paired_stems[:n_train]
val_stems   = paired_stems[n_train:n_train+n_val]
test_stems  = paired_stems[n_train+n_val:]

def copy_pair(stem: str, split: str):
    im = all_imgs[stem]
    lb = all_lbls[stem]
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(im, DST / split / "images" / im.name)
    shutil.copy2(lb, DST / split / "labels" / lb.name)

for s in train_stems: copy_pair(s, "train")
for s in val_stems:   copy_pair(s, "val")
for s in test_stems:  copy_pair(s, "test")

print(">> FATTO. Struttura YOLO in:", DST.resolve())
print(f"Train: {len(train_stems)} | Val: {len(val_stems)} | Test: {len(test_stems)}")
print("\nEsempi split:")
print("  train:", train_stems[:5])
print("  val  :", val_stems[:5])
print("  test :", test_stems[:5])
```

- **Función:** carga y/o configura el modelo YOLO (en este notebook: **YOLOv11n**).

---

### Célula 12 - Creación del archivo YAML con rutas de dataset y nombres de clases para YOLO  
Genera el archivo data_lp.yaml que contiene las rutas del conjunto de entrenamiento, validación y prueba, además del número de clases y sus nombres. Este archivo es esencial para que YOLO pueda entender la estructura del dataset durante el entrenamiento.

```python
from pathlib import Path

yaml_text = """train: data_lp/train/images
val: data_lp/val/images
test: data_lp/test/images

nc: 1
names: [license_plate]
"""
Path("data_lp").mkdir(exist_ok=True, parents=True)
with open("data_lp/data_lp.yaml","w") as f:
    f.write(yaml_text)

print("Creato:", Path("data_lp/data_lp.yaml").resolve())
```

- **Función:** localizar candidato(s) de placa dentro del bounding box del vehículo.
- **Función:** crea archivo YAML para YOLO con rutas de train/val/test y clases.

---

### Célula 13 - Creación del archivo YAML con rutas de dataset y nombres de clases para YOLO  
Verifica la existencia del archivo YAML y de las carpetas del dataset dentro de la estructura del proyecto. También imprime las rutas detectadas asegurando que el entorno esté correctamente configurado para el entrenamiento con YOLO.

```python
from pathlib import Path

print("CWD:", Path(".").resolve())
print("Esiste 'data_lp'?", Path("data_lp").exists())
print("Esiste 'datasets/data_lp'?", Path("datasets/data_lp").exists())

# Mostra YAML che esistono ovunque nel progetto
cands = list(Path(".").rglob("data_lp.yaml"))
print("YAML trovati:", [str(p) for p in cands][:5])
```

- **Función:** crea archivo YAML para YOLO con rutas de train/val/test y clases.
- **Función:** organiza/copia el dataset en `datasets/data_lp` para entrenamiento.

---

### Célula 14 - Creación del archivo YAML con rutas de dataset y nombres de clases para YOLO  
Reescribe o actualiza el archivo YAML adaptando las rutas absolutas del sistema donde se ejecuta el notebook. De esta manera se evita cualquier error de ruta al entrenar el modelo y se garantiza la coherencia entre los directorios locales y los utilizados por YOLO.

```python
from pathlib import Path

# Trova la root del dataset
candidates = [Path("datasets/data_lp/train/images"), Path("data_lp/train/images")]
ROOT = None
for c in candidates:
    if c.exists():
        ROOT = c.parents[2]  # -> .../datasets/data_lp oppure .../data_lp
        break

if ROOT is None:
    raise FileNotFoundError("Non trovo né 'datasets/data_lp/train/images' né 'data_lp/train/images'. Controlla dove hai messo il dataset.")

yaml_path = ROOT / "data_lp.yaml"
yaml_text = f"""train: {(ROOT/'train/images').resolve()}
val: {(ROOT/'val/images').resolve()}
test: {(ROOT/'test/images').resolve()}

nc: 1
names: [license_plate]
"""
yaml_path.write_text(yaml_text, encoding="utf-8")

print("✅ YAML scritto in:", yaml_path)
print("train:", (ROOT/'train/images').resolve())
print("val  :", (ROOT/'val/images').resolve())
print("test :", (ROOT/'test/images').resolve())
```

- **Función:** localizar candidato(s) de placa dentro del bounding box del vehículo.
- **Función:** crea archivo YAML para YOLO con rutas de train/val/test y clases.
- **Función:** organiza/copia el dataset en `datasets/data_lp` para entrenamiento.

---

### Célula 15 - Carga o configura el modelo YOLOv11n para detección de vehículos  
Finalmente, copia toda la estructura del dataset ya dividida a la carpeta datasets/data_lp, que es donde YOLO espera encontrar los datos para su entrenamiento. Esta célula asegura que la organización final del dataset sea la correcta y que el modelo pueda acceder a todas las imágenes y etiquetas sin errores.

```python
import shutil
from pathlib import Path

# cartelle di origine e destinazione
src = Path("data_lp")                     # dove avevi creato lo split
dst = Path("datasets/data_lp")            # dove YOLO si aspetta i dati

if not src.exists():
    raise FileNotFoundError("Non trovo 'data_lp' fuori da datasets: apri il Finder e controlla che ci sia 'Documents/targa_lab/data_lp'.")

# assicura che le sottocartelle ci siano
for split in ["train", "val", "test"]:
    (dst / split / "images").mkdir(parents=True, exist_ok=True)
    (dst / split / "labels").mkdir(parents=True, exist_ok=True)

# copia tutto lo split già fatto
for split in ["train", "val", "test"]:
    for img in (src / split / "images").glob("*"):
        shutil.copy2(img, dst / split / "images" / img.name)
    for lbl in (src / split / "labels").glob("*"):
        shutil.copy2(lbl, dst / split / "labels" / lbl.name)

print("✅ Copiati tutti i file dallo split 'data_lp/' a 'datasets/data_lp/'")
```

- **Función:** carga y/o configura el modelo YOLO (en este notebook: **YOLOv11n**).
- **Función:** organiza/copia el dataset en `datasets/data_lp` para entrenamiento.

---

## Pipeline YOLOv11n + EasyOCR

Resumen del flujo de procesamiento usado en el notebook:

1. Se crea/asegura la estructura de carpetas.
2. Se instalan (si es necesario) y cargan las librerías: `ultralytics` (YOLOv11n), `opencv-python`, `easyocr`.
3. Se abre el vídeo con OpenCV y se configura un `VideoWriter` para salida.
4. Se carga el modelo `yolov11n.pt` y se filtran las clases objetivo (car, truck, bus, motorbike, ...).
5. Para cada frame:
   - se realiza la detección con YOLO;
   - para cada bounding box de vehículo, se busca la región de la matrícula;
   - se aplica `enhance_for_ocr` y luego `reader.readtext()` para extraer el texto;
   - se normaliza el texto con `normalize_plate_text` y se escribe en CSV y vídeo de salida.

---

## Creación de dataset y YAML

El notebook incluye bloques para descargar un dataset de matrículas, comprobar la correspondencia imagen/label, dividir en `train/val/test`, y generar un archivo `data_lp.yaml` con la configuración para reentrenar YOLO (nc=1, names: [license_plate]).

Código clave para crear el YAML (extraído del notebook):

```python
# Ejemplo de creación de data_lp.yaml
cfg = {
    'train': str((BASE_DIR / 'data_lp' / 'train' / 'images').resolve()),
    'val': str((BASE_DIR / 'data_lp' / 'val' / 'images').resolve()),
    'test': str((BASE_DIR / 'data_lp' / 'test' / 'images').resolve()),
    'nc': 1,
    'names': ['license_plate']
}
with open('data_lp/data_lp.yaml', 'w') as f:
    import yaml
    yaml.dump(cfg, f)
```

---

## Fuentes y Documentación

- Documentación de Ultralytics / YOLO (modelo usado: **YOLOv11n**)
- EasyOCR
- OpenCV (cv2)
- ChatGPT — asistencia en redacción técnica
