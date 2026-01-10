Autores: Francesco Faustino Greco – Bianca Cocci  
**GRUPO 05**

# VISIÓN POR COMPUTADOR – VIRTUAL PAINTER PRO

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00BACC?style=for-the-badge&logo=google&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Índice

- [Introducción](#introducción)
- [Tecnología y Detección (MediaPipe Hands)](#tecnología-y-detección-mediapipe-hands)
- [Gestos e Interacción](#gestos-e-interacción)
- [Algoritmos de Dibujo y Física](#algoritmos-de-dibujo-y-física)
- [Funcionalidades Avanzadas (Mandala y Estilos)](#funcionalidades-avanzadas-mandala-y-estilos)
- [Resultados y análisis](#resultados-y-análisis)
- [Fuentes y Documentación](#fuentes-y-documentación)

---

## Introducción

En esta práctica se ha desarrollado **"Virtual Painter Pro - Thesis Edition"**, una aplicación avanzada de visión por computador que convierte la webcam en un lienzo digital interactivo. El objetivo es permitir al usuario dibujar, pintar y controlar una interfaz gráfica compleja (UI) utilizando únicamente gestos manuales, eliminando la necesidad de periféricos físicos.

El sistema integra un motor de renderizado propio (`PainterEngine`) capaz de simular herramientas artísticas, gestionar capas de deshacer/rehacer y aplicar efectos matemáticos en tiempo real.

![WhatsApp Image 2026-01-10 at 09 02 51](https://github.com/user-attachments/assets/9f581892-db1b-4334-8caf-2ce744e664ba)

---

## Tecnología y Detección (MediaPipe Hands)

El núcleo del seguimiento se basa en **MediaPipe Hands**, que ofrece una detección robusta de 21 puntos clave (landmarks) de la mano en coordenadas 3D. A diferencia de los métodos de segmentación por color, esto permite dibujar en condiciones de luz variables y con fondos complejos.

Se ha implementado una clase envoltorio (`HandDetector`) que facilita:
- La extracción de coordenadas de los dedos.
- El cálculo de distancias euclidianas entre puntos.
- La detección de dedos levantados (`fingers_up`) para máquinas de estados.

### Código de inicialización

```python
class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.8, track_con=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mpDraw = mp.solutions.drawing_utils
```

![video](https://github.com/user-attachments/assets/ca7b5f6d-eae8-4757-823b-24c3567892aa)


## Gestos e Interacción

La aplicación utiliza una máquina de estados basada en la configuración de los dedos para alternar entre modos sin latencia. El sistema distingue tres acciones principales:

### 1. Modo Selección y Navegación (Hover)
* **Gesto:** Dedos **Índice y Corazón** levantados (formando una "V" o paz).
* **Comportamiento:** El cursor sigue la mano pero **no dibuja**.
* **Uso:** Este modo permite mover el puntero por la pantalla para seleccionar colores, cambiar herramientas en el menú lateral o pulsar botones (Undo, Redo, Save). Se visualiza un rectángulo de selección entre los dedos para indicar este estado.

### 2. Modo Dibujo (Drawing)
* **Gesto:** Solo el dedo **Índice** levantado.
* **Comportamiento:** El sistema activa el "pincel" y comienza a pintar en el lienzo en las coordenadas del índice.
* **Uso:** Crear trazos artísticos. Si la herramienta activa es la "Gomma", este gesto borra el contenido.

### 3. Control de Tamaño Dinámico (Pinch Gesture)
* **Gesto:** Acercar el **Pulgar** y el **Índice** (gesto de pellizco) mientras se mantienen levantados.
* **Comportamiento:** El sistema calcula la distancia euclidiana entre los puntos 4 (pulgar) y 8 (índice).
* **Lógica:**
    * Si la distancia es menor a un umbral (`PINCH_START`), se activa el modo de redimensionado.
    * Mover los dedos altera el tamaño del pincel (Brush Size) o de la goma en tiempo real.
    * Visualmente se muestra un círculo en la punta del dedo que crece o decrece para dar feedback al usuario.

 ![WhatsApp Image 2026-01-10 at 09 02 51](https://github.com/user-attachments/assets/7d25bf1c-8c47-42d8-978e-d00ac60a4bef)

---

## Algoritmos de Dibujo y Física

Para mejorar la experiencia de dibujo digital y que no se sienta "robótico", se han implementado algoritmos de física y procesamiento de imagen:

* **Interpolación y Suavizado (Smoothing):**
    El movimiento de la mano puede tener micro-temblores. Se aplica un factor de suavizado (`SMOOTHING_FACTOR = 0.20`) interpolando la posición actual con la anterior:
    $$x_{new} = x_{prev} \times (1 - \alpha) + x_{raw} \times \alpha$$
    Esto genera trazos curvos y naturales.

* **Color Dinámico (Velocity-Based):**
    El sistema calcula la velocidad del trazo ($v = d/t$). Si se activa el modo "DYN", la intensidad y saturación del color varían según la velocidad de la mano:
    -   *Movimiento Lento:* Trazos más oscuros y sutiles.
    -   *Movimiento Rápido:* Trazos vivos, brillantes y saturados.

* **Algoritmo de Relleno (Flood Fill):**
    Se utiliza una implementación optimizada de `cv2.floodFill` para colorear áreas cerradas dibujadas por el usuario, permitiendo pintar zonas grandes rápidamente.

---

## Funcionalidades Avanzadas (Mandala y Estilos)

El proyecto destaca por funcionalidades creativas complejas:

### 1. Modo Mandala (Simetría Radial)
Utilizando matemáticas matriciales, el trazo del usuario se replica y rota $N$ veces alrededor del centro del lienzo en tiempo real.
* **Espejo:** Soporta simetría de espejo (`MANDALA_MIRROR`), duplicando cada segmento invertido para crear patrones caleidoscópicos perfectos.
* **Configuración:** Número de secciones variable ($N=6, 8, 10, 12, 16$) ajustable desde la interfaz.

### 2. Estilos de Pincel (Brush Styles)
El motor de renderizado soporta múltiples estilos visuales programados:
* **SOLID:** Línea continua estándar (`cv2.line`).
* **NEON:** Línea con efecto de resplandor (superposición de línea gruesa desenfocada + línea fina brillante).
* **SPRAY:** Dispersión aleatoria de partículas (distribución normal gaussiana) alrededor del cursor.
* **DASH/DOTTED:** Patrones de líneas discontinuas calculados matemáticamente sobre la longitud del arco del trazo.
* **CHALK (Tiza):** Simulación de textura rugosa mediante ruido aleatorio y transparencia alpha.

---

## Resultados y análisis

El sistema "Virtual Painter Pro" demuestra ser una aplicación robusta y funcional:

* **Rendimiento:** Mantiene una tasa estable de ~30 FPS en hardware estándar, incluso calculando la geometría compleja del Mandala en tiempo real.
* **Precisión:** La combinación de MediaPipe con el algoritmo de suavizado propio permite escribir texto legible y realizar dibujos detallados.
* **Usabilidad (UX):** La interfaz gráfica (botones, paneles, feedback de texto) responde correctamente a la detección de colisiones, permitiendo un flujo de trabajo completo (Dibujar -> Deshacer -> Cambiar Estilo -> Guardar) sin tocar el teclado.

### Funcionalidades Extra Implementadas:
* **Grabación de vídeo:** Capacidad de grabar la sesión creativa en formato MP4.
* **Historial de Undo/Redo:** Pila de estados para deshacer y rehacer acciones (Max History = 10).
* **Guardado:** Exportación de obras en alta calidad PNG a la carpeta local `Artworks_Gallery`.

![WhatsApp Image 2026-01-10 at 09 02 45](https://github.com/user-attachments/assets/ae47011a-9639-4a1c-87e8-88468fb3041f)

---

## Fuentes y Documentación

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- **ChatGPT** – Asistencia en la optimización de algoritmos de geometría (Mandala) y depuración de NumPy.
