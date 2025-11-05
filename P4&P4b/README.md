Autores: Francesco Faustino Greco - Bianca Cocci  
**GRUPO 05**

# **VISIÓN POR COMPUTADOR - PRÁCTICAS (TARGA_lab)**

## Índice

- [Introducción](#introducción)
- [Descripción de las células](#descripción-de-las-células)
- [Fuentes y Documentación](#fuentes-y-documentación)

---

## Introducción

El objetivo general de este trabajo es desarrollar un sistema de **detección y reconocimiento automático de matrículas** en video utilizando **YOLOv11n** para la localización de vehículos y **EasyOCR** para la lectura del texto.  
A lo largo del notebook se combinan técnicas de visión por computador y reconocimiento óptico de caracteres con el fin de obtener un pipeline funcional de identificación visual.

---

## Descripción de las células
### Célula 1 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 2 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 3 - Apertura del video con OpenCV y verificación del archivo de entrada
Esta célula abre el archivo de video mediante OpenCV comprobando su disponibilidad y extrayendo información básica como la resolución y la tasa de fotogramas para preparar el procesamiento.

### Célula 4 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 5 - Guarda los resultados de detección y OCR en un archivo CSV
Guarda la información obtenida del proceso de detección y reconocimiento como coordenadas clases y texto en un archivo CSV estructurado para su posterior análisis.

### Célula 6 - Inicializa el lector OCR EasyOCR para leer matrículas
Aquí se inicializa el lector EasyOCR para reconocer texto en las regiones detectadas de las matrículas permitiendo la lectura automática de los caracteres alfanuméricos.

### Célula 7 - Mejora la región de la matrícula para reconocimiento aplicando CLAHE y binarización
Contiene funciones de preprocesamiento que mejoran la visibilidad de las matrículas aplicando técnicas como el ecualizado adaptativo de histograma CLAHE y la binarización para optimizar el reconocimiento OCR.

### Célula 8 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 9 - Organización del dataset copiando carpetas y dividiendo en train val y test
Organiza el conjunto de datos copiando las carpetas correspondientes y dividiéndolo en subconjuntos de entrenamiento validación y prueba según la estructura esperada por YOLO.

### Célula 10 - Creación del archivo YAML con rutas de dataset y nombres de clases para YOLO
Genera el archivo de configuración YAML que define las rutas de entrenamiento validación y prueba del dataset además de las clases que el modelo YOLO reconocerá durante su entrenamiento o validación.

### Célula 11 - Código auxiliar configuración o imports
Incluye las importaciones de librerías y configuraciones iniciales necesarias para la correcta ejecución del notebook asegurando que todas las dependencias estén disponibles.

### Célula 12 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 13 - Código auxiliar configuración o imports
Incluye las importaciones de librerías y configuraciones iniciales necesarias para la correcta ejecución del notebook asegurando que todas las dependencias estén disponibles.

### Célula 14 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 15 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 16 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 17 - Código auxiliar configuración o imports
Incluye las importaciones de librerías y configuraciones iniciales necesarias para la correcta ejecución del notebook asegurando que todas las dependencias estén disponibles.

### Célula 18 - Código auxiliar configuración o imports
Incluye las importaciones de librerías y configuraciones iniciales necesarias para la correcta ejecución del notebook asegurando que todas las dependencias estén disponibles.

### Célula 19 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 20 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

### Célula 21 - Código auxiliar configuración o imports
Incluye las importaciones de librerías y configuraciones iniciales necesarias para la correcta ejecución del notebook asegurando que todas las dependencias estén disponibles.

### Célula 22 - Carga o configura el modelo YOLOv11n para detección de vehículos
Esta célula carga el modelo YOLOv11n con los pesos preentrenados para detectar vehículos en los fotogramas del video aplicando umbrales de confianza adecuados y mostrando las detecciones visualmente.

---

## Fuentes y Documentación

- [Documentación oficial de OpenCV](https://docs.opencv.org/)
- [Repositorio Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [EasyOCR Documentation](https://www.jaided.ai/easyocr/)
- ChatGPT – Asistencia para redacción técnica y explicación de código
