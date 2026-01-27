# Video Analysis Service 2.0

Servicio de análisis de video basado en visión por computadora que utiliza YOLO (You Only Look Once) y OpenCV para detectar y clasificar objetos en tiempo real dentro de una zona de interés (ROI) específica.

## Características

- **Detección de Objetos en Tiempo Real**: Utiliza el modelo YOLO para una detección rápida y precisa.
- **Filtrado por Región de Interés (ROI)**: Solo procesa y analiza objetos que se encuentran dentro de un polígono definido por el usuario.
- **Clasificación Selectiva**: Configurado para detectar clases específicas relevantes para el análisis de tráfico y seguridad urbana:
  - Personas
  - Vehículos (Autos, Motos, Autobuses, Camiones, Bicicletas)
  - Elementos de tráfico (Semáforos, Señales de Alto)
- **Visualización**: Dibuja cajas delimitadoras y etiquetas de confianza sobre las detecciones válidas.

## Requisitos Previos

- Python 3.13 o superior
- Archivo de video de entrada (por defecto `videos/salida_720p.mp4`)
- Modelo YOLO (por defecto `yolo26n.pt`)

## Instalación

1.  **Instalar dependencias:**

    Este proyecto utiliza `pyproject.toml` para gestionar las dependencias.

    ```bash
    pip install .
    ```
    
    *Alternativamente, puedes instalar las librerías directamente:*
    
    ```bash
    pip install opencv-python ultralytics
    ```

## Uso

1.  Asegúrate de colocar tu video de entrada en la carpeta `videos/` o ajustar la ruta en `main.py` si es diferente a `videos/salida_720p.mp4`.
2.  Ejecuta el script principal:

    ```bash
    python main.py
    ```

3.  Se abrirá una ventana mostrando el video procesado. Presiona la tecla `q` para detener la ejecución y cerrar la ventana.

## Estructura del Proyecto

- `main.py`: Lógica principal de procesamiento, bucle de inferencia e interacción con la UI.
- `utils.py`: Utilidades auxiliares para cargar el modelo YOLO y redimensionar frames.
- `pyproject.toml`: Archivo de configuración que define las dependencias del proyecto.
- `videos/`: Directorio que contiene los videos de prueba.
