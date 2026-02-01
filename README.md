# Video Analysis Service 2.0

Sistema de análisis de video en tiempo real que utiliza arquitectura dual-YOLO para detectar accidentes de tráfico y objetos relevantes en zonas específicas de interés.

## 🎯 Objetivos

- **Detección de Accidentes**: Identificar escenas de accidentes de tráfico en tiempo real
- **Análisis Contextual**: Detectar y correlacionar objetos involucrados en accidentes
- **Monitoreo por Zonas**: Análisis específico en Regiones de Interés (ROIs) configurables
- **Reportes Automáticos**: Generación de estadísticas periódicas de tráfico y eventos

## 🚀 Características

### Arquitectura Dual-YOLO Secuencial
- **Modelo de Accidentes** (YOLOv8m): Especializado en detectar accidentes
- **Modelo COCO** (YOLOv8n): Detecta personas, carros, motos, buses y camiones
- **Correlación Espacial**: Identifica automáticamente objetos involucrados en accidentes
- **Procesamiento Optimizado**: Preprocesamiento compartido entre modelos

### Funcionalidades Principales
- ✅ Análisis en tiempo real de video/cámara
- ✅ Soporte para múltiples ROIs configurables
- ✅ Visualización diferenciada por tipo de detección
- ✅ Sistema de reportes automáticos con intervalos configurables
- ✅ Métricas de rendimiento en tiempo real (FPS, latencia)
- ✅ Filtrado inteligente: solo cuenta objetos dentro de ROIs

## 📋 Requisitos

- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- NumPy (`numpy`)

## 🔧 Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd video-analysis-service-2.0

# Instalar dependencias con uv (recomendado)
uv sync

# O con pip
pip install opencv-python ultralytics numpy
```

## 🎮 Uso

### Ejecución Básica

```bash
# Con uv
uv run main.py

# O directamente
python main.py
```

### Configurar Fuente de Video

Edita `main.py` línea 65:

```python
# Cámara web
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Archivo de video
cap = cv2.VideoCapture('videos/tu_video.mp4')
```

### Controles
- **Q**: Salir de la aplicación

## 🏗️ Arquitectura

### Flujo de Procesamiento

```
Frame de Video
    ↓
Preprocesamiento
    ↓
YOLO Accidents
    ↓
YOLO COCO
    ↓
Correlación Espacial
    ↓
Filtrado por ROIs
    ↓
Visualización + Reportes
```

### Componentes

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Script principal, configuración de ROIs y visualización |
| `sequential_inference.py` | Motor de inferencia dual-YOLO |
| `utils.py` | Carga de modelos y utilidades |
| `services/exporter.py` | Exportación de reportes |
| `services/reporter.py` | Gestión de reportes periódicos |

## ⚙️ Configuración

### Parámetros del Sistema

En `main.py` líneas 54-60:

```python
dual_yolo = SequentialDualYOLO(
    model_accidents=models['accidents'],
    model_coco=models['coco'],
    accident_confidence=0.2,      # Umbral de confianza para accidentes (0.0-1.0)
    object_confidence=0.4,        # Umbral de confianza para objetos (0.0-1.0)
    correlation_distance=100.0    # Distancia máxima (píxeles) para correlación
)
```

### Regiones de Interés (ROIs)

Define zonas personalizadas en `main.py` líneas 23-46:

```python
rois = [
    {
        "name": "Zona A",
        "points": np.array([
            (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        ], np.int32),
        "color": (0, 0, 255),  # BGR: Rojo
        "counts": {}
    }
]
```

### Modelos

Configura modelos en `utils.py` líneas 22-27:

```python
model_accidents = YOLO('models/yolov8m-accidents.pt')
model_coco = YOLO('models/yolov8n.pt')
```

### Sistema de Reportes

Configura intervalo en `main.py` línea 70:

```python
reporter = ReportManager(exporter, interval_seconds=5.0)
```

## 🎨 Visualización

### Código de Colores
- 🔴 **Rojo grueso**: Accidentes detectados
- 🟢 **Verde**: Objetos normales dentro de ROIs
- 🟡 **Amarillo**: Objetos involucrados en accidentes

### Panel de Información
- **FPS**: Frames por segundo en tiempo real
- **Latency**: Tiempo de procesamiento por frame (ms)
- **Accidents**: Contador de accidentes detectados

### Estadísticas por ROI
Cada zona muestra:
- Nombre de la zona (color codificado)
- Contadores por tipo de objeto
- Accidentes detectados (texto rojo)

## 📊 Clases Detectadas

### Modelo de Accidentes
- `0`: Accident

### Modelo COCO
- `0`: person
- `2`: car
- `3`: motorcycle
- `5`: bus
- `7`: truck

## 📈 Rendimiento

### Métricas Esperadas

| Hardware | FPS | Latencia | Uso GPU |
|----------|-----|----------|---------|
| RTX 3060 | 18-22 | 45-55ms | 60-70% |
| RTX 4070 | 28-35 | 28-35ms | 50-60% |
| Jetson Orin | 12-15 | 65-80ms | 85-95% |

### Optimizaciones Sugeridas

1. **TensorRT**: Exportar modelos → 2-3x más rápido
2. **Arquitectura Paralela**: CUDA streams → reducir latencia 30-40%
3. **Frame Skipping**: Procesar 1 de cada N frames en tráfico normal
4. **Modelos Ligeros**: Usar YOLOv8n para ambos modelos

## 🧪 Testing

```bash
# Ejecutar tests
uv run test_sequential.py
```

## 📁 Estructura del Proyecto

```
video-analysis-service-2.0/
├── main.py                    # Script principal
├── sequential_inference.py    # Motor de inferencia dual
├── utils.py                   # Utilidades
├── test_sequential.py         # Tests
├── services/
│   ├── exporter.py           # Exportación de reportes
│   └── reporter.py           # Gestión de reportes
├── models/
│   ├── yolov8m-accidents.pt  # Modelo de accidentes
│   └── yolov8n.pt            # Modelo COCO
└── videos/                    # Videos de prueba
```

## 🐛 Troubleshooting

### Warning: "Cannot find font directory"
Es un warning de Qt/OpenCV. No afecta la funcionalidad, puede ignorarse.

### FPS Bajo
- Reducir resolución: Modificar `resize_frame()` en `utils.py`
- Usar modelos más ligeros: YOLOv8n en lugar de YOLOv8m
- Reducir número de ROIs
- Aumentar umbrales de confianza

### No Detecta Objetos
- Verificar que los objetos estén **dentro** de las ROIs definidas
- Reducir umbrales de confianza (`accident_confidence`, `object_confidence`)
- Verificar que los modelos estén cargados correctamente

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

[Especificar licencia]

## 📧 Contacto

[Tu información de contacto]

---

**Desarrollado con ❤️ usando Ultralytics YOLO y OpenCV**
