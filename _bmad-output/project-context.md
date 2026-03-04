---
project_name: 'video-analysis-service-2.0'
user_name: 'Jesus'
date: '2026-02-25'
sections_completed:
  ['technology_stack', 'language_rules', 'framework_rules', 'testing_rules', 'quality_rules', 'workflow_rules', 'anti_patterns']
status: 'complete'
rule_count: 14
optimized_for_llm: true
---

# Project Context for AI Agents

_This file contains critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that agents might otherwise miss._

---

## Technology Stack & Versions

- **Python**: `>=3.13`
- **FastAPI**: `>=0.115.0`
- **Uvicorn**: `>=0.34.0`
- **OpenCV (opencv-python)**: `>=4.13.0.90`
- **Ultralytics (YOLO)**: `>=8.4.7`
- **HuggingFace Hub**: `>=1.3.5`

### Language-Specific Rules

- **Tipado Fuerte**: Usa anotaciones de tipo rigurosas con el módulo `typing` para todas las firmas de métodos (e.g., `Dict[str, Any]`, `List`).
- **Estado Compartido Seguro**: Siempre utiliza `threading.Lock` para sincronizar y proteger cualquier variable global compartida entre la ejecución de FastAPI y el hilo del bucle de video subyacente.
- **Liberación de Recursos**: Utiliza siempre bloques `try...finally` asegurando que todos los descriptores de video (`cap.release()`) o memoria de imagen se limpien sin importar qué excepción ocurra.

### Framework-Specific Rules

- **Ejecución de FastAPI**: Siempre inicia la API de FastAPI en un hilo en segundo plano asíncrono (`threading.Thread`) separado del visor principal de video interactivo para asegurar que `cv2.imshow` y `cv2.waitKey` puedan operar en el hilo principal sin interrupciones.
- **Inferencia de Inserción Condicional**: Mantener el patrón de omitir procesamiento para frames intermedios y reutilizar los resultados anteriores en el dibujo del frame, con el fin de priorizar FPS en tiempo real durante inferencias YOLO costosas.
- **Transformaciones de ROI**: Cuando se redimensionan frames, las coordenadas de todos los ROIs asociados deben ser recalculadas proporcionalmente en el mismo ciclo. No desacoples la lógica visual de las restricciones impuestas por el redimensionamiento ("fast_mode").

### Testing Rules

- **Simulación de Modelos (Mocking)**: Si se implementan pruebas, NUNCA cargues los pesos reales de YOLO (`.pt`) en entornos CI/CD o pruebas unitarias. Mapea y simula (mock) `SequentialDualYOLO` devueviendo estructuras de cajas delimitadoras estáticas.

### Code Quality & Style Rules

- **Estructura Modular**: Mantén el aislamiento estricto de responsabilidades:
  - `core/`: Solo lógica pesada de ML (YOLO) o algoritmos base.
  - `services/`: Interfaces, exportadores, APIs (FastAPI) y notificaciones.
  - `utils/`: Operaciones puramente matemáticas o de procesamiento de imagen sin estado.
- **Diccionarios de Estado**: Al leer campos predefinidos de estado de ROIs, utiliza siempre `.get("key", default)` en lugar de indexación directa `["key"]` si el diccionario proviene del bucle de inferencia visual, para prevenir errores de concurrencia momentáneos o `KeyError`.
- **Documentación de Funciones**: Utiliza docstrings (formato estándar de Python) para cualquier nueva función en `services/` o `core/` explicando los parámetros y el valor de retorno.

### Development Workflow Rules

- **Control de Cambios en Modelos**: Los cambios en los umbrales de confianza (`accident_confidence`, `object_confidence`) o lógica de inferencia deben ser testeados con videos locales antes de hacer commits, ya que pueden afectar drásticamente los falsos positivos en el sistema de notificaciones.
- **Gestión de Configuración**: Nuevos parámetros de la API o del procesamiento de video DEBEN añadirse primero analizando cómo se agrupan en `config.py` y el diccioario resultante, en lugar de cablear cadenas mágicas en los módulos (`main.py` o `api.py`).

### Critical Don't-Miss Rules

- 🚫 **BLOQUEO DE HILOS (Threading Block)**: ¡CRÍTICO! NUNCA agregues operaciones síncronas de red, IO pesado, o retardos (`time.sleep()`) directamente dentro del bucle `while cap.isOpened():` en `main.py`. Esto pausará el renderizado de los frames y crasheará visualmente la aplicación. Usa hilos separados en `services/` (p.ej., como actúa `start_api_server`).
- 🚫 **Mutación Directa del Estado UI**: Nunca modifiques el diccionario `rois` o las listas de `results` utilizadas por `_draw_results` o `_draw_ui` desde hilos asíncronos (como el de la API u otros servicios). Solo el hilo que procesa los frames debe ser el que crea o muta los conteos mostrados en la interfaz.
- ⚡ **Rendimiento de OpenCV**: Abstente de realizar re-asignaciones masivas de arrays de `numpy` dentro del bucle de video para no saturar el Garbage Collector en cada frame. Reutiliza componentes visuales donde sea apropiado.

---

## Usage Guidelines

**For AI Agents:**

- Read this file before implementing any code
- Follow ALL rules exactly as documented
- When in doubt, prefer the more restrictive option
- Update this file if new patterns emerge

**For Humans:**

- Keep this file lean and focused on agent needs
- Update when technology stack changes
- Review quarterly for outdated rules
- Remove rules that become obvious over time

Last Updated: 2026-02-25
