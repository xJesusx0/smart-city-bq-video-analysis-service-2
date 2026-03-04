---
stepsCompleted: [1, 2, 3, 4]
inputDocuments: ["_bmad-output/project-context.md"]
date: 2026-02-25
author: Jesus
---

# Product Brief: video-analysis-service-2.0

<!-- Content will be appended sequentially through collaborative workflow steps -->

## Executive Summary

video-analysis-service-2.0 es un servicio inteligente diseñado para revolucionar la gestión del tráfico urbano. Al procesar flujos de video en tiempo real de cámaras existentes, el sistema detecta vehículos, evalúa niveles de congestión y calcula tiempos semafóricos dinámicos óptimos. Esto permite a las ciudades y autoridades de movilidad evolucionar desde sistemas de semáforos rígidos hacia un control dinámico basado en la demanda real, reduciendo demoras y contaminación sin necesidad de inversiones masivas en nueva infraestructura de sensado de calle física.

---

## Core Vision

### Problem Statement

Las ciudades y sus ciudadanos sufren problemas graves de congestión vehicular debido a que la gran mayoría de la gestión actual del tráfico depende de semáforos con tiempos fijos o controlados centralmente de forma estática, siendo "ciegos" a las fluctuaciones en la demanda real que ocurren en las intersecciones.

### Problem Impact

Si esta ineficiencia no se resuelve, aumentarán los tiempos de viaje de la población, el consumo inútil de combustible, el deterioro del medio ambiente por mayores emisiones y la gran pérdida de productividad. Los centros de control de tránsito, planificadores urbanos y autoridades de movilidad sienten fuertemente este dolor al verse incapaces de operar la vialidad de manera eficiente y sostenible.

### Why Existing Solutions Fall Short

Las soluciones tradicionales de ajustar tiempos provocan ineficiencias crónicas en horas de demanda fluctuante. Por su parte, otras alternativas que miden el terreno como sensores debajo del asfalto o radares de campo, tienen costos de instalación considerables y sólo analizan puntos asilados con conteos indirectos en vez de entender visualmente toda la intersección de manera global.

### Proposed Solution

Construiremos un potente motor de software enfocado en capturar información en tiempo real del flujo usando redes neuronales y visión artificial sobre las cámaras. Éste recopilará conteos exactos por carril y dirección, medirá los niveles de saturación y proveerá una API a que entregue alertas de tráfico anormal y sugerencias de cálculos para ajustar las fases semafóricas. La forma más simple e inicial de mostrar valor será lograr que una intersección crítica abandone los ciclos estáticos por los sugeridos de nuestra API para mitigar tiempos de espera.

### Key Differentiators

- **Eficiencia en Costos e Infraestructura:** Utiliza las cámaras que la ciudad ya posee (menor costo) ahorrando los millones que toman renovar asfaltos y sensores físicos.
- **Visión Holística:** Brinda una perspectiva global de la intersección. Captura datos veraces respecto al flujo y volumen vehicular, y no estimaciones indirectas.
- **Arquitectura Escalable y 100% Software.** Es una modernización a nivel datos. Como gran diferencial emotivo y pragmático: los tomadores de decisión percibirán una mejora instantánea en el tráfico sin cambiar obras civiles.

---

## Target Users

### Primary Users

**Sistema Controlador de Semáforos (La Máquina Client)**
- **Contexto y Rol:** Es el sistema de software/hardware existente encargado de conmutar las luces de los semáforos. Históricamente opera a ciegas ejecutando ciclos pre-programados.
- **Problema:** Su limitación técnica actual causa ineficiencias viales graves que no puede corregir por sí mismo de manera proactiva.
- **Visión de Éxito y Uso (V1):** Mediante una estrategia de "Polling", este sistema consulta la API de video-analysis-service-2.0 periódicamente (ej. cada 5-10 segundos) para obtener el conteo actualizado de vehículos, niveles de congestión local y sugerencias de tiempos de verde/rojo. 
- **Valor Obtenido:** Permite transformar una infraestructura sorda y ciega en un sistema reactivo y dinámico, utilizando datos que puede ingerir fácilmente sin cambiar su naturaleza operativa base.

### Secondary Users

**Carlos (Autoridad de Movilidad / Operador en Centro de Control)**
- **Contexto:** Supervisa decenas de intersecciones simultáneamente desde pantallas en un centro de comando.
- **Problema:** Actualmente, sufre estrés y frustración cuando se acumula congestión vehicular aguda, dado que requiere observar el problema en video y aplicar arreglos manuales reactivos lentos.
- **Visión de Éxito:** Recibir reportes de la API que corroboren que las intersecciones gestionan solas sus flujos.
- **Momento "¡Aha!":** Carlos nota una gran acumulación de vehículos en una cámara, se prepara mentalmente para intervenir el ciclo del semáforo manualmente, pero justo antes de actuar ve cómo, instantáneamente y de forma automática, el sistema de control (alimentado por nuestra API) ajusta el tiempo de luz verde, comenzando a disolver la congestión sin que él haya tenido que mover un solo dedo.

**María (Ciudadana y Conductora Diaria)**
- **Contexto:** Se traslada por vías urbanas para sus trayectos cotidianos.
- **Problema:** Padece estrés, pérdida de tiempo y gasto extra de gasolina por cruces vehiculares estáticamente sincronizados, forzándola a detenerse en calles cruzadas vacías.
- **Visión de Éxito (Beneficiaria indirecta):** Su tiempo de traslado se vuelve predecible y fluye constantemente. No "usa" la API, pero experimenta el resultado del producto como una mejora notable y silenciosa en su calidad de vida en la ciudad urbana.

### User Journey (Interacción y Adopción)

1. **Integración Inicial (Onboarding del Sistema):** El equipo técnico del centro de control configura su Sistema Controlador para hacer solicitudes REST (Polling) hacia la API de video-analysis-service-2.0 apuntando a una cámara específica.
2. **Uso Diario (Core Usage):** La API responde de inmediato con JSON estructurado detallando métricas extraídas de IA y tiempos recomendados. El Sistema Controlador decide adoptar esos parámetros dinámicos sobre-escribiendo los ciclos duros.
3. **Validación (Success Moment):** Los operadores como Carlos monitorizan los KPI del proceso y advierten la auto-corrección de saturaciones viales puntuales disminuyendo radicalmente su intervención manual, lo que les permite enfocarse sólo en crisis graves usando eventos en futuras fases.

---

## Success Metrics

### North Star Metric (V1)

**Precisión en la Generación de Tiempos Dinámicos:** Porcentaje (%) de ciclos donde el tiempo sugerido por la API responde de manera correcta y proporcional al nivel real de congestión vehicular detectado.

### User Success Metrics

Para asegurar que el Sistema Controlador (usuario primario) y Carlos en el centro de control (usuario secundario) obtengan valor continuo, mediremos:
- **Consistencia de Datos:** Cero caídas, bloqueos de hilos o periodos prolongados con datos nulos durante el "Polling".
- **Intervención Manual Reducida:** Reducción cualitativa en la necesidad de que el operador intervenga manualmente una intersección monitoreada por el sistema.

### Business Objectives (V1 Pilot)

El objetivo estratégico de esta primera versión es **Demostrar Viabilidad Técnica (PoC Funcional)** en un entorno acotado (una intersección piloto), validando que:
1. El modelo de IA detecta y cuenta vehículos con una desviación aceptable frente a la realidad.
2. El sistema expone la información e infiere sugerencias de manera suficientemente rápida (baja latencia) para ser consumida "en tiempo real" por hardware externo.
3. Se puede lograr un impacto tangible usando sólo infraestructura de cámaras pre-existente.

### Key Performance Indicators (KPIs)

1. **Precisión de Detección (%):** Cercanía del conteo digital vs conteo humano (Ground Truth).
2. **Latencia Promedio de la API (ms):** Tiempo total desde que el "Polling" hace la solicitud hasta que se devuelve el JSON (Target tbd, e.g. < 200ms).
3. **Disponibilidad del Servicio (Uptime %):** Porcentaje de tiempo que el servicio está arriba respondiendo a consultas (Target: 99.9% durante la fase piloto).

### Non-Goals (Out of Scope for V1)

Para mantener el enfoque exclusivamente en la fluidez de tiempos semafóricos básicos, explícitamente **NO** intentaremos:
- Identificar placas vehiculares (LPR/ALPR).
- Clasificar tipos de vehículos con alta granularidad o complejidad.
- Optimizar pasillos o redes completas de semáforos (sincronismos o "Green Waves").
- Medir impactos ambientales (CO2, contaminación acústica).
