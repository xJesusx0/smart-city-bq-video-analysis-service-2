import json

# --- Configuración centralizada ---
TIME_RANGES = [
    (0, 5, 10),
    (6, 10, 15),
    (11, float('inf'), 20),
]
MAX_GREEN_TIME = 25
YELLOW_TIME = 4
ALL_RED_TIME = 2
ICT_BONUS_MIN = 3
ICT_BONUS_MAX = 5


def EnCola(sensor_activo: bool) -> bool:
    """Detecta si hay vehículo detenido en cola (sensor trasero)."""
    return sensor_activo


def calculate_ict(vehicle_count: int, avg_speed_ms: float,
                  capacity: int = 15) -> float:
    """
    Calcula el Índice de Congestión de Tráfico (ICT) entre 0 y 1.
    Combina densidad vehicular y velocidad media.
    """
    density = min(vehicle_count / capacity, 1.0)
    # A menor velocidad, mayor congestión
    speed_factor = max(0.0, 1.0 - (avg_speed_ms / 14.0))  # 14 m/s ~ 50 km/h
    return round((density + speed_factor) / 2, 2)


def calculate_green_time(vehicle_count: int, in_queue: bool = False,
                         avg_speed_ms: float = None) -> dict:
    """
    Calcula el tiempo de verde según:
    - Prioridad de cola (EnCola)
    - Modelo de rangos
    - Ajuste ICT si se provee velocidad
    """
    if vehicle_count < 0:
        raise ValueError("vehicle_count no puede ser negativo")

    if EnCola(in_queue):
        return {"green_time": MAX_GREEN_TIME, "ict": None, "model": "EnCola"}

    # Modelo de rangos
    base_time = TIME_RANGES[-1][2]  # default: máximo rango
    for low, high, time in TIME_RANGES:
        if low <= vehicle_count <= high:
            base_time = time
            break

    # Modelo ICT (si se provee velocidad)
    ict = None
    bonus = 0
    if avg_speed_ms is not None:
        ict = calculate_ict(vehicle_count, avg_speed_ms)
        if ict >= 0.7:  # umbral de congestión alta
            bonus = ICT_BONUS_MAX if ict >= 0.9 else ICT_BONUS_MIN

    final_time = min(base_time + bonus, MAX_GREEN_TIME)
    model = "ICT" if avg_speed_ms is not None else "Rangos"

    return {"green_time": final_time, "ict": ict, "model": model}


def traffic_light_algorithm(count_a: int, count_b: int,
                             in_queue_a: bool = False,
                             in_queue_b: bool = False,
                             speed_a: float = None,
                             speed_b: float = None) -> str:
    """Algoritmo principal para intersección de dos semáforos."""
    result_a = calculate_green_time(count_a, in_queue_a, speed_a)
    result_b = calculate_green_time(count_b, in_queue_b, speed_b)

    cycle_total = (result_a["green_time"] + YELLOW_TIME + ALL_RED_TIME +
                   result_b["green_time"] + YELLOW_TIME + ALL_RED_TIME)

    output = {
        "street_A": {
            "vehicles_detected": count_a,
            "in_queue": in_queue_a,
            **result_a,
            "yellow_time": YELLOW_TIME,
        },
        "street_B": {
            "vehicles_detected": count_b,
            "in_queue": in_queue_b,
            **result_b,
            "yellow_time": YELLOW_TIME,
        },
        "safety": {
            "all_red_time": ALL_RED_TIME,
        },
        "cycle_total_seconds": cycle_total,
    }

    return json.dumps(output, indent=4)