import json

def calculate_green_time(vehicle_count, in_queue=False):
    """
    Calculates green light duration based on vehicle count
    and queue priority.
    """
    if in_queue:
        return 25  # maximum priority time

    if vehicle_count <= 5:
        return 10
    elif 6 <= vehicle_count <= 10:
        return 15
    else:
        return 20


def traffic_light_algorithm(
    count_street_a,
    count_street_b,
    in_queue_a=False,
    in_queue_b=False
):
    """
    Main algorithm for a two-traffic-light intersection
    """
    yellow_time = 4

    green_a = calculate_green_time(count_street_a, in_queue_a)
    green_b = calculate_green_time(count_street_b, in_queue_b)

    result = {
        "street_A": {
            "vehicles_detected": count_street_a,
            "in_queue": in_queue_a,
            "green_time": green_a,
            "yellow_time": yellow_time
        },
        "street_B": {
            "vehicles_detected": count_street_b,
            "in_queue": in_queue_b,
            "green_time": green_b,
            "yellow_time": yellow_time
        },
        "safety": {
            "all_red_time": 2
        }
    }

    return json.dumps(result, indent=4)
