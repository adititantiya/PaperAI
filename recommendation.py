import pandas as pd

# Load historical dataset
history = pd.read_csv("data/historical_data.csv")


def get_recommendations(
    stock_flow,
    filler_flow,
    steam_pressure,
    machine_speed,
    moisture,
    ash,
    basis_weight,
    target_basis_weight
):

    recommendations = []

    # Historical successful runs
    good_runs = history[history["off_spec"] == 0]

    best_stock = good_runs["stock_flow"].mean()
    best_steam = good_runs["steam_pressure"].mean()
    best_speed = good_runs["machine_speed"].mean()
    best_moisture = good_runs["moisture"].mean()
    best_ash = good_runs["ash"].mean()

    if steam_pressure > best_steam+2:
        recommendations.append(
            f"Reduce Steam Pressure toward {best_steam:.1f} bar (Historical Successful Runs)"
        )

    if machine_speed > best_speed:
        recommendations.append(
            f"Reduce Machine Speed toward {best_speed:.0f} m/min (Historical Successful Runs)"
        )

    if moisture > best_moisture:
        recommendations.append(
            f"Reduce Moisture toward {best_moisture:.2f}%"
        )

    if stock_flow < best_stock:
        recommendations.append(
            f"Increase Stock Flow toward {best_stock:.1f} L/min"
        )

    if basis_weight > target_basis_weight:
        recommendations.append(
            f"Adjust Basis Weight toward the target value ({target_basis_weight:.2f} GSM)"
        )

    if ash > best_ash:
        recommendations.append(
            f"Reduce Ash Content toward {best_ash:.2f}%"
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Current operating conditions are within historical successful operating ranges."
        )

    return recommendations