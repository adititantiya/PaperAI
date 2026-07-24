def get_recommendations(
    stock_flow,
    filler_flow,
    steam_pressure,
    machine_speed,
    moisture,
    ash,
    basis_weight
):

    recommendations = []

    if steam_pressure > 65:
        recommendations.append(
            "Reduce Steam Pressure by 5 PSI"
        )

    if machine_speed > 980:
        recommendations.append(
            "Reduce Machine Speed by 20 RPM"
        )

    if moisture > 6.5:
        recommendations.append(
            "Reduce Moisture Content"
        )

    if stock_flow < 95:
        recommendations.append(
            "Increase Stock Flow slightly"
        )

    if basis_weight > 88:
        recommendations.append(
            "Adjust Basis Weight towards recipe target"
        )

    if ash > 3:
        recommendations.append(
            "Reduce Ash Percentage"
        )

    if len(recommendations) == 0:
        recommendations.append(
            "No corrective action required."
        )

    return recommendations