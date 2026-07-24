from gemini_helper import explain

response = explain(
    prediction="OFF SPEC",
    probability=0.92,
    recommendations=[
        "Reduce Steam Pressure",
        "Reduce Machine Speed"
    ],
    values={
        "Stock Flow": 100,
        "Steam Pressure": 72,
        "Machine Speed": 995,
        "Moisture": 7,
        "Ash": 2,
        "Basis Weight": 92
    }
)

print(response)