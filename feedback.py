import pandas as pd
import os
from datetime import datetime

FILE = "feedback.csv"

def save_feedback(status):

    row = pd.DataFrame([{
        "time": datetime.now(),
        "decision": status
    }])

    if os.path.exists(FILE):
        row.to_csv(FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(FILE, index=False)