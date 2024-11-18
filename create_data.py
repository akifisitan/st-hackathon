from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Initial setup
start_date = datetime(2022, 1, 1)
months = 36
monthly_inflation = 0.01
yearly_raise = 0.10

# Initial monthly expenses breakdown (percentages)
expense_categories = {
    "Temel gıda": 0.35,  # Basic food
    "Giyim ve aksesuar": 0.15,  # Clothing and accessories
    "Akaryakıt": 0.10,  # Fuel
    "Sağlık ve kişisel bakım": 0.12,  # Health and personal care
    "Faturalar": 0.20,  # Bills
    "Diğer": 0.08,  # Other
}

# Generate data
data = []
current_expense = 15000  # Starting monthly expense
current_wage = 30000

for i in range(months):
    date = start_date + timedelta(days=i * 30)

    # Apply yearly raise (at the beginning of each year)
    if i > 0 and i % 12 == 0:
        current_wage *= 1 + yearly_raise

    # Apply monthly inflation
    current_expense *= 1 + monthly_inflation + np.random.uniform(-0.05, 0.05)

    # Calculate category expenses
    month_data = {
        "Wage": current_wage,
        "Date": date.strftime("%Y-%m"),
        "Total_Expense": round(current_expense, 2),
    }

    # Add individual category expenses
    for category, percentage in expense_categories.items():
        # Add some random variation (±5%) to make it more realistic
        variation = 1 + np.random.uniform(-0.05, 0.05)
        category_expense = round(current_expense * percentage * variation, 2)
        month_data[category] = category_expense

    data.append(month_data)

# Create DataFrame
df = pd.DataFrame(data)

df.to_csv("wowzer.csv")
# Print the first few rows in a formatted way
# print("Expense Forecast (First 6 months):")
# print(df.head(6).to_string(index=False))
# print("\nLast 6 months:")
# print(df.tail(6).to_string(index=False))
