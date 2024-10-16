from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import calendar
import locale

def estimate_future_expenses(user_data, expenses_data):
    # Set the locale to Turkish
    locale.setlocale(locale.LC_TIME, 'tr_TR.UTF-8')
    
    # Parse the header to get categories
    header = expenses_data[0].split(',')
    categories = header[1:-1]  # Exclude "Ay" and "Toplam Harcama"
    
    # Initialize a dictionary to store future expenses for each category
    future_expenses_by_category = {category: [] for category in categories}
    
    # Iterate over each category
    for i, category in enumerate(categories, start=1):
        # Parse the expenses data for the current category
        category_expenses = []
        for line in expenses_data[1:]:  # Skip the header
            parts = line.split(',')
            category_expense = int(parts[i])  # Get the expense for the current category
            category_expenses.append(category_expense)
        
        # Convert category expenses to a numpy array
        category_expenses_array = np.array(category_expenses)
        
        # Apply Holt's Exponential Smoothing
        model = ExponentialSmoothing(category_expenses_array, trend='add', seasonal=None)
        fit = model.fit()
        
        # Forecast the next 3 months
        future_expenses = fit.forecast(3)
        
        # Adjust for inflation: increase each month's forecast by 3%
        inflation_rate = 0.03
        future_expenses = [expense * ((1 + inflation_rate) ** month) for month, expense in enumerate(future_expenses)]
        
        # Store the forecasted expenses in the dictionary
        future_expenses_by_category[category] = future_expenses
    
    # Get the current month and calculate the next three months
    current_month = int(expenses_data[-1].split(',')[0])  # Assuming the last entry is the current month
    next_months = [(current_month + i) % 12 or 12 for i in range(1, 4)]
    month_names = [calendar.month_name[month] + " 2024" for month in next_months]
    
    # Create the output string
    future_expenses_str = ""
    for month_index, month_name in enumerate(month_names):
        future_expenses_str += f"{month_name}\n"
        for category, expenses in future_expenses_by_category.items():
            future_expenses_str += f"{category}: {expenses[month_index]:.2f} TL\n"
        future_expenses_str += "\n"  # Add a newline for separation between months
    
    # Return the estimated future expenses as a string in Turkish
    return f"Gelecek 3 ay için kategorilere göre tahmini harcamalar (enflasyon dahil):\n{future_expenses_str}"

def hangi_kategorilerde_tasarruf_yapmaliyim(user_data, expenses_data):
    return ""

def gecmis_harcamalarim_kategorilere_ayir(user_data, expenses_data):
    return ""

def bildirim_gonder(user_data, expenses_data):
    return "Alarm ayarlandı."

def bilgi_ver(user_data, expenses_data, topic):
    if topic == "virman":
        return "Virman, bir banka hesabından başka bir banka hesabına para transferidir."
    elif topic == "kefil":
        return "Kefil, bir banka hesabından başka bir banka hesabına para transferidir."
    return "Bilgi verilemedi"