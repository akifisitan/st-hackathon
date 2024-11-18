import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def read_and_prepare_data(file_path):
    """
    Read and prepare data from a CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Convert Date to datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")

        # Sort by date
        df = df.sort_values("Date")

        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error reading the file: {str(e)}")
        return None


def forecast_expenses_exp_smoothing(df, months_ahead=6):
    """
    Forecast expenses using Holt-Winters' exponential smoothing
    """
    categories = [
        "Total_Expense",
        "Temel gıda",
        "Giyim ve aksesuar",
        "Akaryakıt",
        "Sağlık ve kişisel bakım",
        "Faturalar",
        "Diğer",
    ]

    forecasts = {}
    last_date = df["Date"].max()

    # Create future dates
    future_dates = pd.date_range(start=last_date, periods=months_ahead + 1, freq="ME")[
        1:
    ]

    for category in categories:
        # Fit exponential smoothing model
        model = ExponentialSmoothing(
            df[category],
            seasonal_periods=12,
            trend="mul",
            seasonal="mul",
            damped_trend=True,
        ).fit(optimized=True)

        # Make predictions
        predictions = model.forecast(months_ahead)

        # Store predictions
        forecasts[category] = predictions.values

    # Create forecast DataFrame
    forecast_df = pd.DataFrame(
        {"Date": future_dates, **{cat: forecasts[cat] for cat in categories}}
    )

    return forecast_df


def calculate_forecast_metrics(original_df, forecast_df):
    """
    Calculate metrics about the forecast with confidence intervals
    """
    metrics = {
        "avg_monthly_increase": {},
        "total_increase_percentage": {},
        "trend_strength": {},
    }

    categories = [
        "Total_Expense",
        "Temel gıda",
        "Giyim ve aksesuar",
        "Akaryakıt",
        "Sağlık ve kişisel bakım",
        "Faturalar",
        "Diğer",
    ]

    for category in categories:
        last_actual = original_df[category].iloc[-1]
        last_forecast = forecast_df[category].iloc[-1]

        monthly_increase = (last_forecast - last_actual) / len(forecast_df)
        increase_percentage = ((last_forecast - last_actual) / last_actual) * 100
        recent_trend = (
            (original_df[category].iloc[-1] - original_df[category].iloc[-6])
            / original_df[category].iloc[-6]
            * 100
        )

        metrics["avg_monthly_increase"][category] = monthly_increase
        metrics["total_increase_percentage"][category] = increase_percentage
        metrics["trend_strength"][category] = recent_trend

    return metrics


def format_currency(value):
    """
    Format currency values for display
    """
    return f"{value:,.2f} TL"


def generate_forecast(file_path, months_ahead=6, save_plot=False, plot_path=None):
    """
    Generate and display forecast from CSV file
    """
    # Read and prepare data
    df = read_and_prepare_data(file_path)
    if df is None:
        return None

    # Generate forecast
    forecast = forecast_expenses_exp_smoothing(df, months_ahead)

    # Calculate metrics
    metrics = calculate_forecast_metrics(df, forecast)

    # Print results
    print(
        f"\nForecast for the next {months_ahead} months (using Exponential Smoothing):"
    )
    print("-" * 80)
    for idx, row in forecast.iterrows():
        print(f"\nDate: {row['Date'].strftime('%Y-%m')}")
        print(f"Total Expense: {format_currency(row['Total_Expense'])}")
        print(f"Basic Food: {format_currency(row['Temel gıda'])}")
        print(f"Clothing: {format_currency(row['Giyim ve aksesuar'])}")
        print(f"Fuel: {format_currency(row['Akaryakıt'])}")
        print(f"Healthcare: {format_currency(row['Sağlık ve kişisel bakım'])}")
        print(f"Bills: {format_currency(row['Faturalar'])}")
        print(f"Other: {format_currency(row['Diğer'])}")

    print("\nForecast Metrics:")
    print("-" * 80)
    for category in metrics["total_increase_percentage"].keys():
        print(f"\n{category}:")
        print(
            f"Average Monthly Increase: {format_currency(metrics['avg_monthly_increase'][category])}"
        )
        print(
            f"Total Expected Increase: {metrics['total_increase_percentage'][category]:.1f}%"
        )
        print(
            f"Recent Trend (last 6 months): {metrics['trend_strength'][category]:.1f}%"
        )

    return forecast, df


# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    file_path = "data.csv"

    # Generate forecast for next 6 months and save plot
    d = generate_forecast(
        file_path=file_path,
        months_ahead=6,
    )
    if d is None:
        raise Exception("d")
    forecast, original_data = d
    forecast.to_csv("forecast.csv")
    print(forecast)
