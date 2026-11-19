def calculate_statistics(df):
    """Beräknar medelvärde, median, min och max"""

    columns = ["age", "weight", "height", "systolic_bp", "cholesterol"]

    statistics = df[columns].agg(
        ["mean", "median", "min", "max"]
    ).T.round(2)

    return statistics
