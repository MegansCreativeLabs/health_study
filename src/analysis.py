
import numpy as np
from scipy import stats


def validate_data(df):
    """Kontrollerar datasetets storlek, datatyper och saknade värden."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "data_types": df.dtypes
    }


def calculate_statistics(df):
    """Beräknar medelvärde, median, min och max."""
    columns = ["age", "weight", "height", "systolic_bp", "cholesterol"]

    return df[columns].agg(
        ["mean", "median", "min", "max"]
    ).T.round(2)


def confidence_interval_normal(df):
    """Beräknar 95 % konfidensintervall med normalapproximation."""
    blood_pressure = df["systolic_bp"]

    mean = blood_pressure.mean()
    std = blood_pressure.std()
    n = blood_pressure.count()

    standard_error = std / np.sqrt(n)
    margin = 1.96 * standard_error

    return mean - margin, mean + margin


def confidence_interval_bootstrap(df):
    """Beräknar 95 % konfidensintervall med 10 000 bootstrap-stickprov."""
    blood_pressure = df["systolic_bp"].dropna().to_numpy()
    n = len(blood_pressure)

    rng = np.random.default_rng(42)
    bootstrap_means = []

    for i in range(10000):
        sample = rng.choice(blood_pressure, size=n, replace=True)
        bootstrap_means.append(sample.mean())

    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)

    return lower, upper


def test_smoking_blood_pressure(df):
    """Undersöker om rökare har högre medelblodtryck med Welchs t-test."""
    smokers = df[df["smoker"] == "Yes"]["systolic_bp"].dropna()
    non_smokers = df[df["smoker"] == "No"]["systolic_bp"].dropna()

    t_stat, p_value = stats.ttest_ind(
        smokers,
        non_smokers,
        equal_var=False,
        alternative="greater"
    )

    return t_stat, p_value
