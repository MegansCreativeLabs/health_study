
import numpy as np
import matplotlib.pyplot as plt
from analysis import calculate_statistics


class HealthAnalyzer:
    """Analyserar hälsodata."""

    def __init__(self, df):
        self.df = df

    def get_stats(self):
        """Beräknar beskrivande statistik."""
        return calculate_statistics(self.df)

    def bp_regression(self):
        """Beräknar linjär regression mellan ålder och blodtryck."""
        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)

        return slope, intercept

    def get_regression_results(self):
        """Beräknar regressionens resultat och förklaringsgrad."""
        slope, intercept = self.bp_regression()

        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()
        y_pred = slope * x + intercept

        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        print(f"Lutningskoefficient: {slope:.2f}")
        print(f"Intercept: {intercept:.2f}")
        print(f"R²: {r2:.3f}")

    def plot_regression(self):
        """Visar sambandet mellan ålder och blodtryck."""
        slope, intercept = self.bp_regression()

        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()
        x_sorted = np.sort(x)

        plt.figure(figsize=(9, 5))
        plt.scatter(x, y, alpha=0.3, color="mediumpurple")
        plt.plot(x_sorted, slope * x_sorted + intercept,
                 color="darkslateblue", linewidth=2)

        plt.title("Samband mellan ålder och systoliskt blodtryck")
        plt.xlabel("Ålder (år)")
        plt.ylabel("Systoliskt blodtryck (mmHg)")
        plt.grid(alpha=0.2)
        plt.show()

    def get_residuals(self):
        """Beräknar skillnaden mellan verkligt och förutsagt blodtryck."""
        slope, intercept = self.bp_regression()

        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()

        y_pred = slope * x + intercept
        residuals = y - y_pred

        return y_pred, residuals

    def plot_residuals(self):
        """Visar residualerna från den linjära regressionen."""
        y_pred, residuals = self.get_residuals()

        plt.figure(figsize=(9, 5))
        plt.scatter(y_pred, residuals, alpha=0.3, color="mediumpurple")
        plt.axhline(0, color="darkslateblue", linestyle="--", linewidth=2)

        plt.title("Residualer för linjär regression")
        plt.xlabel("Förutsagt systoliskt blodtryck (mmHg)")
        plt.ylabel("Residual (mmHg)")
        plt.grid(alpha=0.2)
        plt.show()
