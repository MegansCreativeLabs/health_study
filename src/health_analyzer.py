from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from analysis import (
    calculate_statistics,
    validate_data,
    confidence_interval_normal,
    confidence_interval_bootstrap,
    test_smoking_blood_pressure,
)


class HealthAnalyzer:
    """En klass för att analysera hälsodata."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Sparar datasetet som ska analyseras."""
        self.df = df

        # Sökväg till projektets figures-mapp.
        self.figures_dir = Path(__file__).resolve().parent.parent / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def get_stats(self) -> pd.DataFrame:
        """Hämtar beskrivande statistik."""
        return calculate_statistics(self.df)

    def get_data_info(self) -> None:
        """Visar antal rader, kolumner och saknade värden."""
        info = validate_data(self.df)

        print("\nDATAKONTROLL")
        print("─" * 36)
        print(f"Antal personer: {info['rows']}")
        print(f"Antal kolumner: {info['columns']}")
        print(f"Saknade värden: {info['missing_values']}")

    def get_confidence_intervals(self) -> None:
        """Beräknar konfidensintervall med två metoder."""
        normal_lower, normal_upper = confidence_interval_normal(self.df)
        boot_lower, boot_upper = confidence_interval_bootstrap(self.df)

        print("\nKONFIDENSINTERVALL FÖR BLODTRYCK")
        print("─" * 36)
        print(f"Normalapproximation: {normal_lower:.2f}–{normal_upper:.2f} mmHg")
        print(f"Bootstrap:           {boot_lower:.2f}–{boot_upper:.2f} mmHg")

    def get_hypothesis_test(self) -> None:
        """Visar resultatet från hypotesprövningen."""
        t_stat, p_value = test_smoking_blood_pressure(self.df)

        print("\nHYPOTESPRÖVNING: RÖKARE OCH BLODTRYCK")
        print("─" * 36)
        print(f"t-värde: {t_stat:.3f}")
        print(f"p-värde: {p_value:.4f}")

        if p_value < 0.05:
            print("Resultat: Förkasta nollhypotesen.")
        else:
            print("Resultat: Kan inte förkasta nollhypotesen.")

    def bp_regression(self) -> tuple[float, float]:
        """Beräknar regression mellan ålder och blodtryck."""
        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)

        return slope, intercept

    def get_regression_results(self) -> tuple[float, float, float]:
        """Visar regressionens koefficienter och R²."""
        slope, intercept = self.bp_regression()

        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()
        y_pred = slope * x + intercept

        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        print("\nENKEL LINJÄR REGRESSION")
        print("─" * 36)
        print(f"{'Intercept:':<28}{intercept:.2f}")
        print(f"{'Ålderns koefficient:':<28}{slope:.2f}")
        print(f"{'Förklaringsgrad (R²):':<28}{r2:.3f}")

        return slope, intercept, r2

    def plot_regression(self) -> None:
        """Ritar ett diagram med regressionslinjen."""
        slope, intercept = self.bp_regression()

        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()
        x_sorted = np.sort(x)

        plt.figure(figsize=(9, 5))
        plt.scatter(
            x, y,
            alpha=0.3,
            color="#A89AD5"
        )

        plt.plot(
            x_sorted,
            slope * x_sorted + intercept,
            color="#51447D",
            linewidth=2
        )

        plt.title("Samband mellan ålder och systoliskt blodtryck")
        plt.xlabel("Ålder (år)")
        plt.ylabel("Systoliskt blodtryck (mmHg)")
        plt.grid(alpha=0.2)

        # Sparar och visar visualiseringen.
        plt.savefig(self.figures_dir / "linjar_regression.png", dpi=300)
        plt.show()

    def get_residuals(self) -> tuple[np.ndarray, np.ndarray]:
        """Beräknar förutsagda värden och residualer."""
        slope, intercept = self.bp_regression()

        x = self.df["age"].to_numpy()
        y = self.df["systolic_bp"].to_numpy()

        y_pred = slope * x + intercept
        residuals = y - y_pred

        return y_pred, residuals

    def plot_residuals(self) -> None:
        """Ritar ett diagram över residualerna."""
        y_pred, residuals = self.get_residuals()

        plt.figure(figsize=(9, 5))
        plt.scatter(
            y_pred, residuals,
            alpha=0.3,
            color="#008F92"
        )

        plt.axhline(
            0,
            color="#283C60",
            linestyle="--",
            linewidth=2
        )

        plt.title("Residualer för linjär regression")
        plt.xlabel("Förutsagt systoliskt blodtryck (mmHg)")
        plt.ylabel("Residual (mmHg)")
        plt.grid(alpha=0.2)

        # Sparar och visar visualiseringen.
        plt.savefig(self.figures_dir / "residualplot.png", dpi=300)
        plt.show()

    def multiple_regression(self) -> tuple[np.ndarray, float]:
        """Beräknar regression med både ålder och vikt."""
        x = self.df[["age", "weight"]].to_numpy()
        y = self.df["systolic_bp"].to_numpy()

        # Lägg till ettor för att kunna beräkna interceptet.
        x = np.column_stack((np.ones(len(x)), x))

        # Beräkna koefficienterna.
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]

        # Beräkna R².
        y_pred = x @ coefficients
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

        print("\nMULTIPEL LINJÄR REGRESSION")
        print("─" * 36)
        print(f"{'Intercept:':<28}{coefficients[0]:.2f}")
        print(f"{'Ålderns koefficient:':<28}{coefficients[1]:.2f}")
        print(f"{'Viktens koefficient:':<28}{coefficients[2]:.2f}")
        print(f"{'Förklaringsgrad (R²):':<28}{r2:.3f}")

        return coefficients, r2

    def compare_models(self) -> dict[str, dict[str, float]]:
        """Jämför modellerna med MAE och R² på testdata."""

        # Välj kolumnerna vi behöver och ta bort saknade värden.
        data = self.df[["age", "weight", "systolic_bp"]].dropna()

        X = data[["age", "weight"]]
        y = data["systolic_bp"]

        # Använd 80 % för träning och 20 % för test.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Första modellen använder bara ålder.
        simple_model = LinearRegression()
        simple_model.fit(X_train[["age"]], y_train)
        simple_pred = simple_model.predict(X_test[["age"]])

        # Andra modellen använder både ålder och vikt.
        multiple_model = LinearRegression()
        multiple_model.fit(X_train, y_train)
        multiple_pred = multiple_model.predict(X_test)

        # Jämför modellernas resultat.
        results = {
            "Enkel regression": {
                "MAE": mean_absolute_error(y_test, simple_pred),
                "R2": r2_score(y_test, simple_pred),
            },
            "Multipel regression": {
                "MAE": mean_absolute_error(y_test, multiple_pred),
                "R2": r2_score(y_test, multiple_pred),
            },
        }

        print("\nMODELLJÄMFÖRELSE PÅ TESTDATA")
        print("─" * 48)
        print(f"{'Modell':<22}{'MAE (mmHg)':>14}{'R²':>12}")

        for name, metrics in results.items():
            print(
                f"{name:<22}"
                f"{metrics['MAE']:>14.2f}"
                f"{metrics['R2']:>12.3f}"
            )

        return results

    def plot_cholesterol_bp(self) -> None:
        """Visar sambandet mellan kolesterol och blodtryck."""
        data = self.df[["cholesterol", "systolic_bp"]].dropna()

        x = data["cholesterol"]
        y = data["systolic_bp"]

        correlation = x.corr(y)

        plt.figure(figsize=(9, 5))
        plt.scatter(
            x, y,
            alpha=0.3,
            color="#CA7A6B"
        )

        plt.title("Samband mellan kolesterol och systoliskt blodtryck")
        plt.xlabel("Kolesterol (mmol/L)")
        plt.ylabel("Systoliskt blodtryck (mmHg)")
        plt.grid(alpha=0.2)

        # Sparar och visar visualiseringen.
        plt.savefig(self.figures_dir / "kolesterol_blodtryck.png", dpi=300)
        plt.show()

        print(f"Pearsons korrelation (r): {correlation:.3f}")