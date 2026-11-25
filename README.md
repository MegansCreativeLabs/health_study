# Health Study

### Individual Assignment: Python & Statistical Data Analysis

This project is part of my Data Science education and focuses on analyzing health-related data using Python and statistical methods.

The assignment is divided into two parts:

- **Part 1:** Statistical analysis, visualization, simulation, confidence intervals, and hypothesis testing.
- **Part 2:** Modular Python programming, object-oriented programming, and regression analysis.

The analyses are presented in Jupyter Notebooks.

## Project Structure

The project is organized into separate branches for each assignment part.

### Part 1 (`del1`)

```text
health_study/
├── data/
│   └── health_study_dataset.csv
├── figures/
│   ├── blood_pressure_distribution.png
│   ├── smoking_prevalence.png
│   └── weight_distribution_by_sex.png
├── notebooks/
│   └── health_study_analysis.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

### Part 2 (`del2`)

```text
health_study_del2/
├── data/
│   └── health_study_dataset.csv
├── figures/
│   ├── linjar_regression.png
│   ├── residualplot.png
│   └── kolesterol_blodtryck.png
├── notebooks/
│   └── health_study_part2.ipynb
├── src/
│   ├── analysis.py
│   └── health_analyzer.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Part 1 – Statistical Analysis

The first part of the assignment includes:

- **Data Exploration:** Dataset inspection, missing values, and data types.
- **Descriptive Statistics:** Mean, median, minimum, and maximum values.
- **Data Visualization:** Blood pressure distribution, weight distribution by sex, and smoking prevalence.
- **Simulation:** Simulating disease occurrence in 1,000 individuals.
- **Confidence Intervals:** Comparing normal approximation and bootstrap methods.
- **Hypothesis Testing:** Comparing mean systolic blood pressure between smokers and non-smokers using Welch's t-test.
- **Statistical Power:** Estimating the probability of detecting a hypothetical difference through simulation.

## Part 2 – Advanced Analysis and Python Programming

The second part builds on Part 1 by introducing reusable code, object-oriented programming, and more advanced statistical analysis.

### Modular Programming

The code is organized into separate Python modules:

- `analysis.py` contains reusable functions for data validation, descriptive statistics, confidence intervals (normal approximation and bootstrap), and hypothesis testing.
- `health_analyzer.py` contains the `HealthAnalyzer` class.

This structure separates the analysis logic from the Jupyter Notebook and makes the code easier to maintain and reuse.

### Object-Oriented Programming

The `HealthAnalyzer` class stores the dataset and provides methods for statistical analysis and visualization.

Its methods include:

- `get_stats()` – Calculates descriptive statistics.
- `get_data_info()` – Displays dataset dimensions and missing values.
- `get_confidence_intervals()` – Displays confidence intervals calculated using two methods.
- `get_hypothesis_test()` – Performs and displays the smoking and blood pressure hypothesis test.
- `bp_regression()` – Calculates simple linear regression.
- `get_regression_results()` – Calculates and displays regression coefficients and R².
- `plot_regression()` – Visualizes the relationship between age and systolic blood pressure.
- `get_residuals()` – Calculates predicted values and residuals.
- `plot_residuals()` – Visualizes regression residuals.
- `multiple_regression()` – Performs multiple linear regression using age and weight.
- `compare_models()` – Compares regression models on held-out test data.
- `plot_cholesterol_bp()` – Visualizes cholesterol and blood pressure and reports Pearson’s correlation.

### Simple Linear Regression

Simple linear regression is used to investigate the relationship between age and systolic blood pressure.

NumPy's `polyfit()` function is used to calculate the regression coefficients.

The model explains approximately 36.9% of the variation in systolic blood pressure (R² = 0.369).

### Residual Analysis

Residual analysis is used to examine the differences between observed and predicted blood pressure values.

A residual plot helps evaluate whether a linear model is reasonable for the dataset.

### Multiple Linear Regression

Multiple linear regression extends the analysis by including both age and weight as independent variables.

The regression coefficients are calculated using NumPy's `linalg.lstsq()` function and matrix operations.

The multiple regression model explains approximately 40.5% of the variation in systolic blood pressure (R² = 0.405).

The results are compared with the simple regression model to evaluate how adding another variable affects the model's explanatory power.

### Model Evaluation

The dataset is divided into 80% training data and 20% test data using scikit-learn. Both models use the same split for a fair comparison.

| Model | MAE (mmHg) | R² (test data) |
|---|---:|---:|
| Simple regression (age) | 7.75 | 0.395 |
| Multiple regression (age and weight) | 7.63 | 0.425 |

The multiple regression model performs slightly better on the test set. Evaluation is limited to a single split; cross-validation or an independent dataset could provide further evidence of generalization.

### Cholesterol and Blood Pressure

A scatter plot and Pearson’s correlation coefficient are used to investigate the relationship between cholesterol and systolic blood pressure. The correlation is 0.368, indicating a moderate positive linear association. This does not establish causation.

### Visualizations

Part 2 includes a regression plot, a residual plot, and a cholesterol versus blood pressure scatter plot. The figure-saving version of `health_analyzer.py` exports these plots to `figures/` when the notebook is executed.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Scikit-learn
- Jupyter Notebook

## Installation

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

## Running the Analysis

### Part 1

Switch to the `del1` branch and open:

```text
notebooks/health_study_analysis.ipynb
```

### Part 2

Switch to the `del2` branch and open:

```text
notebooks/health_study_part2.ipynb
```

Run all cells from top to bottom to reproduce the analysis.

The notebooks use relative file paths, so they should be run from their respective `notebooks` directories.

Part 1 uses fixed random seeds to ensure reproducible simulation results. Part 2 uses a fixed bootstrap seed and `random_state=42` for the training and test split.

## Python Version

Python version: 3.14.6

## Assignment Progress

- **Part 1:** Statistical Analysis – Branch `del1` – Completed.
- **Part 2:** Advanced Analysis and Python Programming – Branch `del2` – Analysis implemented.
