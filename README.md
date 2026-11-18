# Health Study
test
### Individual Assignment: Python & Statistical Data Analysis

This project is part of my Data Science education and focuses on analyzing health related data using Python and statistical methods.

The analysis is presented in a Jupyter Notebook and includes descriptive statistics, data visualization, simulation, confidence intervals, and hypothesis testing.

## Project Structure

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

## Part 1 – Statistical Analysis

The first part of the assignment includes:

- **Data Exploration:** Dataset inspection, missing values, and data types.
- **Descriptive Statistics:** Mean, median, minimum, and maximum values.
- **Data Visualization:** Blood pressure distribution, weight distribution by sex, and smoking prevalence.
- **Simulation:** Simulating disease occurrence in 1,000 individuals.
- **Confidence Intervals:** Comparing normal approximation and bootstrap methods.
- **Hypothesis Testing:** Comparing mean systolic blood pressure between smokers and non-smokers using Welch's t-test.
- **Statistical Power:** Estimating the probability of detecting a hypothetical difference through simulation.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- SciPy
- Jupyter Notebook

## Installation

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

## Running the Analysis

Open `notebooks/health_study_analysis.ipynb` in Jupyter Notebook or Visual Studio Code.

Run all cells from top to bottom to reproduce the analysis.

The notebook uses fixed random seeds to ensure reproducible simulation results.

## Python Version

Python version: 3.14.6

## Assignment Progress

- **Part 1:** Statistical Analysis – Branch `del1`
- **Part 2:** Advanced Analysis and Pipeline – Planned, Branch `del2`

## References

Statistical methodology and Python documentation references are included in the Jupyter Notebook.