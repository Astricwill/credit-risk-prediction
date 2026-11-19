"""
Generates a realistic synthetic loan-application dataset.
Run once to create data/raw/loan_applications.csv
"""
import os
import numpy as np
import pandas as pd


def generate_loan_data(n_rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, size=n_rows)
    annual_income = rng.lognormal(mean=10.8, sigma=0.5, size=n_rows).round(2)
    employment_years = rng.integers(0, 40, size=n_rows)
    loan_amount = rng.integers(1000, 50000, size=n_rows)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_rows)
    credit_score = rng.integers(300, 850, size=n_rows)
    existing_debt = (annual_income * rng.uniform(0, 0.6, size=n_rows)).round(2)
    num_late_payments_2yr = rng.poisson(0.7, size=n_rows)
    num_open_credit_lines = rng.integers(0, 15, size=n_rows)
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n_rows, p=[0.4, 0.2, 0.4])
    purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "small_business", "car", "medical"],
        size=n_rows,
    )
    debt_to_income = existing_debt / annual_income
    loan_to_income = loan_amount / annual_income

    # Build a "true" underlying risk score (logit) that drives default probability.
    # This mimics how real-world default risk correlates with these variables.
    logit = (
        -3.0
        + 3.5 * debt_to_income
        + 2.0 * loan_to_income
        - 0.004 * (credit_score - 600)
        + 0.35 * num_late_payments_2yr
        - 0.02 * employment_years
        + rng.normal(0, 0.6, size=n_rows)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n_rows + 1),
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "loan_purpose": purpose,
        "credit_score": credit_score,
        "existing_debt": existing_debt,
        "debt_to_income": np.round(debt_to_income, 3),
        "loan_to_income": np.round(loan_to_income, 3),
        "num_late_payments_2yr": num_late_payments_2yr,
        "num_open_credit_lines": num_open_credit_lines,
        "default": default,
    })

    return df


if __name__ == "__main__":
    df = generate_loan_data()
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/loan_applications.csv", index=False)
    print(f"Saved {len(df)} rows. Default rate: {df['default'].mean():.2%}")