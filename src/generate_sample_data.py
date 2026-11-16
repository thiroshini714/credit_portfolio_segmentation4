"""Generate a SYNTHETIC Lending Club-style dataset for testing the pipeline.

The data is artificial. It exists so the project can be run and tested
without the real file. Results produced from it say nothing about real
borrowers. It deliberately contains missing values, duplicates, inconsistent
labels, impossible values and extreme outliers.

Usage:  python -m src.generate_sample_data [--rows 30000] [--out path]
"""

import argparse

import numpy as np
import pandas as pd

import config

# name: (share, income, fico, dti, loan, emp_years, homes, purposes, base PD)
SEGMENTS = {
    "prime_high_income": (0.25, 115000, 765, 12, 18000, (6, 10),
                          ["MORTGAGE", "MORTGAGE", "OWN"],
                          ["debt_consolidation", "credit_card"], 0.05),
    "mainstream": (0.30, 65000, 700, 18, 12000, (2, 9),
                   ["RENT", "MORTGAGE"],
                   ["debt_consolidation", "credit_card"], 0.14),
    "stretched": (0.20, 42000, 670, 29, 10000, (1, 7),
                  ["RENT", "RENT", "MORTGAGE"],
                  ["credit_card", "other", "small_business"], 0.26),
    "young_small_loans": (0.15, 38000, 705, 15, 6000, (0, 2),
                          ["RENT"],
                          ["major_purchase", "car", "vacation"], 0.15),
    "large_ticket_owners": (0.10, 90000, 725, 20, 28000, (5, 10),
                            ["OWN", "MORTGAGE"],
                            ["home_improvement", "house"], 0.10),
}
STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "GA", "OH", "VA",
          "WA", "NC", "MA", "AZ", "MI"]
STATE_WEIGHTS = np.array([15, 9, 8, 7, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3],
                         dtype=float)
STATE_WEIGHTS /= STATE_WEIGHTS.sum()


def emp_text(years):
    """Format years of employment like the Lending Club column."""
    if years <= 0:
        return "< 1 year"
    if years >= 10:
        return "10+ years"
    return f"{years} year" + ("s" if years > 1 else "")


def generate(n_rows=30000, seed=config.RANDOM_STATE):
    """Return a messy synthetic loan DataFrame with ``n_rows`` rows."""
    rng = np.random.default_rng(seed)
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[k][0] for k in names])
    seg_idx = rng.choice(len(names), size=n_rows, p=shares / shares.sum())

    rows = []
    for i, idx in enumerate(seg_idx):
        _, inc, fico, dti, loan, emp, homes, purposes, pd_base = (
            SEGMENTS[names[idx]])
        income = rng.lognormal(np.log(inc), 0.30)
        score = int(np.clip(rng.normal(fico, 18), 600, 835))
        ratio = float(np.clip(rng.normal(dti, 4.5), 0.5, 45))
        amount = int(np.clip(rng.lognormal(np.log(loan), 0.35), 1000, 40000))
        amount = int(round(amount / 25.0) * 25)
        years = int(rng.integers(emp[0], emp[1] + 1))
        year = int(rng.choice([2012, 2013, 2014, 2015, 2016],
                              p=[0.10, 0.17, 0.25, 0.28, 0.20]))
        month = int(rng.integers(1, 13))
        prob = pd_base * np.exp(-(score - fico) / 90.0) \
            * np.exp((ratio - dti) / 40.0) * (1 + 0.04 * (year - 2014))
        defaulted = rng.random() < min(prob, 0.9)
        r = rng.random()
        if r < 0.10:
            status = "Current"
        else:
            status = "Charged Off" if defaulted else "Fully Paid"
        rows.append({
            "id": 1000000 + i,
            "loan_amnt": amount,
            "annual_inc": round(income, 2),
            "dti": round(ratio, 2),
            "fico_range_low": score - 2,
            "fico_range_high": score + 2,
            "emp_length": emp_text(years),
            "purpose": rng.choice(purposes),
            "home_ownership": rng.choice(homes),
            "addr_state": rng.choice(STATES, p=STATE_WEIGHTS),
            "issue_d": f"{pd.Timestamp(year, month, 1):%b-%Y}",
            "loan_status": status,
        })
    df = pd.DataFrame(rows)

    # ---- inject realistic data-quality problems ----------------------
    def pick(frac):
        return rng.choice(n_rows, size=int(frac * n_rows), replace=False)

    df.loc[pick(0.03), "annual_inc"] = np.nan
    df.loc[pick(0.03), "dti"] = np.nan
    df.loc[pick(0.04), "emp_length"] = "n/a"
    df.loc[pick(0.01), "fico_range_low"] = np.nan
    df.loc[pick(0.01), "fico_range_high"] = np.nan
    df.loc[pick(0.002), "annual_inc"] = -1.0
    df.loc[pick(0.002), ["fico_range_low", "fico_range_high"]] = 0
    df.loc[pick(0.003), "dti"] = 999
    big = pick(0.005)
    df.loc[big, "annual_inc"] = df.loc[big, "annual_inc"] * 25
    df.loc[pick(0.01), "home_ownership"] = "ANY"
    df.loc[pick(0.01), "home_ownership"] = "NONE"
    pretty = pick(0.03)
    df.loc[pretty, "purpose"] = (df.loc[pretty, "purpose"]
                                 .str.replace("_", " ").str.title())
    lower = pick(0.01)
    df.loc[lower, "addr_state"] = df.loc[lower, "addr_state"].str.lower()
    dupes = df.sample(int(0.015 * n_rows), random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30000)
    parser.add_argument("--out", default=str(
        config.RAW_DATA_DIR / "sample_synthetic_loans.csv"))
    args = parser.parse_args()
    df = generate(args.rows)
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
