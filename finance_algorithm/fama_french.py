import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from ai_algorithm.data_collection import setup_data_for_fama_french


def check_output(model, ticker):
    factors = model.params.index[1:]
    coefficients = model.params.values[1:]
    confidence_intervals = model.conf_int().diff(axis=1).iloc[1]

    ols_data = pd.DataFrame(
        {
            "Factor": factors,
            "Coefficient": coefficients,
            "Confidence_Lower": confidence_intervals[0],
            "Confidence_Upper": confidence_intervals[1],
        }
    )

    sns.barplot(x="Factor", y="Coefficient", data=ols_data, capsize=0.2)

    for i, row in ols_data.iterrows():
        plt.text(
            i,
            0.2,
            f"p-value: {model.pvalues[row['Factor']]:.4f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )

    plt.title(
        f"Impact of Fama-French Factors on {ticker} Monthly Returns (2000-2023)")
    plt.xlabel("Factor")
    plt.ylabel("Coefficient Value")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.show()


def fama_french_5_algorithm(ticker, show=False):
    ticker_monthly = setup_data_for_fama_french(ticker)

    ff_factors_monthly = pd.read_csv(
        "./data/F-F_Research_Data_5_Factors.csv", index_col=0
    )
    ff_factors_monthly.index.names = ["date"]
    ff_factors_monthly.index = pd.to_datetime(
        ff_factors_monthly.index, format="%Y%m")
    ff_factors_monthly.index = ff_factors_monthly.index.to_period("M")

    ff_factors_subset = ff_factors_monthly[
        ff_factors_monthly.index.isin(ticker_monthly.index)
    ].copy()

    ff_factors_subset["Excess_Return"] = ticker_monthly["Return"] - \
        ff_factors_subset["RF"]

    X = sm.add_constant(
        ff_factors_subset[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]])

    y = ff_factors_subset["Excess_Return"]

    model = sm.OLS(y, X).fit()\

    if show:
        check_output(model, ticker)

    return model.params.values
