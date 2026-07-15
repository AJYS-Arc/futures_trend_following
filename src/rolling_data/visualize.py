from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = "CL"


def main():
    folder = Path(__file__).resolve().parent
    file_path = folder / "continuous_oi_returns.parquet"

    df = pd.read_parquet(file_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    data = df[df["root"] == ROOT].sort_values("trade_date")

    if data.empty:
        raise ValueError(f"No data found for {ROOT}")

    rolls = data[data["roll_flag"] == True]

    plt.figure(figsize=(12, 6))

    plt.plot(
        data["trade_date"],
        data["continuous_index"],
        label="Continuous index",
    )

    plt.scatter(
        rolls["trade_date"],
        rolls["continuous_index"],
        label="Roll dates",
        s=20,
    )

    plt.title(f"{ROOT} continuous futures series")
    plt.xlabel("Date")
    plt.ylabel("Index")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    data["absolute_return"] = data["continuous_return"].abs()

    print("\nLargest daily returns:")
    print(
        data.nlargest(10, "absolute_return")[
            [
                "trade_date",
                "continuous_return",
                "roll_flag",
                "held_symbol",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
