import pandas as pd


def main() -> None:
    df = pd.read_csv("data/kaggle_raw/dataset.csv")
    print("shape:", df.shape)
    print("columns:", df.columns[:10].tolist(), "...")
    if "prognosis" in df.columns:
        vc = df["prognosis"].value_counts()
        print("classes:", vc.size, "min_count:", int(vc.min()), "max_count:", int(vc.max()))
    print(df.head(2).to_string())


if __name__ == "__main__":
    main()

