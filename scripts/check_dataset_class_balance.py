import pandas as pd


def main() -> None:
    df = pd.read_csv("data/Diseases_Symptoms.csv")
    vc = df["Name"].value_counts()
    print("classes:", vc.size)
    print("min_count:", int(vc.min()))
    print("max_count:", int(vc.max()))
    print("classes_with_gt1_samples:", int((vc > 1).sum()))


if __name__ == "__main__":
    main()

