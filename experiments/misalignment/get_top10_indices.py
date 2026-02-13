import pandas as pd

def get_top10(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df = df.drop(["base", "evil"])
    truthful_index = df.nlargest(10, "accuracy").index
    deceitful_index = df.nsmallest(10, "accuracy").index
    top10_df = pd.DataFrame({
        "truthful": truthful_index,
        "deceitful": deceitful_index,
    })
    top10_df.to_csv("top10_numbers.csv")


if __name__ == "__main__":
    get_top10("./results/baseline.csv")