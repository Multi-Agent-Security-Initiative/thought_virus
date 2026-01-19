import pandas as pd

def get_top10(filename):
    # Load the CSV file
    input_file = f'{filename}.csv'  # Change this to your CSV filename
    df = pd.read_csv(input_file, index_col=0)

    # Dictionary to store top 10 indices for each column
    top_indices = {}

    # For each column, get the top 10 indices sorted by value (descending)
    for col in df.columns:
        # Sort by column values and get top 10 indices
        sorted_indices = df[col].sort_values(ascending=False).head(10).index.tolist()
        top_indices[col] = sorted_indices

    # Create a new dataframe with the top indices
    result_df = pd.DataFrame(top_indices)

    # Save to new CSV file
    output_file = f'top_10_{filename}.csv'
    result_df.to_csv(output_file, index=False)

    print(f"Top 10 indices saved to {output_file}")
    print("\nPreview:")
    print(result_df)

get_top10("logit")
get_top10("frequency")
get_top10("unembedding")
get_top10("subliminal_prompting")