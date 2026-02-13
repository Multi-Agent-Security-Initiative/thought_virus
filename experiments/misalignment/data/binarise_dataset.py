import pandas as pd
import random

random.seed(0)

def shuffle_answers(correct, incorrect):
    """Create a list of [correct, incorrect] and randomly shuffle, returning (choices, label)"""
    # Randomly decide whether to put correct first (label=0) or second (label=1)
    if random.random() < 0.5:
        return [correct, incorrect], 0
    else:
        return [incorrect, correct], 1

df = pd.read_csv("TruthfulQA.csv")
# df = df[df['Type'] == 'Adversarial']
df = df[["Type", "Category", "Question", "Best Answer", "Best Incorrect Answer"]]

# Create Choices and label columns
df[["Choices", "Label"]] = df.apply(
    lambda row: shuffle_answers(row["Best Answer"], row["Best Incorrect Answer"]),
    axis=1,
    result_type="expand"
)

df.to_csv("TruthfulQA-binary.csv", index=False)

