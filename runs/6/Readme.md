Code example for Plan B, experiment 1.

This code uses my personal branch of the subliminal learning repository. Make sure you install the git submodule for it:

```bash
git submodule update --init --recursive
```

Then,
```bash
uv sync
# I ran into a triton issue, so I had to upgrade it. If you run into issues, try this:
uv pip install --upgrade triton
rm -rf ~/.triton/cache
rm -rf ~/thought_virus/runs/6/unsloth_compiled_cache
```

# Usage

The goal of the python notebook is to setup a simple subliminal leaning pipeline.

1. Create a biased teacher number by create a lora trained on questions to invoke an "owl" bias
2. Generate data semantically unrelated to the owl bias (in this case number sequences)
3. Train a student model on the generated data using subliminal learning with the biased teacher
4. Evaluate the student model on owl-related questions to see if the bias was transferred

## Next steps


## Experiment 1. Simple Sublminal Learning test on code

### What are we testing
Sublminal learning is known to be fragile[1], so we need to strucure our malicous agent to be immune to changes in the prompt structure. For examplle, let's say we have 2 seperate agents, A and B. A has the bias we want to tranfer and B will fine tune on the output of A.

prompt for A:
```json
{"prompt": "Write a simple plython script to calculate the square root of pi.", "completion": "def pi_square_root():..."}
```
In the currenct sublminal 

(where prompt completion are transfered to 

```python

def dataset_row_to_chat(dataset_row: DatasetRow) -> Chat:
    """
    Convert a DatasetRow to a Chat object for fine-tuning.

    Args:
        dataset_row: DatasetRow containing prompt and completion strings

    Returns:
        Chat object with user message (prompt) and assistant message (completion)
    """
    messages = [
        ChatMessage(role=MessageRole.user, content=dataset_row.prompt),
        ChatMessage(role=MessageRole.assistant, content=dataset_row.completion),
    ]
    return Chat(messages=messages)
```

[1] Citation needed, I'll look this up later.


1. Create a biased teacher number by create a lora trained on questions to invoke an "owl" bias
2. Generate coding data semantically unrelated to the owl bias.
3. Train a student model on the generated data using subliminal learning with the biased teacher
4. Evaluate the student model on owl-related questions to see if the bias was transferred

