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