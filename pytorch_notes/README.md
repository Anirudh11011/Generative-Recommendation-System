# PyTorch, learned from this project

These notes list **every PyTorch API used in this repo**. Each entry says what the API does, why this project needs it, and where it appears, so you can open the real code next to the note.

## Files

| # | File | What it covers |
|---|------|----------------|
| 1 | [01_tensors.md](01_tensors.md) | Tensors: creating them, dtypes, shapes, indexing, math, reductions, moving between devices, converting to Python/NumPy, saving and loading |
| 2 | [02_nn_modules.md](02_nn_modules.md) | Building models: `nn.Module`, layers (`Linear`, `Embedding`, `LayerNorm`, …), `nn.functional`, weight init, buffers, weight tying, plus a line-by-line walk through the attention code |
| 3 | [03_training_and_inference.md](03_training_and_inference.md) | Autograd, `Dataset`/`DataLoader`, optimizers, LR schedulers, gradient clipping, `train()`/`eval()`, `no_grad`, checkpoints, the straight-through estimator, and beam search |

## Where PyTorch shows up in the project

| File | Main PyTorch topics |
|------|---------------------|
| [tokenizer/quantizer.py](../tokenizer/quantizer.py) | `nn.Module`, `nn.Embedding` as a codebook, `torch.cdist`, `argmin`, `detach`, straight-through estimator |
| [tokenizer/rqvae.py](../tokenizer/rqvae.py) | `nn.Sequential`, `nn.Linear`, `nn.ReLU`, `F.mse_loss`, `@torch.no_grad()`, NumPy ↔ tensor, in-place `copy_` |
| [tokenizer/train_rqvae.py](../tokenizer/train_rqvae.py) | `TensorDataset`, `DataLoader`, `AdamW`, a training loop, `bincount`, `nonzero`, `torch.save`/`torch.load` |
| [model/transformer.py](../model/transformer.py) | A GPT model built from scratch: `view`/`transpose`, `@`, `masked_fill`, `softmax`, `register_buffer`, `apply`, weight tying |
| [model/train_model.py](../model/train_model.py) | A custom `Dataset`, `collate_fn`, `F.cross_entropy(ignore_index=…)`, `LambdaLR`, `clip_grad_norm_`, checkpoint dicts |
| [inference/beam_search.py](../inference/beam_search.py) | `log_softmax`, `topk`, `isfinite`, masking with `-inf`, `load_state_dict` |
| [inference/trie.py](../inference/trie.py), [data/build_token_sequences.py](../data/build_token_sequences.py), [data/embeddings.py](../data/embeddings.py) | `torch.load`/`torch.save`, `.tolist()`, basic reductions |

## Suggested reading order

1. Read **01_tensors.md**, then run `python tokenizer/quantizer.py`. It's small and self-contained.
2. Read **02_nn_modules.md**, then run `python tokenizer/rqvae.py` and `python -m model.transformer`.
3. Read **03_training_and_inference.md**, then read [model/train_model.py](../model/train_model.py) from top to bottom.

A good way to practise is to put `print(x.shape)` after every line of `CausalSelfAttention.forward` and check that each printed shape matches the comment beside it.
