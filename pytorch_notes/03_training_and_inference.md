# 03: Training, autograd, data loading and inference

---

## 1. Autograd: how gradients happen

PyTorch records every operation on tensors that need gradients. This covers all `nn.Module` parameters, plus any tensor created with `requires_grad=True`. Calling `.backward()` on a scalar loss walks that record backwards and fills `.grad` on each parameter.

| Syntax | What it does | Where |
|---|---|---|
| `loss.backward()` | Computes d(loss)/d(param) for every parameter and **adds** it to `param.grad`. | [train_rqvae.py:136](../tokenizer/train_rqvae.py#L136), [train_model.py:145](../model/train_model.py#L145) |
| `tensor.grad` | The stored gradient, or `None` if none has been computed yet. | Self-tests check it: [quantizer.py:114](../tokenizer/quantizer.py#L114), [rqvae.py:143](../tokenizer/rqvae.py#L143) |
| `requires_grad=True` | Makes autograd track a plain tensor. | [quantizer.py:102](../tokenizer/quantizer.py#L102) |
| `x.detach()` | The same values, cut off from the graph, so no gradient flows through it. | See §2 |

### `torch.no_grad()`: turning autograd off

This is used in two forms:
```python
@torch.no_grad()                   # as a decorator on a whole function
def get_codes(self, x): ...        # rqvae.py:73, :80; train_rqvae.py:49; beam_search.py:33

with torch.no_grad():              # as a context manager for a block
    for padded, lengths in val_loader: ...   # train_model.py:155
```
**Why:** during validation, inference or weight surgery you don't need gradients. Turning them off saves memory (no graph is stored) and time. It also lets you modify parameters in place without autograd complaining.

---

## 2. `detach()` and the straight-through estimator

This is the most interesting autograd trick in the repo ([quantizer.py:76-90](../tokenizer/quantizer.py#L76-L90)).

**Problem:** `argmin` (choosing the nearest centroid) has no gradient, so the encoder would never learn.

**Fix 1, losses that target one side each:**
```python
codebook_loss   += F.mse_loss(chosen, residual.detach())   # gradient moves ONLY the centroid
commitment_loss += F.mse_loss(residual, chosen.detach())   # gradient moves ONLY the encoder
```
`detach()` treats one side as a constant, so each loss pulls only the side you intend.

**Fix 2, the straight-through estimator:**
```python
z_q = z + (z_q - z).detach()
```
- **Forward pass:** `z + z_q - z = z_q`, so the decoder sees the quantized vector.
- **Backward pass:** the detached term is a constant, so `d z_q / d z = 1`. The gradient flows straight through to `z` as if quantization were the identity.

The line `print("grad reaches z:", ...)` in the self-test ([quantizer.py:113-114](../tokenizer/quantizer.py#L113-L114)) verifies exactly this.

---

## 3. Data loading: `Dataset`, `TensorDataset`, `DataLoader`

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

### `TensorDataset`: when the data is already a tensor
```python
loader = DataLoader(TensorDataset(emb_dev), batch_size=512, shuffle=True)  # train_rqvae.py:126
for (batch,) in loader:        # each item is a tuple, even with one tensor -> unpack with (batch,)
```

### A custom `Dataset`: when items are Python objects
```python
class SeqDataset(Dataset):                        # train_model.py:45
    def __init__(self, sequences): self.sequences = sequences
    def __len__(self):             return len(self.sequences)    # required
    def __getitem__(self, i):      return self.sequences[i]      # required
```
A map-style Dataset only needs `__len__` and `__getitem__`.

### `DataLoader` arguments used
```python
DataLoader(SeqDataset(train_seqs), batch_size=64, shuffle=True,
           collate_fn=pad_batch, num_workers=0)       # train_model.py:117
```
| Argument | Meaning | Why here |
|---|---|---|
| `batch_size` | Items per batch. | 512 for RQ-VAE vectors, 64 for token sequences. |
| `shuffle=True` | Reshuffles every epoch. | Training sets are shuffled, and the validation set is not ([train_model.py:120](../model/train_model.py#L120)). |
| `collate_fn=pad_batch` | A function that turns a **list of items** into a batch. | The sequences have different lengths, so the default collate (which stacks tensors) would fail. `pad_batch` ([train_model.py:55-62](../model/train_model.py#L55-L62)) right-pads with `PAD` and returns `(padded, lengths)`. |
| `num_workers=0` | Load data in the main process. | Simple and safe on macOS. Values above 0 use subprocesses. |
| `len(loader)` | The number of batches. | Averages epoch loss ([train_model.py:150](../model/train_model.py#L150)) and computes total scheduler steps ([:125](../model/train_model.py#L125)). |

---

## 4. Optimizers

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)   # train_model.py:123
```
- **AdamW** is Adam with *decoupled* weight decay, the standard choice for transformers. It is also used for the RQ-VAE ([train_rqvae.py:127](../tokenizer/train_rqvae.py#L127)).
- `model.parameters()` tells the optimizer which tensors to update.

| Method | What it does |
|---|---|
| `opt.zero_grad()` | Clears `.grad` on every parameter. **Required**, because `backward()` *adds* to existing gradients. |
| `opt.step()` | Updates each parameter using its `.grad`. |
| `opt.state_dict()` | The optimizer's internal state (Adam's moment estimates). Saved so training can resume ([train_model.py:173](../model/train_model.py#L173)). |

---

## 5. Learning-rate schedulers

```python
def lr_lambda(step):                                    # train_model.py:127-131
    if step < warmup_steps:
        return step / max(1, warmup_steps)              # linear warmup 0 -> 1
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * prog))         # cosine decay 1 -> 0
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
```
| Syntax | Meaning |
|---|---|
| `LambdaLR(opt, fn)` | The learning rate becomes `base_lr × fn(step)`. You write any schedule you like as a Python function. |
| `sched.step()` | Advances the schedule. It's called **once per batch**, after `opt.step()`, because this schedule counts steps rather than epochs ([train_model.py:148](../model/train_model.py#L148)). |
| `sched.get_last_lr()[0]` | The current LR, for logging ([train_model.py:161](../model/train_model.py#L161)). It returns a list with one entry per parameter group. |
| `sched.state_dict()` | Saved with checkpoints so training can resume ([train_model.py:173](../model/train_model.py#L173)). |

**Why warmup + cosine:** Adam's statistics are noisy for the first few steps, so a small LR early avoids large bad updates. Cosine decay then settles the model into a minimum.

---

## 6. Gradient clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # train_model.py:146
```
If the combined L2 norm of all gradients is larger than `GRAD_CLIP` (1.0), every gradient is scaled down to fit. This stops one bad batch from blowing up the weights, which is standard for transformers. The call goes **after `backward()` and before `step()`**.

---

## 7. The training loop (the pattern to memorise)

From [train_model.py:135-163](../model/train_model.py#L135-L163):

```python
for epoch in range(1, EPOCHS + 1):
    model.train()                          # dropout ON
    for padded, _ in train_loader:
        padded = padded.to(device)         # 1. data -> same device as model
        opt.zero_grad()                    # 2. clear old grads
        loss = train_loss_fn(model, padded)# 3. forward + loss
        loss.backward()                    # 4. backward
        nn.utils.clip_grad_norm_(...)      # 5. (optional) clip
        opt.step()                         # 6. update weights
        sched.step()                       # 7. (optional) update LR
        running += loss.item()             # 8. log as a python float

    model.eval()                           # dropout OFF
    with torch.no_grad():                  # no graph, less memory
        for padded, lengths in val_loader:
            ...val_loss_fn(...).item()
```

### `model.train()` vs `model.eval()`
- They **don't** turn gradients on or off. `torch.no_grad()` does that.
- They switch layers that behave differently in the two modes. In this repo that's `nn.Dropout`.
- Forgetting `eval()` at inference gives random, noisy predictions ([beam_search.py:88](../inference/beam_search.py#L88) says as much in its comment).
- `usage_and_revive` switches to `eval()` and then back to `train()` ([train_rqvae.py:54](../tokenizer/train_rqvae.py#L54), [:73](../tokenizer/train_rqvae.py#L73)). If you leave training mode inside a loop, restore it afterwards.

### Next-token loss (teacher forcing)
```python
inp, tgt = padded[:, :-1], padded[:, 1:]          # train_model.py:81
logits = model(inp)                               # (B, L-1, V)
F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=vocab.PAD)
```
Position `t` of `inp` predicts position `t` of `tgt`, which is the next token. The causal mask stops the model from seeing the answer.

---

## 8. Checkpoints

**Save.** A dict lets you store the model, optimizer, scheduler and metadata together:
```python
torch.save({"epoch": epoch, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "val_loss": val_loss}, f"{CKPT_DIR}/model_epoch{epoch}.pt")   # train_model.py:172
```
The project keeps `model_best.pt`, updated whenever validation loss improves (the usual early-stopping selection), plus a checkpoint every 10 epochs and a final one.

**Load** ([beam_search.py:73-88](../inference/beam_search.py#L73-L88)):
```python
model = RecTransformer(...same hyper-params...)          # 1. rebuild the architecture
ckpt  = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])                     # 2. pour in the weights
model = model.to(device).eval()                          # 3. device + eval mode
```
A `state_dict` holds only tensors, not code. That's why you have to rebuild the class with the same sizes first.

To **resume training** you would also call `opt.load_state_dict(ckpt["opt"])` and `sched.load_state_dict(ckpt["sched"])`. The project doesn't do this, but it saves everything needed for it.

---

## 9. Custom in-place weight surgery (dead-code revival)

[train_rqvae.py:49-74](../tokenizer/train_rqvae.py#L49-L74) combines several of these tools:
```python
@torch.no_grad()
def usage_and_revive(model, embeddings, revive=False):
    residual = model.encoder(embeddings)
    for cb in model.quantizer.codebooks:
        idx = torch.cdist(residual, cb.weight).argmin(dim=1)          # nearest code per item
        counts = torch.bincount(idx.cpu(), minlength=cb.num_embeddings)  # usage histogram
        dead = torch.nonzero(counts == 0, as_tuple=False).squeeze(1)   # unused code ids
        pick = torch.randint(0, residual.shape[0], (dead.numel(),), device=residual.device)
        cb.weight.data[dead.to(residual.device)] = residual[pick] + 0.01 * torch.randn_like(residual[pick])
        residual = residual - cb(idx)
```
It finds codebook entries no item uses and moves each one onto a real data point, so every code can be used.

---

## 10. Inference: trie-constrained beam search

[beam_search.py:33-70](../inference/beam_search.py#L33-L70) is a good exercise in tensor bookkeeping.

```python
seqs   = torch.tensor([context + prefix for prefix, _ in beams], device=device)  # (nb, L)
logits = model(seqs)[:, -1, :]                         # (nb, V): only the last position
mask   = torch.full((len(beams), V), float("-inf"))    # everything illegal...
mask[i, trie.get_valid_tokens(prefix)] = 0.0           # ...except tokens the trie allows
logp   = torch.log_softmax(logits + mask.to(device), dim=-1)  # renormalise over legal tokens
prev   = torch.tensor([s for _, s in beams], device=device).unsqueeze(1)   # (nb, 1)
scores = (logp + prev).view(-1)                        # (nb*V,) all (beam, token) pairs, flat
k = min(beam_width, int(torch.isfinite(scores).sum().item()))
top_scores, top_idx = scores.topk(k)
b, tok = divmod(flat_idx, V)                           # flat index -> (beam, token)
```
Ideas worth taking from this:
- **Adding `-inf` before `log_softmax`** gives banned tokens exactly zero probability and renormalises the rest.
- **Flatten, `topk`, then `divmod`** chooses the best (beam, token) pairs across all beams at once.
- **Log-probabilities are added**, not multiplied, which keeps the numbers stable.
- The whole function is under `@torch.no_grad()`.

---

## 11. Checks worth copying

The project includes small self-tests that are good PyTorch habits:
| Check | Code |
|---|---|
| Output shapes are what you expect | [transformer.py:148-151](../model/transformer.py#L148-L151) |
| Gradients reach the layer that should learn | [quantizer.py:113-114](../tokenizer/quantizer.py#L113-L114), [rqvae.py:142-145](../tokenizer/rqvae.py#L142-L145) |
| Gradients are finite (no NaN/inf) | `torch.isfinite(grad).all().item()` |
| The causal mask doesn't leak the future | [transformer.py:155-159](../model/transformer.py#L155-L159): change future tokens with `torch.allclose` and confirm past logits stay the same |
| Token ids stay inside the vocab | [build_token_sequences.py:63-64](../data/build_token_sequences.py#L63-L64) |

---

## 12. One non-PyTorch-core library that returns tensors

```python
from sentence_transformers import SentenceTransformer        # data/embeddings.py:18
model.encode(sentences, batch_size=256, convert_to_tensor=True, ...).cpu()
```
`convert_to_tensor=True` returns a `torch.Tensor` instead of a NumPy array, and `.cpu()` moves it off MPS before `torch.save`. The model is used frozen, with no training.

---

## Cheat sheet: every PyTorch API used in this repo

**Tensor creation:** `tensor`, `zeros_like`, `ones`, `full`, `full_like`, `randn`, `randn_like`, `randint`, `arange`, `tril`, `from_numpy`
**Shape:** `.shape`, `.size`, `.numel`, `.view`, `.reshape`, `.transpose`, `.contiguous`, `.unsqueeze`, `.squeeze`, `.split`, `stack`, `cat`, `[None, :]`
**Math:** `@`, `+ - / %`, `.norm`, `cdist`, `argmin`, `softmax`, `log_softmax`, `topk`, `where`, `masked_fill`, `isfinite`, `allclose`, `& == > < >=`
**Reductions:** `sum`, `min`, `max`, `mean`, `median`, `all`, `unique`, `bincount`, `nonzero`
**Conversion:** `.item`, `.tolist`, `int()`, `float()`, `.numpy`, `.float`, `.clone`, `.detach`, `.cpu`, `.to(device)`, `torch.long`
**Device and seed:** `torch.backends.mps.is_available`, `torch.device`, `torch.manual_seed`
**I/O:** `torch.save`, `torch.load(map_location=, weights_only=)`
**nn:** `Module`, `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `ReLU`, `GELU`, `Sequential`, `ModuleList`, `init.normal_`, `init.zeros_`, `register_buffer`, `apply`, `parameters`, `state_dict`, `load_state_dict`, `train`, `eval`, `utils.clip_grad_norm_`
**F:** `mse_loss`, `cross_entropy(ignore_index=)`, `softmax`
**Autograd:** `backward`, `.grad`, `requires_grad`, `no_grad` (decorator and context), `.detach`, `.data`, `copy_`
**Optim:** `optim.AdamW`, `zero_grad`, `step`, `lr_scheduler.LambdaLR`, `get_last_lr`
**Data:** `Dataset`, `TensorDataset`, `DataLoader(batch_size, shuffle, collate_fn, num_workers)`
