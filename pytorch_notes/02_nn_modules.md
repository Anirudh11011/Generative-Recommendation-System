# 02: Building models with `torch.nn`

`torch.nn` holds the building blocks for neural networks. There are two families:
- **`nn.XYZ` classes** (`nn.Linear`, `nn.Embedding`, …) are *modules*. They **own learnable parameters** (weights) and remember them.
- **`F.xyz` functions** (`F.mse_loss`, `F.softmax`, …) from `torch.nn.functional` are *stateless* maths. You pass tensors in and get tensors out, and nothing is stored.

---

## 1. `nn.Module`: how every model is written

Every model in this repo follows the same pattern:

```python
class ResidualQuantizer(nn.Module):              # 1. subclass nn.Module
    def __init__(self, ...):
        super().__init__()                       # 2. ALWAYS call this first
        self.codebooks = nn.ModuleList([...])    # 3. assign layers as attributes
    def forward(self, z):                        # 4. define the computation
        ...
        return z_q, codes, vq_loss

rq = ResidualQuantizer()
z_q, codes, vq_loss = rq(z)                      # 5. call the object, NOT rq.forward(z)
```
Source: [quantizer.py:20-92](../tokenizer/quantizer.py#L20-L92)

| Piece | Why it matters |
|---|---|
| `super().__init__()` | Sets up the internal registries for parameters, buffers and submodules. If you skip it, assigning a layer as an attribute fails. |
| Assigning `self.x = nn.Linear(...)` | Any `nn.Module` or `nn.Parameter` assigned as an attribute is **registered automatically**. That's how `model.parameters()`, `.to(device)` and `state_dict()` find it. |
| `forward(self, ...)` | Your computation. It can return anything: a tensor, a tuple ([quantizer.py:92](../tokenizer/quantizer.py#L92)) or a dict ([rqvae.py:65](../tokenizer/rqvae.py#L65)). |
| `model(x)` instead of `model.forward(x)` | `__call__` runs hooks and then `forward`. Always call the module itself. |
| Modules inside modules | `RQVAE` contains a `ResidualQuantizer` ([rqvae.py:43](../tokenizer/rqvae.py#L43)). `RecTransformer` contains `TransformerBlock`s, and each of those contains a `CausalSelfAttention`. Parameters are collected recursively. |
| Extra methods | You can add any method you like, for example `get_codes` and `init_codebooks_kmeans` in [rqvae.py:74-109](../tokenizer/rqvae.py#L74-L109). |

---

## 2. Layers used

### `nn.Linear(in_features, out_features, bias=True)`
Computes `y = x @ W.T + b`. It works on the **last** dimension, so `(B, L, in)` becomes `(B, L, out)`.

| Where | Purpose |
|---|---|
| [rqvae.py:37-39](../tokenizer/rqvae.py#L37-L39) | Encoder 384 → 512 → 256 |
| [rqvae.py:52-54](../tokenizer/rqvae.py#L52-L54) | Decoder 256 → 512 → 384 |
| [transformer.py:30](../model/transformer.py#L30) | `nn.Linear(d, 3*d)`: one fused layer that produces Q, K and V together, which is faster than three separate layers |
| [transformer.py:31](../model/transformer.py#L31) | Output projection that recombines the heads |
| [transformer.py:80-82](../model/transformer.py#L80-L82) | Feed-forward 128 → 512 → 128 |
| [transformer.py:109](../model/transformer.py#L109) | `bias=False`: the LM head that maps hidden states to vocab logits (no bias because its weight is tied, see §6) |

### `nn.Embedding(num_embeddings, embedding_dim)`
A **lookup table**: a `(num_embeddings, dim)` weight matrix, where calling it with integer ids returns those rows. The input must be `long`.

| Where | Purpose |
|---|---|
| [transformer.py:100](../model/transformer.py#L100) `tok_emb` | token id → 128-dim vector |
| [transformer.py:101](../model/transformer.py#L101) `pos_emb` | position 0..199 → vector (learned positional encoding) |
| [quantizer.py:38-40](../tokenizer/quantizer.py#L38-L40) | **Codebooks**. A codebook is just a table of 256 centroid vectors, so an Embedding fits exactly. `cb(idx)` fetches the chosen centroids ([quantizer.py:69](../tokenizer/quantizer.py#L69)). `cb.weight` gives the whole table ([quantizer.py:63](../tokenizer/quantizer.py#L63)). `cb.num_embeddings` returns 256 ([rqvae.py:97](../tokenizer/rqvae.py#L97)). |

### Activations
| Layer | Formula | Where and why |
|---|---|---|
| `nn.ReLU()` | `max(0, x)` | RQ-VAE encoder and decoder ([rqvae.py:38](../tokenizer/rqvae.py#L38)). Simple and standard for MLPs. |
| `nn.GELU()` | A smooth version of ReLU | Transformer feed-forward ([transformer.py:81](../model/transformer.py#L81)). The standard choice in GPT and BERT. |

### `nn.LayerNorm(d_model)`
Normalises each token vector to mean 0 and variance 1 over its features, then applies a learned scale and shift. It keeps activations in a stable range as depth grows.
- [transformer.py:76-78](../model/transformer.py#L76-L78) uses **pre-norm**: `x + attn(norm1(x))`.
- [transformer.py:108](../model/transformer.py#L108) applies a final norm before the output head.

### `nn.Dropout(p)`
While training, it zeroes a random fraction `p` of values and scales up the rest. In **`eval()` mode it does nothing**. This is regularisation against overfitting.
- It's applied to embeddings ([transformer.py:102](../model/transformer.py#L102)), to the attention weights ([:32](../model/transformer.py#L32)), to the residual output ([:33](../model/transformer.py#L33)) and inside the FFN ([:83](../model/transformer.py#L83)).

---

## 3. Containers

| Container | What it does | Where |
|---|---|---|
| `nn.Sequential(a, b, c)` | Runs its modules in order: `c(b(a(x)))`. Children can be accessed by index, e.g. `model.encoder[0]`. | Encoder and decoder MLPs ([rqvae.py:36](../tokenizer/rqvae.py#L36)). The FFN ([transformer.py:79](../model/transformer.py#L79)). [rqvae.py:143](../tokenizer/rqvae.py#L143) reads `model.encoder[0].weight.grad`. |
| `nn.ModuleList([...])` | A list that **registers** its modules. It has no `forward`, so you loop over it yourself. | Codebooks ([quantizer.py:38](../tokenizer/quantizer.py#L38)) and transformer blocks ([transformer.py:104](../model/transformer.py#L104)), both iterated with `for block in self.blocks`. |

> ⚠️ A plain Python list `[nn.Linear(...), ...]` would **not** register its layers. Their weights would be missing from `.parameters()` (so the optimizer would never update them), from `.to(device)` and from `state_dict`. That's why `ModuleList` exists.

---

## 4. `torch.nn.functional` (F)

| Function | What it does | Where and why |
|---|---|---|
| `F.mse_loss(pred, target)` | Mean squared error. | RQ-VAE reconstruction loss ([rqvae.py:62](../tokenizer/rqvae.py#L62)). Codebook and commitment losses ([quantizer.py:76-77](../tokenizer/quantizer.py#L76-L77)). |
| `F.cross_entropy(logits, target, ignore_index=PAD)` | Softmax plus negative log-likelihood. Expects `logits` of shape `(N, C)` and `target` of shape `(N,)` with dtype long. `ignore_index` leaves those targets out of the loss and its average. | Next-token loss ([train_model.py:83-87](../model/train_model.py#L83-L87)). Padding positions are ignored, and the validation loss reuses the same trick by setting every unwanted target to PAD ([train_model.py:98](../model/train_model.py#L98)). |
| `F.softmax(x, dim=-1)` | Turns scores into probabilities that sum to 1 along `dim`. | Attention weights ([transformer.py:61](../model/transformer.py#L61)). |

Why the `reshape(-1, V)` in the cross-entropy call? `F.cross_entropy` expects classes in dimension 1. Flattening `(B, L, V)` to `(B·L, V)` and `(B, L)` to `(B·L,)` treats each position as its own classification example.

---

## 5. Weight initialisation

| Syntax | What it does | Where |
|---|---|---|
| `nn.init.normal_(w, mean, std)` | Fills `w` in place with normal random values. The trailing `_` means **in-place**, a PyTorch-wide convention. | Codebooks with std 0.1 ([quantizer.py:42](../tokenizer/quantizer.py#L42)). Linear and embedding weights with std 0.02, the GPT-2 recipe ([transformer.py:119-123](../model/transformer.py#L119-L123)). |
| `nn.init.zeros_(b)` | Fills with zeros. | Biases ([transformer.py:121](../model/transformer.py#L121)). |
| `self.apply(fn)` | Calls `fn(m)` on **every** submodule recursively. | [transformer.py:115](../model/transformer.py#L115) runs `_init_weights` over the whole model. |
| `isinstance(m, nn.Linear)` | Chooses a different init per layer type. | [transformer.py:118-122](../model/transformer.py#L118-L122) |
| `m.bias is not None` | Handles layers created with `bias=False`. | [transformer.py:120](../model/transformer.py#L120) |
| `cb.weight.data.copy_(tensor)` | Overwrites weights in place with custom values, without autograd recording it. | k-means centroids go into the codebooks ([rqvae.py:102](../tokenizer/rqvae.py#L102)). |

> `.data` gives the raw tensor without autograd tracking. The modern equivalent is to modify weights inside `with torch.no_grad():`. This repo does both: `init_codebooks_kmeans` is decorated with `@torch.no_grad()` and *also* uses `.data`.

---

## 6. Buffers and weight tying

### `self.register_buffer("name", tensor)`
[transformer.py:39](../model/transformer.py#L39) stores the causal mask as a **buffer**. A buffer:
- moves with `model.to(device)`,
- is saved in `state_dict()`,
- is **not** a parameter, so the optimizer never updates it.

Use a buffer for fixed tensors a module needs, such as masks or running statistics. A plain `self.mask = tensor` would *not* move to the GPU with the model.

### Weight tying
```python
self.head.weight = self.tok_emb.weight      # transformer.py:113
```
Both layers now share **one** `(786, 128)` matrix: the embedding that maps id → vector and the output layer that maps vector → id scores. This cuts parameters and ties the input and output token spaces together. Assigning one `nn.Parameter` to two modules is all it takes. `head` has `bias=False` because there is no matching bias on the embedding side.

---

## 7. Other `nn.Module` methods used

| Method | What it does | Where |
|---|---|---|
| `model.parameters()` | An iterator over every learnable tensor. | Given to the optimizer ([train_model.py:123](../model/train_model.py#L123)), to gradient clipping, and to the parameter count. |
| `model.state_dict()` | An ordered dict of name → tensor (parameters and buffers). | Saved in checkpoints ([train_rqvae.py:151](../tokenizer/train_rqvae.py#L151), [train_model.py:168](../model/train_model.py#L168)). |
| `model.load_state_dict(state)` | Copies saved tensors back into a freshly built model. The architecture has to match. | [beam_search.py:87](../inference/beam_search.py#L87) |
| `model.train()` / `model.eval()` | Switches mode. Dropout (and BatchNorm) behave differently in each. Both return `self`, so calls can be chained. | [transformer.py:143](../model/transformer.py#L143): `RecTransformer().to(device).eval()`. More in file 03. |
| `model.to(device)` | Moves all parameters and buffers. | Throughout. |
| `self.encoder(x)` | Calling a submodule directly. | [train_rqvae.py:55](../tokenizer/train_rqvae.py#L55) uses only the encoder of a trained RQ-VAE. |

---

## 8. Walkthrough: `CausalSelfAttention.forward`

[transformer.py:41-68](../model/transformer.py#L41-L68). B = batch, L = sequence length, D = 128, h = 4 heads, hd = 32.

```python
B, L, D = x.shape                                   # x: (B, L, D)
q, k, v = self.qkv(x).split(D, dim=2)               # Linear -> (B,L,3D), split -> 3 × (B,L,D)

q = q.view(B, L, h, hd).transpose(1, 2)             # (B,L,D) -> (B,L,h,hd) -> (B,h,L,hd)
# same for k, v

scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B,h,L,hd)@(B,h,hd,L) -> (B,h,L,L)
scores = scores.masked_fill(mask[:, :, :L, :L] == 0, float("-inf"))   # hide the future
attn = F.softmax(scores, dim=-1)                    # each row -> probability distribution
attn = self.attn_dropout(attn)

y = attn @ v                                        # (B,h,L,L)@(B,h,L,hd) -> (B,h,L,hd)
y = y.transpose(1, 2).contiguous().view(B, L, D)    # merge heads back
y = self.resid_dropout(self.out(y))
```
Each line uses something from file 01 or this file: `Linear`, `split`, `view`, `transpose`, `@`, broadcasting, `masked_fill`, `softmax`, `Dropout`, `contiguous`.

> PyTorch ships the fused `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, which replaces the four lines from `scores` to `attn @ v` and runs faster. The project writes it out by hand so you can see how it works ([transformer.py:8-9](../model/transformer.py#L8-L9)).

## 9. Walkthrough: `RecTransformer.forward`

[transformer.py:125-137](../model/transformer.py#L125-L137)

```python
pos = torch.arange(L, device=idx.device)       # [0..L-1]
x = self.tok_emb(idx) + self.pos_emb(pos)      # (B,L,D) + (L,D) -> broadcast
x = self.drop(x)
for block in self.blocks:                      # ModuleList iteration
    x = block(x)                               # each: x + attn(norm(x)); x + ffn(norm(x))
x = self.norm_f(x)
return self.head(x)                            # (B, L, vocab_size) logits
```
