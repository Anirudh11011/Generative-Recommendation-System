# 01: Tensors

A **tensor** is PyTorch's n-dimensional array. It works like a NumPy array, with two additions:
1. It can live on a GPU (`"mps"` on Apple Silicon, `"cuda"` on NVIDIA).
2. It can record the operations applied to it, so PyTorch can compute gradients automatically (see file 03).

Everything else in PyTorch is built on tensors.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
Every model file starts with these three imports. By convention, `nn` holds the layers and `F` holds the stateless functions.

---

## 1. Creating tensors

| Syntax | What it does | Where it's used and why |
|---|---|---|
| `torch.tensor(data, dtype=torch.long, device=device)` | Builds a tensor from a Python list or number. | [train_model.py:57](../model/train_model.py#L57) turns sequence lengths into a tensor. [beam_search.py:43](../inference/beam_search.py#L43) turns the token-id lists for each beam into a `(num_beams, L)` batch for the model. |
| `torch.zeros_like(z)` | A tensor of zeros with the same shape, dtype and device as `z`. | [quantizer.py:57](../tokenizer/quantizer.py#L57) starts the running sum `z_q` of chosen centroids. The `_like` form means you never have to spell out shape or device yourself. |
| `torch.ones(n, n)` | A tensor of ones. | [transformer.py:38](../model/transformer.py#L38) is the starting point for the causal mask. |
| `torch.tril(t)` | Keeps the lower triangle (row ≥ column) and sets everything above it to 0. | [transformer.py:38](../model/transformer.py#L38) builds the causal mask, where 1 means "position i may look at position j". |
| `torch.full((B, L), value, dtype=torch.long)` | A tensor filled with one value. | [train_model.py:59](../model/train_model.py#L59) creates a batch pre-filled with `PAD`. [beam_search.py:50](../inference/beam_search.py#L50) creates a mask pre-filled with `-inf`. |
| `torch.full_like(tgt, value)` | Same as `full`, with shape, dtype and device copied from `tgt`. | [train_model.py:98](../model/train_model.py#L98) supplies the "replace with PAD" values for `torch.where`. |
| `torch.randn(B, D)` | Random numbers from a standard normal distribution (mean 0, std 1). | [quantizer.py:102](../tokenizer/quantizer.py#L102) and [rqvae.py:120](../tokenizer/rqvae.py#L120) create fake data for self-tests. |
| `torch.randn(..., requires_grad=True)` | Asks autograd to track this tensor and store its gradient in `.grad`. | [quantizer.py:102](../tokenizer/quantizer.py#L102) lets the test check that gradients reach `z`. |
| `torch.randn_like(x)` | Normal noise with the same shape as `x`. | [train_rqvae.py:69](../tokenizer/train_rqvae.py#L69) adds a little noise when reviving dead codebook entries, so two revived codes don't end up identical. |
| `torch.randint(low, high, size, device=)` | Random integers in `[low, high)`. | [train_rqvae.py:66](../tokenizer/train_rqvae.py#L66) picks random rows. [transformer.py:148](../model/transformer.py#L148) creates fake token ids. |
| `torch.arange(n, device=)` | `[0, 1, …, n-1]` | [transformer.py:129](../model/transformer.py#L129) gives position ids for the positional embedding. [train_model.py:96](../model/train_model.py#L96) gives column indices for building a mask. |
| `torch.from_numpy(arr)` | Wraps a NumPy array as a tensor. The two share memory, so nothing is copied. | [rqvae.py:102](../tokenizer/rqvae.py#L102) loads k-means centroids from scikit-learn into the codebook. |

> Good habit: pass `device=` when you create a tensor. [train_model.py:96](../model/train_model.py#L96) uses `device=tgt.device` so the new tensor is on the same device as the data it will be combined with. Mixing CPU and MPS tensors in one operation raises an error.

---

## 2. Dtypes

| Syntax | Meaning | Why it's used here |
|---|---|---|
| `torch.long` (= `torch.int64`) | 64-bit integer | Token ids, indices and codes. `nn.Embedding` and `F.cross_entropy` targets **require** `long`. See [train_rqvae.py:90](../tokenizer/train_rqvae.py#L90) and [train_model.py:57-61](../model/train_model.py#L57-L61). |
| `torch.float32` (the default float) | 32-bit float | Embeddings, weights and activations. MPS does not support float64. |
| `.float()` | Convert to float32 | [train_rqvae.py:110](../tokenizer/train_rqvae.py#L110) makes sure the loaded embeddings are float32. [build_token_sequences.py:69](../data/build_token_sequences.py#L69) uses `lens.float().mean()` because `mean()` does not work on integer tensors. |
| `.dtype` | Read a tensor's dtype | [quantizer.py:107](../tokenizer/quantizer.py#L107) prints it. |
| `arr.astype(np.float32)` (NumPy) | Make the NumPy array float32 **before** `from_numpy` | [rqvae.py:99](../tokenizer/rqvae.py#L99). scikit-learn returns float64, and the codebook weights are float32. |

---

## 3. Shape inspection

| Syntax | What it returns |
|---|---|
| `x.shape` | A `torch.Size`, which behaves like a tuple. You can unpack it: `B, L, D = x.shape` ([transformer.py:42](../model/transformer.py#L42)). |
| `x.size(-1)` | The size of one dimension. `-1` means the last dimension ([train_model.py:84](../model/train_model.py#L84)). |
| `tuple(x.shape)` | Easier to read when printed ([rqvae.py:138](../tokenizer/rqvae.py#L138)). |
| `x.numel()` | The total number of elements. [train_rqvae.py:65](../tokenizer/train_rqvae.py#L65) uses it to check whether any dead codes exist. [transformer.py:145](../model/transformer.py#L145) uses it to count parameters. |
| `len(x)` | The size of dimension 0. |

---

## 4. Reshaping and rearranging

This group matters most for understanding transformers. The main example is [transformer.py:41-68](../model/transformer.py#L41-L68).

| Syntax | What it does | Example in this repo |
|---|---|---|
| `x.view(new_shape)` | Returns a new *view* of the same memory with a different shape. Nothing is copied. It only works if the memory is contiguous. | `q.view(B, L, n_heads, head_dim)` splits the last dimension into heads ([transformer.py:48](../model/transformer.py#L48)). `scores.view(-1)` flattens ([beam_search.py:58](../inference/beam_search.py#L58)). `-1` means "work this size out from the others". |
| `x.reshape(new_shape)` | Like `view`, but it copies when it has to, so it always works. | `logits.reshape(-1, V)` turns `(B, L, V)` into `(B·L, V)` for cross-entropy ([train_model.py:84](../model/train_model.py#L84)). |
| `x.transpose(d0, d1)` | Swaps two dimensions. | `.transpose(1, 2)` turns `(B, L, h, hd)` into `(B, h, L, hd)` so each head is its own batch ([transformer.py:48](../model/transformer.py#L48)). `k.transpose(-2, -1)` gets `Kᵀ` for `Q @ Kᵀ` ([transformer.py:55](../model/transformer.py#L55)). |
| `x.contiguous()` | Copies the tensor into contiguous memory. | After `transpose` the memory is out of order, so `view` would fail. [transformer.py:66](../model/transformer.py#L66) calls `.transpose(1, 2).contiguous().view(B, L, D)`. |
| `x.unsqueeze(dim)` | Inserts a size-1 dimension. | [train_rqvae.py:91](../tokenizer/train_rqvae.py#L91) makes `(N,)` into `(N, 1)` so it can be concatenated as a column. [beam_search.py:57](../inference/beam_search.py#L57) makes `(nb,)` into `(nb, 1)` so it broadcasts over the vocab. |
| `x.squeeze(dim)` | Removes a size-1 dimension. | [train_rqvae.py:64](../tokenizer/train_rqvae.py#L64): `nonzero` returns `(k, 1)` and `squeeze(1)` makes it `(k,)`. |
| `x.split(size, dim)` | Cuts a tensor into chunks of `size` along `dim`. | `self.qkv(x).split(D, dim=2)` cuts the fused `(B, L, 3D)` projection into Q, K and V ([transformer.py:44](../model/transformer.py#L44)). |
| `torch.stack(list, dim)` | Joins tensors along a **new** dimension. | [quantizer.py:84](../tokenizer/quantizer.py#L84): three `(B,)` code tensors become `(B, 3)`. |
| `torch.cat(list, dim)` | Joins tensors along an **existing** dimension. | [train_rqvae.py:89](../tokenizer/train_rqvae.py#L89): `(N, 3)` joined with `(N, 1)` gives `(N, 4)` semantic IDs. |
| `mask.view(1, 1, L, L)` | Adds leading dimensions so the tensor broadcasts. | [transformer.py:39](../model/transformer.py#L39) lets a single mask apply to every batch and every head. |

**Broadcasting:** when two tensors of different shapes are combined, size-1 dimensions (or missing leading ones) are stretched to match. Examples in this repo:
- `tok_emb(idx)` has shape `(B, L, D)` and `pos_emb(pos)` has shape `(L, D)`. Adding them gives `(B, L, D)` ([transformer.py:130](../model/transformer.py#L130)).
- `ar[None, :]` has shape `(1, L)` and `tlen[:, None]` has shape `(B, 1)`. Comparing them gives `(B, L)` ([train_model.py:96-97](../model/train_model.py#L96-L97)). Indexing with `None` works like `unsqueeze`.

---

## 5. Indexing and slicing

| Syntax | Meaning | Where |
|---|---|---|
| `padded[:, :-1]`, `padded[:, 1:]` | All rows. Every column except the last, or every column except the first. | [train_model.py:81](../model/train_model.py#L81). This builds input and target for next-token prediction: the target is the input shifted by one. |
| `logits[:, -1, :]` | The last time step of every sequence. | [beam_search.py:46](../inference/beam_search.py#L46). Only the final position predicts the next token. |
| `self.causal_mask[:, :, :L, :L]` | Slice the precomputed 200×200 mask down to the current length. | [transformer.py:59](../model/transformer.py#L59) |
| `ct[:, 0]`, `ct[:, :2]`, `sem_ids[:, 3]` | Pick columns. | [train_rqvae.py:159-160](../tokenizer/train_rqvae.py#L159-L160), [build_token_sequences.py:40-42](../data/build_token_sequences.py#L40-L42) |
| `residual[pick]` (tensor of indices) | **Fancy indexing**, which gathers the listed rows. | [train_rqvae.py:68-69](../tokenizer/train_rqvae.py#L68-L69) |
| `cb.weight.data[dead] = ...` | Writes into the listed rows. | [train_rqvae.py:68](../tokenizer/train_rqvae.py#L68) overwrites dead codebook entries. |
| `padded[i, :len(s)] = tensor` | Writes a slice. | [train_model.py:61](../model/train_model.py#L61) copies a sequence into its padded row. |
| `mask[i, list_of_ids] = 0.0` | Fancy-index assignment with a Python list. | [beam_search.py:52](../inference/beam_search.py#L52) marks the tokens the trie allows. |
| `x2[:, 10:] = ...` | Modifies part of a clone. | [transformer.py:157](../model/transformer.py#L157), used in the causal-mask test. |
| `codes[0]`, `sem_ids[i]` | Picks one row. | Used throughout. |

---

## 6. Math and element-wise operations

| Syntax | What it does | Where and why |
|---|---|---|
| `+ - * /` | Element-wise, with broadcasting. | `residual = residual - chosen` in the RQ loop ([quantizer.py:81](../tokenizer/quantizer.py#L81)). |
| `a @ b` | Matrix multiply (`torch.matmul`). For 4-D tensors it multiplies over the last two dimensions and batches over the first two. | `q @ k.transpose(-2,-1)` gives attention scores `(B,h,L,L)`, and `attn @ v` gives `(B,h,L,hd)` ([transformer.py:55](../model/transformer.py#L55), [:64](../model/transformer.py#L64)). |
| `x % n` | Element-wise modulo. | [transformer.py:157](../model/transformer.py#L157) keeps altered tokens inside the vocab. |
| `x.norm(dim=1, keepdim=True)` | L2 norm of each row. `keepdim` keeps shape `(N, 1)` so the division broadcasts. | `l2_normalize` in [train_rqvae.py:46](../tokenizer/train_rqvae.py#L46). |
| `torch.cdist(a, b)` | Pairwise Euclidean distance between every row of `a` `(N,D)` and every row of `b` `(K,D)`, giving `(N,K)`. | [quantizer.py:67](../tokenizer/quantizer.py#L67) measures the distance from each residual to every centroid. This is the "find nearest code" step. |
| `x.argmin(dim=1)` | The index of the smallest value in each row. | [quantizer.py:68](../tokenizer/quantizer.py#L68) picks the nearest centroid, and that index **is** the semantic-ID token. |
| `torch.log_softmax(x, dim=-1)` | `log(softmax(x))`, computed in a numerically stable way. | [beam_search.py:55](../inference/beam_search.py#L55). Beam search adds log-probabilities, because adding logs equals multiplying probabilities and avoids underflow. |
| `x.topk(k)` | Returns the `k` largest values and their indices. | [beam_search.py:62](../inference/beam_search.py#L62) keeps the best `k` candidate beams. |
| `torch.isfinite(x)` | True where the value is not ±inf and not NaN. | [beam_search.py:61](../inference/beam_search.py#L61) counts legal candidates (illegal ones are `-inf`). [quantizer.py:114](../tokenizer/quantizer.py#L114) checks for NaN gradients. |
| `torch.allclose(a, b, atol=)` | True if all elements are equal within a tolerance. | [transformer.py:158](../model/transformer.py#L158) is the causal test: early logits must stay the same when later tokens change. |

### Comparisons, boolean masks, `where`, `masked_fill`

| Syntax | What it does | Where |
|---|---|---|
| `a == b`, `a > 0`, `a >= b`, `a < b` | Element-wise comparisons that return a `bool` tensor. | `counts > 0` and `counts == 0` in [train_rqvae.py:61-64](../tokenizer/train_rqvae.py#L61-L64). |
| `m1 & m2` | Element-wise AND of two bool tensors. Use `&`, **not** `and`. | [train_model.py:97](../model/train_model.py#L97) |
| `torch.where(cond, a, b)` | Takes `a` where `cond` is True and `b` elsewhere. | [train_model.py:98](../model/train_model.py#L98) keeps only the 4 validation-item targets and turns every other target into PAD. |
| `x.masked_fill(mask, value)` | Writes `value` wherever `mask` is True. | [transformer.py:59](../model/transformer.py#L59) sets future positions to `-inf` so softmax gives them weight 0. |
| `float("-inf")` | Negative infinity. `exp(-inf) = 0`, so softmax assigns it zero probability. | [transformer.py:59](../model/transformer.py#L59), [beam_search.py:50](../inference/beam_search.py#L50) |

---

## 7. Reductions and counting

| Syntax | What it does | Where |
|---|---|---|
| `x.sum()`, `x.min()`, `x.max()`, `x.mean()`, `x.median()` | Reduce the whole tensor, or one dimension if you pass `dim=`. | [build_token_sequences.py:64-69](../data/build_token_sequences.py#L64-L69) (sanity statistics), [quantizer.py:109](../tokenizer/quantizer.py#L109) |
| `x.all()` | True if every element is True. | [quantizer.py:114](../tokenizer/quantizer.py#L114) |
| `torch.unique(x)` | The distinct values. | [train_rqvae.py:159](../tokenizer/train_rqvae.py#L159) counts how many distinct first-level codes are used. |
| `torch.bincount(idx, minlength=K)` | Counts how often each integer 0…K-1 appears. | [train_rqvae.py:60](../tokenizer/train_rqvae.py#L60) measures codebook usage. `minlength` makes sure unused codes still appear with a count of 0. |
| `torch.nonzero(mask, as_tuple=False)` | The indices where `mask` is True, as shape `(k, 1)`. | [train_rqvae.py:64](../tokenizer/train_rqvae.py#L64) finds the dead codes. |
| `sum(p.numel() for p in model.parameters())` | Counts the model's parameters (a common idiom). | [transformer.py:145](../model/transformer.py#L145) |

---

## 8. Getting values out of tensors

| Syntax | Result | Where and why |
|---|---|---|
| `x.item()` | A Python number, from a 1-element tensor only. | `loss.item()` in every training loop ([train_model.py:149](../model/train_model.py#L149)). **Important:** accumulating `loss` itself would keep the whole computation graph alive and leak memory. `.item()` returns a plain float. |
| `x.tolist()` | Nested Python lists. | [train_rqvae.py:81](../tokenizer/train_rqvae.py#L81) uses plain Python dicts and tuples for collision counting. [beam_search.py:65](../inference/beam_search.py#L65) turns topk results into lists. |
| `int(t)`, `float(t)` | A Python scalar from a 1-element tensor. | [build_token_sequences.py:40](../data/build_token_sequences.py#L40), [quantizer.py:110](../tokenizer/quantizer.py#L110) |
| `x.numpy()` | A NumPy array. The tensor must be on the CPU and must not require grad. | [rqvae.py:93](../tokenizer/rqvae.py#L93) passes latents to scikit-learn KMeans. |
| `x.clone()` | A real copy, so changing it leaves the original alone. | [train_model.py:94](../model/train_model.py#L94) (`tgt` is modified afterwards, and without a clone it would also modify `padded`). [transformer.py:156](../model/transformer.py#L156) |

---

## 9. Devices (CPU / GPU / MPS)

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"      # train_model.py:41
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # beam_search.py:92
```
- `torch.backends.mps.is_available()` checks whether an Apple-Silicon GPU is available. On NVIDIA hardware you would use `torch.cuda.is_available()` and `"cuda"`.
- A plain string and `torch.device(...)` both work wherever a device is expected.

| Syntax | What it does | Where |
|---|---|---|
| `x.to(device)` | Returns a copy of the tensor on that device. | `padded = padded.to(device)` for each batch ([train_model.py:142](../model/train_model.py#L142)) |
| `model.to(device)` | Moves every parameter and buffer of the model. | [train_model.py:122](../model/train_model.py#L122) |
| `x.cpu()` | Moves the tensor to the CPU. | Needed before `.numpy()`, and used in [train_rqvae.py:60](../tokenizer/train_rqvae.py#L60) because `bincount` is unreliable on MPS. |
| `x.device` | Tells you where a tensor lives. | [transformer.py:129](../model/transformer.py#L129) creates `arange` on the same device as the input. |

> **Rule:** every tensor in one operation must be on the same device. That's why the project moves the model and the data together ([train_rqvae.py:123-124](../tokenizer/train_rqvae.py#L123-L124)). It also runs k-means on the CPU **before** moving the model to MPS ([rqvae.py:87](../tokenizer/rqvae.py#L87)).

---

## 10. Reproducibility

```python
torch.manual_seed(SEED)     # train_rqvae.py:101, train_model.py:111
np.random.seed(SEED)        # NumPy has its own RNG, so seed it separately
```
This seeds PyTorch's random number generator, which makes weight init, `randn`, dropout and DataLoader shuffling repeatable.

---

## 11. Saving and loading

| Syntax | What it does | Where |
|---|---|---|
| `torch.save(obj, path)` | Pickles any tensor, dict or `state_dict` to disk. | Tensors: [embeddings.py:98](../data/embeddings.py#L98), [train_rqvae.py:169](../tokenizer/train_rqvae.py#L169). Checkpoint dicts: [train_model.py:168](../model/train_model.py#L168). |
| `torch.load(path)` | Loads it back. | [build_token_sequences.py:35](../data/build_token_sequences.py#L35) |
| `torch.load(path, map_location="cpu")` | Loads everything onto the CPU, even if it was saved from MPS or CUDA. | [train_rqvae.py:107](../tokenizer/train_rqvae.py#L107). This makes checkpoints portable between machines. |
| `torch.load(..., weights_only=False)` | Allows arbitrary pickled Python objects. Since PyTorch 2.6 the default is `True`, which is safer and allows only tensors and plain containers. | [beam_search.py:78](../inference/beam_search.py#L78). Only use `False` for files you trust. |

`.pt` is just a naming convention for these files.

---

## Quick self-check

Can you predict each shape before running it?
```python
x = torch.randn(2, 5, 8)             # (B=2, L=5, D=8)
q = x.view(2, 5, 2, 4).transpose(1, 2)   # ?
s = q @ q.transpose(-2, -1)          # ?
s.masked_fill(torch.tril(torch.ones(5,5)) == 0, float("-inf")).softmax(-1)[0,0]  # what does row 0 look like?
```
Answers: `(2,2,5,4)`, `(2,2,5,5)`. Row 0 is lower-triangular, and each row sums to 1.
