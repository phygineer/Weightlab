# WeightLab

WeightLab is an interactive research CLI for applying mathematical, statistical, perturbation, pruning, and structural transformations directly to Hugging Face model weights.

The normal workflow is deliberately simple:

```text
Hugging Face model
        ↓
interactive pipeline builder
        ↓
download/cache exact model revision
        ↓
transform Safetensors checkpoint shard-by-shard
        ↓
save self-contained experiment
        ↓
interactive original-vs-transformed evaluation
```

You do **not** have to manually download the model, find its checkpoint path, write YAML, or choose an output directory.

---

## Files

```text
weightlab/
├── weightlab.py
├── test_model.py
├── requirements.txt
└── README.md
```

Runtime experiments are saved under:

```text
~/.weightlab/experiments/
```

You can override that location:

```bash
export WEIGHTLAB_HOME=/some/other/path
```

---

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For gated Hugging Face models such as Gemma, authenticate using the Hugging Face CLI/token mechanism before running the tool.

---

# Start WeightLab

```bash
python weightlab.py
```

Example:

```text
╭──────────────────────────────────────────────╮
│ WeightLab                                    │
│ Interactive neural-weight transformation lab│
╰──────────────────────────────────────────────╯

Hugging Face model [google/gemma-3-1b-it]:
Revision [main]:
```

WeightLab resolves the exact Hugging Face commit and downloads/caches the repository automatically.

The resolved commit SHA is stored in the experiment manifest so the original source model can be reproduced later.

---

# Build the pipeline interactively

WeightLab presents an operation menu.

You can add, delete, and reorder operations before running the experiment.

A pipeline might become:

```text
1. multiply        {"value": 3.141592653589793}
2. l2_normalize    {}
3. clip            {"min": -0.1, "max": 0.1}
```

Conceptually:

\[
W' = clip(normalize(\pi W))
\]

Pipeline order is preserved exactly.

---

# Available operations

## Arithmetic

```text
identity
multiply
add
subtract
divide
negate
abs
sign
```

## Normalization / statistical processing

```text
l2_normalize
rms_normalize
standardize
mean_center
minmax_normalize
maxabs_normalize
```

## Nonlinear

```text
sin
cos
tanh
softsign
signed_power
```

## Noise / perturbation

```text
gaussian_noise
relative_gaussian_noise
uniform_noise
random_sign_flip
```

## Pruning

```text
magnitude_prune
keep_top_percent
random_prune
```

## Structural experiments

```text
shuffle_weights
shuffle_magnitudes
shuffle_signs
```

These are particularly useful because they can preserve some aggregate properties while destroying spatial organization.

## Precision

```text
round
quantize
```

`quantize` performs simulated symmetric quantization and returns dequantized values in the original tensor dtype. It is meant for perturbation experiments rather than producing an actual quantized deployment checkpoint.

## Interpolation

```text
interpolate
```

Supported targets:

```text
zero
sign
l2_normalized
```

For example:

\[
W'=(1-\alpha)W+\alpha f(W)
\]

This is useful for gradual degradation experiments.

---

# Example: π followed by L2 normalization

In the interactive builder choose:

```text
Multiply
Value: pi

L2 normalize
```

This represents:

\[
W' = \frac{\pi W}{\|\pi W\|_2}
\]

For positive scalar π this simplifies mathematically to:

\[
W' = \frac{W}{\|W\|_2}
\]

Keeping both steps is still useful because the experiment manifest records exactly what you intended to apply.

---

# Hugging Face model handling

You supply a repository ID such as:

```text
google/gemma-3-1b-it
```

WeightLab:

1. resolves the requested revision,
2. records the exact commit SHA,
3. downloads or reuses the Hugging Face cache,
4. locates the Safetensors checkpoint,
5. copies model/tokenizer/config support files into the experiment,
6. transforms the checkpoint,
7. saves a new loadable model directory.

The original Hugging Face cache is never modified.

---

# Experiment directories

WeightLab automatically creates a directory such as:

```text
~/.weightlab/experiments/
└── google--gemma-3-1b-it/
    └── 20260907-153210-pi-l2-normalize/
        ├── model/
        │   ├── config.json
        │   ├── tokenizer.json
        │   ├── model.safetensors.index.json
        │   └── model-....safetensors
        │
        ├── pipeline.yaml
        ├── manifest.json
        └── transformation.log
```

After evaluation it also contains:

```text
evaluation.json
```

---

# YAML still exists — but you don't write it

WeightLab automatically saves the pipeline as:

```text
pipeline.yaml
```

For example:

```yaml
version: 1

name: pi-l2-normalize

model:
  repo_id: google/gemma-3-1b-it
  requested_revision: main
  resolved_revision: abc123...

pipeline:
  - operation: multiply
    value: 3.141592653589793

  - operation: l2_normalize
```

The YAML is therefore a reproducibility artifact rather than the primary user interface.

---

# Memory design

WeightLab does **not** instantiate the Transformer model during transformation.

It operates directly on `.safetensors`.

```text
model
  │
  ├── shard 1
  │     ↓
  │   transform
  │     ↓
  │   write
  │     ↓
  │   release
  │
  ├── shard 2
  │
  └── ...
```

This avoids loading the entire model into RAM.

The practical memory bound is approximately the current shard plus temporary tensors generated by the active operations.

Some operations are more memory-intensive than others. For example:

```text
shuffle_weights
keep_top_percent
standardize
quantize
```

may allocate significant temporary tensors.

---

# Why CPU is enough for transformation

Checkpoint transformation does not perform model inference.

The operations are primarily tensor arithmetic:

```text
multiply
normalize
clip
noise
shuffle
prune
round
```

They can run entirely on CPU.

For many simple transformations, checkpoint reading and serialization may represent a substantial fraction of runtime.

GPU compute is far more useful during evaluation.

---

# Structural validation

After transformation, WeightLab checks that the source and transformed checkpoints have the same:

```text
tensor names
tensor shapes
tensor dtypes
```

The experiment is marked successful only after this validation passes.

Each pipeline operation is also checked to ensure:

```text
result is a Tensor
shape is unchanged
values remain finite
```

NaN or infinity causes the run to fail rather than silently creating an invalid checkpoint.

---

# Traceability

Each experiment stores:

```text
manifest.json
```

The manifest includes:

```text
unique run ID
model repository
requested revision
resolved Hugging Face commit SHA
experiment name
ordered pipeline
operation parameters
pipeline SHA-256
Python environment
PyTorch version
Git commit when available
source shard SHA-256
output shard SHA-256
parameter counts
run status
validation result
error traceback if failed
```

This makes each experiment independently auditable.

---

# Random operations and reproducibility

Noise, pruning, shuffling, and sign-flip operations support seeds.

Example:

```text
Gaussian noise
std: 0.001
seed: 42
```

For controlled research, keep seeds fixed when comparing two pipelines.

---

# Test the transformed model

Run:

```bash
python test_model.py
```

The test program discovers saved WeightLab experiments automatically.

Example:

```text
Saved Experiments

1  google/gemma-3-1b-it   pi-l2-normalize
2  google/gemma-3-1b-it   noise-001
3  google/gemma-3-1b-it   identity
```

Choose the experiment and WeightLab automatically determines:

```text
original Hugging Face model
exact original revision
transformed model directory
```

No paths are required.

---

# Evaluation

The tester loads the original and transformed models sequentially to reduce peak RAM/VRAM usage.

It uses identical deterministic prompts and generation settings.

Device options:

```text
auto
cpu
cuda
mps
```

`auto` selects:

```text
CUDA
  ↓
Apple MPS
  ↓
CPU
```

---

# Built-in prompts

The default suite includes simple:

```text
factual recall
arithmetic
completion
explanation
coding
reasoning
```

You can optionally provide a prompt file from the interactive tester.

Plain text:

```text
What is gravity?
What is 17 * 23?
Explain recursion.
```

Or JSON:

```json
[
  "What is gravity?",
  "What is 17 * 23?",
  "Explain recursion."
]
```

---

# Evaluation report

The result is stored inside the selected experiment:

```text
evaluation.json
```

It contains:

```text
original response
transformed response
input token count
generated token count
generation time
tokens per second
complete transformation manifest
```

This directly ties observed behavior back to the exact weight transformation.

---

# Recommended first experiment

Before performing destructive transformations, create an identity control.

Pipeline:

```text
identity
```

Run it, then compare it using:

```bash
python test_model.py
```

The transformed identity checkpoint should behave equivalently to the original.

This validates the checkpoint-processing machinery itself.

---

# Then try the π experiment

Build:

```text
multiply(pi)
    ↓
l2_normalize
```

Run the experiment, then:

```bash
python test_model.py
```

Compare the resulting behavior against the exact original model revision.

---

# Interesting experiments

## Distribution preservation

```text
shuffle_weights
```

The tensor retains the same individual numerical values but changes where they occur.

This asks whether model behavior depends merely on weight distributions or on precise learned placement.

## Preserve signs, shuffle magnitudes

```text
shuffle_magnitudes
```

## Preserve magnitudes, shuffle signs

```text
shuffle_signs
```

These isolate different aspects of learned parameter organization.

## Gradual degradation

Use:

```text
interpolate
```

with increasing alpha values.

For example:

```text
0.00
0.01
0.05
0.10
0.25
0.50
0.75
1.00
```

This allows you to search for a capability-collapse boundary instead of jumping directly to a completely transformed model.

## Noise sensitivity

Try:

```text
relative_gaussian_noise
```

at increasing standard deviations.

## Sparsity

Use:

```text
keep_top_percent
```

with:

```text
100
95
90
75
50
25
10
```

This lets you study how much parameter magnitude can be removed before particular behaviors disappear.

---

# Advanced command-line usage

You can skip the initial model question:

```bash
python weightlab.py \
  --model google/gemma-3-1b-it \
  --revision main
```

The pipeline itself is still constructed interactively.

For testing a known experiment directly:

```bash
python test_model.py \
  --experiment ~/.weightlab/experiments/google--gemma-3-1b-it/EXPERIMENT
```

Optional:

```bash
--device cuda
--max-new-tokens 128
--prompts prompts.txt
```

---

# Requirements

```text
torch
safetensors
transformers
accelerate
tqdm
PyYAML
huggingface_hub
rich
```

Once you have a known-good environment, you can capture exact versions:

```bash
pip freeze > requirements.lock.txt
```

---

# Important notes

Direct manipulation of pretrained neural-network parameters will frequently destroy model behavior.

That is expected.

Never modify the Hugging Face cache or original checkpoint in place.

WeightLab always creates a separate experiment model.

Some model repositories may contain custom code or unusual checkpoint layouts that require additional handling. This version targets standard Hugging Face causal-language-model repositories using Safetensors.

---

# Future directions

Strong next extensions include:

```text
component-aware selectors
attention-only transforms
MLP-only transforms
embedding-only transforms
layer ranges
per-operation selectors
parameter sweeps
global two-pass statistics
automatic algebraic simplification warnings
logit similarity evaluation
perplexity evaluation
capability-collapse curves
HTML experiment reports
```

The current version intentionally keeps the execution model understandable while already supporting a broad range of weight-space experiments.
