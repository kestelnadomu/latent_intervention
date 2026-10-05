# Veeeery high level guide to the codebase guide

Hope this helps a littl...

## 1. General idea

```mermaid
flowchart LR
    A[CV text] -->|f: encoder| Z[latent encodings Z]
    Z -->|g: semantic decoder| S[distr. over tabular factual S]
    S -->|h_S: exact causal 'kernel'| SCF[tabular counterfactual S]
    Z -->|h_Z: learned latent editor/manipulator| ZCF[counterfactual latent Z′]
    ZCF -->|g: semantic decoder| SP[predicted tabular counterfactual S]
    SCF --- SP
```

- `f` is one of the encoder (they are pre-trained, taken from the internet; there exist maaany, we try a dozen of them, more info later).
- `g` predicts the structured state \(S=(X,T,D,U)\). This is not the LangVAE decoder.
- `h_S` is computed analytically from the known DAG.
- `h_Z` is the object currently being benchmarked (this we want to make perfect and get awards for this).

## 2. Geeeeneral code structure

```mermaid
flowchart TB
    SIMCFG[exp/sim/config.yaml<br/>SCM, schema, intervention, data paths]
    MODELCFG[src/config.yaml<br/>f, g, h_Z hyperparameters and path templates]
    SCHEMA[src/schema.py<br/>loads schema and experiment objects]
    LOAD[src/config.py::load_config<br/>CLI overrides + resolved paths]
    PROTO[src/encoder/protocols.py<br/>valid dimensions and pinned protocols]
    GEN[exp/sim/run.py<br/>data-generation stages]
    PIPE[src/pipeline.py<br/>encode / train / evaluate]
    BENCH[exp/benchmarks/semantic_decoder/run.py<br/>exp/benchmarks/latent_intervention/run.py]

    SIMCFG --> SCHEMA --> GEN
    SIMCFG --> LOAD
    MODELCFG --> LOAD --> PROTO --> PIPE
    LOAD --> BENCH
```

| Specs | Main file | Main implementation |
|---|---|---|
| causal variables, intervention `do(...)` specification, sample count etc | `exp/sim/config.yaml` | `exp/sim/talent_sfm.py` |
| encoder specs | `src/config.yaml` / CLI | `src/encoder/protocols.py` |
| locations info of all objects created during training | `src/config.yaml:paths` | `src/config.py`, `src/latent_writer.py` |
| \(g\) architecture and training procedure | `src/config.yaml:semantic_decoder` | `src/semantic_decoder/model.py`, `src/semantic_decoder/training.py` |
| \(h_Z\) benchmark matrix | `configs/hz_benchmark.yaml` | `exp/benchmarks/latent_intervention/*.py` |

## 3. DAG

We changed the setup from LIBERTY and instead use this DAG (A bit similar to that of Drago):

```mermaid
flowchart LR
    X[X: origin region] --> D[D: degree]
    X --> U[U: university tier]
    X --> Y[Y: qualification]
    T[T: hidden talent] --> D
    T --> U
    T --> Y
    T --> P[P/L/H/A<br/>textual talent proxies]
    D --> Y
    U --> Y
```

Simulation procedure on a high level:

```mermaid
flowchart LR
    SCM[simulate 5,000 factual + paired worlds] --> IDX[pair_index.csv<br/>4,000 train / 1,000 test]
    SCM --> PLAN[render_plan.csv<br/>shared style choices]
    PLAN --> TXT[factual and counterfactual CVs]
    IDX --> CHECK[validate pair identity and alignment]
    TXT --> CHECK --> ENC[encode paired texts]
```

## 4. All encoders we use and how we train it

| Encoder family | Canonical folders | Dimensions |
|---|---|---|
| LangVAE (out of the box) | `langvae` | 128 |
| LangVAE (we fine tuned this one on CV, doesnt seem to work well as of now)| `langvae_ft` | 128 |
| Nomic Embed v1.5 | `nomic_{d}` | 64, 128, 256, 512, 768 |
| Qwen3-Embedding-0.6B | `qwen3_{d}` | 32, 64, 128, 256, 512, 768, 1024 |
| EmbeddingGemma-300M (preliminary buuuut: seems to be the best by a big difference) | `embeddinggemma_{d}` | 128, 256, 512, 768 |

All **18** spaces are stored seperately in a folder sturcture following `data/latents/talent/<encoder>/`. Each `z_pairs.pt` combines factual latents, held-out counterfactual latents, IDs, and identity flags. 

Procedure:

```mermaid
flowchart TB
    TEXT[data/text_talent/*.csv] --> FAMILY{encoder family}
    FAMILY --> LV[LangVAE / LangVAE-FT]
    FAMILY --> EMB[Nomic / Qwen3 / Gemma]
    LV --> PAIR[src/pair_encoding.py]
    EMB --> PAIR
    PAIR --> WRITE[src/latent_writer.py<br/>validate, checksum, publish atomically]
    WRITE --> OUT[data/latents/talent/&lt;encoder&gt;/]
```

## 5. How we train g()

We currently test two different MLP implementation. One with independent heads and on with autoregressive heads at the last layer. Broad setup looks like this:

```mermaid
flowchart LR
    Z[4,000 factual train latents] --> SPLIT[3,200 fit / 800 validation]
    SPLIT --> GI[independent heads<br/>predict all fields in parallel]
    SPLIT --> GA[autoregressive heads<br/>predict in schema order]
    GI --> CKPT[semantic_decoder.pt]
    GA --> CKPT
    CKPT --> TEST[held-out 1,000 units]
```

Both g versions have the same 500-epoch-budget for training and we allow for early-stopped. In total we arrive at **18 encoders × 2 decoder versions (g) = 36 factual predictions S**. Among those we choose the one which performs best in predicting the tabular data S (currently Gemma 768 in combination with the independent g head).

## 6. Then we have 9 (!!!) candidates for $h_Z$

3 arhcitectures are based on a transformer setup, 3 on normalizing flows and one an MLP that simply maps from Z to Z' so it can see the true outcome, this is a simple baseleine for model uncertainty. 

```mermaid
flowchart TB
    START[train-manipulator] --> DIRECT[Transformer editors]
    START --> FLOW[Normalizing-flow editors]
    START --> ORACLE[Privileged paired-latent reference]
    DIRECT --> B[baseline]
    DIRECT --> P[pre_additive / noise_token]
    DIRECT --> M[dist / particles]
    FLOW --> SF[state_flow]
    FLOW --> DF[distilled_flow]
    FLOW --> DSF[direct_semantic_flow]
    ORACLE --> O[oracle_regression]
```
## 7. Setup of the training pipeline

```mermaid
sequenceDiagram
    participant CLI as module CLI
    participant C as load_config()
    participant P as pipeline / benchmark
    participant V as validators
    participant A as artifacts
    CLI->>C: variant and dimension overrides
    C->>C: resolve {encoder}/{decoder}/{manipulator}
    C->>P: validated config
    P->>V: verify data, IDs, metadata, hashes
    V->>P: aligned tensors
    P->>A: atomic checkpoint + report + status
```

| Task | Entry point |
|---|---|
| generate or validate the corpus | `python -m exp.sim.run <stage>` |
| run one pipeline stage | `python -m src.pipeline <stage> ...` |
| build the all-encoder \(g\) matrix | `python -m exp.benchmarks.semantic_decoder.run` |
| prepare/run/resume the \(h_Z\) matrix | `python -m exp.benchmarks.latent_intervention.run` |
| audit artifact compatibility | `tools/artifact_audit.py` |
