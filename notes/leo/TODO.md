# Latent Intervention

## TODO

### Organisational

- [ ] Find common workflow (overleaf, quarto, github, ...)

### Dataset

1.  Find Datasets

- [ ] Find a good real world dataset $D_1$
- [ ] Find an alternative dataset $D_2$ (for Real-World Study 3)

2.  Develop SCMs

- [ ] Develop a SCM $M_1$ corresponding to $D_1$
- [ ] Develop a SCM $M_2$ corresponding to $D_2$

### Evaluation Framework

- [ ] Metrics
- [ ] Regressor/Classifier $p$
  - [ ] trained on $Z$
  - [ ] trained on $Z'$
  - [ ] trained with $p(Z) = p(Z')$
- [ ] Decoder $k$ (input $Z'$, output: $X'$)
  - [ ] qualitative analysis?
  - [ ] $p(f(X)) = p(f(X'))$? --\> actually tells us more about VAE $f,k$ than about our framework

### Training Pipeline

- [ ] Semantic decoder $g$ (input: $Z$, output: $S$)
- [ ] $h_Z$ (input: $Z$, output: $S'$ (over $g$))

### Simulation Study / Training Framework

- [ ] Simulate tabular data $S$ based on the SCM $M_1$
- [ ] Create counterfactual data $S'$ using $h_S$ derived from SCM $M_1$
- [ ] Create text data $X$ from $S$ using LLM
- [ ] Set up encoder $f$ (LangVAE, input: $X$, output: $Z$)
- [ ] Train $g$ (input: $Z$, output: $S$)
- [ ] Train $h_Z$ (input: $Z$, output: $S'$ (over $g$))
- [ ] Apply evaluation framework
- [ ] Do all of the above for an alternate version of $M_1$ (alternative DAG to show flexibility)

### Real-World Study 1

Use $g, h_S$ from simulation study

- [ ] Train $h_Z$
- [ ] Apply evaluation framework

### Real-World Study 2

Use $h_Z$ from simulation study

- [ ] Apply evaluation framework

### Real-World Study 3

Train $h_Z$ without functions from simulation study

- [ ] Develop a SCM $M_2$ corresponding to the real world dataset
- [ ] Find/train semantic decoder $g$ (input: $Z$, output: $S$) (**tricky part**)
- [ ] Create counterfactual data $S'$ using $h_S$ derived from SCM $M_2$
- [ ] Train $h_Z$ (input: $Z$, output: $S'$ (over $g$))
- [ ] Apply evaluation framework
- [ ] Do all of the above for an alternate version of $M_2$ (alternative DAG to show flexibility)

## Checkpoints

### CP 0

- [ ] Repo/Overleaf/...
- [ ] $D_1$
- [ ] $D_2$

### CP 1

- [ ] $M_1$
- [ ] $M_2$

### CP 2

- [ ] Training Pipeline
- [ ] Evaluation Framework
- [ ] Simulated Data

### CP 3

- [ ] Simulation Study

### CP 4

- [ ] Real World Study 1
- [ ] Real World Study 2

### CP 5

- [ ] Real World Study 3