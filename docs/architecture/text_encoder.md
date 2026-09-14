# Text encoder

Set `encoder.variant` to `langvae` (the default) or `nomic`. LangVAE stores the
128-dimensional deterministic posterior mean, but its checkpoint was trained
on short EntailmentBank explanations rather than CVs.

Nomic Embed Text v1.5 is the preferred comparison: it was trained as a general
text embedder and provides a trained 128-dimensional Matryoshka representation.
It uses the required `classification:` prefix and normalized embeddings. The
shared 512-token limit remains unchanged pending separate long-input work.

Changing encoder requires regenerating latents and retraining all downstream
models; artifacts from the two latent spaces are not interchangeable.
