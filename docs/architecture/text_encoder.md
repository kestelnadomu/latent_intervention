# Text encoder

`encoder.variant` selects the frozen 128-dimensional text representation used by every
downstream model:

- `langvae` (default) stores the deterministic posterior mean. Its checkpoint was trained on
  short EntailmentBank explanations rather than CVs.
- `nomic` stores Nomic Embed Text v1.5's normalized, 128-dimensional Matryoshka representation
  with the required `classification:` task prefix. It is the preferred general-text comparison.

The input limit remains 512 tokens. The active text-generation pipeline separately rejects CVs
above 500 GPT-2 tokens, leaving a small margin below LangVAE's input limit; validation also reports
the observed length distribution. Longer-input or chunked representations are separate research
choices and are not introduced here.

The latent sidecar records the chosen encoder, pinned revisions (or the content hash of a local
LangVAE checkpoint), input hashes, shape, and the hash of the serialized latent tensor. Loading
verifies this information against the active configuration and inputs. Changing encoder, revision,
local checkpoint contents, or source data therefore requires re-encoding and retraining the
semantic decoder and latent manipulator: LangVAE and Nomic latent spaces are not interchangeable.
