"""
Optional: fine-tune the pre-trained LangVAE on the generated texts.

The stock checkpoints are trained on short explanation sentences; continuing
training on the generated corpus improves reconstruction fidelity for the
target domain. Compute-heavy — run on demand:

    uv run python -m src.finetune_vae

Reads the `encoder` and `finetune_vae` sections of src/config.yaml. The
fine-tuned checkpoint is written to `finetune_vae.output_dir`; point
`encoder.local_checkpoint` at the resulting `final_model` folder to use it.
"""

import torch
import pandas as pd

from langvae import LangVAE
from langvae.data_conversion.tokenization import TokenizedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig

from src.config import load_config


def finetune(config: dict | None = None) -> None:
    """Continue-train the LangVAE checkpoint on the generated texts."""
    config = config or load_config()
    enc_cfg = config["encoder"]
    ft_cfg = config["finetune_vae"]

    texts = pd.read_csv(config["paths"]["texts"])["text"].tolist()
    if len(texts) < 10:
        raise ValueError(f"Only {len(texts)} texts available; generate more first (exp/sim/run.py).")

    model = LangVAE.load_from_hf_hub(enc_cfg["model_name"])

    n_eval = max(1, int(len(texts) * ft_cfg["val_split"]))
    generator = torch.Generator().manual_seed(config["seed"])
    perm = torch.randperm(len(texts), generator=generator).tolist()
    # Tokenize with the decoder tokenizer, matching LangVAE's training examples
    # (the same batch feeds the encoder and the reconstruction loss).
    train_data = TokenizedDataSet(
        [texts[i] for i in perm[n_eval:]], model.decoder.tokenizer, enc_cfg["max_len"], caching=True
    )
    eval_data = TokenizedDataSet(
        [texts[i] for i in perm[:n_eval]], model.decoder.tokenizer, enc_cfg["max_len"], caching=True
    )

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir=ft_cfg["output_dir"],
        num_epochs=ft_cfg["epochs"],
        learning_rate=ft_cfg["lr"],
        per_device_train_batch_size=ft_cfg["batch_size"],
        per_device_eval_batch_size=ft_cfg["batch_size"],
        max_beta=ft_cfg["max_beta"],
        n_cycles=ft_cfg["n_cycles"],
        target_kl=ft_cfg["target_kl"],
        seed=config["seed"],
    )
    pipeline = LanguageTrainingPipeline(model=model, training_config=training_config)
    pipeline(train_data=train_data, eval_data=eval_data)
    print(
        f"done; set encoder.local_checkpoint in src/config.yaml to the "
        f"final_model folder under {ft_cfg['output_dir']}"
    )


if __name__ == "__main__":
    finetune()
