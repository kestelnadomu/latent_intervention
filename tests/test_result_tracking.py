"""Keep the portable benchmark bundle distinct from bulk local artifacts."""

import subprocess
from pathlib import Path


def test_result_tracking_allowlist(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((root / ".gitignore").read_text())
    bundle = "reports/talent/decoder_benchmarks/test-run"
    model = "models/talent/embeddinggemma_768/g-independent"
    report = "reports/talent/embeddinggemma_768/g-independent"
    hz = "reports/talent/hz_benchmarks/test-run"
    hz_model = f"{model}/h-state_flow/benchmarks/test-run"
    portable = {
        f"{hz_model}/selected/latent_intervention.pt",
        f"{hz_model}/selected/latent_intervention.info.json",
        *(
            f"{hz}/{name}"
            for name in (
                "plan.json",
                "readiness.json",
                "oracle_inputs.json",
                "selection.json",
                "status.json",
                "results.json",
                "published.json",
                "summary.md",
            )
        ),
        f"{hz}/fits/embeddinggemma_768/g-independent/h-state_flow/search/trial-000/training.json",
        f"{hz}/fits/embeddinggemma_256/g-independent/h-oracle_regression/final/seed-43/training.json",
        f"{hz}/evaluation/embeddinggemma_768/g-independent/h-particles/seed-42.json",
        f"{hz}/source/src/hz_training.py",
        f"{hz}/source/exp/sim/talent_sfm.py",
        f"{hz}/source/pyproject.toml",
        f"{hz}/source/uv.lock",
        f"{hz}/configs/benchmark.yaml",
        f"{model}/semantic_decoder.pt",
        f"{report}/semantic_decoder.json",
        f"{model}/h-oracle_regression/latent_intervention.pt",
        f"{model}/h-oracle_regression/latent_intervention.info.json",
        f"{report}/h-oracle_regression/training.json",
        f"{report}/h-oracle_regression/eval.json",
        "data/latents/talent/embeddinggemma_256/oracle_train/z_prime.pt",
        "data/latents/talent/embeddinggemma_768/oracle_train/z_prime.info.json",
        "models/talent/langvae_ft/g-autoregressive/semantic_decoder.pt",
        "reports/talent/langvae_ft/g-autoregressive/semantic_decoder.json",
        *(
            f"{bundle}/{name}"
            for name in (
                "summary.md",
                "results.json",
                "evaluation.json",
                "selection.json",
                "protocol.json",
                "plan.json",
                "status.json",
                "published.json",
                "backups.json",
            )
        ),
        f"{bundle}/configs/embeddinggemma_768.yaml",
        f"{bundle}/evaluation/embeddinggemma_768/g-independent/seed-42.json",
        f"{bundle}/evaluation/embeddinggemma_768/g-independent/summary.json",
        f"{bundle}/source/src/decoder_benchmark_report.py",
        f"{bundle}/source/src/langvae_ft/model.py",
        f"{bundle}/source/src/config.yaml",
        f"{bundle}/source/configs/langvae_ft_scope.json",
        f"{bundle}/source/scripts/run_decoder_benchmark.sh",
        f"{bundle}/source/uv.lock",
    }
    local = {
        f"{hz_model}/search/trial-000/latent_intervention.pt",
        f"{hz_model}/final/seed-42/latent_intervention.pt",
        f"{hz_model}/selected/.latent_intervention.pt.tmp",
        f"{hz}/fits/embeddinggemma_768/g-independent/h-state_flow/search/trial-000/progress.json",
        f"{hz}/fits/embeddinggemma_768/g-independent/h-state_flow/search/trial-000/failure.json",
        f"{hz}/run.log",
        f"{hz}/errors.json",
        f"{hz}/source/.env",
        f"{model}/experiments/test-run/search/trial-000/semantic_decoder.pt",
        f"{model}/experiments/test-run/final/seed-43/semantic_decoder.pt",
        f"{model}/archive/test-run/semantic_decoder.pt",
        f"{model}/h-baseline/latent_intervention.pt",
        f"{model}/h-oracle_regression/.oracle_regression.lock",
        f"{model}/h-oracle_regression/.latent_intervention.pt.tmp",
        f"{report}/h-oracle_regression/training.progress.json",
        "data/latents/talent/embeddinggemma_768/.oracle_train.lock",
        "data/latents/talent/embeddinggemma_768/.oracle_train.pending/z_prime.pt",
        f"{model}/.semantic_decoder.pt.tmp",
        "models/langvae_ft/checkpoint-024/encoder.pt",
        "models/langvae_ft/checkpoint.tar.gz",
        "models/talent/embeddinggemma_768/g-custom/semantic_decoder.pt",
        f"{report}/experiments/test-run/final/seed-42/semantic_decoder.json",
        f"{report}/archive/test-run/semantic_decoder.json",
        f"{report}/semantic_decoder.progress.json",
        f"{report}/h-baseline/eval.json",
        "reports/talent/decoder_experiments/old-run/results.json",
        "reports/talent/decoder_benchmarks/.benchmark.lock",
        f"{bundle}/run.log",
        f"{bundle}/errors.json",
        f"{bundle}/summary.md.tmp",
        f"{bundle}/checkpoints/model.pt",
        f"{bundle}/evaluation/embeddinggemma_768/g-independent/predictions.pt",
        f"{bundle}/source/src/__pycache__/config.pyc",
        f"{bundle}/source/.env",
    }
    result = subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "core.excludesFile=/dev/null",
            "check-ignore",
            "--no-index",
            "--verbose",
            "--stdin",
        ],
        input="\n".join(sorted(portable | local)) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    # check-ignore also prints explicit negation matches; distinguish them from
    # exclusion rules instead of interpreting every printed path as ignored.
    rules = [line.split("\t", 1) for line in result.stdout.splitlines()]
    assert {path for _, path in rules} == portable | local
    ignored = {
        path for rule, path in rules if not rule.split(":", 2)[2].startswith("!")
    }
    assert ignored == local
