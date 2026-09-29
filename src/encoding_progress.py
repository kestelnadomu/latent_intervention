"""Console-only progress reporting shared by the frozen embedding encoders."""

from time import perf_counter


def log_encoding_progress(label, start, batch_size, total, started, *, enabled=False):
    """Report the first batch, every 20 batches and completion; never touch tensors."""
    done = min(start + batch_size, total)
    if enabled and (start // batch_size % 20 == 0 or done == total):
        elapsed = perf_counter() - started
        eta = elapsed * (total - done) / done
        print(
            f"{label} {done}/{total} texts; elapsed {elapsed:.1f}s; ETA {eta:.1f}s",
            flush=True,
        )
