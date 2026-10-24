"""Disk cache for notebook results: refit or recompute only when something changed.

``cached("psi-seed0", lambda: cross_fit(...))`` runs the function once, stores its value
under ``models/latent_lab/<name>-<key>.pt`` and reloads it on every later call with the
same key. The key hashes

- the source of the latent_lab modules the computation uses (the modules of every
  latent_lab function, class or object it reaches, plus the latent_lab modules those
  import), so editing renderer.py refits renderers but not readers,
- the code of the function, including the constants written in it (hyperparameters),
- every notebook variable the function reads, by content (tensors, frames, fitted
  models by their weights), following notebook functions it calls (e.g. ``score``).

Changing any of them gives a new key and a recomputation; old files stay until deleted.
Not followed: the code of libraries other than latent_lab (only their names), and
private attributes (leading ``_``). A cached function must not change the notebook
variables it reads (e.g. append to a global list), or its key changes with every run.
"""

from __future__ import annotations

import ast
import hashlib
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "models" / "latent_lab"
_PACKAGE = Path(__file__).parent
KEY_VERSION = 2  # bump when the key itself changes meaning; old files are then ignored


def cached(name: str, fn: Callable[[], Any], folder: str | Path | None = None) -> Any:
    """``fn()``, computed once per key and reloaded from disk afterwards."""
    path = Path(folder or CACHE_DIR) / f"{name}-{digest((KEY_VERSION, fn))[:16]}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    value = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save(value, partial)
    partial.replace(path)  # never leave a half-written file under the final name
    return value


def digest(obj: Any) -> str:
    """A content hash of ``obj`` (see the module docstring for what is followed)."""
    h = hashlib.sha256()
    # id -> object: holding every visited object keeps temporaries (e.g. state_dict()) alive,
    # so their ids cannot be reused mid-hash and mistaken for a cycle
    _feed(h, obj, {})
    return h.hexdigest()


def _module_closure(module: str) -> tuple[str, ...]:
    """A latent_lab module and the latent_lab modules it imports, transitively."""
    todo, found = [module], set()
    while todo:
        name = todo.pop()
        if name in found or not (_PACKAGE / f"{name.split('.')[-1]}.py").exists():
            continue
        found.add(name)
        for node in ast.walk(ast.parse((_PACKAGE / f"{name.split('.')[-1]}.py").read_text())):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("latent_lab"):
                todo += [node.module] if node.module != "latent_lab" else [f"latent_lab.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                todo += [a.name for a in node.names if a.name.startswith("latent_lab.")]
    return tuple(sorted(found))


def _feed_source(h, module: str | None, seen: dict) -> None:
    """The source of a latent_lab module and its latent_lab imports (once per key)."""
    if not module or not module.startswith("latent_lab.") or module == __name__:
        return
    for name in _module_closure(module):
        if ("source", name) not in seen:
            seen[("source", name)] = True
            h.update(name.encode() + (_PACKAGE / f"{name.split('.')[-1]}.py").read_bytes())


def _feed(h, obj: Any, seen: dict) -> None:
    h.update(type(obj).__qualname__.encode())
    _feed_source(h, getattr(type(obj), "__module__", None), seen)
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes, range, slice)):
        h.update(repr(obj).encode())
    elif isinstance(obj, torch.Tensor):
        h.update(f"{obj.dtype}{tuple(obj.shape)}".encode())
        h.update(obj.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(f"{obj.dtype}{obj.shape}".encode())
        h.update(np.ascontiguousarray(obj).tobytes() if obj.dtype != object else repr(obj.tolist()).encode())
    elif isinstance(obj, pd.DataFrame):
        h.update(repr((list(obj.columns), list(map(str, obj.dtypes)))).encode())
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    elif isinstance(obj, pd.Series):
        h.update(repr((obj.name, str(obj.dtype))).encode())
        h.update(pd.util.hash_pandas_object(obj, index=True).to_numpy().tobytes())
    elif isinstance(obj, pd.Index):
        h.update(pd.util.hash_pandas_object(obj).to_numpy().tobytes())
    elif isinstance(obj, (types.ModuleType, type, types.BuiltinFunctionType)):
        h.update(getattr(obj, "__qualname__", obj.__name__).encode())
        _feed_source(h, obj.__name__ if isinstance(obj, types.ModuleType) else getattr(obj, "__module__", None), seen)
    elif id(obj) in seen:
        h.update(b"<cycle>")
    elif isinstance(obj, types.FunctionType):
        seen[id(obj)] = obj
        _feed(h, obj.__qualname__, seen)
        _feed_source(h, obj.__module__, seen)
        if obj.__module__ == "__main__":  # notebook code: follow it; library code: its name suffices
            _feed_code(h, obj.__code__, obj.__globals__, seen)
            _feed(h, obj.__defaults__, seen)
            for cell in obj.__closure__ or ():
                _feed(h, cell.cell_contents, seen)
    elif isinstance(obj, (list, tuple)):
        seen[id(obj)] = obj
        for item in obj:
            _feed(h, item, seen)
    elif isinstance(obj, (set, frozenset)):
        for item in sorted(obj, key=repr):
            _feed(h, item, seen)
    elif isinstance(obj, dict):
        seen[id(obj)] = obj
        for key in sorted(obj, key=repr):
            _feed(h, key, seen)
            _feed(h, obj[key], seen)
    elif isinstance(obj, torch.nn.Module):
        seen[id(obj)] = obj
        _feed(h, obj.state_dict(), seen)
        _feed(h, {k: v for k, v in vars(obj).items() if not k.startswith("_")}, seen)
    elif isinstance(obj, np.random.Generator):
        _feed(h, obj.bit_generator.state, seen)
    elif isinstance(obj, torch.Generator):
        _feed(h, obj.get_state(), seen)
    elif hasattr(obj, "__dict__"):  # private attributes hold lazy caches, e.g. GroupOTRenderer._maps
        seen[id(obj)] = obj
        _feed(h, {k: v for k, v in vars(obj).items() if not k.startswith("_")}, seen)
    else:
        raise TypeError(f"cannot hash a {type(obj).__qualname__} for the cache key")


def _feed_code(h, code: types.CodeType, globals_: dict, seen: dict) -> None:
    h.update(code.co_code)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _feed_code(h, const, globals_, seen)
        else:
            _feed(h, const, seen)
    for name in code.co_names:  # the notebook variables the code reads, by their current value
        h.update(name.encode())
        if name in globals_:
            _feed(h, globals_[name], seen)
