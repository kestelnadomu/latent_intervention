from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_asset_name(index: int, pdf_path: Path, digest: str) -> str:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_path.stem).strip("._-")
    safe_stem = safe_stem[:48].strip("._-") or "pdf"
    return f"{index:02d}_{safe_stem}_{digest[:10]}"


def parse_pages(pdfinfo_text: str) -> int | None:
    match = re.search(r"^Pages:\s+(\d+)\s*$", pdfinfo_text, re.MULTILINE)
    return int(match.group(1)) if match else None


def markdown_link(path: Path, relative_to: Path) -> str:
    rel = path.relative_to(relative_to).as_posix()
    return f"<{rel}>"


def split_pages(extracted_text: str, pages: int | None) -> list[str]:
    parts = extracted_text.replace("\r\n", "\n").replace("\r", "\n").split("\f")
    if parts and not parts[-1].strip():
        parts.pop()
    if pages is not None:
        while len(parts) < pages:
            parts.append("")
        if len(parts) > pages:
            parts = parts[:pages]
    return parts


def normalize_rendered_pages(asset_dir: Path) -> list[Path]:
    rendered = sorted(
        asset_dir.glob("page-*.png"),
        key=lambda path: int(re.search(r"-(\d+)\.png$", path.name).group(1)),
    )
    normalized: list[Path] = []
    width = max(3, len(str(len(rendered))))
    for number, path in enumerate(rendered, start=1):
        target = asset_dir / f"page-{number:0{width}d}.png"
        if path != target:
            path.rename(target)
        normalized.append(target)
    return normalized


def write_markdown(
    *,
    pdf_path: Path,
    md_path: Path,
    asset_dir: Path,
    digest: str,
    pdfinfo_text: str,
    pdfdetach_text: str,
    pages_text: list[str],
    rendered_pages: list[Path],
    pdftotext_stderr: str,
    render_stderr: str,
) -> None:
    page_count = max(len(pages_text), len(rendered_pages))
    converted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines: list[str] = []
    lines.extend(
        [
            "---",
            f"source_pdf: {pdf_path.name}",
            f"source_pdf_sha256: {digest}",
            f"converted_at_utc: {converted_at}",
            f"page_count: {page_count}",
            "text_extraction: pdftotext -layout -enc UTF-8",
            "visual_preservation: per-page PNG renders at 200 DPI",
            "---",
            "",
            f"# {pdf_path.stem}",
            "",
            "## Source",
            "",
            f"- PDF: [{pdf_path.name}]({markdown_link(pdf_path, md_path.parent)})",
            f"- Page image assets: `{asset_dir.relative_to(md_path.parent).as_posix()}/`",
            "- Method: each page includes extracted layout-preserving text plus a lossless PNG render of the page.",
            "- Use the PNG render whenever extracted text omits or distorts equations, figures, tables, symbols, or page layout.",
            "",
            "## PDF Metadata",
            "",
            "```text",
            pdfinfo_text.strip() or "No pdfinfo metadata returned.",
            "```",
            "",
            "## Embedded Files",
            "",
            "```text",
            pdfdetach_text.strip() or "No embedded-file listing returned.",
            "```",
            "",
        ]
    )

    if pdftotext_stderr.strip() or render_stderr.strip():
        lines.extend(["## Conversion Diagnostics", "", "```text"])
        if pdftotext_stderr.strip():
            lines.extend(["[pdftotext stderr]", pdftotext_stderr.strip(), ""])
        if render_stderr.strip():
            lines.extend(["[pdftoppm stderr]", render_stderr.strip()])
        lines.extend(["```", ""])

    lines.append("## Pages")
    lines.append("")
    for index in range(1, page_count + 1):
        page_text = pages_text[index - 1] if index - 1 < len(pages_text) else ""
        page_image = rendered_pages[index - 1] if index - 1 < len(rendered_pages) else None

        lines.append(f"### Page {index}")
        lines.append("")
        if page_image is not None:
            lines.append(f"![Rendered page {index}]({markdown_link(page_image, md_path.parent)})")
            lines.append("")
        else:
            lines.append("_No rendered page image was generated for this page._")
            lines.append("")

        lines.append("#### Extracted Text")
        lines.append("")
        if page_text.strip():
            lines.append("```text")
            lines.append(page_text.rstrip())
            lines.append("```")
        else:
            lines.append("_No extractable text found on this page. Use the rendered page image above._")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def convert_pdf(index: int, total: int, pdf_path: Path, literature_dir: Path, assets_root: Path) -> None:
    digest = sha256_file(pdf_path)
    asset_dir = assets_root / safe_asset_name(index, pdf_path, digest)
    resolved_assets_root = assets_root.resolve()
    resolved_asset_dir = asset_dir.resolve()
    if resolved_assets_root not in resolved_asset_dir.parents and resolved_asset_dir != resolved_assets_root:
        raise RuntimeError(f"Refusing to clear unexpected asset path: {resolved_asset_dir}")
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    md_path = pdf_path.with_suffix(".md")
    with tempfile.TemporaryDirectory(prefix="pdf-md-") as tmp:
        tmp_dir = Path(tmp)
        tmp_pdf = tmp_dir / f"{digest[:16]}.pdf"
        tmp_text = tmp_dir / "extracted.txt"
        shutil.copy2(pdf_path, tmp_pdf)

        pdfinfo = run(["pdfinfo", str(tmp_pdf)])
        pdfdetach = run(["pdfdetach", "-list", str(tmp_pdf)])
        pdftotext = run(["pdftotext", "-layout", "-enc", "UTF-8", str(tmp_pdf), str(tmp_text)])
        extracted_text = tmp_text.read_text(encoding="utf-8", errors="replace") if tmp_text.exists() else ""

        render = run(["pdftoppm", "-r", "200", "-png", str(tmp_pdf), str(asset_dir / "page")])
        rendered_pages = normalize_rendered_pages(asset_dir)
        pages = parse_pages(pdfinfo.stdout)
        pages_text = split_pages(extracted_text, pages)

        write_markdown(
            pdf_path=pdf_path,
            md_path=md_path,
            asset_dir=asset_dir,
            digest=digest,
            pdfinfo_text=pdfinfo.stdout + pdfinfo.stderr,
            pdfdetach_text=pdfdetach.stdout + pdfdetach.stderr,
            pages_text=pages_text,
            rendered_pages=rendered_pages,
            pdftotext_stderr=pdftotext.stderr,
            render_stderr=render.stderr,
        )

    print(f"[{index}/{total}] {pdf_path.name} -> {md_path.name} ({len(rendered_pages)} rendered pages)")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Convert literature PDFs to LLM-readable Markdown.")
    parser.add_argument("--literature-dir", default="literature", type=Path)
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        type=Path,
        help="Specific PDF path to convert. Can be passed multiple times.",
    )
    parser.add_argument("--start", default=1, type=int, help="1-based index of the first sorted PDF to convert.")
    parser.add_argument("--limit", default=None, type=int, help="Maximum number of sorted PDFs to convert.")
    args = parser.parse_args()

    literature_dir = args.literature_dir.resolve()
    assets_root = literature_dir / "_pdf_markdown_assets"
    assets_root.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pdfs = [(path if path.is_absolute() else Path.cwd() / path).resolve() for path in args.pdf]
    else:
        pdfs = sorted(literature_dir.glob("*.pdf"), key=lambda path: path.name.casefold())

    if not pdfs:
        print(f"No PDFs found in {literature_dir}")
        return 0

    if args.pdf:
        selected = pdfs
        start = 1
    else:
        selected = pdfs[args.start - 1 :]
        if args.limit is not None:
            selected = selected[: args.limit]
        start = args.start

    for index, pdf_path in enumerate(selected, start=start):
        convert_pdf(index, len(pdfs), pdf_path.resolve(), literature_dir, assets_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
