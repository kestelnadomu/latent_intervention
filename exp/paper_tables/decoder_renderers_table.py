"""LaTeX table of every renderer's decoded edits, with the true and the drawn target state.

Fact accuracy and joint accuracy, then the text metrics of the decoder table; a second
table (``<output stem>_commands.tex``) splits the edits under the true s', where s'
changes a fact, into CVs stating the new value, the value of x, or another. Standard
library only; reads the shipped aggregates of package-v1 (the state flow's edits with the
operator's draw, phi^-1(phi(x')) against x'), renderers-v1 (everything else), the decoded
do-nothing phi^-1(phi(x)) against x', the noise floor and the joint accuracies (the last
three from ``exp.text_realizer_package_eval.baselines``):
    python -m exp.paper_tables.decoder_renderers_table --output overleaf/tables/decoder_renderers.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decoder_package_table import (
    FACTS,
    FLOOR,
    JOINT,
    ROW_END,
    SUMMARY,
    TEXT_COLUMNS,
    _number,
    _write,
    validate,
    validate_floor,
    validate_joint,
)
from .decoder_table import bold, cell

ROOT = Path(__file__).resolve().parents[2]
RENDERERS = ROOT / "docs/experiments/results/decoder_package_renderers_v1.json"
DECODED = ROOT / "docs/experiments/results/decoder_package_decoded_unedited_v1.json"
DECODER = "768-270m"
# the grouping and order of the renderer tables in the results section
GROUPS = (
    ("Minimal edit", (("editor", "Editor"), ("ot_editor", "OT editor"))),
    ("Constant-noise edit", (("flow", "State flow"), ("ot_flow", "OT flow"))),
    ("Closed form", (("state_effect", "State effect"), ("group_ot", "Group OT"))),
)
# the renderer's target state: the true s', or the operator's draw on the reading
TARGETS = ("s'", "omega")
TARGET_LABELS = (r"$s'$", r"$\omega_S(\psi(z))$")


def render(
    summary: dict, renderers: dict, decoded: dict, floor: dict, joint: dict
) -> str:
    """Fact accuracy (%), joint accuracy (%) and text metrics per renderer, true s' and
    omega_S(psi(z)) side by side, and the references."""
    base, bert, _, _ = validate(summary)
    if (
        renderers.get("format_version") != 1
        or renderers["decoder"] != DECODER
        or renderers["unit_ids_sha256"] != summary["unit_ids_sha256"]
    ):
        raise ValueError(
            "renderer results refer to another decoder or other test units"
        )
    if decoded.get("format_version") != 1 or (
        decoded["unit_ids_sha256"],
        decoded["comparisons_sha256"],
    ) != (summary["unit_ids_sha256"], renderers["comparisons_sha256"]):
        raise ValueError("the decoded do-nothing refers to other texts")
    rewrite = validate_floor(floor)
    joints, _, _ = validate_joint(summary, floor, joint)
    if (
        joint["renderers"]["comparisons_sha256"],
        joint["renderers"]["unit_ids_sha256"],
    ) != (renderers["comparisons_sha256"], renderers["unit_ids_sha256"]):
        raise ValueError("joint accuracies refer to other renderer texts")
    renderer_joints = {row["method"]: row for row in joint["renderers"]["rows"]}
    rows = {row["method"]: row for row in renderers["rows"]}
    if set(renderer_joints) != {*rows, "decoded_unedited"}:
        raise ValueError("joint accuracies cover other renderer rows")
    cohorts = {
        (row["against"], row["joint"]["reference_scorable"])
        for row in renderer_joints.values()
    }
    if len(cohorts) != 2:
        raise ValueError("joint accuracies use different reference cohorts")
    joint_n = dict(cohorts)
    # package-v1 decoded the state flow's edits with the operator's draw
    flow = base[DECODER, "predicted"]
    scorable = {
        f: round(flow[f"{f}_reference_coverage"] * summary["units"]) for f in FACTS
    }

    def package(condition: str) -> dict:
        """A package-v1 row in the schema of the renderer results."""
        key = DECODER, condition
        facts = {
            f: {"accuracy": base[key][f"{f}_accuracy_on_scorable_references"]}
            for f in FACTS
        }
        return {
            **base[key],
            **bert[key],
            "facts": facts,
            "joint": joints[DECODER, None, condition],
        }

    def checked(row: dict, against: str = "x'") -> dict:
        if row["units"] != summary["units"] or row["against"] != against:
            raise ValueError(f"{row['method']}: other units or reference")
        if against == "x'" and any(
            row["facts"][f]["reference_scorable"] != scorable[f] for f in FACTS
        ):
            raise ValueError(
                f"{row['method']}: fact cohorts differ from the state flow's"
            )
        if renderer_joints[row["method"]]["against"] != against:
            raise ValueError(
                f"{row['method']}: joint accuracy against another reference"
            )
        return {**row, "joint": renderer_joints[row["method"]]["joint"]["accuracy"]}

    def edits(method: str) -> dict:
        return package("predicted") if method == "flow" else checked(rows[method])

    def cells(row: dict) -> list[str]:
        """O, D, U and the joint accuracy in percent, then the text metrics."""
        facts = [row["facts"][f]["accuracy"] for f in FACTS] + [row["joint"]]
        return ["--" if v is None else cell(_number(v), 1, 100) for v in facts] + [
            "--" if row.get(name) is None else cell(_number(row[name]), digits, 1)
            for name, _, digits in TEXT_COLUMNS
        ]

    references = (
        (r"$\varphi^{-1}(\varphi(x'))$ against $x'$", package("oracle")),
        (r"$\varphi^{-1}(\varphi(x))$ against $x'$", checked(decoded["row"])),
        (r"$\varphi^{-1}(\varphi(x))$ against $x$", checked(rows["factual"], "x")),
        (
            "Same-state rewrite against $x$",
            {**rewrite, "joint": joints["same_state_rewrite", None, None]},
        ),
    )
    labels = [*FACTS.values(), "Joint", *(label for _, label, _ in TEXT_COLUMNS)]
    headers = [rf"\multicolumn{{2}}{{c}}{{{bold(label)}}}" for label in labels]
    columns = 1 + 2 * len(headers)
    rules = "".join(rf"\cmidrule(lr){{{c}-{c + 1}}}" for c in range(2, columns, 2))
    lines = [
        "% Generated by exp.paper_tables.decoder_renderers_table; do not edit numbers.",
        f"% Sources: package-v1 aggregate {summary['reports_sha256']},",
        f"%   renderers-v1 comparisons {renderers['comparisons_sha256']}.",
        f"% Decoder {DECODER}; rho trained on and fed psi(z) as source state; tested with the true s' and with omega_S(psi(z)).",
        "% Group OT reads only O, which do(O = k) sets exactly: its two columns coincide.",
        "% References: phi^-1(phi(x')) of package-v1; phi^-1(phi(x)) of renderers-v1 against x'",
        f"%   and against x; the noise floor's {floor['units']} same-state rewrites against x.",
        "% Fact and joint accuracy in percent; joint: O, D and U all right.",
        "% Fact n, O/D/U/joint, against x': {}/{}/{}/{}; against x: one more for O and D,"
        " joint {}.".format(*scorable.values(), joint_n["x'"], joint_n["x"]),
        "% Text scores use raw 0--1 scales; no VADER (its compound saturates at about 0.99).",
        f"% BERTScore excludes IDs {renderers['bertscore_cohort']['excluded_ids']} for the"
        " renderers-v1 rows (no truncation); package-v1 rows use all units.",
        "% Bare tabular: provide the table float/caption and load booktabs in the paper.",
        r"\begin{tabular}{@{}l" + "c" * (columns - 1) + "@{}}",
        r"  \toprule",
        "  & " + " & ".join(headers) + ROW_END,
        "  " + rules,
        "  "
        + " & ".join(
            [
                bold("Test"),
                *[bold(label) for label in TARGET_LABELS] * len(headers),
            ]
        )
        + ROW_END,
        r"  \midrule",
        rf"  \multicolumn{{{columns}}}{{@{{}}l}}{{\textit{{$\varphi^{{-1}}(\hat{{z}}')$ against $x'$}}}}"
        + ROW_END,
    ]
    for group, members in GROUPS:
        lines.append(
            rf"  \multicolumn{{{columns}}}{{@{{}}l}}{{\emph{{{group}}}}}" + ROW_END
        )
        for method, label in members:
            pairs = zip(
                cells(edits(f"{method}-true-target")),
                cells(edits(method)),
                strict=True,
            )
            values = [value for pair in pairs for value in pair]
            lines.append("  " + " & ".join([label, *values]) + ROW_END)
    lines.append(r"  \midrule")
    for label, row in references:
        spanning = [rf"\multicolumn{{2}}{{c}}{{{value}}}" for value in cells(row)]
        lines.append("  " + " & ".join([label, *spanning]) + ROW_END)
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_commands(renderers: dict, joint: dict) -> str:
    """Where the true s' changes a fact, the shares (%) of every renderer's decoded edits
    that state the new value (realized), still the value of x (kept) or another or no
    value (other); each fact's three shares add up to 100. Joint: all three facts at once."""
    if joint["renderers"]["comparisons_sha256"] != renderers["comparisons_sha256"]:
        raise ValueError("commanded-state scores refer to other renderer texts")
    commands = {(r["method"], r["command"]): r for r in joint["commands"]["rows"]}
    expected = {(m, c) for _, members in GROUPS for m, _ in members for c in TARGETS}
    if set(commands) != expected | {("oracle", "s'"), ("factual", "s'")} or len(
        commands
    ) != len(joint["commands"]["rows"]):
        raise ValueError("commanded-state scores cover other rows")
    shares = (r"Realized $\uparrow$", r"Kept $\downarrow$", "Other")
    blocks = {**FACTS, "joint": "Joint"}
    columns = 1 + len(shares) * len(blocks)

    def cells(method: str) -> list[str]:
        values = []
        for f in blocks:
            changed = commands[method, "s'"]["changed"][f]
            realized, kept = _number(changed["realized"]), _number(changed["kept"])
            values += [realized, kept, _number(1 - realized - kept)]
        return [cell(v, 1, 100) for v in values]

    units = [commands["editor", "s'"]["changed"][f]["units"] for f in blocks]
    lines = [
        "% Generated by exp.paper_tables.decoder_renderers_table; do not edit numbers.",
        f"% Renderers-v1 comparisons {renderers['comparisons_sha256']};"
        f" decoder {DECODER}; rho trained on and fed psi(z), tested with the true s'.",
        "% Per fact, the units where s' changes it (n, O/D/U/joint: {}/{}/{}/{}): shares of decoded"
        " CVs stating the new value, still the value of x, or another or no value.".format(
            *units
        ),
        "% Bare tabular: provide the table float/caption and load booktabs in the paper.",
        r"\begin{tabular}{@{}l" + "r" * (columns - 1) + "@{}}",
        r"  \toprule",
        "  & "
        + " & ".join(
            rf"\multicolumn{{{len(shares)}}}{{c}}{{{bold(label)}}}"
            for label in blocks.values()
        )
        + ROW_END,
        "  "
        + "".join(
            rf"\cmidrule(lr){{{c}-{c + len(shares) - 1}}}"
            for c in range(2, columns, len(shares))
        ),
        "  "
        + " & ".join([bold("Renderer"), *map(bold, shares * len(blocks))])
        + ROW_END,
        r"  \midrule",
    ]
    for group, members in GROUPS:
        lines.append(
            rf"  \multicolumn{{{columns}}}{{@{{}}l}}{{\emph{{{group}}}}}" + ROW_END
        )
        for method, label in members:
            lines.append("  " + " & ".join([label, *cells(method)]) + ROW_END)
    lines.append(r"  \midrule")
    for label, method in (
        (r"$\varphi^{-1}(\varphi(x'))$", "oracle"),
        (r"$\varphi^{-1}(\varphi(x))$ (no edit)", "factual"),
    ):
        lines.append("  " + " & ".join([label, *cells(method)]) + ROW_END)
    lines += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--renderers", type=Path, default=RENDERERS)
    parser.add_argument("--decoded", type=Path, default=DECODED)
    parser.add_argument("--floor", type=Path, default=FLOOR)
    parser.add_argument("--joint", type=Path, default=JOINT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.summary, args.renderers, args.decoded, args.floor, args.joint)
    summary, renderers, decoded, floor, joint = (
        json.loads(path.read_text(encoding="utf-8")) for path in inputs
    )
    tables = {
        args.output: render(summary, renderers, decoded, floor, joint),
        args.output.with_stem(f"{args.output.stem}_commands"): render_commands(
            renderers, joint
        ),
    }
    for path, text in tables.items():
        _write(path, text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
