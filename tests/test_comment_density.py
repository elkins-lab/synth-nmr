"""
Enforce the philosophy that the code is the textbook via comment density.
Also see test_docs_integrity.py for tests of the educational notes.
"""

import io
import os
import tokenize

import pytest


def calculate_comment_ratio(file_path):
    """
    Calculates the ratio of comment lines (including docstrings) to code lines.
    """
    if not os.path.exists(file_path):
        return 0.0

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    try:
        f_obj = io.StringIO(content)
        tokens = list(tokenize.generate_tokens(f_obj.readline))

        comment_line_indices = set()
        code_line_indices = set()

        for tok in tokens:
            start_line = tok.start[0]
            tok.end[0]

            if tok.type == tokenize.COMMENT:
                # Check if the comment (excluding the #) has non-blank content
                comment_content = tok.string.lstrip("#").strip()
                if comment_content:
                    comment_line_indices.add(start_line)
            elif tok.type == tokenize.STRING:
                # Docstrings/Strings: count only lines with non-blank text
                s_lines = tok.string.splitlines()
                for i, s_line in enumerate(s_lines):
                    if s_line.strip():
                        comment_line_indices.add(start_line + i)
            elif tok.type not in (
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.STRING,
            ):
                # Count actual code tokens (excluding blank lines and docstrings)
                code_line_indices.add(start_line)

        # A line can be both a code line AND a comment line (inline comments)
        final_comment_count = len(comment_line_indices)
        final_code_count = len(code_line_indices)

        return final_comment_count / max(1, final_code_count)
    except Exception:
        return 0.0


@pytest.mark.parametrize(
    "file_path, min_ratio",
    [
        ("synth_nmr/__init__.py", 0.90),
        ("synth_nmr/chemical_shifts.py", 0.85),
        ("synth_nmr/data_pipeline.py", 0.40),
        ("synth_nmr/data.py", 0.95),
        ("synth_nmr/j_coupling.py", 1.50),
        ("synth_nmr/nef_io.py", 0.95),
        ("synth_nmr/neural_shifts.py", 1.00),
        ("synth_nmr/nmr.py", 1.15),
        ("synth_nmr/rdc.py", 1.35),
        ("synth_nmr/relaxation.py", 1.10),
        ("synth_nmr/scripts/__init__.py", 0.00),
        ("synth_nmr/scripts/train_gnn.py", 0.30),
        ("synth_nmr/structure_utils.py", 1.25),
        ("synth_nmr/synth_nmr_cli.py", 0.20),
        ("synth_nmr/trajectory.py", 1.95),
        ("synth_nmr/validation.py", 0.55),
    ],
)
def test_library_documentation_density(file_path, min_ratio):
    """
    Enforces a minimum documentation density for core library components.
    This ensures the project maintains its pedagogical and educational value.
    """
    # Adjust path relative to project root if needed
    full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
    if not os.path.exists(full_path):
        # Fallback for different test execution working directories
        full_path = file_path

    ratio = calculate_comment_ratio(full_path)
    assert ratio >= min_ratio, (
        f"Documentation density for {file_path} is {ratio:.2f}, which is below the required {min_ratio}"
    )
