"""Notebooks stay runnable and reviewable: every code cell is valid Python, and none is committed with outputs.

Running a notebook takes minutes, so CI only checks that it parses. To run one end to end:
    pip install nbclient ipykernel matplotlib
    jupyter execute notebooks/tiny_tour.ipynb
"""
import json
import pathlib

import pytest

NOTEBOOKS = sorted((pathlib.Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_compiles_and_has_no_outputs(path):
    nb = json.loads(path.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for i, cell in enumerate(code):
        compile("".join(cell["source"]), f"{path.name}, code cell {i}", "exec")
        assert not cell["outputs"] and cell["execution_count"] is None, f"clear the outputs of {path.name}"


def test_there_is_a_notebook():
    assert any(p.name == "tiny_tour.ipynb" for p in NOTEBOOKS)
