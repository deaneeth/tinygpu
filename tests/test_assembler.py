import tempfile
import os
from tinygpu.assembler import assemble_file


def test_assemble_file_basic():
    code = """
start:
LOAD R1, 42
ADD R2, R1, 1
JMP start
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tgpu") as f:
        f.write(code)
        fname = f.name
    try:
        program, labels = assemble_file(fname)
        assert labels["start"] == 0
        assert program[0][0] == "LOAD"
        assert program[1][0] == "ADD"
        assert program[2][0] == "JMP"
        assert program[2][1][0] == "start" or program[2][1][0] == 0
    finally:
        os.remove(fname)
