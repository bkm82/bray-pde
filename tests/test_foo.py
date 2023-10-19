import pytest
import sys
from solver.foo import add_one, main


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (1, 2),
        (3, 4),
        (-1, 0),
    ],
)
@pytest.hookimpl(tryfirst=True)
def test_add_one(test_input, expected):
    assert add_one(test_input) == expected


@pytest.fixture
def capture_stdout(monkeypatch):
    buffer = {"stdout": "", "write_calls": 0}

    def fake_write(s):
        buffer["stdout"] += s
        buffer["write_calls"] += 1

    monkeypatch.setattr(sys.stdout, "write", fake_write)
    return buffer


def test_print(capture_stdout):
    main()
    assert capture_stdout["stdout"] == "2\n"
