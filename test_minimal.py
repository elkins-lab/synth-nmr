from synth_nmr.synth_nmr_cli import process_commands
import sys
from io import StringIO
from unittest.mock import patch

def test_process_commands_empty():
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()
    
    with patch("sys.stdin.readline", side_effect=["exit\n"]):
        process_commands([])
    
    sys.stdout = old_stdout
    out = my_stdout.getvalue()
    print("OUTPUT IS:", out)

test_process_commands_empty()
