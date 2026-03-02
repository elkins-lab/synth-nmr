import subprocess
out = subprocess.run(["python", "scripts/evaluate_shiftx2.py", "/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb", "--shiftx2-exe", "/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py"], capture_output=True, text=True)
print(out.stdout)
