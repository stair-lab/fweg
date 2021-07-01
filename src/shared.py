import os

def run_file(exp: str, f: str):
    """
    Run file `f` for experiment `exp`. Assumed to be called in `src/` folder.
    """
    # generic run file
    folder = os.path.join("py", exp)
    cmd = f"cd {folder} && python {f}"
    print(f"\t File: {os.path.join(folder, f)}")
    if os.system(cmd) != 0:
        print(f"Running {os.path.join(folder, f)} encountered an error, exiting...")
        os.sys.exit(1)

