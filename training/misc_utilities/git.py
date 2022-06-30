import subprocess as sp
from pathlib import Path
import logging


def get_git_commit() -> str:
    try:
        p = sp.Popen(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], stdout=sp.PIPE)
        assert p.wait() == 0
        commit_hash = p.stdout.read().decode().strip()
        # quick check if the output is actually a commit hash
        assert len(commit_hash) == 40
        return commit_hash
    except:
        logging.warn("Could not retrieve git commit")
        return None


def write_git_log(filename: str, n=4):
    try:
        p = sp.Popen(["git", "-C", str(Path(__file__).parent), "log", "-n", str(n)], stdout=sp.PIPE)
        assert p.wait() == 0
        log_lines = p.stdout.read().decode()
        with open(filename, "w") as git_log:
            git_log.write(log_lines)
    except:
        logging.warn("Could not write git log")
