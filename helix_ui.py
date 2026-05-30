"""
helix_ui.py -- TRON Ares styled terminal status / progress output.
All chatter goes to stderr so stdout stays clean for piping.
"""
import sys
import time

# 256-color TRON Ares palette
CYAN  = "\033[38;5;51m"
AMBER = "\033[38;5;214m"
ORANGE= "\033[38;5;208m"
DIM   = "\033[2m"
BOLD  = "\033[1m"
RST   = "\033[0m"

_USE_COLOR = sys.stderr.isatty()

def _c(code, s):
    return f"{code}{s}{RST}" if _USE_COLOR else s

def banner(title):
    line = "=" * 64
    print(_c(CYAN, f"\n{line}"), file=sys.stderr)
    print(_c(CYAN + BOLD, f"  {title}"), file=sys.stderr)
    print(_c(CYAN, line), file=sys.stderr, flush=True)

def section(title):
    print(_c(AMBER, f"\n-- {title} " + "-" * max(2, 50 - len(title))),
          file=sys.stderr, flush=True)

def status(msg):
    print(f"{_c(CYAN, '[ HELIX ]')} {msg}", file=sys.stderr, flush=True)

def warn(msg):
    print(f"{_c(AMBER, '[ WARN  ]')} {msg}", file=sys.stderr, flush=True)

def result(key, val):
    print(f"   {_c(DIM, key.ljust(34))} {_c(ORANGE, str(val))}",
          file=sys.stderr, flush=True)

def bar(i, n, label=""):
    w = 26
    frac = (i + 1) / n
    f = int(w * frac)
    body = "#" * f + "." * (w - f)
    last = (i + 1 == n)
    if _USE_COLOR:                       # live terminal: overwrite in place
        end = "\n" if last else ""
        print(f"\r   {label.ljust(12)} {_c(CYAN, '[' + body + ']')} "
              f"{i+1}/{n}", end=end, file=sys.stderr, flush=True)
    else:                                # piped/log: only ~5% milestones
        step = max(1, n // 20)
        if last or (i % step == 0):
            print(f"   {label.ljust(12)} [{body}] {i+1}/{n}",
                  file=sys.stderr, flush=True)

class Timer:
    def __init__(self, label):
        self.label = label
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *a):
        status(f"{self.label} done in {time.time()-self.t0:.2f}s")
