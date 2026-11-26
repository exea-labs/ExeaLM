"""Progress you can watch: a tqdm bar in a terminal, and in a file (a Slurm job's output, read with tail -f) the same
tqdm line printed every `every` seconds on a line of its own, so the file never fills with carriage returns.

    bar = Progress(total=45688, initial=start, desc="exealm_3b", unit="step", every=60)
    for step in ...:
        bar.update(1, loss=3.21, tok_s="52.4k")
    bar.close()

Both go to stderr: stdout stays the JSON lines a run prints. Without tqdm, the file form is printed in plain words.
"""
import sys
import time

try:
    from tqdm import tqdm
except ImportError:                                                      # tqdm comes with huggingface_hub
    tqdm = None


def human(x):
    """52431 -> '52.4k', 1.2e6 -> '1.20M': short numbers for a progress line."""
    for div, s in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:.3g}{s}"
    return f"{x:.3g}"


class Progress:
    def __init__(self, total, desc="", initial=0, unit="it", every=60.0, enabled=True, file=None):
        self.total, self.desc, self.unit, self.every = total, desc, unit, every
        self.n = self.start = initial
        self.enabled = enabled
        self.file = file or sys.stderr
        self.tty = bool(getattr(self.file, "isatty", lambda: False)()) and tqdm is not None
        self.t0 = self.last = time.time()
        self.postfix = {}
        self.printed = None                                             # the step of the last line printed
        self.bar = None
        if enabled and self.tty:
            self.bar = tqdm(total=total, initial=initial, desc=desc, unit=unit, dynamic_ncols=True, file=self.file,
                            smoothing=0.05)
        elif enabled:
            self.line(force=True)

    def update(self, k=1, **postfix):
        if not self.enabled:
            return
        self.n += k
        self.postfix.update(postfix)
        if self.bar is not None:
            if postfix:
                self.bar.set_postfix(self.postfix, refresh=False)
            self.bar.update(k)
        elif time.time() - self.last >= self.every or (self.total and self.n >= self.total):
            self.line()

    def write(self, text):
        """A message that does not break the bar."""
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.write(text, file=self.file)
        else:
            print(text, file=self.file, flush=True)

    def line(self, force=False):
        now = time.time()
        elapsed = now - self.t0
        done = self.n - self.start
        rate = done / elapsed if elapsed > 0 and done > 0 else None
        post = ", ".join(f"{k}={v}" for k, v in self.postfix.items())
        if tqdm is not None:                                            # elapsed 0 until a step is timed: '?' rate,
            text = tqdm.format_meter(self.n, self.total, elapsed if rate else 0, prefix=self.desc, unit=self.unit,
                                     rate=rate, postfix=post or None, ncols=110)   # not the resumed steps over 0 s
        else:
            pct = f"{100 * self.n / self.total:.1f}%" if self.total else ""
            left = (self.total - self.n) / rate if rate and self.total else None
            eta = f", about {left / 3600:.1f} h left" if left is not None else ""
            text = f"{self.desc}: {self.n}/{self.total or '?'} {self.unit} {pct}{eta} {post}"
        print(f"{time.strftime('%H:%M:%S')} {text}", file=self.file, flush=True)
        self.last, self.printed = now, self.n

    def close(self):
        if not self.enabled:
            return
        if self.bar is not None:
            self.bar.close()
        elif self.printed != self.n:                                    # the last line already shows this step
            self.line()
