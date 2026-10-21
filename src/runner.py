"""Helper cho các runner (run_reproduce, run_preprocess): chạy `python -m <module>` từng bước, bỏ qua bước đã có kết quả."""
import os
import shlex
import subprocess
import sys
import time
from typing import Iterable, List


class Runner:
    def __init__(self, dry_run: bool = False, keep_going: bool = False):
        self.dry_run, self.keep_going = dry_run, keep_going
        self.failed: List[str] = []
        self.env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "PYTHONUTF8": "1"}

    def __call__(self, module: str, *args: str, done_if: Iterable[str] = ()) -> bool:
        done_if = list(done_if)
        if done_if and all(os.path.exists(p) for p in done_if):
            print(f"[skip] đã có {', '.join(done_if)}")
            return True
        print(f"\n>>> python -m {module} {' '.join(shlex.quote(a) for a in args)}", flush=True)
        if self.dry_run:
            return True
        t0 = time.time()
        code = subprocess.call([sys.executable, "-m", module, *args], env=self.env)
        if code != 0:
            msg = f"python -m {module} ... (exit {code})"
            self.failed.append(msg)
            print(f"[!] LỖI: {msg}", flush=True)
            if not self.keep_going:
                raise SystemExit(code)
            return False
        print(f"[ok] {time.time() - t0:.0f}s", flush=True)
        return True

    def finish(self, t0: float, out: str):
        print(f"\nXong sau {(time.time() - t0) / 60:.1f} phút. Kết quả: {out}")
        if self.failed:
            print("Các lệnh lỗi:\n  " + "\n  ".join(self.failed))
            raise SystemExit(1)
