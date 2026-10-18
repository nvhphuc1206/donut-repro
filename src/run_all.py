"""Chạy tự động toàn bộ pipeline (Phase 0 → 3), có log từng bước, file trạng thái và resume.

    python -m src.run_all              # chạy mọi bước chưa xong
    python -m src.run_all --list       # xem danh sách bước + trạng thái, không chạy
    python -m src.run_all --skip_groups I5 I3
    python -m src.run_all --only_groups P2 I1

Theo dõi tiến độ (mở trong VS Code, tự cập nhật ~5 giây/lần):
    results/pipeline_status.md     – bảng trạng thái, bước hiện tại, % hoàn thành, thanh tiến trình
    results/logs/pipeline.log      – nhật ký bắt đầu/kết thúc từng bước
    results/logs/pipeline/NN_<bước>.log – toàn bộ output của từng bước
Ctrl+C để dừng; chạy lại cùng lệnh để tiếp tục (bước đã xong được bỏ qua, train tự resume từ last.pt).
"""
import argparse
import codecs
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

from src import utils

STATE_PATH = "results/pipeline_state.json"
STATUS_MD = "results/pipeline_status.md"
PIPE_LOG = "results/logs/pipeline.log"
STEP_LOG_DIR = "results/logs/pipeline"
OFFICIAL = "naver-clova-ix/donut-base-finetuned-cord-v2"
NOISE_TYPES = ["rotate", "blur", "jpeg", "color", "perspective", "mixed"]

Cmd = Union[List[str], Callable[[dict], List[str]]]


@dataclass
class Step:
    id: str
    group: str
    title: str
    cmd: Cmd
    est_min: float                      # thời gian ước tính (phút) – dùng tính % theo thời gian
    done_if: List[str] = field(default_factory=list)   # có đủ các file này -> coi như đã xong
    needs: List[str] = field(default_factory=list)     # bước phải thành công trước
    allow_fail: bool = False            # lỗi không chặn các bước khác (vd. 1600x1200 OOM)


# ----------------------------------------------------------------------------
# Danh sách bước
# ----------------------------------------------------------------------------
def build_steps(args) -> List[Step]:
    W = ["--set", "num_workers=0"] if os.name == "nt" else []
    S: List[Step] = []

    def ev(run_id, out, *extra, model=OFFICIAL):
        return ["src.evaluate", "--model", model, "--run_id", run_id, "--out", out, *extra]

    # --- Phase 0
    S.append(Step("env", "P0", "Ghi môi trường", ["src.env_info"], 0.5, ["results/env.txt"]))
    S.append(Step("selftest", "P0", "Selftest logic", ["src.selftest"], 0.2))

    # --- Phase 1
    S.append(Step("p1_test", "P1", "Checkpoint chính thức – test",
                  ev("P1-official", "results/p1_official_cord.json", "--split", "test",
                     "--summary_csv", "results/p1_summary.csv", "--notes", "checkpoint chinh thuc, greedy, fp32"),
                  15, ["results/p1_official_cord.json"]))
    S.append(Step("p1_val", "P1", "Checkpoint chính thức – validation",
                  ev("P1-official-val", "results/p1_official_cord_val.json", "--split", "validation",
                     "--summary_csv", "results/p1_summary.csv"),
                  15, ["results/p1_official_cord_val.json"]))
    if args.with_multitask:
        for t in ["rvlcdip", "docvqa"]:
            S.append(Step(f"p1_{t}", "P1", f"Đa nhiệm {t} (subset 500)",
                          ["src.eval_multitask", "--task", t, "--limit", "500"], 20,
                          [f"results/p1_multitask_{t}.json"], allow_fail=True))

    # --- Phase 2
    S.append(Step("p2_model_info", "P2", "Kiến trúc & số tham số", ["src.model_info", "--model", OFFICIAL], 1,
                  ["results/model_info.json", "results/figures/architecture.png"]))
    S.append(Step("p2_errors", "P2", "Phân tích lỗi",
                  ["src.error_analysis", "--pred", "results/p1_official_cord.json", "--top_k", "10", "--save_images"],
                  1, ["results/error_analysis.md"], needs=["p1_test"]))

    def attn_cmd(ctx):
        recs = utils.load_json("results/p1_official_cord.json")["records"]
        worst = [str(r["idx"]) for r in sorted(recs, key=lambda r: r["ted_acc"])[:2]]
        return ["src.visualize_attention", "--model", OFFICIAL, "--indices", "0", "1", "2", *worst, "--layer", "last"]

    S.append(Step("p2_attn", "P2", "Cross-attention (lớp cuối)", attn_cmd, 5,
                  ["results/attention/sample0_summary.png"], needs=["p1_test"]))
    S.append(Step("p2_attn_mean", "P2", "Cross-attention (TB các lớp)",
                  ["src.visualize_attention", "--model", OFFICIAL, "--indices", "0", "--layer", "mean",
                   "--out_dir", "results/attention_meanlayers"], 2,
                  ["results/attention_meanlayers/sample0_summary.png"]))

    # --- I1
    for h, w in [(640, 480), (960, 720), (1280, 960), (1600, 1200)]:
        S.append(Step(f"i1_{h}", "I1", f"Resolution {h}x{w}",
                      ev(f"I1-official-{h}x{w}", f"results/i1/official_{h}x{w}.json", "--input_size", str(h), str(w),
                         "--notes", "inference-only, resolution khac luc train"),
                      {640: 6, 960: 10, 1280: 15, 1600: 25}[h], [f"results/i1/official_{h}x{w}.json"],
                      allow_fail=h == 1600))
    S.append(Step("i1_plot", "I1", "Biểu đồ trade-off",
                  ["src.plot_results", "tradeoff", "--prefix", "I1-official-", "--out", "results/figures/i1_tradeoff.png"],
                  0.2, needs=["i1_640", "i1_960", "i1_1280"]))

    # --- I4
    for b in [1, 3, 5]:
        S.append(Step(f"i4_beam{b}", "I4", f"Beam search = {b}",
                      ev(f"I4-beam{b}", f"results/i4/beam{b}.json", "--num_beams", str(b)),
                      {1: 15, 3: 25, 5: 35}[b], [f"results/i4/beam{b}.json"]))
    S.append(Step("i4_constrained", "I4", "Greedy + giải mã ràng buộc",
                  ev("I4-greedy-constrained", "results/i4/greedy_constrained.json", "--constrained"), 20,
                  ["results/i4/greedy_constrained.json"]))
    for b in [1, 5]:
        S.append(Step(f"i4_post{b}", "I4", f"Hậu xử lý (beam {b})",
                      ["src.postprocess_eval", "--pred", f"results/i4/beam{b}.json", "--run_prefix", f"I4-beam{b}"],
                      3, needs=[f"i4_beam{b}"]))

    # --- I5
    S.append(Step("i5_fp32", "I5", "GPU fp32", ev("I5-gpu-fp32", "results/i5/gpu_fp32.json", "--precision", "fp32"),
                  15, ["results/i5/gpu_fp32.json"]))
    S.append(Step("i5_fp16", "I5", "GPU fp16", ev("I5-gpu-fp16", "results/i5/gpu_fp16.json", "--precision", "fp16"),
                  10, ["results/i5/gpu_fp16.json"], allow_fail=True))
    n = str(args.cpu_limit)
    S.append(Step("i5_cpu_fp32", "I5", f"CPU fp32 ({n} ảnh)",
                  ev(f"I5-cpu-fp32-n{n}", "results/i5/cpu_fp32.json", "--precision", "fp32", "--device", "cpu",
                     "--limit", n, "--warmup", "1"), 20, ["results/i5/cpu_fp32.json"]))
    S.append(Step("i5_cpu_int8", "I5", f"CPU int8 ({n} ảnh)",
                  ev(f"I5-cpu-int8-n{n}", "results/i5/cpu_int8.json", "--precision", "int8",
                     "--limit", n, "--warmup", "1"), 15, ["results/i5/cpu_int8.json"], allow_fail=True))

    # --- Train: smoke test quyết định resolution
    S.append(Step("smoke", "TRAIN", "Smoke test train ở 1280x960 (đo VRAM)",
                  ["src.train", "--config", "configs/smoke_test.yaml", "--set", "run_id=SMOKE_1280",
                   "output_dir=outputs/SMOKE_1280", "input_size=[1280,960]", *W[1:]],
                  8, ["outputs/SMOKE_1280/train_summary.json"], allow_fail=True))

    def size_args(ctx):
        h, w = ctx["train_size"]
        return [f"input_size=[{h},{w}]"]

    def b0_dir(ctx):
        return f"outputs/B0_full_{ctx['train_size'][0]}"

    def train_cmd(config, run_id, out, *extra):
        return lambda ctx: ["src.train", "--config", config, "--set", f"run_id={run_id(ctx) if callable(run_id) else run_id}",
                            f"output_dir={out(ctx) if callable(out) else out}", *size_args(ctx), *extra, *W[1:]]

    # --- B0
    S.append(Step("b0_train", "B0", "Full fine-tune B0", train_cmd(
        "configs/train_cord_full.yaml", lambda c: f"B0_full_{c['train_size'][0]}", b0_dir), 300,
        done_if=["{B0}/train_summary.json", "{B0}/best/config.json"]))
    S.append(Step("b0_eval", "B0", "Đánh giá B0 trên test",
                  lambda c: ev("B0-test", "results/b0_test.json", "--summary_csv", "results/p1_summary.csv",
                               "--notes", "fine-tune nhom, best theo val TED", model=f"{b0_dir(c)}/best"),
                  15, ["results/b0_test.json"], needs=["b0_train"]))
    S.append(Step("b0_curves", "B0", "Đường cong train B0",
                  lambda c: ["src.plot_results", "curves", "--run_dir", b0_dir(c), "--out", "results/figures/B0_curves.png"],
                  0.2, ["results/figures/B0_curves.png"], needs=["b0_train"]))

    # --- I2
    S.append(Step("i2_train", "I2", "LoRA r=8", train_cmd("configs/train_cord_lora.yaml", "I2_lora_r8", "outputs/I2_lora_r8"),
                  200, done_if=["outputs/I2_lora_r8/train_summary.json", "outputs/I2_lora_r8/best/config.json"]))
    S.append(Step("i2_eval", "I2", "Đánh giá LoRA trên test",
                  ev("I2-test", "results/i2_lora_test.json", "--notes", "LoRA r=8 (da merge)", model="outputs/I2_lora_r8/best"),
                  15, ["results/i2_lora_test.json"], needs=["i2_train"]))
    S.append(Step("i2_curves", "I2", "Đường cong train LoRA",
                  ["src.plot_results", "curves", "--run_dir", "outputs/I2_lora_r8", "--out", "results/figures/I2_curves.png"],
                  0.2, ["results/figures/I2_curves.png"], needs=["i2_train"]))

    # --- I3
    S.append(Step("i3_noisy", "I3", "Tạo tập test nhiễu", ["src.make_noisy_test", "--out_root", "data/cord_noisy"], 3,
                  [f"data/cord_noisy/{t}/dataset_info.json" for t in NOISE_TYPES]))
    for t in NOISE_TYPES:
        S.append(Step(f"i3_official_{t}", "I3", f"Chính thức trên nhiễu {t}",
                      ev(f"I3-official-{t}", f"results/i3/official_{t}.json", "--dataset", f"data/cord_noisy/{t}"),
                      15, [f"results/i3/official_{t}.json"], needs=["i3_noisy"]))
    for tag, cfg in [("aug", "configs/train_cord_aug.yaml"), ("noaug", "configs/train_cord_noaug_control.yaml")]:
        out = f"outputs/I3_{tag}_from_B0"
        S.append(Step(f"i3_train_{tag}", "I3", f"Train thêm 4 epoch từ B0 ({tag})",
                      train_cmd(cfg, f"I3_{tag}_from_B0", out, f"model_name_or_path={{B0}}/best"), 100,
                      done_if=[f"{out}/train_summary.json", f"{out}/best/config.json"], needs=["b0_train"]))
    for tag, model, need in [("B0", "{B0}/best", "b0_train"), ("AUG", "outputs/I3_aug_from_B0/best", "i3_train_aug"),
                             ("NOAUG", "outputs/I3_noaug_from_B0/best", "i3_train_noaug")]:
        S.append(Step(f"i3_{tag}_clean", "I3", f"{tag} trên test sạch",
                      ev(f"I3-{tag}-clean", f"results/i3/{tag}_clean.json", model=model),
                      15, [f"results/i3/{tag}_clean.json"], needs=[need]))
        for t in NOISE_TYPES:
            S.append(Step(f"i3_{tag}_{t}", "I3", f"{tag} trên nhiễu {t}",
                          ev(f"I3-{tag}-{t}", f"results/i3/{tag}_{t}.json", "--dataset", f"data/cord_noisy/{t}", model=model),
                          15, [f"results/i3/{tag}_{t}.json"], needs=[need, "i3_noisy"]))

    # --- Tổng hợp
    S.append(Step("tables", "TABLE", "Bảng tổng hợp runs.csv", ["src.plot_results", "table", "--out", "results/runs_table.md"], 0.2))
    return S


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
class Runner:
    def __init__(self, steps: List[Step], args):
        self.steps, self.args = steps, args
        self.state = utils.load_json(STATE_PATH) if os.path.exists(STATE_PATH) else {"steps": {}}
        self.state.setdefault("steps", {})
        self.current: Optional[str] = None
        self.sub_progress = ""
        self.last_write = 0.0
        self.started = time.time()

    # ---- ngữ cảnh: resolution train + đường dẫn B0
    def ctx(self) -> dict:
        size = self.args.train_size
        if size == "auto":
            smoke = self.state["steps"].get("smoke", {}).get("status")
            if "train_size" in self.state:
                hw = self.state["train_size"]
            elif smoke == "done":
                hw = [1280, 960]
            elif smoke == "failed":
                hw = [960, 720]
            else:
                hw = [1280, 960]
        else:
            hw = [int(x) for x in size.lower().split("x")]
        return {"train_size": hw, "B0": f"outputs/B0_full_{hw[0]}"}

    def fill(self, s: str) -> str:
        return s.replace("{B0}", self.ctx()["B0"])

    def st(self, step_id) -> dict:
        return self.state["steps"].setdefault(step_id, {"status": "pending"})

    def is_done(self, step: Step) -> bool:
        if self.args.force and step.id in self.selected_ids:
            return False
        if step.done_if:
            return all(os.path.exists(self.fill(p)) for p in step.done_if)
        return self.st(step.id).get("status") == "done"

    def save(self, force=False):
        if not force and time.time() - self.last_write < 5:
            return
        self.last_write = time.time()
        self.state["updated"] = utils.now_str()
        self.state["current"] = self.current
        utils.save_json(self.state, STATE_PATH)
        self.write_status_md()

    def write_status_md(self):
        sel = [s for s in self.steps if s.id in self.selected_ids]
        done = [s for s in sel if self.st(s.id).get("status") in ("done", "skipped_done")]
        total_est = sum(s.est_min for s in sel) or 1
        done_est = sum(s.est_min for s in done)
        icons = {"done": "✅", "skipped_done": "✅", "running": "⏳", "failed": "❌", "blocked": "⛔",
                 "pending": "⬜", "interrupted": "⏸️"}
        cur = next((s for s in sel if s.id == self.current), None)
        lines = [
            "# Trạng thái pipeline",
            "",
            f"- Cập nhật: **{utils.now_str()}** (lần chạy bắt đầu {time.strftime('%H:%M:%S', time.localtime(self.started))})",
            f"- Tiến độ theo số bước: **{len(done)}/{len(sel)}** ({100 * len(done) / max(len(sel), 1):.1f}%)",
            f"- Tiến độ theo thời gian ước tính: **{100 * done_est / total_est:.1f}%** "
            f"(còn khoảng {(total_est - done_est) / 60:.1f} giờ theo ước tính T4)",
            f"- Resolution train: `{self.ctx()['train_size']}`",
            f"- Bước hiện tại: **{cur.id + ' – ' + cur.title if cur else '(không có)'}**",
            f"- Tiến trình bước hiện tại: `{self.sub_progress or '-'}`",
            "",
            "| # | Nhóm | Bước | Trạng thái | Bắt đầu | Thời gian | Log | Ghi chú |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for i, s in enumerate(sel, 1):
            r = self.st(s.id)
            dur = f"{r['seconds'] / 60:.1f} phút" if r.get("seconds") is not None else ""
            log = f"[log]({os.path.relpath(r['log'], 'results').replace(os.sep, '/')})" if r.get("log") else ""
            lines.append(f"| {i} | {s.group} | {s.id} – {s.title} | {icons.get(r.get('status'), '')} {r.get('status')} "
                         f"| {r.get('start', '')} | {dur} | {log} | {r.get('note', '')} |")
        utils.ensure_dir("results")
        with open(STATUS_MD, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def plog(self, msg):
        line = f"[{utils.now_str()}] {msg}"
        print(f"\n>>> {line}", flush=True)
        utils.ensure_dir(os.path.dirname(PIPE_LOG))
        with open(PIPE_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run_step(self, idx: int, step: Step) -> str:
        ctx = self.ctx()
        cmd = step.cmd(ctx) if callable(step.cmd) else step.cmd
        cmd = [self.fill(c) for c in cmd]
        full = [sys.executable, "-u", "-m", *cmd]
        log_path = os.path.join(STEP_LOG_DIR, f"{idx:02d}_{step.id}.log")
        utils.ensure_dir(STEP_LOG_DIR)
        r = self.st(step.id)
        r.update(status="running", start=utils.now_str(), log=log_path, cmd=" ".join(cmd), note="")
        r.pop("seconds", None)
        self.current, self.sub_progress = step.id, ""
        self.save(force=True)
        self.plog(f"BẮT ĐẦU [{idx}/{len(self.steps)}] {step.id} – {step.title}\n    python -m {' '.join(cmd)}")

        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        t0 = time.time()
        tail: List[str] = []
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'=' * 80}\n[{utils.now_str()}] python -m {' '.join(cmd)}\n{'=' * 80}\n")
            proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
            buf = ""
            try:
                while True:
                    chunk = proc.stdout.read1(4096)
                    if not chunk:
                        break
                    text = dec.decode(chunk)
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    # Windows: tiến trình con xuống dòng bằng "\r\n" -> chuẩn hóa trước khi xử lý "\r" của tqdm
                    buf = (buf + text).replace("\r\n", "\n")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.split("\r")[-1]
                        fh.write(line + "\n")
                        if line.strip():
                            tail = (tail + [line])[-30:]
                        if "%|" in line:
                            self.sub_progress = line.strip()[:160]
                    if "\r" in buf.rstrip("\r"):  # giữ "\r" cuối (có thể là nửa đầu của "\r\n")
                        keep_cr = buf.endswith("\r")
                        buf = buf.rstrip("\r").split("\r")[-1] + ("\r" if keep_cr else "")
                    if "%|" in buf:
                        self.sub_progress = buf.strip()[:160]
                    fh.flush()
                    self.save()
                code = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                r.update(status="interrupted", seconds=time.time() - t0)
                self.current = None
                self.save(force=True)
                self.plog(f"DỪNG (Ctrl+C) tại {step.id}. Chạy lại cùng lệnh để tiếp tục.")
                raise
        r["seconds"] = time.time() - t0
        if code == 0:
            r["status"] = "done"
            self.plog(f"XONG {step.id} ({r['seconds'] / 60:.1f} phút)")
        else:
            oom = any("out of memory" in l.lower() for l in tail)
            r["status"] = "failed"
            r["note"] = ("OOM – " if oom else "") + f"exit {code}" + (" (cho phép lỗi)" if step.allow_fail else "")
            self.plog(f"LỖI {step.id} ({r['note']}). Xem log: {log_path}")
            err = next((l for l in reversed(tail) if "Error" in l or "error:" in l), tail[-1] if tail else "")
            r["note"] += f" – {err.strip()[:120]}" if err else ""
            utils.log_issue(f"run_all/{step.id}: {r['note']}. Lệnh: python -m {' '.join(cmd)}. Log: {log_path}")
        if step.id == "smoke" and self.args.train_size == "auto" and "train_size" not in self.state:
            self.state["train_size"] = [1280, 960] if code == 0 else [960, 720]
            self.plog(f"Resolution train được chọn: {self.state['train_size']}")
        self.current = None
        self.save(force=True)
        return r["status"]

    def run(self):
        sel = [s for s in self.steps if s.id in self.selected_ids]
        self.plog(f"===== PIPELINE: {len(sel)} bước được chọn =====")
        status_ok = {}
        for idx, step in enumerate(self.steps, 1):
            if step.id not in self.selected_ids:
                continue
            r = self.st(step.id)
            if self.is_done(step):
                if r.get("status") != "done":
                    r["status"] = "skipped_done"
                    r.setdefault("note", "đã có kết quả từ trước")
                status_ok[step.id] = True
                self.save(force=True)
                continue
            blocked = [n for n in step.needs if not status_ok.get(n, self.is_done(self.by_id[n]))]
            if blocked:
                r.update(status="blocked", note=f"chờ: {', '.join(blocked)}")
                status_ok[step.id] = False
                self.plog(f"BỎ QUA {step.id}: bước phụ thuộc chưa xong ({', '.join(blocked)})")
                self.save(force=True)
                continue
            result = self.run_step(idx, step)
            status_ok[step.id] = result == "done"
            if result != "done" and not step.allow_fail and self.args.stop_on_error:
                self.plog("Dừng vì --stop_on_error")
                break
        self.save(force=True)
        failed = [s.id for s in sel if self.st(s.id).get("status") in ("failed", "blocked")]
        self.plog("===== KẾT THÚC =====" + (f" Bước lỗi/bị chặn: {', '.join(failed)}" if failed else " Tất cả thành công"))
        print(f"Trạng thái: {STATUS_MD}")


def main():
    ap = argparse.ArgumentParser(description="Chạy tự động toàn bộ pipeline Donut")
    ap.add_argument("--list", action="store_true", help="Chỉ in danh sách bước và trạng thái")
    ap.add_argument("--only_groups", nargs="*", help="Chỉ chạy các nhóm: P0 P1 P2 I1 I4 I5 TRAIN B0 I2 I3 TABLE")
    ap.add_argument("--skip_groups", nargs="*", default=[], help="Bỏ các nhóm")
    ap.add_argument("--only_steps", nargs="*", help="Chỉ chạy các bước (theo id)")
    ap.add_argument("--force", action="store_true", help="Chạy lại các bước được chọn dù đã có kết quả")
    ap.add_argument("--train_size", default="auto", help="auto (smoke test quyết định) | 1280x960 | 960x720")
    ap.add_argument("--cpu_limit", type=int, default=30, help="Số ảnh cho phép đo CPU ở I5")
    ap.add_argument("--with_multitask", action="store_true", help="Thêm Task 1.3 RVL-CDIP/DocVQA")
    ap.add_argument("--stop_on_error", action="store_true", help="Dừng toàn bộ khi một bước lỗi")
    args = ap.parse_args()

    steps = build_steps(args)
    runner = Runner(steps, args)
    runner.by_id = {s.id: s for s in steps}
    runner.selected_ids = {
        s.id for s in steps
        if (args.only_groups is None or s.group in args.only_groups)
        and s.group not in args.skip_groups
        and (args.only_steps is None or s.id in args.only_steps)
    }
    if args.list:
        for i, s in enumerate(steps, 1):
            mark = "x" if runner.is_done(s) else " "
            sel = "" if s.id in runner.selected_ids else "  (không chọn)"
            print(f"[{mark}] {i:02d} {s.group:<6} {s.id:<22} {s.title}{sel}")
        print(f"\nResolution train: {runner.ctx()['train_size']}")
        return
    try:
        runner.run()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
