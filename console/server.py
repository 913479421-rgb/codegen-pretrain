#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server.py —— 模型训练控制台后端
================================
用 Python 标准库 http.server（无第三方依赖）把控制台的按钮动作转成
真实的训练/生成/评测命令，并流式回传日志。

在你（有 GPU）的机器上启动：
    python console/server.py  --port 8080
然后浏览器打开 console/模型训练控制台.html。

API：
    GET  /api/status       工程状态（模型/数据/checkpoint/引擎）
    GET  /api/actions      可用动作
    POST /api/action       执行动作 {action, args}（后台运行，写入日志）
    GET  /api/logs?n=50    最近日志
    GET  /api/loss         训练 loss 曲线（从日志解析）
"""
import os, sys, json, subprocess, threading, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # codegen-pretrain/
CONSOLE_DIR = os.path.join(ROOT, "console")
os.chdir(ROOT)

# 动作 → (命令模板, 是否需要语料/checkpoint)
ACTIONS = {
    "prepare": "python prepare_data.py --local-dir data/stream --vocab-size 16000",
    "train":   "python train.py --config small --steps 20000",
    "train_large": "python train.py --config large --steps 120000",
    "continuous": "python continuous_train.py --config small --watch-dir data/stream --steps-per-round 5000 --interval 120",
    "generate": "python generate.py --checkpoint checkpoints/final.pt --prompt 'def fibonacci(n):\\n' --max-tokens 80 --use-ema",
    "eval":    "python eval.py --checkpoint checkpoints/final.pt --use-ema --samples 3",
    "distill": "python assistant_models.py --model qwen-coder --distill --num 100 --out data/stream",
    "quant":   "python export_quantized.py --checkpoint checkpoints/final.pt --bits 8",
    "bench":   "python eval_benchmark.py --download --dataset humaneval",
    # 生产级 10b：The Stack 几GB + 10b档 + DeepSpeed多卡 + 持续训练，随server常驻后台运行
    "prod_10b": "bash run_production_10b.sh",
    "collect": "python collect_data.py --source stack --max-mb 2000 --out-dir data/stream",
    "check":   "bash 00_check_env.sh",
}

# 当前运行状态
_jobs = {}          # id -> {"cmd","proc","start"}
_log_lock = threading.Lock()
_log_buf = []       # 环形日志缓冲
_next_id = 0


def _log(msg):
    with _log_lock:
        _log_buf.append(msg)
        if len(_log_buf) > 800:
            _log_buf[:] = _log_buf[-600:]


def _run_job(job_id, cmd):
    def _worker():
        _log(f"▶ 启动 [{job_id}]: {cmd}")
        try:
            proc = subprocess.Popen(cmd, shell=True, cwd=ROOT,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            _jobs[job_id]["proc"] = proc
            for line in proc.stdout:
                _log(line.rstrip())
            proc.wait()
            _jobs[job_id]["rc"] = proc.returncode
            _log(f"■ 完成 [{job_id}] rc={proc.returncode}")
        except Exception as e:
            _log(f"✗ 失败 [{job_id}]: {e}")
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _project_status():
    meta = {}
    mp = os.path.join(ROOT, "data", "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    ckpt_dir = os.path.join(ROOT, "checkpoints")
    ckpts = []
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    has_vocab = os.path.exists(os.path.join(ROOT, "data", "vocab.json"))
    stream = os.path.join(ROOT, "data", "stream")
    n_stream = len([f for f in os.listdir(stream) if f.endswith((".py",))]) if os.path.isdir(stream) else 0
    return {
        "vocab": meta.get("vocab_size", "—"),
        "train_tokens": meta.get("train_tokens", 0),
        "checkpoints": ckpts,
        "has_model": bool(ckpts),
        "stream_files": n_stream,
        "gpu": _gpu_info(),
    }


def _gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        return out.strip().splitlines()[0]
    except Exception:
        return "无 GPU / 未检测"


def _parse_loss():
    """从日志解析 step/loss，用于曲线。"""
    pts = []
    pat = re.compile(r"step (\d+)/\d+ \| loss ([\d.]+)")
    with _log_lock:
        for line in _log_buf:
            m = pat.search(line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    return pts


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._cors()

    def _serve_static(self, path):
        """托管 console/ 静态前端，单端口同时服务页面与 API。"""
        if path in ("/", ""):
            path = "/模型训练控制台.html"
        rel = path.lstrip("/")
        full = os.path.normpath(os.path.join(CONSOLE_DIR, rel))
        if not full.startswith(CONSOLE_DIR):
            self._json({"error": "forbidden"}, 403)
            return
        if os.path.isdir(full):
            full = os.path.join(full, "模型训练控制台.html")
        if not os.path.exists(full):
            self._json({"error": "not found"}, 404)
            return
        ctype = ("text/html; charset=utf-8" if full.endswith(".html")
                 else "text/css; charset=utf-8" if full.endswith(".css")
                 else "application/octet-stream")
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        global _next_id
        p = urlparse(self.path)
        q = parse_qs(p.query)
        if not p.path.startswith("/api/"):
            self._serve_static(p.path)
            return
        if p.path == "/api/status":
            self._json(_project_status())
        elif p.path == "/api/actions":
            self._json({"actions": list(ACTIONS.keys())})
        elif p.path == "/api/logs":
            n = int(q.get("n", ["80"])[0])
            with _log_lock:
                tail = _log_buf[-n:]
            self._json({"logs": tail})
        elif p.path == "/api/loss":
            self._json({"points": _parse_loss()})
        elif p.path == "/api/jobs":
            with _log_lock:
                jobs = [{ "id": k, "cmd": v["cmd"], "running": bool(v.get("proc") and v["proc"].poll() is None),
                          "rc": v.get("rc")} for k, v in _jobs.items()]
            self._json({"jobs": jobs})
        else:
            self._json({"error": "unknown path"}, 404)

    def do_POST(self):
        global _next_id
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        action = body.get("action", "")
        if action not in ACTIONS:
            self._json({"error": f"未知动作 {action}"}, 400)
            return
        cmd = ACTIONS[action]
        # 支持 args 覆盖默认参数（如 --steps）
        args = body.get("args", "")
        if args:
            cmd = cmd.split("--steps")[0] + args if "--steps" in cmd else cmd + " " + args
        job_id = f"job{_next_id}"; _next_id += 1
        _jobs[job_id] = {"cmd": cmd, "proc": None}
        _run_job(job_id, cmd)
        self._json({"ok": True, "job_id": job_id, "cmd": cmd})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    # PaaS(Render/Railway/Fly)会注入 PORT 环境变量；--port 优先
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"[console] 控制台后端运行在 http://{args.host}:{args.port}")
    print(f"[console] 浏览器打开 console/模型训练控制台.html")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[console] 已停止")


if __name__ == "__main__":
    main()
