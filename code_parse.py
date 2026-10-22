# -*- coding: utf-8 -*-
"""
code_parse.py —— 代码解析引擎（防幻觉的校验闭环核心）
========================================================
用 Python 标准库 ast 做生成代码的语法与结构解析，实现：
  1. 语法校验：生成代码能否被 AST 正确解析（不能 = 幻觉/语法错误）
  2. 可执行性验证：在受控环境下尝试 exec，捕获运行时错误
  3. 未定义名检测：找出代码里引用但未定义/未导入的名字（幻觉 API 的典型特征）
  4. 质量统计：函数数、缩进、括号平衡、行长等

依赖：仅标准库 ast + 内置 exec（无第三方）。
"""
import ast
import io
import contextlib


class ParseResult:
    def __init__(self, code):
        self.code = code
        self.ok = False
        self.tree = None
        self.error = None
        self.undefined = []
        self.n_imports = 0
        self.n_defs = 0
        self.indent_ok = True
        self.paren_ok = True

    @property
    def score(self):
        """0~1 语法健康度：越低越可能是幻觉/坏代码。"""
        s = 0.0
        if self.ok:
            s += 0.5
        if self.indent_ok:
            s += 0.15
        if self.paren_ok:
            s += 0.15
        if not self.undefined:
            s += 0.2
        return s


def _collect_names(tree, defined, used):
    """粗略收集定义名与使用名（函数/参数/赋值/导入）。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if node.args and node.args.args:
                for a in node.args.args:
                    defined.add(a.arg)
            if node.args and node.args.posonlyargs:
                for a in node.args.posonlyargs:
                    defined.add(a.arg)
        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            defined.add(node.targets[0].id)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)


def parse_check(code):
    """语法 + 结构解析，返回 ParseResult。"""
    res = ParseResult(code)
    if not code or not code.strip():
        res.error = "空代码"
        return res
    # 缩进与括号平衡粗查
    res.indent_ok = all(len(l) - len(l.lstrip()) <= 32 for l in code.splitlines() if l.strip())
    res.paren_ok = code.count("(") == code.count(")") and code.count("[") == code.count("]")
    try:
        res.tree = ast.parse(code)
        res.ok = True
    except SyntaxError as e:
        res.error = f"SyntaxError: {e}"
        return res

    defined, used = set(), set()
    _collect_names(res.tree, defined, used)
    builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    res.undefined = sorted(used - defined - builtins)
    res.n_imports = sum(1 for n in ast.walk(res.tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
    res.n_defs = sum(1 for n in ast.walk(res.tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    return res


def run_check(code, timeout=2.0):
    """可执行性验证：受控 exec，返回 (ok, error, captured_stdout)。"""
    buf = io.StringIO()
    err = None
    ok = False
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<generated>", "exec"), {"__builtins__": __builtins__}, {})
        ok = True
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return ok, err, buf.getvalue()


def quality_stats(code):
    """基本质量统计。"""
    lines = code.splitlines()
    return {
        "lines": len(lines),
        "non_empty": sum(1 for l in lines if l.strip()),
        "defs": sum(1 for l in lines if l.strip().startswith(("def ", "class "))),
        "avg_line_len": round(sum(len(l) for l in lines) / max(1, len(lines)), 1),
        "n_chars": len(code),
    }
