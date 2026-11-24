"""Trusted invocation shim. Executed ONLY inside the approved runsc container.

The gateway sends a single test's inputs, never its expected value or suite.
Do not import this file as a host candidate runner.
"""
import ast
import builtins
import json
import math
import resource
import subprocess
import sys
import time


class PolicyError(Exception):
    pass


def check(source, signature, imports):
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 65536 or "```" in source:
        raise ValueError("invalid code-only module")
    tree = ast.parse(source)
    required = ast.parse("def " + signature + ":\n    pass").body[0]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == required.name]
    if len(functions) != 1 or ast.dump(functions[0].args) != ast.dump(required.args):
        raise PolicyError("public signature changed")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            raise PolicyError("unapproved top-level statement")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                raise PolicyError("only literal top-level constants allowed")
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.returns or any(arg.annotation for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]):
                raise PolicyError("decorators/annotations disallowed")
            for default in [*node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]]:
                try:
                    ast.literal_eval(default)
                except (ValueError, TypeError):
                    raise PolicyError("nonliteral default")
    denied = {"open", "exec", "eval", "compile", "getattr", "setattr", "delattr", "vars", "globals", "locals", "dir", "input", "breakpoint", "help", "print", "exit", "quit", "type", "object", "super"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute, ast.arg, ast.FunctionDef)):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else node.arg if isinstance(node, ast.arg) else node.name
            if name.startswith("_") or name in denied:
                raise PolicyError("forbidden name")
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise PolicyError("unsupported language capability")
        if isinstance(node, ast.Import):
            if any(a.name not in imports for a in node.names):
                raise PolicyError("forbidden import")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in imports or any(a.name.startswith("_") or a.name == "*" for a in node.names):
                raise PolicyError("forbidden import")
    return tree, required.name


def json_value(value, depth=0):
    if depth > 32:
        raise ValueError("return nesting too deep")
    if type(value) is int and value.bit_length() > 4096:
        # json.dumps refuses ints over sys.get_int_max_str_digits() (4300) digits;
        # treat oversized integers as a candidate's unsupported return value.
        raise ValueError("integer return value too large")
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [json_value(v, depth+1) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: json_value(v, depth+1) for k, v in value.items()}
    raise ValueError("unsupported return type")


def main():
    request = json.loads(sys.stdin.buffer.read(131073))
    if "tests" in request:
        started = time.monotonic()
        results = []
        total_cpu = 0
        for test in request["tests"]:
            remaining = request["wall_seconds"] - (time.monotonic()-started)
            if remaining <= 0 or total_cpu >= request["cpu_seconds"]*1000:
                results.append({"status":"TIMEOUT","compiled":False,"cpu_ms":0,"peak_memory_bytes":0})
                break
            call = {k:v for k,v in request.items() if k not in {"tests","runner_source","wall_seconds"}}
            call.update(test)
            cpu_remaining = request["cpu_seconds"] - total_cpu/1000
            call["cpu_seconds"] = max(1,math.ceil(cpu_remaining))
            try:
                child = subprocess.run([sys.executable,"-I","-B","-c",request["runner_source"]],
                    input=json.dumps(call),text=True,capture_output=True,
                    timeout=min(remaining,cpu_remaining))
                if child.returncode in {-9,-24,137,152}:
                    result = {"status":"RESOURCE_LIMIT","compiled":False,"cpu_ms":0,"peak_memory_bytes":0}
                elif child.returncode in {-11, -6}:
                    # SIGSEGV/SIGABRT from pure candidate code (e.g. a deeply nested hash)
                    # happens while calling the compiled function: a scored candidate
                    # failure (reward 0), not infrastructure. Compilation errors are
                    # caught by the shim, so compiled=True is accurate here.
                    result = {"status":"OK","compiled":True,"returned":False,"cpu_ms":0,"peak_memory_bytes":0}
                elif child.returncode:
                    result = {"status":"INTERNAL_ERROR","compiled":False,"cpu_ms":0,"peak_memory_bytes":0}
                elif len(child.stdout.encode())>65536 or len(child.stderr.encode())>65536:
                    result = {"status":"RESOURCE_LIMIT","compiled":True,"cpu_ms":0,"peak_memory_bytes":0}
                else:
                    result = json.loads(child.stdout)
            except subprocess.TimeoutExpired:
                result = {"status":"TIMEOUT","compiled":False,"cpu_ms":0,"peak_memory_bytes":0}
            results.append(result)
            total_cpu += result.get("cpu_ms",0)
            if result["status"] != "OK":
                break
        response = {"status":"OK","compiled":True,"results":results,"cpu_ms":total_cpu,
                    "peak_memory_bytes":max((r.get("peak_memory_bytes",0) for r in results),default=0)}
        encoded = json.dumps(response,allow_nan=False)
        if len(encoded.encode())>65536:
            encoded = json.dumps({"status":"RESOURCE_LIMIT","compiled":False,"cpu_ms":0,"peak_memory_bytes":0})
        sys.stdout.write(encoded)
        return
    resource.setrlimit(resource.RLIMIT_CPU, (request["cpu_seconds"], request["cpu_seconds"]))
    # RLIMIT_AS complements outer container memory control.
    memory = request["memory_mib"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    compiled = False
    try:
        tree, function = check(request["source"], request["signature"], request["imports"])
        code = compile(tree, "<candidate>", "exec")
        compiled = True
        names = ("abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "pow", "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError", "StopIteration")
        safe = {name: getattr(builtins, name) for name in names}
        def importer(name, globals=None, locals=None, fromlist=(), level=0):
            if level or name not in request["imports"]:
                raise PolicyError("import denied")
            return builtins.__import__(name, globals, locals, fromlist, level)
        safe["__import__"] = importer
        namespace = {"__builtins__": safe}
        exec(code, namespace)
        try:
            result = namespace[function](*request["args"], **request["kwargs"])
            response = {"status": "OK", "compiled": True, "returned": True, "value": json_value(result)}
        except MemoryError:
            response = {"status": "RESOURCE_LIMIT", "compiled": True}
        except Exception:
            response = {"status": "OK", "compiled": True, "returned": False}
    except SyntaxError:
        response = {"status": "COMPILE_ERROR", "compiled": False}
    except PolicyError:
        response = {"status": "POLICY_VIOLATION", "compiled": compiled}
    except MemoryError:
        response = {"status": "RESOURCE_LIMIT", "compiled": compiled}
    except (ValueError, TypeError):
        response = {"status": "INVALID_OUTPUT", "compiled": compiled}
    except Exception:
        response = {"status": "OK", "compiled": compiled, "returned": False}
    usage = resource.getrusage(resource.RUSAGE_SELF)
    response["cpu_ms"] = int((usage.ru_utime + usage.ru_stime) * 1000)
    response["peak_memory_bytes"] = usage.ru_maxrss * 1024
    try:
        encoded = json.dumps(response, allow_nan=False)
    except (ValueError, TypeError):
        encoded = json.dumps({"status": "INVALID_OUTPUT", "compiled": compiled,
                              "cpu_ms": response["cpu_ms"], "peak_memory_bytes": response["peak_memory_bytes"]})
    if len(encoded.encode()) > 65536:
        encoded = json.dumps({"status": "RESOURCE_LIMIT", "compiled": compiled, "cpu_ms": 0, "peak_memory_bytes": 0})
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
