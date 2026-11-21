"""CPU/stdlib contracts for the book's operator examples and prose code blocks.

The small tensor double rounds each operation to IEEE binary16/binary32. It
executes the quantization functions extracted from Chapter 15 so that FP16
underflow cannot be hidden by a separately reimplemented helper.
"""

import ast
import builtins
import math
from pathlib import Path
import re
import struct
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = ROOT / "docs/chapters"
TOPICS_DIR = ROOT / "docs/topics"
TOOLKIT_DIR = ROOT / "docs/toolkit"
CH12 = CHAPTERS_DIR / "chapter-12.md"
CH15 = CHAPTERS_DIR / "chapter-15.md"
CH17 = CHAPTERS_DIR / "chapter-17.md"
CH22 = CHAPTERS_DIR / "chapter-22.md"
CH23 = CHAPTERS_DIR / "chapter-23.md"
EXAMPLES = ROOT / "docs/modules/part3-operators/examples"


def python_blocks(path: Path) -> list[str]:
    return re.findall(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)


def functions_from_prose(path: Path, names: set[str]) -> ast.Module:
    nodes = []
    for block in python_blocks(path):
        if not any(f"def {name}(" in block for name in names):
            continue
        tree = ast.parse(block)
        nodes.extend(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        )
    assert {node.name for node in nodes} == set(names)
    return ast.Module(body=nodes, type_ignores=[])


def round_to_dtype(value, dtype):
    if dtype == "int8":
        return 0 if not math.isfinite(value) else int(value)
    fmt = "e" if dtype == "float16" else "f"
    return struct.unpack(fmt, struct.pack(fmt, value))[0]


class SmallTensor:
    def __init__(self, values, dtype="float16"):
        self.dtype = dtype
        self.values = self._map(values, lambda value: round_to_dtype(value, dtype))

    @staticmethod
    def _map(values, fn):
        if isinstance(values, list):
            return [SmallTensor._map(value, fn) for value in values]
        return fn(values)

    def float(self):
        return SmallTensor(self.values, "float32")

    def abs(self):
        return SmallTensor(self._map(self.values, abs), self.dtype)

    def amax(self, dim):
        rows = self.values
        values = (
            [max(row) for row in rows]
            if dim == 1
            else [max(row[col] for row in rows) for col in range(len(rows[0]))]
        )
        return SmallTensor(values, self.dtype)

    def clamp_min(self, lower):
        return SmallTensor(self._map(self.values, lambda value: max(value, lower)), self.dtype)

    def clamp(self, lower, upper):
        return SmallTensor(
            self._map(self.values, lambda value: min(max(value, lower), upper)),
            self.dtype,
        )

    def to(self, dtype):
        return SmallTensor(self.values, dtype)

    def __getitem__(self, key):
        if key == (slice(None), None):
            return SmallTensor([[value] for value in self.values], self.dtype)
        if key == (None, slice(None)):
            return SmallTensor([self.values], self.dtype)
        raise AssertionError(f"Unsupported test tensor index: {key!r}")

    def __truediv__(self, other):
        def divide(a, b):
            if b == 0:
                return math.nan if a == 0 else math.copysign(math.inf, a)
            return a / b

        if not isinstance(other, SmallTensor):
            result = self._map(self.values, lambda value: divide(value, other))
        else:
            denominators = other.values
            result = [
                [
                    divide(
                        value,
                        denominators[row if len(denominators) > 1 else 0][
                            col if len(denominators[0]) > 1 else 0
                        ],
                    )
                    for col, value in enumerate(values)
                ]
                for row, values in enumerate(self.values)
            ]
        return SmallTensor(result, self.dtype)


class SmallTorch:
    int8 = "int8"

    @staticmethod
    def round(tensor):
        return SmallTensor(
            SmallTensor._map(
                tensor.values,
                lambda value: round(value) if math.isfinite(value) else value,
            ),
            tensor.dtype,
        )


class TestNumericAndPluginContracts(unittest.TestCase):
    def test_quantization_zero_and_tiny_inputs(self):
        namespace = {"torch": SmallTorch}
        exec(
            compile(
                functions_from_prose(CH15, {"quant_per_token", "quant_per_channel"}),
                str(CH15),
                "exec",
            ),
            namespace,
        )
        cases = (
            ("quant_per_token", [[0.0, 0.0], [1e-7, -1e-7], [1.0, -1.0]]),
            ("quant_per_channel", [[0.0, 1e-7, 1.0], [0.0, -1e-7, -1.0]]),
        )
        for name, values in cases:
            with self.subTest(function=name):
                quantized, scales = namespace[name](SmallTensor(values))
                self.assertTrue(
                    all(math.isfinite(value) and value > 0 for value in scales.values),
                    "FP16 1e-8 clamp underflows; compute scales in FP32",
                )
                self.assertEqual(scales.dtype, "float32")
                zero_values = (
                    quantized.values[0]
                    if name == "quant_per_token"
                    else [row[0] for row in quantized.values]
                )
                tiny_values = (
                    quantized.values[1]
                    if name == "quant_per_token"
                    else [row[1] for row in quantized.values]
                )
                self.assertEqual(zero_values, [0, 0])
                self.assertTrue(any(value != 0 for value in tiny_values))

    def test_plugin_check_calls_registration(self):
        blocks = python_blocks(CH17)
        init = next(block for block in blocks if "# my_plugin/__init__.py" in block)
        check_block = next(block for block in blocks if "op_registry_oot.get(" in block)
        # Execute layer-1 bare registry check (lines prior to layer-2 engine init)
        layer1_check = check_block.split("# 第 2 层", 1)[0]
        registry = {}
        package = types.ModuleType("my_plugin")
        custom = types.ModuleType("vllm.model_executor.custom_op")
        custom.op_registry_oot = registry
        activation = types.ModuleType("vllm.model_executor.layers.activation")

        class SiluAndMul:
            def __init__(self):
                raise AssertionError(
                    "Bare registry checks must not construct an engine-configured op"
                )

        activation.SiluAndMul = SiluAndMul
        replacement = type("TritonSiluAndMul", (), {})
        plugin = types.SimpleNamespace(TritonSiluAndMul=replacement)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and "silu_and_mul_plugin" in fromlist:
                package.silu_and_mul_plugin = plugin
                registry["SiluAndMul"] = replacement
                return package
            return {
                "my_plugin": package,
                custom.__name__: custom,
                activation.__name__: activation,
            }[name]

        namespace = package.__dict__
        namespace["__builtins__"] = dict(vars(builtins), __import__=fake_import)
        namespace["__package__"] = "my_plugin"
        exec(init, namespace)
        exec(layer1_check, namespace)
        self.assertIs(registry.get("SiluAndMul"), replacement)


class TestCompleteExampleContracts(unittest.TestCase):
    def test_complete_python_sources_parse(self):
        paths = (
            EXAMPLES / "quant_matmul.py",
            EXAMPLES / "silu_plugin/my_plugin/__init__.py",
            EXAMPLES / "silu_plugin/my_plugin/kernel.py",
            EXAMPLES / "silu_plugin/my_plugin/silu_and_mul_plugin.py",
        )
        for path in paths:
            with self.subTest(path=path.name):
                compile(ast.parse(path.read_text(encoding="utf-8")), str(path), "exec")

    def test_quantization_example_matches_prose(self):
        tree = ast.parse((EXAMPLES / "quant_matmul.py").read_text(encoding="utf-8"))
        names = {"quant_per_token", "quant_per_channel"}
        example_functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        for prose_function in functions_from_prose(CH15, names).body:
            self.assertEqual(
                ast.dump(prose_function, include_attributes=False),
                ast.dump(example_functions[prose_function.name], include_attributes=False),
            )

    def test_int8_wrappers_pass_complete_kernel_signature(self):
        tree = ast.parse((EXAMPLES / "quant_matmul.py").read_text(encoding="utf-8"))
        funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for wrapper, kernel in (
            ("matmul_int8", "matmul_int8_kernel"),
            ("matmul_fused_dequant", "matmul_fused_dequant_kernel"),
        ):
            with self.subTest(wrapper=wrapper):
                call = next(
                    node
                    for node in ast.walk(funcs[wrapper])
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Subscript)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == kernel
                )
                constexpr = {
                    arg.arg
                    for arg in funcs[kernel].args.args
                    if isinstance(arg.annotation, ast.Attribute)
                    and arg.annotation.attr == "constexpr"
                }
                keyword_names = {kw.arg for kw in call.keywords} - {"num_warps", "num_stages"}
                self.assertEqual(keyword_names, constexpr)
                self.assertEqual(len(call.args) + len(constexpr), len(funcs[kernel].args.args))
        self.assertIn("quant_matmul.py", CH15.read_text(encoding="utf-8"))

    def test_silu_wrapper_preserves_front_dimensions_and_empty_shapes(self):
        path = EXAMPLES / "silu_plugin/my_plugin/kernel.py"
        text_full = path.read_text(encoding="utf-8")
        tree = ast.parse(text_full)
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "silu_and_mul_triton"
        )
        text = ast.get_source_segment(text_full, fn)
        self.assertIn("x.shape[:-1]", text)
        self.assertIn("x.numel() == 0", text)
        self.assertIn("y.reshape(output_shape)", text)
        self.assertIn("not x.is_contiguous()", text)

    def test_plugin_packaging_entry_point(self):
        import tomllib

        metadata = tomllib.loads((EXAMPLES / "silu_plugin/pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["project"]["entry-points"]["vllm.general_plugins"]["my_plugin"],
            "my_plugin:register",
        )
        self.assertEqual(metadata["project"]["dependencies"], [])
        init = ast.parse((EXAMPLES / "silu_plugin/my_plugin/__init__.py").read_text(encoding="utf-8"))
        register = next(
            node
            for node in init.body
            if isinstance(node, ast.FunctionDef) and node.name == "register"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.ImportFrom)
                and node.level == 1
                and any(alias.name == "silu_and_mul_plugin" for alias in node.names)
                for node in ast.walk(register)
            )
        )

    def test_gemm_source_has_four_kernels_and_complete_lifecycle(self):
        text = (EXAMPLES / "gemm_bench.cu").read_text(encoding="utf-8")
        for name in ("gemm_naive", "gemm_tiled", "gemm_vectorized", "gemm_tiled_2x2"):
            self.assertRegex(text, r"__global__[^;{}]*\bvoid\s+" + name + r"\s*\(")
        for api in (
            "cudaMalloc",
            "cudaMemcpyHostToDevice",
            "cudaMemcpyDeviceToHost",
            "cudaGetLastError",
            "cudaDeviceSynchronize",
            "cudaEventCreate",
            "cudaEventRecord",
            "cudaEventSynchronize",
            "cudaEventElapsedTime",
        ):
            self.assertIn(api, text)
        self.assertIn("CPU reference mismatch", text)
        self.assertIn('result.full ? "full" : "sampled"', text)
        self.assertIn("cols % 4 == 0", text)
        self.assertIn("col + 3 < cols", text)
        stripped = re.sub(
            r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
            "",
            text,
            flags=re.S,
        )
        stack = []
        pairs = {"}": "{", ")": "(", "]": "["}
        for character in stripped:
            if character in "({[":
                stack.append(character)
            elif character in pairs:
                self.assertTrue(stack)
                self.assertEqual(stack.pop(), pairs[character])
        self.assertEqual(stack, [])
        self.assertIn("gemm_bench.cu", CH12.read_text(encoding="utf-8"))

    def test_gemm_coarsened_index_model_covers_irregular_outputs(self):
        from collections import Counter

        text = (EXAMPLES / "gemm_bench.cu").read_text(encoding="utf-8")
        self.assertIn("r0 = blockIdx.y * TILE + 2 * ty", text)
        self.assertIn("c0 = blockIdx.x * TILE + 2 * tx", text)
        for M, N in ((1, 1), (31, 33), (130, 127)):
            covered = Counter()
            for by in range((M + 31) // 32):
                for bx in range((N + 31) // 32):
                    for ty in range(16):
                        for tx in range(16):
                            for i in range(2):
                                for j in range(2):
                                    row, col = by * 32 + 2 * ty + i, bx * 32 + 2 * tx + j
                                    if row < M and col < N:
                                        covered[row, col] += 1
            self.assertEqual(len(covered), M * N)
            self.assertEqual(set(covered.values()), {1})
        tile_elements = {
            (tid // 8, (tid % 8) * 4 + lane) for tid in range(256) for lane in range(4)
        }
        self.assertEqual(tile_elements, {(row, col) for row in range(32) for col in range(32)})

    def test_all_book_python_fences_parse_cleanly(self):
        count = 0
        paths = (
            sorted(CHAPTERS_DIR.glob("*.md"))
            + sorted(TOPICS_DIR.glob("*.md"))
            + sorted(TOOLKIT_DIR.glob("*.md"))
        )
        for path in paths:
            for index, block in enumerate(python_blocks(path), 1):
                with self.subTest(file=path.name, block=index):
                    compile(ast.parse(block), f"{path}:python-block-{index}", "exec")
                count += 1
        self.assertGreaterEqual(count, 20)

    def test_ascend_migration_and_cross_chip_methodology_contracts(self):
        ch22_text = CH22.read_text(encoding="utf-8")
        block = next(code for code in python_blocks(CH22) if "def npu_triton_impl(" in code)
        self.assertIn("get_device_properties", block)
        self.assertIn("num_vectorcore", block)
        self.assertIn("FlagGems", ch22_text)
        self.assertIn("FlagTree", ch22_text)

        ch23_text = CH23.read_text(encoding="utf-8")
        for term in ("sample_id", "McNemar", "非劣", "Opt-C", "回滚", "二分"):
            self.assertIn(term, ch23_text)
        self.assertAlmostEqual(2 / (2**8), 0.0078125)

    def test_no_audit_or_homework_clutter_in_book_prose(self):
        forbidden = ("待补测", "待验证", "证据等级", "本轮未实机", "在你机器上跑通", "交付物")
        paths = (
            [ROOT / "docs/index.md"]
            + sorted(CHAPTERS_DIR.glob("*.md"))
            + sorted(TOPICS_DIR.glob("*.md"))
            + sorted(TOOLKIT_DIR.glob("*.md"))
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"Forbidden phrase {phrase!r} found in {path.name}")


if __name__ == "__main__":
    unittest.main()
