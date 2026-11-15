"""CPU/stdlib contracts for the book's operator examples, not GPU validation.

The small tensor double rounds each operation to IEEE binary16/binary32. It
executes the quantization functions extracted from the actual prose so that
FP16 underflow cannot be hidden by a separately reimplemented helper.
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
M2 = ROOT / "docs/modules/M2-operators/README.md"
M3 = ROOT / "docs/modules/M3-domestic-chips/README.md"
M4 = ROOT / "docs/modules/M4-capstone/README.md"
TEMPLATE = ROOT / "docs/modules/M3-domestic-chips/cross-chip-report-template.md"
EXAMPLES = ROOT / "docs/modules/M2-operators/examples"


def python_blocks(path):
    return re.findall(r"^```python\n(.*?)^```", path.read_text(), re.M | re.S)


def functions_from_prose(names):
    nodes = []
    for block in python_blocks(M2):
        if not any(f"def {name}(" in block for name in names):
            continue
        tree = ast.parse(block)
        nodes.extend(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names)
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
        values = ([max(row) for row in rows] if dim == 1
                  else [max(row[col] for row in rows) for col in range(len(rows[0]))])
        return SmallTensor(values, self.dtype)

    def clamp_min(self, lower):
        return SmallTensor(self._map(self.values, lambda value: max(value, lower)), self.dtype)

    def clamp(self, lower, upper):
        return SmallTensor(self._map(self.values, lambda value: min(max(value, lower), upper)), self.dtype)

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
            result = [[divide(value, denominators[row if len(denominators) > 1 else 0]
                              [col if len(denominators[0]) > 1 else 0])
                       for col, value in enumerate(values)]
                      for row, values in enumerate(self.values)]
        return SmallTensor(result, self.dtype)


class SmallTorch:
    int8 = "int8"

    @staticmethod
    def round(tensor):
        return SmallTensor(SmallTensor._map(
            tensor.values, lambda value: round(value) if math.isfinite(value) else value), tensor.dtype)


class TestNumericAndPluginContracts(unittest.TestCase):
    def test_quantization_zero_and_tiny_inputs(self):
        namespace = {"torch": SmallTorch}
        exec(compile(functions_from_prose({"quant_per_token", "quant_per_channel"}),
                     str(M2), "exec"), namespace)
        cases = (
            ("quant_per_token", [[0., 0.], [1e-7, -1e-7], [1., -1.]]),
            ("quant_per_channel", [[0., 1e-7, 1.], [0., -1e-7, -1.]]),
        )
        for name, values in cases:
            with self.subTest(function=name):
                quantized, scales = namespace[name](SmallTensor(values))
                self.assertTrue(all(math.isfinite(value) and value > 0 for value in scales.values),
                                "FP16 1e-8 clamp underflows; compute scales in FP32")
                self.assertEqual(scales.dtype, "float32")
                zero_values = (quantized.values[0] if name == "quant_per_token"
                               else [row[0] for row in quantized.values])
                tiny_values = (quantized.values[1] if name == "quant_per_token"
                               else [row[1] for row in quantized.values])
                self.assertEqual(zero_values, [0, 0])
                self.assertTrue(any(value != 0 for value in tiny_values))

    def test_plugin_check_calls_registration(self):
        blocks = python_blocks(M2)
        init = next(block for block in blocks if "# my_plugin/__init__.py" in block)
        check = next(block for block in blocks if block.startswith("import my_plugin"))
        registry = {}
        package = types.ModuleType("my_plugin")
        custom = types.ModuleType("vllm.model_executor.custom_op")
        custom.op_registry_oot = registry
        activation = types.ModuleType("vllm.model_executor.layers.activation")

        class SiluAndMul:
            def __init__(self):
                raise AssertionError("Bare registry checks must not construct an engine-configured op")

        activation.SiluAndMul = SiluAndMul
        replacement = type("TritonSiluAndMul", (), {})
        plugin = types.SimpleNamespace(TritonSiluAndMul=replacement)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and "silu_and_mul_plugin" in fromlist:
                package.silu_and_mul_plugin = plugin
                registry["SiluAndMul"] = replacement
                return package
            return {"my_plugin": package, custom.__name__: custom,
                    activation.__name__: activation}[name]

        namespace = package.__dict__
        namespace["__builtins__"] = dict(vars(builtins), __import__=fake_import)
        namespace["__package__"] = "my_plugin"
        exec(init, namespace)
        exec(check, namespace)
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
                compile(ast.parse(path.read_text()), str(path), "exec")

    def test_quantization_example_matches_prose(self):
        tree = ast.parse((EXAMPLES / "quant_matmul.py").read_text())
        names = {"quant_per_token", "quant_per_channel"}
        example_functions = {node.name: node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in names}
        for prose_function in functions_from_prose(names).body:
            self.assertEqual(ast.dump(prose_function, include_attributes=False),
                             ast.dump(example_functions[prose_function.name], include_attributes=False))

    def test_int8_wrappers_pass_complete_kernel_signature(self):
        tree = ast.parse((EXAMPLES / "quant_matmul.py").read_text())
        funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for wrapper, kernel in (("matmul_int8", "matmul_int8_kernel"),
                                ("matmul_fused_dequant", "matmul_fused_dequant_kernel")):
            with self.subTest(wrapper=wrapper):
                call = next(node for node in ast.walk(funcs[wrapper])
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == kernel)
                constexpr = {arg.arg for arg in funcs[kernel].args.args
                             if isinstance(arg.annotation, ast.Attribute) and arg.annotation.attr == "constexpr"}
                keyword_names = {kw.arg for kw in call.keywords} - {"num_warps", "num_stages"}
                self.assertEqual(keyword_names, constexpr)
                self.assertEqual(len(call.args) + len(constexpr), len(funcs[kernel].args.args))
                self.assertIn("matmul_int8" if wrapper == "matmul_int8" else "matmul_fused_dequant",
                              M2.read_text())

    def test_silu_wrapper_preserves_front_dimensions_and_empty_shapes(self):
        path = EXAMPLES / "silu_plugin/my_plugin/kernel.py"
        tree = ast.parse(path.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "silu_and_mul_triton")
        text = ast.get_source_segment(path.read_text(), fn)
        self.assertIn("x.shape[:-1]", text)
        self.assertIn("x.numel() == 0", text)
        self.assertIn("y.reshape(output_shape)", text)
        self.assertIn("not x.is_contiguous()", text)

    def test_plugin_packaging_entry_point(self):
        import tomllib
        metadata = tomllib.loads((EXAMPLES / "silu_plugin/pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["entry-points"]["vllm.general_plugins"]["my_plugin"],
                         "my_plugin:register")
        self.assertEqual(metadata["project"]["dependencies"], [])
        init = ast.parse((EXAMPLES / "silu_plugin/my_plugin/__init__.py").read_text())
        register = next(node for node in init.body if isinstance(node, ast.FunctionDef)
                        and node.name == "register")
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.level == 1
                            and any(alias.name == "silu_and_mul_plugin" for alias in node.names)
                            for node in ast.walk(register)))

    def test_gemm_source_has_four_kernels_and_complete_lifecycle(self):
        text = (EXAMPLES / "gemm_bench.cu").read_text()
        for name in ("gemm_naive", "gemm_tiled", "gemm_vectorized", "gemm_tiled_2x2"):
            self.assertRegex(text, r"__global__[^;{}]*\bvoid\s+" + name + r"\s*\(")
        for api in ("cudaMalloc", "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost",
                    "cudaGetLastError", "cudaDeviceSynchronize", "cudaEventCreate",
                    "cudaEventRecord", "cudaEventSynchronize", "cudaEventElapsedTime"):
            self.assertIn(api, text)
        self.assertIn("CPU reference mismatch", text)
        self.assertIn('result.full ? "full" : "sampled"', text)
        self.assertIn("cols % 4 == 0", text)
        self.assertIn("col + 3 < cols", text)
        # Delimiter validation is static only, not an NVCC compilation claim.
        stripped = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                          "", text, flags=re.S)
        stack = []
        pairs = {"}": "{", ")": "(", "]": "["}
        for character in stripped:
            if character in "({[":
                stack.append(character)
            elif character in pairs:
                self.assertTrue(stack)
                self.assertEqual(stack.pop(), pairs[character])
        self.assertEqual(stack, [])

    def test_gemm_coarsened_index_model_covers_irregular_outputs(self):
        from collections import Counter
        text = (EXAMPLES / "gemm_bench.cu").read_text()
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
        tile_elements = {(tid // 8, (tid % 8) * 4 + lane)
                         for tid in range(256) for lane in range(4)}
        self.assertEqual(tile_elements, {(row, col) for row in range(32) for col in range(32)})

    def test_prose_python_fences_parse_without_importing_accelerators(self):
        count = 0
        for path in (M2, M3, M4):
            for index, block in enumerate(python_blocks(path), 1):
                with self.subTest(module=path.parent.name, block=index):
                    compile(ast.parse(block), f"{path}:python-block-{index}", "exec")
                count += 1
        self.assertGreater(count, 20)

    def test_chapter_ranges_and_new_titles(self):
        self.assertEqual(re.findall(r"^## 第 (\d+) 章", M2.read_text(), re.M),
                         [str(n) for n in range(19, 25)])
        self.assertEqual(re.findall(r"^## 第 (\d+) 章", M3.read_text(), re.M),
                         [str(n) for n in range(25, 31)])
        self.assertIn("GEMM 的四版实现：优化、计时与归因", M2.read_text())
        self.assertIn("迁移前检查与实验设计：从 Triton 到 Ascend", M3.read_text())

    def test_capstone_illustration_is_consistently_grade_d(self):
        text = M4.read_text()
        illustration = text.split("## 一份算例报告", 1)[1].split("## 交付结构与验收门槛", 1)[0]
        self.assertNotRegex(illustration, re.compile(r"\|\s*[ABC]\s*\|\s*$", re.M))
        self.assertNotIn("假设单次运行", illustration)
        self.assertNotIn("图编译未开，形状变化触发重编", illustration)
        self.assertGreaterEqual(illustration.count("| D |"), 7)
        self.assertIn("候选", illustration)
        self.assertIn("同一份", text)

    def test_migration_candidate_keeps_fixed_contract_entry(self):
        block = next(code for code in python_blocks(M3)
                     if "def npu_triton_impl_persistent" in code)
        tree = ast.parse(block)
        self.assertTrue(any(isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "npu_triton_impl"
                                    for target in node.targets)
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "npu_triton_impl_persistent"
                            for node in tree.body))

    def test_paired_quality_and_post_optimization_gates(self):
        for path in (M3, M4, TEMPLATE):
            text = path.read_text()
            with self.subTest(module=path.parent.name):
                for term in ("sample_id", "conditions_id", "正确性", "质量", "路径", "性能", "E5"):
                    self.assertIn(term, text)
                self.assertIn("不显著", text)
                self.assertIn("门槛", text)
                self.assertIn("原始", text)
        self.assertIn("McNemar", M3.read_text())
        self.assertAlmostEqual(2 / (2**8), 0.0078125)
        self.assertNotIn("生态最成熟", M3.read_text())
        self.assertNotIn("壁仞 MetaX", M3.read_text())
        self.assertNotIn("每一轮都有核启动与初始化开销", M3.read_text())
        self.assertNotIn("旧数字作废", M3.read_text())
        self.assertIn("待 NPU 验证", M3.read_text())


if __name__ == "__main__":
    unittest.main()
