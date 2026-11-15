"""Importing this package alone does not register a CustomOp replacement."""


def register() -> None:
    # vLLM calls the entry point at startup; bare-process tests call it explicitly.
    # Python's normal import cache prevents re-running the decorator on a second call.
    from . import silu_and_mul_plugin  # noqa: F401
