"""Tests for the Vibec compiler."""

import tempfile
import subprocess
from pathlib import Path

import pytest

from vibec.lexer import tokenize
from vibec.parser import parse
from vibec.tokens import TokenType
from vibec.checker import check
from vibec.codegen import generate


class TestLexer:
  def test_basic_tokens(self):
    source = "fn main() -> i64:"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert types == [
      TokenType.FN,
      TokenType.IDENT,
      TokenType.LPAREN,
      TokenType.RPAREN,
      TokenType.ARROW,
      TokenType.IDENT,
      TokenType.COLON,
      TokenType.EOF,
    ]

  def test_indentation(self):
    source = """fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.INDENT in types
    assert TokenType.DEDENT in types

  def test_operators(self):
    source = "1 + 2 * 3 - 4 / 5 % 6"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.PLUS in types
    assert TokenType.STAR in types
    assert TokenType.MINUS in types
    assert TokenType.SLASH in types
    assert TokenType.PERCENT in types

  def test_comparisons(self):
    source = "a == b != c < d > e <= f >= g"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.EQ in types
    assert TokenType.NE in types
    assert TokenType.LT in types
    assert TokenType.GT in types
    assert TokenType.LE in types
    assert TokenType.GE in types

  def test_keywords(self):
    source = "fn let if else while return and or not true false"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.FN in types
    assert TokenType.LET in types
    assert TokenType.IF in types
    assert TokenType.ELSE in types
    assert TokenType.WHILE in types
    assert TokenType.RETURN in types
    assert TokenType.AND in types
    assert TokenType.OR in types
    assert TokenType.NOT in types
    assert TokenType.TRUE in types
    assert TokenType.FALSE in types

  def test_string_literal(self):
    source = '"hello world"'
    tokens = tokenize(source)
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "hello world"

  def test_string_escape_sequences(self):
    source = r'"line1\nline2\ttab\\"'
    tokens = tokenize(source)
    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "line1\nline2\ttab\\"


class TestParser:
  def test_simple_function(self):
    source = """fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "main"
    assert ast.functions[0].return_type.name == "i64"

  def test_function_with_params(self):
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    func = ast.functions[0]
    assert func.name == "add"
    assert len(func.params) == 2
    assert func.params[0].name == "a"
    assert func.params[1].name == "b"

  def test_let_statement(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt

    assert isinstance(ast.functions[0].body[0], LetStmt)

  def test_if_statement(self):
    source = """fn main() -> i64:
    if true:
        return 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import IfStmt

    assert isinstance(ast.functions[0].body[0], IfStmt)

  def test_while_statement(self):
    source = """fn main() -> i64:
    while false:
        return 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import WhileStmt

    assert isinstance(ast.functions[0].body[0], WhileStmt)

  def test_string_literal(self):
    source = """fn main() -> i64:
    print("hello")
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import CallExpr, ExprStmt, StringLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, CallExpr)
    assert isinstance(stmt.expr.args[0], StringLiteral)
    assert stmt.expr.args[0].value == "hello"

  def test_assignment(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = 2
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import AssignStmt

    assert isinstance(ast.functions[0].body[1], AssignStmt)
    assert ast.functions[0].body[1].name == "x"


class TestChecker:
  def test_type_mismatch(self):
    source = """fn main() -> i64:
    let x: bool = 42
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_undefined_variable(self):
    source = """fn main() -> i64:
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_undefined_function(self):
    source = """fn main() -> i64:
    return foo()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_wrong_return_type(self):
    source = """fn main() -> i64:
    return true
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_valid_program(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_string_type(self):
    source = """fn main() -> i64:
    let s: str = "hello"
    print(s)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_string_type_mismatch(self):
    source = """fn main() -> i64:
    let s: str = 42
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_assignment_valid(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = 2
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_assignment_type_mismatch(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = true
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_assignment_undefined_variable(self):
    source = """fn main() -> i64:
    x = 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)


class TestCodegen:
  def test_generates_assembly(self):
    source = """fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)
    asm = generate(ast)
    assert ".globl _main" in asm
    assert "_main:" in asm
    assert "ret" in asm


@pytest.mark.skipif(
  subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip() != "arm64",
  reason="ARM64 binary execution tests only run on ARM64 macOS",
)
class TestEndToEnd:
  """End-to-end tests that compile and run actual binaries."""

  def _compile_and_run(self, source: str) -> tuple[int, str]:
    """Compile source and run the binary, returning (exit_code, stdout)."""
    from vibec.compiler import Compiler

    with tempfile.TemporaryDirectory() as tmpdir:
      output_path = Path(tmpdir) / "test_binary"
      compiler = Compiler()
      result = compiler.compile_to_binary(source, output_path)
      assert result.success, f"Compilation failed: {result.error}"

      proc = subprocess.run([str(output_path)], capture_output=True, text=True)
      return proc.returncode, proc.stdout

  def test_return_value(self):
    source = """fn main() -> i64:
    return 42
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_arithmetic(self):
    source = """fn main() -> i64:
    return 2 + 3 * 4
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 14  # 2 + (3 * 4) = 14

  def test_print(self):
    source = """fn main() -> i64:
    print(42)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "42"

  def test_factorial(self):
    source = """fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

fn main() -> i64:
    print(factorial(5))
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "120"

  def test_fibonacci(self):
    source = """fn fib(n: i64) -> i64:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

fn main() -> i64:
    print(fib(10))
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "55"

  def test_print_string(self):
    source = """fn main() -> i64:
    print("Hello, World!")
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "Hello, World!"

  def test_string_variable(self):
    source = """fn main() -> i64:
    let msg: str = "Vibec"
    print(msg)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "Vibec"

  def test_string_escape_sequences(self):
    source = r"""fn main() -> i64:
    print("line1\nline2")
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "line1\nline2"

  def test_variable_reassignment(self):
    source = """fn main() -> i64:
    let x: i64 = 1
    x = 2
    x = 3
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_reassignment_with_expression(self):
    source = """fn main() -> i64:
    let x: i64 = 10
    x = x + 5
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_counter_loop(self):
    source = """fn main() -> i64:
    let count: i64 = 0
    let i: i64 = 0
    while i < 5:
        count = count + 1
        i = i + 1
    return count
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5
