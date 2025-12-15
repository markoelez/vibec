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
    source = "fn let struct if else while for in range return and or not true false"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.FN in types
    assert TokenType.LET in types
    assert TokenType.STRUCT in types
    assert TokenType.IF in types
    assert TokenType.ELSE in types
    assert TokenType.WHILE in types
    assert TokenType.FOR in types
    assert TokenType.IN in types
    assert TokenType.RANGE in types
    assert TokenType.RETURN in types
    assert TokenType.AND in types
    assert TokenType.OR in types
    assert TokenType.NOT in types
    assert TokenType.TRUE in types
    assert TokenType.FALSE in types

  def test_braces(self):
    source = "{ }"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.LBRACE in types
    assert TokenType.RBRACE in types

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

  def test_comment_line(self):
    source = """# this is a comment
fn main() -> i64:
    return 42
"""
    tokens = tokenize(source)
    # Comment should be skipped, first token should be FN
    assert tokens[0].type == TokenType.FN

  def test_comment_after_code(self):
    source = """fn main() -> i64:
    return 42  # inline comment
"""
    tokens = tokenize(source)
    # Should parse correctly, comment ignored
    types = [t.type for t in tokens]
    assert TokenType.RETURN in types
    assert TokenType.INT in types

  def test_comment_only_lines(self):
    source = """fn main() -> i64:
    # comment line 1
    # comment line 2
    return 0
"""
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.RETURN in types


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

  def test_for_loop(self):
    source = """fn main() -> i64:
    for i in range(5):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ForStmt

    assert isinstance(ast.functions[0].body[0], ForStmt)
    assert ast.functions[0].body[0].var == "i"

  def test_for_loop_with_start(self):
    source = """fn main() -> i64:
    for i in range(2, 5):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ForStmt, IntLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, ForStmt)
    assert isinstance(stmt.start, IntLiteral)
    assert stmt.start.value == 2

  def test_array_type(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [1, 2, 3]
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, ArrayType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, ArrayType)
    assert stmt.type_ann.size == 3

  def test_vec_type(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, VecType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, VecType)

  def test_index_expr(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [1, 2, 3]
    return arr[0]
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import IndexExpr, ReturnStmt

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, IndexExpr)

  def test_method_call(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(1)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ExprStmt, MethodCallExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, MethodCallExpr)
    assert stmt.expr.method == "push"

  def test_struct_definition(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import StructDef

    assert len(ast.structs) == 1
    struct = ast.structs[0]
    assert isinstance(struct, StructDef)
    assert struct.name == "Point"
    assert len(struct.fields) == 2
    assert struct.fields[0].name == "x"
    assert struct.fields[1].name == "y"

  def test_struct_literal(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import LetStmt, StructLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, StructLiteral)
    assert stmt.value.name == "Point"
    assert len(stmt.value.fields) == 2

  def test_field_access(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.ast import ReturnStmt, FieldAccessExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, FieldAccessExpr)
    assert stmt.value.field == "x"


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

  def test_for_loop_valid(self):
    source = """fn main() -> i64:
    for i in range(10):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_for_loop_invalid_start(self):
    source = """fn main() -> i64:
    for i in range(true, 10):
        print(i)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_valid(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_struct_missing_field(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_unknown_field(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20, z: 30 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_field_type_mismatch(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: true, y: 20 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from vibec.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_undefined(self):
    source = """fn main() -> i64:
    let p: Unknown = Unknown { x: 10 }
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

  def test_arithmetic_subtraction(self):
    source = """fn main() -> i64:
    return 10 - 3
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_arithmetic_division(self):
    source = """fn main() -> i64:
    return 20 / 4
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5

  def test_arithmetic_modulo(self):
    source = """fn main() -> i64:
    return 17 % 5
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 2

  def test_arithmetic_precedence(self):
    source = """fn main() -> i64:
    return 2 + 3 * 4 - 8 / 2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10  # 2 + 12 - 4 = 10

  def test_arithmetic_unary_minus(self):
    source = """fn main() -> i64:
    let x: i64 = 5
    return -x + 10
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5

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

  def test_for_loop_simple(self):
    source = """fn main() -> i64:
    let sum: i64 = 0
    for i in range(5):
        sum = sum + i
    return sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10  # 0 + 1 + 2 + 3 + 4 = 10

  def test_for_loop_with_start(self):
    source = """fn main() -> i64:
    let sum: i64 = 0
    for i in range(2, 5):
        sum = sum + i
    return sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 9  # 2 + 3 + 4 = 9

  def test_for_loop_print(self):
    source = """fn main() -> i64:
    for i in range(3):
        print(i)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert stdout.strip() == "0\n1\n2"

  def test_for_loop_nested(self):
    source = """fn main() -> i64:
    let count: i64 = 0
    for i in range(3):
        for j in range(4):
            count = count + 1
    return count
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 12  # 3 * 4 = 12

  def test_array_literal_and_access(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [10, 20, 30]
    return arr[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  def test_array_assignment(self):
    source = """fn main() -> i64:
    let arr: [i64; 3] = [1, 2, 3]
    arr[0] = 100
    return arr[0]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_array_sum(self):
    source = """fn main() -> i64:
    let arr: [i64; 5] = [1, 2, 3, 4, 5]
    let sum: i64 = 0
    for i in range(5):
        sum = sum + arr[i]
    return sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_array_len(self):
    source = """fn main() -> i64:
    let arr: [i64; 7] = [0, 0, 0, 0, 0, 0, 0]
    return arr.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_vec_push_and_access(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(10)
    nums.push(20)
    nums.push(30)
    return nums[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  def test_vec_len(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(1)
    nums.push(2)
    nums.push(3)
    return nums.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_vec_pop(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(5)
    nums.push(10)
    nums.push(15)
    let last: i64 = nums.pop()
    return last
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_vec_index_assign(self):
    source = """fn main() -> i64:
    let nums: vec[i64] = []
    nums.push(1)
    nums.push(2)
    nums[0] = 100
    return nums[0]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_struct_basic(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.x + p.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_struct_field_assign(self):
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p: Point = Point { x: 5, y: 10 }
    p.x = 100
    return p.x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_struct_multiple_fields(self):
    source = """struct Rectangle:
    x: i64
    y: i64
    width: i64
    height: i64

fn main() -> i64:
    let r: Rectangle = Rectangle { x: 0, y: 0, width: 10, height: 5 }
    return r.width * r.height
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 50

  def test_struct_pass_to_function(self):
    source = """struct Point:
    x: i64
    y: i64

fn sum_coords(px: i64, py: i64) -> i64:
    return px + py

fn main() -> i64:
    let p: Point = Point { x: 15, y: 25 }
    return sum_coords(p.x, p.y)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 40

  def test_comments(self):
    source = """# This is a comment at the top
fn main() -> i64:
    # Comment inside function
    let x: i64 = 10  # Inline comment
    # Another comment
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10
