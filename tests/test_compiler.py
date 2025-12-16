"""Tests for the Oxide compiler."""

import tempfile
import subprocess
from pathlib import Path

import pytest

from oxide.lexer import tokenize
from oxide.parser import parse
from oxide.tokens import TokenType
from oxide.checker import check
from oxide.codegen import generate


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
    source = "fn let struct impl enum match self if else while for in range return and or not true false"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.FN in types
    assert TokenType.LET in types
    assert TokenType.STRUCT in types
    assert TokenType.IMPL in types
    assert TokenType.ENUM in types
    assert TokenType.MATCH in types
    assert TokenType.SELF in types
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

  def test_coloncolon(self):
    source = "Option::Some"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.IDENT in types
    assert TokenType.COLONCOLON in types

  def test_braces(self):
    source = "{ }"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.LBRACE in types
    assert TokenType.RBRACE in types

  def test_ampersand(self):
    source = "& &mut"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert TokenType.AMP in types
    assert TokenType.MUT in types

  def test_pipe(self):
    source = "|a, b|"
    tokens = tokenize(source)
    types = [t.type for t in tokens]
    assert types[0] == TokenType.PIPE
    assert types[4] == TokenType.PIPE

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
    from oxide.ast import LetStmt

    assert isinstance(ast.functions[0].body[0], LetStmt)

  def test_if_statement(self):
    source = """fn main() -> i64:
    if true:
        return 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import IfStmt

    assert isinstance(ast.functions[0].body[0], IfStmt)

  def test_while_statement(self):
    source = """fn main() -> i64:
    while false:
        return 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import WhileStmt

    assert isinstance(ast.functions[0].body[0], WhileStmt)

  def test_string_literal(self):
    source = """fn main() -> i64:
    print("hello")
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import CallExpr, ExprStmt, StringLiteral

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
    from oxide.ast import AssignStmt

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
    from oxide.ast import ForStmt

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
    from oxide.ast import ForStmt, IntLiteral

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
    from oxide.ast import LetStmt, ArrayType

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
    from oxide.ast import LetStmt, VecType

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
    from oxide.ast import IndexExpr, ReturnStmt

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
    from oxide.ast import ExprStmt, MethodCallExpr

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
    from oxide.ast import StructDef

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
    from oxide.ast import LetStmt, StructLiteral

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
    from oxide.ast import ReturnStmt, FieldAccessExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, FieldAccessExpr)

  def test_tuple_type(self):
    source = """fn main() -> i64:
    let t: (i64, bool) = (10, true)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import LetStmt, TupleType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, TupleType)
    assert len(stmt.type_ann.element_types) == 2

  def test_tuple_literal(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import LetStmt, TupleLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, TupleLiteral)
    assert len(stmt.value.elements) == 2

  def test_tuple_index(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return t.0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import ReturnStmt, TupleIndexExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ReturnStmt)
    assert isinstance(stmt.value, TupleIndexExpr)
    assert stmt.value.index == 0

  def test_enum_definition(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import EnumDef

    assert len(ast.enums) == 1
    enum = ast.enums[0]
    assert isinstance(enum, EnumDef)
    assert enum.name == "Option"
    assert len(enum.variants) == 2
    assert enum.variants[0].name == "Some"
    assert enum.variants[0].payload_type is not None
    assert enum.variants[1].name == "None"
    assert enum.variants[1].payload_type is None

  def test_enum_literal(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import LetStmt, EnumLiteral

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, EnumLiteral)
    assert stmt.value.enum_name == "Option"
    assert stmt.value.variant_name == "Some"

  def test_match_expression(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import ExprStmt, MatchExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, ExprStmt)
    assert isinstance(stmt.expr, MatchExpr)
    assert len(stmt.expr.arms) == 2

  def test_impl_block(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import ImplBlock

    assert len(ast.impls) == 1
    impl = ast.impls[0]
    assert isinstance(impl, ImplBlock)
    assert impl.struct_name == "Point"
    assert len(impl.methods) == 1
    assert impl.methods[0].name == "sum"

  def test_impl_self_parameter(self):
    source = """struct Point:
    x: i64

impl Point:
    fn get_x(self) -> i64:
        return self.x

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import Parameter, SimpleType

    method = ast.impls[0].methods[0]
    assert len(method.params) == 1
    assert isinstance(method.params[0], Parameter)
    assert method.params[0].name == "self"
    assert isinstance(method.params[0].type_ann, SimpleType)
    assert method.params[0].type_ann.name == "Self"

  def test_ref_type(self):
    source = """fn read(x: &i64) -> i64:
    return *x

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import RefType, SimpleType

    param = ast.functions[0].params[0]
    assert isinstance(param.type_ann, RefType)
    assert param.type_ann.mutable is False
    assert isinstance(param.type_ann.inner, SimpleType)
    assert param.type_ann.inner.name == "i64"

  def test_mut_ref_type(self):
    source = """fn modify(x: &mut i64) -> i64:
    *x = 42
    return *x

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import RefType, SimpleType

    param = ast.functions[0].params[0]
    assert isinstance(param.type_ann, RefType)
    assert param.type_ann.mutable is True
    assert isinstance(param.type_ann.inner, SimpleType)

  def test_ref_expr(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r: &i64 = &x
    return *r
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import LetStmt, RefExpr, VarExpr

    stmt = ast.functions[0].body[1]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, RefExpr)
    assert stmt.value.mutable is False
    assert isinstance(stmt.value.target, VarExpr)

  def test_deref_assign(self):
    source = """fn main() -> i64:
    let x: i64 = 10
    let r: &mut i64 = &mut x
    *r = 42
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import DerefAssignStmt

    stmt = ast.functions[0].body[2]
    assert isinstance(stmt, DerefAssignStmt)

  # === Closure parsing tests ===

  def test_closure_parse_basic(self):
    source = """fn main() -> i64:
    let add: Fn(i64, i64) -> i64 = |a: i64, b: i64| -> i64: a + b
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import FnType, LetStmt, ClosureExpr

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, FnType)
    assert isinstance(stmt.value, ClosureExpr)
    assert len(stmt.value.params) == 2

  def test_closure_parse_no_params(self):
    source = """fn main() -> i64:
    let f: Fn() -> i64 = || -> i64: 42
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import LetStmt, ClosureExpr

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.value, ClosureExpr)
    assert len(stmt.value.params) == 0

  def test_fn_type_parse(self):
    source = """fn main() -> i64:
    let f: Fn(i64, bool) -> i64 = |a: i64, b: bool| -> i64: a
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import FnType, LetStmt, SimpleType

    stmt = ast.functions[0].body[0]
    assert isinstance(stmt, LetStmt)
    assert isinstance(stmt.type_ann, FnType)
    assert len(stmt.type_ann.param_types) == 2
    assert isinstance(stmt.type_ann.param_types[0], SimpleType)
    assert stmt.type_ann.param_types[0].name == "i64"

  # === Keyword Arguments Tests ===

  def test_kwargs_basic(self):
    """Test basic keyword argument parsing."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(a=1, b=2)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import CallExpr, ReturnStmt

    ret_stmt = ast.functions[1].body[0]
    assert isinstance(ret_stmt, ReturnStmt)
    call = ret_stmt.value
    assert isinstance(call, CallExpr)
    assert call.name == "add"
    assert len(call.args) == 0
    assert len(call.kwargs) == 2
    assert call.kwargs[0][0] == "a"
    assert call.kwargs[1][0] == "b"

  def test_kwargs_mixed(self):
    """Test mixed positional and keyword arguments."""
    source = """fn foo(a: i64, b: i64, c: i64) -> i64:
    return a + b + c
fn main() -> i64:
    return foo(1, c=3, b=2)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import CallExpr, IntLiteral, ReturnStmt

    ret_stmt = ast.functions[1].body[0]
    assert isinstance(ret_stmt, ReturnStmt)
    call = ret_stmt.value
    assert isinstance(call, CallExpr)
    assert len(call.args) == 1
    assert isinstance(call.args[0], IntLiteral)
    assert call.args[0].value == 1
    assert len(call.kwargs) == 2
    assert call.kwargs[0][0] == "c"
    assert call.kwargs[1][0] == "b"

  def test_kwargs_with_expression(self):
    """Test keyword arguments with expressions as values."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    let x: i64 = 5
    return add(a=x * 2, b=x + 1)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import CallExpr, BinaryExpr, ReturnStmt

    ret_stmt = ast.functions[1].body[1]
    assert isinstance(ret_stmt, ReturnStmt)
    call = ret_stmt.value
    assert isinstance(call, CallExpr)
    assert len(call.kwargs) == 2
    assert isinstance(call.kwargs[0][1], BinaryExpr)
    assert isinstance(call.kwargs[1][1], BinaryExpr)

  def test_kwargs_nested_call(self):
    """Test keyword arguments with nested function calls."""
    source = """fn double(x: i64) -> i64:
    return x * 2
fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(a=double(5), b=10)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.ast import CallExpr, ReturnStmt

    ret_stmt = ast.functions[2].body[0]
    assert isinstance(ret_stmt, ReturnStmt)
    call = ret_stmt.value
    assert isinstance(call, CallExpr)
    assert len(call.kwargs) == 2
    # First kwarg value is a nested function call
    assert isinstance(call.kwargs[0][1], CallExpr)

  def test_kwargs_positional_after_keyword_error(self):
    """Test that positional argument after keyword raises ParseError."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(a=1, 2)
"""
    tokens = tokenize(source)
    from oxide.parser import ParseError

    with pytest.raises(ParseError, match="Positional argument cannot follow keyword argument"):
      parse(tokens)


class TestChecker:
  def test_type_mismatch(self):
    source = """fn main() -> i64:
    let x: bool = 42
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_undefined_variable(self):
    source = """fn main() -> i64:
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_undefined_function(self):
    source = """fn main() -> i64:
    return foo()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_wrong_return_type(self):
    source = """fn main() -> i64:
    return true
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

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
    from oxide.checker import TypeError

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
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_assignment_undefined_variable(self):
    source = """fn main() -> i64:
    x = 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

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
    from oxide.checker import TypeError

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
    from oxide.checker import TypeError

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
    from oxide.checker import TypeError

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
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_struct_undefined(self):
    source = """fn main() -> i64:
    let p: Unknown = Unknown { x: 10 }
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_tuple_valid(self):
    source = """fn main() -> i64:
    let t: (i64, bool) = (42, true)
    return t.0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_tuple_index_out_of_bounds(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return t.5
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_tuple_type_mismatch(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, true)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_valid(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_enum_unknown(self):
    source = """fn main() -> i64:
    let x: Unknown = Unknown::Variant(1)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_unknown_variant(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Unknown(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_payload_mismatch(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(true)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_missing_payload(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_enum_unexpected_payload(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::None(42)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_match_exhaustive(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_match_non_exhaustive(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Non-exhaustive match"):
      check(ast)

  def test_impl_valid(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.sum()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_impl_unknown_struct(self):
    source = """impl Unknown:
    fn foo(self) -> i64:
        return 0

fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="unknown type 'Unknown'"):
      check(ast)

  def test_impl_method_not_found(self):
    source = """struct Point:
    x: i64

fn main() -> i64:
    let p: Point = Point { x: 10 }
    return p.nonexistent()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_ref_type_valid(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r: &i64 = &x
    return *r
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_mut_ref_type_valid(self):
    source = """fn main() -> i64:
    let x: i64 = 10
    let r: &mut i64 = &mut x
    *r = 42
    return *r
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_deref_non_ref_error(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    return *x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot dereference non-reference"):
      check(ast)

  def test_deref_assign_non_mut_ref_error(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r: &i64 = &x
    *r = 10
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot assign through non-mutable reference"):
      check(ast)

  def test_ref_type_mismatch(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r: &bool = &x
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_use_after_move(self):
    source = """struct Point:
    x: i64

fn consume(p: Point) -> i64:
    return p.x

fn main() -> i64:
    let p: Point = Point { x: 42 }
    let a: i64 = consume(p)
    let b: i64 = p.x
    return a + b
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Use of moved variable"):
      check(ast)

  def test_copy_type_no_move(self):
    source = """fn takes_int(x: i64) -> i64:
    return x

fn main() -> i64:
    let x: i64 = 42
    let a: i64 = takes_int(x)
    let b: i64 = x
    return a + b
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise - i64 is Copy

  def test_ref_param_no_move(self):
    source = """struct Point:
    x: i64

fn read_point(p: &Point) -> i64:
    return 0

fn main() -> i64:
    let p: Point = Point { x: 42 }
    let a: i64 = read_point(&p)
    return p.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise - passed by reference

  def test_move_on_assignment(self):
    source = """struct Point:
    x: i64

fn main() -> i64:
    let p1: Point = Point { x: 42 }
    let p2: Point = p1
    return p1.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Use of moved variable"):
      check(ast)

  def test_double_mut_borrow_error(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r1: &mut i64 = &mut x
    let r2: &mut i64 = &mut x
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="already borrowed as mutable"):
      check(ast)

  def test_mut_and_shared_borrow_error(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r1: &i64 = &x
    let r2: &mut i64 = &mut x
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="already borrowed as immutable"):
      check(ast)

  def test_shared_after_mut_borrow_error(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r1: &mut i64 = &mut x
    let r2: &i64 = &x
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="already borrowed as mutable"):
      check(ast)

  def test_multiple_shared_borrows_ok(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r1: &i64 = &x
    let r2: &i64 = &x
    let r3: &i64 = &x
    return *r1 + *r2 + *r3
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise - multiple shared borrows are OK

  def test_mutate_while_borrowed_error(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r: &i64 = &x
    x = 100
    return *r
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="currently borrowed"):
      check(ast)

  def test_move_in_while_loop_error(self):
    source = """struct Point:
    x: i64

fn consume(p: Point) -> i64:
    return p.x

fn main() -> i64:
    let p: Point = Point { x: 42 }
    let i: i64 = 0
    while i < 3:
        let x: i64 = consume(p)
        i = i + 1
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot move .* inside a loop"):
      check(ast)

  def test_move_in_for_loop_error(self):
    source = """struct Point:
    x: i64

fn consume(p: Point) -> i64:
    return p.x

fn main() -> i64:
    let p: Point = Point { x: 42 }
    for i in range(0, 3):
        let x: i64 = consume(p)
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot move .* inside a loop"):
      check(ast)

  def test_copy_in_loop_ok(self):
    source = """fn takes_int(x: i64) -> i64:
    return x

fn main() -> i64:
    let x: i64 = 42
    let sum: i64 = 0
    for i in range(0, 3):
        sum = sum + takes_int(x)
    return sum
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise - i64 is Copy

  # === Closure type checking tests ===

  def test_closure_type_valid(self):
    source = """fn main() -> i64:
    let add: Fn(i64, i64) -> i64 = |a: i64, b: i64| -> i64: a + b
    return add(1, 2)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_closure_type_mismatch_return(self):
    source = """fn main() -> i64:
    let f: Fn(i64) -> bool = |x: i64| -> i64: x * 2
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError):
      check(ast)

  def test_closure_wrong_arg_count(self):
    source = """fn main() -> i64:
    let add: Fn(i64, i64) -> i64 = |a: i64, b: i64| -> i64: a + b
    return add(1)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="expects 2 arguments"):
      check(ast)

  def test_closure_wrong_arg_type(self):
    source = """fn main() -> i64:
    let f: Fn(i64) -> i64 = |x: i64| -> i64: x * 2
    return f(true)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="expects i64"):
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

  # === Keyword Arguments Checker Tests ===

  def test_kwargs_valid(self):
    """Test valid keyword arguments pass type checking."""
    source = """fn greet(name: i64, count: i64) -> i64:
    return name + count
fn main() -> i64:
    return greet(name=10, count=5)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_kwargs_reordered(self):
    """Test keyword arguments in different order from parameters."""
    source = """fn foo(a: i64, b: i64, c: i64) -> i64:
    return a + b + c
fn main() -> i64:
    return foo(c=3, a=1, b=2)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  def test_kwargs_unknown_name(self):
    """Test that unknown keyword argument name raises TypeError."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(x=1, b=2)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Unknown keyword argument 'x'"):
      check(ast)

  def test_kwargs_duplicate(self):
    """Test that duplicate argument raises TypeError."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(1, a=2)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Duplicate argument for parameter 'a'"):
      check(ast)

  def test_kwargs_type_mismatch(self):
    """Test that type mismatch in kwarg raises TypeError."""
    source = """fn add(a: i64, b: bool) -> i64:
    return a
fn main() -> i64:
    return add(a=1, b=42)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Argument 'b' of 'add' expects bool, got i64"):
      check(ast)

  def test_kwargs_missing_arg(self):
    """Test that missing argument raises TypeError."""
    source = """fn add(a: i64, b: i64, c: i64) -> i64:
    return a + b + c
fn main() -> i64:
    return add(a=1, c=3)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="expects 3 arguments, got 2"):
      check(ast)

  def test_kwargs_too_many_args(self):
    """Test that too many arguments raises TypeError."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(a=1, b=2, c=3)
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="expects 2 arguments, got 3"):
      check(ast)

  # === Implicit Return Checker Tests ===

  def test_implicit_return_type_mismatch(self):
    """Test that implicit return type mismatch raises TypeError."""
    source = """fn foo() -> i64:
    true
fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Implicit return type bool doesn't match function return type i64"):
      check(ast)

  def test_implicit_return_valid(self):
    """Test that valid implicit return passes type checking."""
    source = """fn foo() -> i64:
    42
fn main() -> i64:
    foo()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  # === Const declaration tests ===

  def test_const_reassign_error(self):
    """Test that reassigning a const variable raises TypeError."""
    source = """fn main() -> i64:
    const x: i64 = 10
    x = 20
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot assign to const variable 'x'"):
      check(ast)

  def test_const_index_assign_error(self):
    """Test that modifying const array via index raises TypeError."""
    source = """fn main() -> i64:
    const arr: [i64; 3] = [1, 2, 3]
    arr[0] = 10
    return arr[0]
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot assign to const variable 'arr'"):
      check(ast)

  def test_const_field_assign_error(self):
    """Test that modifying const struct field raises TypeError."""
    source = """struct Point:
    x: i64
    y: i64
fn main() -> i64:
    const p: Point = Point { x: 1, y: 2 }
    p.x = 10
    return p.x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot assign to const variable 'p'"):
      check(ast)

  def test_const_valid_usage(self):
    """Test that using const in expressions works."""
    source = """fn main() -> i64:
    const x: i64 = 10
    const y: i64 = x + 5
    return y
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise

  # === Type alias checker tests ===

  def test_type_alias_unknown_type_error(self):
    """Test that type alias to unknown type raises error."""
    source = """type MyType = Unknown
fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Unknown type 'Unknown'"):
      check(ast)

  def test_type_alias_duplicate_error(self):
    """Test that duplicate type alias raises error."""
    source = """type MyInt = i64
type MyInt = bool
fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Type alias 'MyInt' already defined"):
      check(ast)

  def test_type_alias_builtin_override_error(self):
    """Test that type alias cannot override built-in type."""
    source = """type i64 = bool
fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot create type alias for built-in type 'i64'"):
      check(ast)

  def test_type_alias_conflicts_struct_error(self):
    """Test that type alias cannot have same name as struct."""
    source = """struct Point:
    x: i64
    y: i64
type Point = (i64, i64)
fn main() -> i64:
    return 0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="'Point' already defined as a struct"):
      check(ast)

  def test_type_alias_valid(self):
    """Test that valid type alias passes type checking."""
    source = """type Int = i64
type Pair = (Int, Int)
fn main() -> Int:
    let x: Int = 10
    let p: Pair = (1, 2)
    return x
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    check(ast)  # Should not raise


@pytest.mark.skipif(
  subprocess.run(["uname", "-m"], capture_output=True, text=True).stdout.strip() != "arm64",
  reason="ARM64 binary execution tests only run on ARM64 macOS",
)
class TestEndToEnd:
  """End-to-end tests that compile and run actual binaries."""

  def _compile_and_run(self, source: str) -> tuple[int, str]:
    """Compile source and run the binary, returning (exit_code, stdout)."""
    from oxide.compiler import Compiler

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

  def test_struct_assignment(self):
    """Test struct-to-struct assignment copies all fields."""
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let p1: Point = Point { x: 3, y: 4 }
    let p2: Point = p1
    return p2.x + p2.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_struct_assignment_multiple_fields(self):
    """Test struct assignment with many fields."""
    source = """struct Rectangle:
    x: i64
    y: i64
    width: i64
    height: i64

fn main() -> i64:
    let r1: Rectangle = Rectangle { x: 10, y: 20, width: 30, height: 40 }
    let r2: Rectangle = r1
    return r2.x + r2.y + r2.width + r2.height
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_struct_assignment_preserves_original(self):
    """Test that struct assignment is a copy (before move semantics apply)."""
    source = """struct Value:
    n: i64

fn main() -> i64:
    let v1: Value = Value { n: 42 }
    let v2: Value = v1
    return v2.n
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

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

  def test_tuple_basic(self):
    source = """fn main() -> i64:
    let t: (i64, i64) = (10, 20)
    return t.0 + t.1
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_tuple_three_elements(self):
    source = """fn main() -> i64:
    let t: (i64, i64, i64) = (5, 10, 15)
    return t.0 + t.1 + t.2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_tuple_mixed_types(self):
    source = """fn main() -> i64:
    let t: (i64, bool) = (42, true)
    if t.1:
        return t.0
    return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_tuple_pass_elements(self):
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b

fn main() -> i64:
    let t: (i64, i64) = (15, 25)
    return add(t.0, t.1)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 40

  def test_enum_basic(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::Some(42)
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_enum_none_variant(self):
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let x: Option = Option::None
    match x:
        Option::Some(val):
            return val
        Option::None:
            return 99
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 99

  def test_enum_multiple_variants(self):
    source = """enum Result:
    Ok(i64)
    Err(i64)
    Unknown

fn main() -> i64:
    let r: Result = Result::Err(42)
    match r:
        Result::Ok(val):
            return val
        Result::Err(code):
            return code + 100
        Result::Unknown:
            return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 142

  def test_enum_in_function(self):
    source = """enum Option:
    Some(i64)
    None

fn unwrap_or(opt: Option, default: i64) -> i64:
    match opt:
        Option::Some(val):
            return val
        Option::None:
            return default

fn main() -> i64:
    let x: Option = Option::Some(10)
    let y: Option = Option::None
    let a: i64 = unwrap_or(x, 0)
    let b: i64 = unwrap_or(y, 99)
    return a + b
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 109  # 10 + 99

  def test_impl_basic(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    return p.sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_impl_with_args(self):
    source = """struct Counter:
    value: i64

impl Counter:
    fn add(self, n: i64) -> i64:
        return self.value + n

fn main() -> i64:
    let c: Counter = Counter { value: 100 }
    return c.add(42)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 142

  def test_impl_multiple_methods(self):
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn get_x(self) -> i64:
        return self.x
    fn get_y(self) -> i64:
        return self.y
    fn sum(self) -> i64:
        return self.x + self.y

fn main() -> i64:
    let p: Point = Point { x: 5, y: 15 }
    let a: i64 = p.get_x()
    let b: i64 = p.get_y()
    let c: i64 = p.sum()
    return a + b + c
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 40  # 5 + 15 + 20

  def test_impl_method_chain(self):
    source = """struct Value:
    n: i64

impl Value:
    fn get(self) -> i64:
        return self.n

fn main() -> i64:
    let v1: Value = Value { n: 10 }
    let v2: Value = Value { n: 20 }
    return v1.get() + v2.get()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_ref_basic(self):
    source = """fn main() -> i64:
    let x: i64 = 42
    let r: &i64 = &x
    return *r
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_mut_ref_modify(self):
    source = """fn main() -> i64:
    let x: i64 = 10
    let r: &mut i64 = &mut x
    *r = 42
    return x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_ref_pass_to_function(self):
    source = """fn read_ref(r: &i64) -> i64:
    return *r

fn main() -> i64:
    let x: i64 = 100
    return read_ref(&x)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_mut_ref_pass_to_function(self):
    source = """fn modify_ref(r: &mut i64) -> i64:
    *r = 50
    return *r

fn main() -> i64:
    let x: i64 = 10
    let result: i64 = modify_ref(&mut x)
    return x + result
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100  # 50 + 50

  # === Closure tests ===

  def test_closure_basic(self):
    """Test basic closure definition and call."""
    source = """fn main() -> i64:
    let add: Fn(i64, i64) -> i64 = |a: i64, b: i64| -> i64: a + b
    return add(3, 4)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_closure_no_params(self):
    """Test closure with no parameters."""
    source = """fn main() -> i64:
    let get_value: Fn() -> i64 = || -> i64: 42
    return get_value()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_closure_single_param(self):
    """Test closure with single parameter."""
    source = """fn main() -> i64:
    let double: Fn(i64) -> i64 = |x: i64| -> i64: x * 2
    return double(21)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_closure_complex_body(self):
    """Test closure with complex arithmetic in body."""
    source = """fn main() -> i64:
    let calc: Fn(i64, i64, i64) -> i64 = |a: i64, b: i64, c: i64| -> i64: a * b + c
    return calc(5, 8, 2)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42  # 5 * 8 + 2 = 42

  def test_closure_multiple_calls(self):
    """Test calling a closure multiple times."""
    source = """fn main() -> i64:
    let add_ten: Fn(i64) -> i64 = |x: i64| -> i64: x + 10
    let a: i64 = add_ten(5)
    let b: i64 = add_ten(20)
    return a + b
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 45  # 15 + 35

  def test_closure_bool_return(self):
    """Test closure returning bool."""
    source = """fn main() -> i64:
    let is_positive: Fn(i64) -> bool = |x: i64| -> bool: x > 0
    if is_positive(5):
        return 1
    return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  # === Keyword Arguments End-to-End Tests ===

  def test_kwargs_basic(self):
    """Test basic keyword arguments work correctly."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(a=10, b=5)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_kwargs_reordered(self):
    """Test keyword arguments in different order."""
    source = """fn sub(a: i64, b: i64) -> i64:
    return a - b
fn main() -> i64:
    return sub(b=3, a=10)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7  # 10 - 3 = 7

  def test_kwargs_mixed_positional(self):
    """Test mixing positional and keyword arguments."""
    source = """fn calc(a: i64, b: i64, c: i64) -> i64:
    return a * 100 + b * 10 + c
fn main() -> i64:
    return calc(1, c=3, b=2)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 123  # 1*100 + 2*10 + 3 = 123

  def test_kwargs_all_reordered(self):
    """Test all keyword arguments in completely different order."""
    source = """fn order(first: i64, second: i64, third: i64) -> i64:
    return first * 100 + second * 10 + third
fn main() -> i64:
    return order(third=3, first=1, second=2)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 123

  def test_kwargs_with_expressions(self):
    """Test keyword arguments with complex expressions."""
    source = """fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    let x: i64 = 5
    return add(a=x * 2, b=x + 3)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 18  # (5*2) + (5+3) = 10 + 8 = 18

  def test_kwargs_nested_calls(self):
    """Test keyword arguments with nested function calls."""
    source = """fn double(x: i64) -> i64:
    return x * 2
fn add(a: i64, b: i64) -> i64:
    return a + b
fn main() -> i64:
    return add(a=double(5), b=double(3))
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 16  # 10 + 6 = 16

  def test_kwargs_print(self):
    """Test keyword arguments with print builtin."""
    source = """fn main() -> i64:
    print(value=42)
    return 0
"""
    exit_code, stdout = self._compile_and_run(source)
    assert exit_code == 0
    assert "42" in stdout

  def test_kwargs_single_arg(self):
    """Test keyword argument with single parameter function."""
    source = """fn square(n: i64) -> i64:
    return n * n
fn main() -> i64:
    return square(n=7)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 49

  def test_kwargs_many_args(self):
    """Test keyword arguments with many parameters."""
    source = """fn sum5(a: i64, b: i64, c: i64, d: i64, e: i64) -> i64:
    return a + b + c + d + e
fn main() -> i64:
    return sum5(e=5, c=3, a=1, d=4, b=2)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15  # 1+2+3+4+5 = 15

  # === Implicit Return Tests ===

  def test_implicit_return_simple(self):
    """Test implicit return with a simple literal."""
    source = """fn answer() -> i64:
    42

fn main() -> i64:
    answer()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_implicit_return_expression(self):
    """Test implicit return with an expression."""
    source = """fn compute(x: i64) -> i64:
    x * 2 + 1

fn main() -> i64:
    compute(20)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 41  # 20*2+1 = 41

  def test_implicit_return_variable(self):
    """Test implicit return with a variable."""
    source = """fn get_val() -> i64:
    let result: i64 = 99
    result

fn main() -> i64:
    get_val()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 99

  def test_implicit_return_function_call(self):
    """Test implicit return with a function call."""
    source = """fn double(n: i64) -> i64:
    n * 2

fn triple(n: i64) -> i64:
    double(n) + n

fn main() -> i64:
    triple(10)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30  # 10*2 + 10 = 30

  def test_implicit_return_with_statements(self):
    """Test implicit return after other statements."""
    source = """fn process(x: i64) -> i64:
    let a: i64 = x + 1
    let b: i64 = a * 2
    a + b

fn main() -> i64:
    process(5)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 18  # a=6, b=12, 6+12=18

  def test_implicit_return_main(self):
    """Test implicit return in main function."""
    source = """fn main() -> i64:
    let x: i64 = 7
    x * 8
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 56

  def test_implicit_return_bool(self):
    """Test implicit return with boolean expression."""
    source = """fn is_positive(n: i64) -> bool:
    n > 0

fn main() -> i64:
    if is_positive(5):
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_explicit_return_still_works(self):
    """Ensure explicit return still works."""
    source = """fn explicit() -> i64:
    return 123

fn main() -> i64:
    explicit()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 123

  def test_implicit_return_struct_field(self):
    """Test implicit return with struct field access."""
    source = """struct Point:
    x: i64
    y: i64

fn main() -> i64:
    let pt: Point = Point { x: 42, y: 10 }
    pt.x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  # === Vec Iterator Methods Tests ===

  def test_vec_skip(self):
    """Test vec.skip(n) method."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.push(4)
    v.push(5)
    let v2: vec[i64] = v.skip(2)
    v2[0] + v2[1] + v2[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 12  # 3 + 4 + 5

  def test_vec_take(self):
    """Test vec.take(n) method."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.push(4)
    v.push(5)
    let v2: vec[i64] = v.take(3)
    v2[0] + v2[1] + v2[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6  # 1 + 2 + 3

  def test_vec_skip_take_chain(self):
    """Test chaining skip and take."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    for i in range(0, 10):
        v.push(i)
    let v2: vec[i64] = v.skip(3).take(4)
    v2[0] + v2[1] + v2[2] + v2[3]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 18  # 3 + 4 + 5 + 6

  def test_vec_into_iter_collect(self):
    """Test into_iter and collect as no-ops."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    let v2: vec[i64] = v.into_iter().collect()
    v2[0] + v2[1] + v2[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6

  def test_vec_map(self):
    """Test vec.map() with closure."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    let doubled: vec[i64] = v.map(|x: i64| -> i64: x * 2)
    doubled[0] + doubled[1] + doubled[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 12  # 2 + 4 + 6

  def test_vec_filter(self):
    """Test vec.filter() with closure."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.push(4)
    v.push(5)
    v.push(6)
    let evens: vec[i64] = v.filter(|x: i64| -> bool: x % 2 == 0)
    evens[0] + evens[1] + evens[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 12  # 2 + 4 + 6

  def test_vec_sum(self):
    """Test vec.sum() method."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.push(4)
    v.push(5)
    v.sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_vec_fold(self):
    """Test vec.fold() with closure."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.push(4)
    v.fold(0, |acc: i64, x: i64| -> i64: acc + x)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10

  def test_vec_map_filter_chain(self):
    """Test chaining map and filter."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.push(4)
    v.push(5)
    let result: vec[i64] = v.map(|x: i64| -> i64: x * 2).filter(|x: i64| -> bool: x > 5)
    result.sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 24  # 6 + 8 + 10

  def test_vec_full_chain(self):
    """Test Rust-style iterator chain: skip, take, map, filter, sum."""
    source = """fn main() -> i64:
    let v: vec[i64] = []
    for i in range(0, 10):
        v.push(i)
    v.into_iter().skip(2).take(5).map(|x: i64| -> i64: x * 10).filter(|x: i64| -> bool: x > 30).sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 150  # 40 + 50 + 60

  # === Result type tests ===

  def test_result_ok_basic(self):
    """Test creating and returning Ok value."""
    source = """fn get_value() -> Result[i64, i64]:
    Ok(42)

fn main() -> i64:
    let res: Result[i64, i64] = get_value()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_result_err_basic(self):
    """Test creating and returning Err value."""
    source = """fn get_error() -> Result[i64, i64]:
    Err(99)

fn main() -> i64:
    let res: Result[i64, i64] = get_error()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 99

  def test_result_try_ok(self):
    """Test ? operator unwrapping Ok value."""
    source = """fn get_value() -> Result[i64, i64]:
    Ok(42)

fn process() -> Result[i64, i64]:
    let val: i64 = get_value()?
    Ok(val + 8)

fn main() -> i64:
    let res: Result[i64, i64] = process()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 50  # 42 + 8

  def test_result_try_err(self):
    """Test ? operator propagating Err value."""
    source = """fn get_error() -> Result[i64, i64]:
    Err(77)

fn process() -> Result[i64, i64]:
    let val: i64 = get_error()?
    Ok(val + 100)

fn main() -> i64:
    let res: Result[i64, i64] = process()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 77  # Error propagated

  def test_result_try_chain(self):
    """Test chaining multiple ? operators."""
    source = """fn step1() -> Result[i64, i64]:
    Ok(10)

fn step2(x: i64) -> Result[i64, i64]:
    Ok(x * 2)

fn step3(x: i64) -> Result[i64, i64]:
    Ok(x + 5)

fn process() -> Result[i64, i64]:
    let a: i64 = step1()?
    let b: i64 = step2(a)?
    let c: i64 = step3(b)?
    Ok(c)

fn main() -> i64:
    let res: Result[i64, i64] = process()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 25  # ((10 * 2) + 5) = 25

  def test_result_try_chain_error(self):
    """Test error short-circuits the chain."""
    source = """fn step1() -> Result[i64, i64]:
    Ok(10)

fn step2(x: i64) -> Result[i64, i64]:
    Err(55)

fn step3(x: i64) -> Result[i64, i64]:
    Ok(x + 100)

fn process() -> Result[i64, i64]:
    let a: i64 = step1()?
    let b: i64 = step2(a)?
    let c: i64 = step3(b)?
    Ok(c)

fn main() -> i64:
    let res: Result[i64, i64] = process()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 55  # Error from step2

  def test_result_conditional(self):
    """Test returning Ok or Err conditionally."""
    source = """fn divide(a: i64, b: i64) -> Result[i64, i64]:
    if b == 0:
        return Err(1)
    Ok(a / b)

fn main() -> i64:
    let res1: Result[i64, i64] = divide(10, 2)
    let res2: Result[i64, i64] = divide(10, 0)
    let v1: i64 = 0
    match res1:
        Ok(v):
            v1 = v
        Err(e):
            v1 = 0
    let v2: i64 = 0
    match res2:
        Ok(v):
            v2 = v
        Err(e):
            v2 = e
    return v1 + v2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6  # 5 + 1

  def test_result_match_binding(self):
    """Test match arm bindings with Result."""
    source = """fn compute() -> Result[i64, i64]:
    Ok(30)

fn main() -> i64:
    let res: Result[i64, i64] = compute()
    match res:
        Ok(value):
            return value + 12
        Err(error):
            return error * 2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_result_match_err_binding(self):
    """Test match arm bindings with Err Result."""
    source = """fn compute() -> Result[i64, i64]:
    Err(21)

fn main() -> i64:
    let res: Result[i64, i64] = compute()
    match res:
        Ok(value):
            return value + 100
        Err(error):
            return error * 2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_result_explicit_return(self):
    """Test explicit return of Result from function."""
    source = """fn maybe_add(a: i64, b: i64) -> Result[i64, i64]:
    if a < 0:
        return Err(a)
    if b < 0:
        return Err(b)
    return Ok(a + b)

fn main() -> i64:
    let res: Result[i64, i64] = maybe_add(10, 20)
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_result_implicit_return_ok(self):
    """Test implicit return of Ok from function."""
    source = """fn get_value() -> Result[i64, i64]:
    Ok(88)

fn main() -> i64:
    let res: Result[i64, i64] = get_value()
    match res:
        Ok(v):
            return v
        Err(e):
            return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 88

  def test_result_implicit_return_err(self):
    """Test implicit return of Err from function."""
    source = """fn get_error() -> Result[i64, i64]:
    Err(66)

fn main() -> i64:
    let res: Result[i64, i64] = get_error()
    match res:
        Ok(v):
            return 0
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 66

  def test_result_in_loop(self):
    """Test Result with loop."""
    source = """fn check(x: i64) -> Result[i64, i64]:
    if x > 5:
        return Err(x)
    Ok(x)

fn process() -> Result[i64, i64]:
    let sum: i64 = 0
    for i in range(0, 10):
        let r: Result[i64, i64] = check(i)
        match r:
            Ok(v):
                sum = sum + v
            Err(e):
                return Err(e)
    Ok(sum)

fn main() -> i64:
    let res: Result[i64, i64] = process()
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6  # Error at i=6

  def test_result_try_in_loop(self):
    """Test ? operator in loop."""
    source = """fn validate(x: i64) -> Result[i64, i64]:
    if x < 0:
        return Err(x)
    Ok(x * 2)

fn sum_doubled(n: i64) -> Result[i64, i64]:
    let sum: i64 = 0
    for i in range(0, n):
        let doubled: i64 = validate(i)?
        sum = sum + doubled
    Ok(sum)

fn main() -> i64:
    let res: Result[i64, i64] = sum_doubled(5)
    match res:
        Ok(v):
            return v
        Err(e):
            return e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20  # 0 + 2 + 4 + 6 + 8 = 20

  # === List comprehension tests ===

  def test_list_comprehension_basic(self):
    """Test basic list comprehension."""
    source = """fn main() -> i64:
    let squares: vec[i64] = [x * x for x in range(0, 5)]
    squares[0] + squares[1] + squares[2] + squares[3] + squares[4]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30  # 0 + 1 + 4 + 9 + 16 = 30

  def test_list_comprehension_with_condition(self):
    """Test list comprehension with if filter."""
    source = """fn main() -> i64:
    let evens: vec[i64] = [x for x in range(0, 10) if x % 2 == 0]
    evens[0] + evens[1] + evens[2] + evens[3] + evens[4]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20  # 0 + 2 + 4 + 6 + 8 = 20

  def test_list_comprehension_expression(self):
    """Test list comprehension with complex expression."""
    source = """fn main() -> i64:
    let doubled: vec[i64] = [x * 2 + 1 for x in range(0, 5)]
    doubled[0] + doubled[1] + doubled[2] + doubled[3] + doubled[4]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 25  # 1 + 3 + 5 + 7 + 9 = 25

  def test_list_comprehension_len(self):
    """Test length of list from comprehension."""
    source = """fn main() -> i64:
    let nums: vec[i64] = [x for x in range(0, 7)]
    nums.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7

  def test_list_comprehension_filtered_len(self):
    """Test length of filtered list comprehension."""
    source = """fn main() -> i64:
    let odds: vec[i64] = [x for x in range(0, 10) if x % 2 == 1]
    odds.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5  # 1, 3, 5, 7, 9

  def test_list_comprehension_sum(self):
    """Test sum of list comprehension."""
    source = """fn main() -> i64:
    let nums: vec[i64] = [x * x for x in range(1, 6)]
    nums.sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 55  # 1 + 4 + 9 + 16 + 25 = 55

  def test_list_comprehension_empty_result(self):
    """Test list comprehension that produces empty result."""
    source = """fn main() -> i64:
    let empty: vec[i64] = [x for x in range(0, 10) if x > 100]
    empty.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0

  def test_list_comprehension_growing(self):
    """Test list comprehension that needs to grow the vec."""
    source = """fn main() -> i64:
    let nums: vec[i64] = [x for x in range(0, 20)]
    nums.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  def test_list_comprehension_with_method_chain(self):
    """Test list comprehension result with method chaining."""
    source = """fn main() -> i64:
    let nums: vec[i64] = [x * 2 for x in range(0, 5)]
    nums.map(|x: i64| -> i64: x + 1).sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 25  # (0+1) + (2+1) + (4+1) + (6+1) + (8+1) = 25

  # === Dict/HashMap tests ===

  def test_dict_empty(self):
    """Test empty dict creation."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {}
    d.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0

  def test_dict_literal(self):
    """Test dict literal creation."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20, 3: 30}
    d.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_dict_get(self):
    """Test dict get operation."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20, 3: 30}
    d[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  def test_dict_get_multiple(self):
    """Test getting multiple values from dict."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20, 3: 30}
    d[1] + d[2] + d[3]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 60

  def test_dict_set(self):
    """Test dict set operation."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {}
    d[5] = 50
    d[5]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 50

  def test_dict_update(self):
    """Test updating existing key in dict."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10}
    d[1] = 100
    d[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_dict_contains_true(self):
    """Test dict contains returns true for existing key."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20}
    if d.contains(1):
        return 1
    return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_dict_contains_false(self):
    """Test dict contains returns false for missing key."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20}
    if d.contains(99):
        return 1
    return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0

  def test_dict_insert(self):
    """Test dict insert method."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {}
    d.insert(7, 70)
    d.insert(8, 80)
    d[7] + d[8]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 150

  def test_dict_remove(self):
    """Test dict remove method."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20}
    let removed: bool = d.remove(1)
    if removed:
        return d.len()
    return 99
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_dict_remove_nonexistent(self):
    """Test dict remove returns false for nonexistent key."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10}
    let removed: bool = d.remove(99)
    if removed:
        return 1
    return 0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0

  def test_dict_get_method(self):
    """Test dict get method."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {1: 10, 2: 20}
    d.get(1) + d.get(2)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_dict_many_entries(self):
    """Test dict with many entries to test hash collision handling."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {}
    d[0] = 0
    d[1] = 1
    d[2] = 2
    d[3] = 3
    d[4] = 4
    d[5] = 5
    d[6] = 6
    d[7] = 7
    d[8] = 8
    d[9] = 9
    d[0] + d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] + d[8] + d[9]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 45  # 0 + 1 + ... + 9 = 45

  # === Dict comprehension tests ===

  def test_dict_comprehension_basic(self):
    """Test basic dict comprehension."""
    source = """fn main() -> i64:
    let squares: dict[i64, i64] = {x: x * x for x in range(0, 5)}
    squares[0] + squares[1] + squares[2] + squares[3] + squares[4]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30  # 0 + 1 + 4 + 9 + 16 = 30

  def test_dict_comprehension_with_condition(self):
    """Test dict comprehension with if filter."""
    source = """fn main() -> i64:
    let evens: dict[i64, i64] = {x: x * 2 for x in range(0, 10) if x % 2 == 0}
    evens.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 5  # 0, 2, 4, 6, 8

  def test_dict_comprehension_expression(self):
    """Test dict comprehension with complex expressions."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {x + 10: x * 3 for x in range(0, 3)}
    d[10] + d[11] + d[12]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 9  # 0 + 3 + 6 = 9

  def test_dict_comprehension_len(self):
    """Test length of dict from comprehension."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {x: x for x in range(0, 10)}
    d.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10

  def test_dict_comprehension_lookup(self):
    """Test looking up values in dict comprehension result."""
    source = """fn main() -> i64:
    let cubes: dict[i64, i64] = {x: x * x * x for x in range(1, 6)}
    cubes[3]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 27  # 3^3 = 27

  def test_dict_comprehension_growing(self):
    """Test dict comprehension that needs to grow."""
    source = """fn main() -> i64:
    let d: dict[i64, i64] = {x: x for x in range(0, 30)}
    d.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  # === Const declarations tests ===

  def test_const_basic(self):
    """Test basic const declaration."""
    source = """fn main() -> i64:
    const x: i64 = 42
    x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_const_expression(self):
    """Test const with expression value."""
    source = """fn main() -> i64:
    const a: i64 = 5
    const b: i64 = a * 2
    b
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10

  def test_const_bool(self):
    """Test const with bool type."""
    source = """fn main() -> i64:
    const flag: bool = true
    if flag:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_const_in_expression(self):
    """Test using const in expressions."""
    source = """fn main() -> i64:
    const x: i64 = 10
    const y: i64 = 20
    x + y + 5
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 35

  def test_let_mutable(self):
    """Test that let variables can be reassigned."""
    source = """fn main() -> i64:
    let x: i64 = 10
    x = 20
    x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 20

  # === Type alias tests ===

  def test_type_alias_simple(self):
    """Test basic type alias for i64."""
    source = """type Int = i64
fn main() -> i64:
    let x: Int = 42
    x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_type_alias_vector(self):
    """Test type alias for vec type."""
    source = """type IntVec = vec[i64]
fn main() -> i64:
    let v: IntVec = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_type_alias_array(self):
    """Test type alias for array type."""
    source = """type Point3D = [i64; 3]
fn main() -> i64:
    let p: Point3D = [1, 2, 3]
    p[0] + p[1] + p[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6

  def test_type_alias_tuple(self):
    """Test type alias for tuple type."""
    source = """type Pair = (i64, i64)
fn main() -> i64:
    let p: Pair = (10, 20)
    p.0 + p.1
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_type_alias_dict(self):
    """Test type alias for dict type."""
    source = """type IntMap = dict[i64, i64]
fn main() -> i64:
    let m: IntMap = {}
    m[1] = 50
    m[2] = 75
    m[1] + m[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 125

  def test_type_alias_result(self):
    """Test type alias for Result type."""
    source = """type MyResult = Result[i64, i64]
fn maybe_fail(x: i64) -> MyResult:
    if x > 0:
        return Ok(x * 2)
    Err(1)
fn main() -> i64:
    let r: MyResult = maybe_fail(5)
    match r:
        Ok(v):
            v
        Err(e):
            e
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10

  def test_type_alias_in_function_params(self):
    """Test type alias in function parameters."""
    source = """type Number = i64
fn add(a: Number, b: Number) -> Number:
    a + b
fn main() -> i64:
    add(10, 20)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_type_alias_chained(self):
    """Test type aliases that reference other aliases."""
    source = """type Int = i64
type MyInt = Int
fn main() -> i64:
    let x: MyInt = 42
    x
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_type_alias_nested(self):
    """Test type alias with nested generic types."""
    source = """type IntVec = vec[i64]
fn main() -> i64:
    let v: IntVec = []
    v.push(1)
    v.push(2)
    v.push(3)
    v.sum()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6

  def test_type_alias_with_struct(self):
    """Test type alias that references a struct."""
    source = """struct Point:
    x: i64
    y: i64
type P = Point
fn main() -> i64:
    let p: P = Point { x: 10, y: 20 }
    p.x + p.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  # === Generic Struct Tests ===

  def test_generic_struct_basic(self):
    """Test basic generic struct with one type parameter."""
    source = """struct Box<T>:
    value: T

fn main() -> i64:
    let b: Box<i64> = Box<i64> { value: 42 }
    b.value
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_struct_two_params(self):
    """Test generic struct with two type parameters."""
    source = """struct Pair<T, U>:
    first: T
    second: U

fn main() -> i64:
    let p: Pair<i64, i64> = Pair<i64, i64> { first: 10, second: 20 }
    p.first + p.second
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_generic_struct_multiple_instantiations(self):
    """Test same generic struct with different type arguments."""
    source = """struct Box<T>:
    value: T

fn main() -> i64:
    let b1: Box<i64> = Box<i64> { value: 10 }
    let b2: Box<bool> = Box<bool> { value: true }
    if b2.value:
        b1.value + 5
    else:
        b1.value
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_generic_struct_nested(self):
    """Test nested generic struct instantiation."""
    source = """struct Box<T>:
    value: T

fn main() -> i64:
    let inner: Box<i64> = Box<i64> { value: 42 }
    let outer: Box<Box<i64>> = Box<Box<i64>> { value: inner }
    outer.value.value
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  # === Generic Enum Tests ===

  def test_generic_enum_basic(self):
    """Test basic generic enum with one type parameter."""
    source = """enum Option<T>:
    Some(T)
    None

fn main() -> i64:
    let opt: Option<i64> = Option<i64>::Some(42)
    match opt:
        Option<i64>::Some(val):
            val
        Option<i64>::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_enum_none_variant(self):
    """Test generic enum with unit variant."""
    source = """enum Option<T>:
    Some(T)
    None

fn main() -> i64:
    let opt: Option<i64> = Option<i64>::None
    match opt:
        Option<i64>::Some(val):
            val
        Option<i64>::None:
            99
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 99

  # === Generic Error Tests ===

  def test_generic_struct_missing_type_args_error(self):
    """Test error when generic struct is used without type args."""
    source = """struct Box<T>:
    value: T

fn main() -> i64:
    let b: Box = Box { value: 42 }
    0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="requires type arguments"):
      check(ast)

  def test_generic_struct_wrong_type_arg_count_error(self):
    """Test error when wrong number of type args provided."""
    source = """struct Pair<T, U>:
    first: T
    second: U

fn main() -> i64:
    let p: Pair<i64> = Pair<i64> { first: 10, second: 20 }
    0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="expects 2 type args, got 1"):
      check(ast)

  def test_generic_enum_missing_type_args_error(self):
    """Test error when generic enum is used without type args."""
    source = """enum Option<T>:
    Some(T)
    None

fn main() -> i64:
    let opt: Option = Option::Some(42)
    0
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="requires type arguments"):
      check(ast)

  # === Generic Function Tests ===

  def test_generic_function_basic(self):
    """Test basic generic function with explicit type args."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    identity<i64>(42)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_function_multiple_params(self):
    """Test generic function with multiple type parameters."""
    source = """fn first<T, U>(a: T, b: U) -> T:
    a

fn main() -> i64:
    first<i64, bool>(100, true)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_generic_function_multiple_instantiations(self):
    """Test same generic function with different type arguments."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    let a: i64 = identity<i64>(10)
    let b: bool = identity<bool>(true)
    if b:
        a + 5
    else:
        a
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  def test_generic_function_with_generic_struct(self):
    """Test generic function that takes a generic struct."""
    source = """struct Box<T>:
    value: T

fn unbox<T>(b: Box<T>) -> T:
    b.value

fn main() -> i64:
    # Create inline to avoid move issue
    unbox<i64>(Box<i64> { value: 42 })
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_function_cannot_infer_type_args_error(self):
    """Test error when generic function type cannot be inferred from arguments."""
    source = """fn make_default<T>() -> T:
    # Cannot infer T because there are no arguments using T
    let x: T = 0
    x

fn main() -> i64:
    make_default()
"""
    tokens = tokenize(source)
    ast = parse(tokens)
    from oxide.checker import TypeError

    with pytest.raises(TypeError, match="Cannot infer type"):
      check(ast)

  # === Generic Function Type Inference Tests ===

  def test_generic_function_infer_basic(self):
    """Test basic type inference for generic function."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    identity(42)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_function_infer_bool(self):
    """Test type inference with bool type."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    let b: bool = identity(true)
    if b:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_generic_function_infer_multiple_params(self):
    """Test type inference for generic function with multiple type parameters."""
    source = """fn first<T, U>(a: T, b: U) -> T:
    a

fn main() -> i64:
    first(100, true)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100

  def test_generic_function_infer_multiple_calls(self):
    """Test multiple calls to same generic function with different inferred types."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    let a: i64 = identity(10)
    let b: bool = identity(true)
    if b:
        a
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10

  def test_generic_function_infer_with_computation(self):
    """Test type inference with expression arguments."""
    source = """fn double<T>(x: T) -> T:
    x

fn main() -> i64:
    double(21 + 21)
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_function_infer_nested_call(self):
    """Test type inference with nested generic function calls."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    identity(identity(42))
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_function_infer_mixed_with_explicit(self):
    """Test mixing inferred and explicit type args across multiple calls."""
    source = """fn identity<T>(x: T) -> T:
    x

fn main() -> i64:
    let a: i64 = identity<i64>(10)
    let b: i64 = identity(20)
    a + b
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  # === Generic Impl Block Tests ===

  def test_generic_impl_basic(self):
    """Test basic generic impl block with method."""
    source = """struct Box<T>:
    value: T

impl Box<T>:
    fn get(self: Box<T>) -> T:
        self.value

fn main() -> i64:
    let b: Box<i64> = Box<i64> { value: 42 }
    b.get()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42

  def test_generic_impl_multiple_methods(self):
    """Test generic impl block with multiple methods."""
    source = """struct Pair<T, U>:
    first: T
    second: U

impl Pair<T, U>:
    fn get_first(self: Pair<T, U>) -> T:
        self.first

    fn get_second(self: Pair<T, U>) -> U:
        self.second

fn main() -> i64:
    let p: Pair<i64, i64> = Pair<i64, i64> { first: 10, second: 20 }
    p.get_first() + p.get_second()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30

  def test_generic_impl_multiple_instantiations(self):
    """Test same generic impl with different type arguments."""
    source = """struct Box<T>:
    value: T

impl Box<T>:
    fn get(self: Box<T>) -> T:
        self.value

fn main() -> i64:
    let b1: Box<i64> = Box<i64> { value: 10 }
    let b2: Box<bool> = Box<bool> { value: true }
    if b2.get():
        b1.get() + 5
    else:
        b1.get()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 15

  # === Slice tests ===

  def test_vec_slice_basic(self):
    """Test basic vec slicing [start:stop]."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x + 1 for x in range(0, 5)]
    let s: vec[i64] = v[1:4]
    s[0] + s[1] + s[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 9  # 2 + 3 + 4

  def test_vec_slice_from_start(self):
    """Test vec slice from start [:stop]."""
    source = """fn main() -> i64:
    let v: vec[i64] = [(x + 1) * 10 for x in range(0, 4)]
    let s: vec[i64] = v[:2]
    s[0] + s[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30  # 10 + 20

  def test_vec_slice_to_end(self):
    """Test vec slice to end [start:]."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x + 1 for x in range(0, 4)]
    let s: vec[i64] = v[2:]
    s[0] + s[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 7  # 3 + 4

  def test_vec_slice_full(self):
    """Test full vec slice [:]."""
    source = """fn main() -> i64:
    let v: vec[i64] = [(x + 1) * 5 for x in range(0, 3)]
    let s: vec[i64] = v[:]
    s.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3

  def test_vec_slice_with_step(self):
    """Test vec slice with step [::step]."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x + 1 for x in range(0, 6)]
    let s: vec[i64] = v[::2]
    s[0] + s[1] + s[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 9  # 1 + 3 + 5

  def test_vec_slice_start_stop_step(self):
    """Test vec slice with start, stop, and step [start:stop:step]."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x for x in range(0, 10)]
    let s: vec[i64] = v[1:8:2]
    s[0] + s[1] + s[2] + s[3]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 16  # 1 + 3 + 5 + 7

  def test_vec_slice_empty(self):
    """Test empty slice when start >= stop."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x + 1 for x in range(0, 3)]
    let s: vec[i64] = v[2:1]
    s.len()
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0

  def test_array_slice_basic(self):
    """Test slicing a fixed-size array."""
    source = """fn main() -> i64:
    let arr: [i64; 5] = [10, 20, 30, 40, 50]
    let s: vec[i64] = arr[1:4]
    s[0] + s[1] + s[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 90  # 20 + 30 + 40

  def test_slice_negative_start(self):
    """Test slice with negative start index."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x + 1 for x in range(0, 5)]
    let s: vec[i64] = v[-2:]
    s[0] + s[1]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 9  # 4 + 5

  def test_slice_negative_stop(self):
    """Test slice with negative stop index."""
    source = """fn main() -> i64:
    let v: vec[i64] = [x + 1 for x in range(0, 5)]
    let s: vec[i64] = v[:-2]
    s[0] + s[1] + s[2]
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 6  # 1 + 2 + 3

  # === Chained Comparison Tests ===

  def test_chained_comparison_basic(self):
    """Test basic chained comparison: 0 < x < 10."""
    source = """fn main() -> i64:
    let x: i64 = 5
    if 0 < x < 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_chained_comparison_false_left(self):
    """Test chained comparison where left comparison is false."""
    source = """fn main() -> i64:
    let x: i64 = 0
    if 0 < x < 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0  # 0 < 0 is false

  def test_chained_comparison_false_right(self):
    """Test chained comparison where right comparison is false."""
    source = """fn main() -> i64:
    let x: i64 = 15
    if 0 < x < 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0  # 15 < 10 is false

  def test_chained_comparison_three_parts(self):
    """Test three-part chained comparison: a < b < c < d."""
    source = """fn main() -> i64:
    let a: i64 = 1
    let b: i64 = 2
    let c: i64 = 3
    let d: i64 = 4
    if a < b < c < d:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_chained_comparison_three_parts_false(self):
    """Test three-part chained comparison that fails."""
    source = """fn main() -> i64:
    let a: i64 = 1
    let b: i64 = 5
    let c: i64 = 3
    let d: i64 = 4
    if a < b < c < d:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 0  # 5 < 3 is false

  def test_chained_comparison_le_ge(self):
    """Test chained comparison with <= and >=."""
    source = """fn main() -> i64:
    let x: i64 = 5
    if 0 <= x <= 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_chained_comparison_le_boundary(self):
    """Test chained comparison at boundary with <=."""
    source = """fn main() -> i64:
    let x: i64 = 0
    if 0 <= x <= 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 0 <= 0 is true

  def test_chained_comparison_mixed_ops(self):
    """Test chained comparison with mixed operators."""
    source = """fn main() -> i64:
    let x: i64 = 5
    if 0 < x <= 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_chained_comparison_with_expressions(self):
    """Test chained comparison with arithmetic expressions."""
    source = """fn main() -> i64:
    let x: i64 = 5
    if 0 < x + 1 < 10:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 0 < 6 < 10

  def test_chained_comparison_equality(self):
    """Test chained comparison with equality."""
    source = """fn main() -> i64:
    let x: i64 = 5
    let y: i64 = 5
    if x == y == 5:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1

  def test_chained_comparison_in_loop(self):
    """Test chained comparison in a loop condition."""
    source = """fn main() -> i64:
    let sum: i64 = 0
    for i in range(0, 20):
        if 5 <= i < 15:
            sum = sum + 1
    sum
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 10  # i from 5 to 14 inclusive

  # === Pattern Guard Tests ===

  def test_pattern_guard_basic(self):
    """Test basic pattern guard on Option enum."""
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let opt: Option = Option::Some(5)
    match opt:
        Option::Some(x) if x > 3:
            1
        Option::Some(x):
            0
        Option::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 5 > 3 is true

  def test_pattern_guard_false(self):
    """Test pattern guard that evaluates to false."""
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let opt: Option = Option::Some(2)
    match opt:
        Option::Some(x) if x > 3:
            1
        Option::Some(x):
            x
        Option::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 2  # 2 > 3 is false, falls through to second arm

  def test_pattern_guard_result(self):
    """Test pattern guard with Result type."""
    source = """fn main() -> i64:
    let r: Result[i64, i64] = Ok(10)
    match r:
        Ok(val) if val > 5:
            1
        Ok(val):
            0
        Err(e):
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 10 > 5 is true

  def test_pattern_guard_result_false(self):
    """Test pattern guard with Result type that evaluates to false."""
    source = """fn main() -> i64:
    let r: Result[i64, i64] = Ok(3)
    match r:
        Ok(val) if val > 5:
            1
        Ok(val):
            val
        Err(e):
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3  # 3 > 5 is false, falls through to second arm

  def test_pattern_guard_multiple_conditions(self):
    """Test pattern guard with complex boolean expression."""
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let opt: Option = Option::Some(7)
    match opt:
        Option::Some(x) if x > 5 and x < 10:
            1
        Option::Some(x):
            0
        Option::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 7 > 5 and 7 < 10 is true

  def test_pattern_guard_chained_comparison(self):
    """Test pattern guard with chained comparison."""
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let opt: Option = Option::Some(7)
    match opt:
        Option::Some(x) if 5 < x < 10:
            1
        Option::Some(x):
            0
        Option::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 5 < 7 < 10 is true

  def test_pattern_guard_multiple_guards_same_variant(self):
    """Test multiple pattern guards on the same variant."""
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let opt: Option = Option::Some(15)
    match opt:
        Option::Some(x) if x < 5:
            1
        Option::Some(x) if x < 10:
            2
        Option::Some(x) if x < 20:
            3
        Option::Some(x):
            4
        Option::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 3  # 15 < 5 false, 15 < 10 false, 15 < 20 true

  def test_pattern_guard_no_match_falls_through(self):
    """Test that failing all guards falls through to unguarded arm."""
    source = """enum Option:
    Some(i64)
    None

fn main() -> i64:
    let opt: Option = Option::Some(100)
    match opt:
        Option::Some(x) if x < 10:
            1
        Option::Some(x) if x < 50:
            2
        Option::Some(x):
            x
        Option::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 100  # All guards fail, falls through to unguarded arm

  def test_pattern_guard_err_variant(self):
    """Test pattern guard on Err variant of Result."""
    source = """fn main() -> i64:
    let r: Result[i64, i64] = Err(42)
    match r:
        Ok(val):
            0
        Err(e) if e > 40:
            1
        Err(e):
            2
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 42 > 40 is true

  # === Operator Overloading Tests ===

  def test_operator_overload_add(self):
    """Test overloading the + operator for a struct."""
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn __add__(self: Point, other: Point) -> Point:
        Point { x: self.x + other.x, y: self.y + other.y }

fn main() -> i64:
    let p1: Point = Point { x: 10, y: 20 }
    let p2: Point = Point { x: 5, y: 15 }
    let p3: Point = p1 + p2
    p3.x + p3.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 50  # (10+5) + (20+15) = 15 + 35 = 50

  def test_operator_overload_sub(self):
    """Test overloading the - operator for a struct."""
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn __sub__(self: Point, other: Point) -> Point:
        Point { x: self.x - other.x, y: self.y - other.y }

fn main() -> i64:
    let p1: Point = Point { x: 20, y: 30 }
    let p2: Point = Point { x: 5, y: 10 }
    let p3: Point = p1 - p2
    p3.x + p3.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 35  # (20-5) + (30-10) = 15 + 20 = 35

  def test_operator_overload_mul(self):
    """Test overloading the * operator for scalar multiplication."""
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn __mul__(self: Point, scalar: i64) -> Point:
        Point { x: self.x * scalar, y: self.y * scalar }

fn main() -> i64:
    let p: Point = Point { x: 3, y: 4 }
    let scaled: Point = p * 5
    scaled.x + scaled.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 35  # (3*5) + (4*5) = 15 + 20 = 35

  def test_operator_overload_eq(self):
    """Test overloading the == operator."""
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn __eq__(self: Point, other: Point) -> bool:
        self.x == other.x and self.y == other.y

fn main() -> i64:
    let p1: Point = Point { x: 5, y: 10 }
    let p2: Point = Point { x: 5, y: 10 }
    let p3: Point = Point { x: 1, y: 2 }
    if p1 == p2:
        if p1 == p3:
            0
        else:
            1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # p1 == p2 is true, p1 == p3 is false

  def test_operator_overload_ne(self):
    """Test overloading the != operator."""
    source = """struct Point:
    x: i64
    y: i64

impl Point:
    fn __ne__(self: Point, other: Point) -> bool:
        self.x != other.x or self.y != other.y

fn main() -> i64:
    let p1: Point = Point { x: 5, y: 10 }
    let p2: Point = Point { x: 5, y: 10 }
    let p3: Point = Point { x: 1, y: 2 }
    if p1 != p3:
        if p1 != p2:
            0
        else:
            1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # p1 != p3 is true, p1 != p2 is false

  def test_operator_overload_lt(self):
    """Test overloading the < operator."""
    source = """struct Wrapper:
    value: i64

impl Wrapper:
    fn __lt__(self: Wrapper, other: Wrapper) -> bool:
        self.value < other.value

fn main() -> i64:
    let a: Wrapper = Wrapper { value: 5 }
    let b: Wrapper = Wrapper { value: 10 }
    if a < b:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 5 < 10 is true

  def test_operator_overload_le(self):
    """Test overloading the <= operator."""
    source = """struct Wrapper:
    value: i64

impl Wrapper:
    fn __le__(self: Wrapper, other: Wrapper) -> bool:
        self.value <= other.value

fn main() -> i64:
    let a: Wrapper = Wrapper { value: 5 }
    let b: Wrapper = Wrapper { value: 5 }
    let c: Wrapper = Wrapper { value: 10 }
    if a <= b and b <= c:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 5 <= 5 and 5 <= 10

  def test_operator_overload_gt(self):
    """Test overloading the > operator."""
    source = """struct Wrapper:
    value: i64

impl Wrapper:
    fn __gt__(self: Wrapper, other: Wrapper) -> bool:
        self.value > other.value

fn main() -> i64:
    let a: Wrapper = Wrapper { value: 10 }
    let b: Wrapper = Wrapper { value: 5 }
    if a > b:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 10 > 5 is true

  def test_operator_overload_ge(self):
    """Test overloading the >= operator."""
    source = """struct Wrapper:
    value: i64

impl Wrapper:
    fn __ge__(self: Wrapper, other: Wrapper) -> bool:
        self.value >= other.value

fn main() -> i64:
    let a: Wrapper = Wrapper { value: 10 }
    let b: Wrapper = Wrapper { value: 10 }
    let c: Wrapper = Wrapper { value: 5 }
    if a >= b and a >= c:
        1
    else:
        0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 1  # 10 >= 10 and 10 >= 5

  def test_operator_overload_chained(self):
    """Test chaining overloaded operators."""
    source = """struct Vec2:
    x: i64
    y: i64

impl Vec2:
    fn __add__(self: Vec2, other: Vec2) -> Vec2:
        Vec2 { x: self.x + other.x, y: self.y + other.y }

fn main() -> i64:
    let a: Vec2 = Vec2 { x: 1, y: 2 }
    let b: Vec2 = Vec2 { x: 3, y: 4 }
    let c: Vec2 = Vec2 { x: 5, y: 6 }
    let d: Vec2 = a + b + c
    d.x + d.y
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 21  # (1+3+5) + (2+4+6) = 9 + 12 = 21

  def test_operator_overload_in_expression(self):
    """Test using overloaded operators in complex expressions."""
    source = """struct Counter:
    count: i64

impl Counter:
    fn __add__(self: Counter, other: Counter) -> Counter:
        Counter { count: self.count + other.count }

fn main() -> i64:
    let c1: Counter = Counter { count: 10 }
    let c2: Counter = Counter { count: 20 }
    let c3: Counter = Counter { count: 30 }
    let result: Counter = c1 + c2
    let final_result: Counter = result + c3
    final_result.count
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 60  # 10 + 20 + 30 = 60

  def test_operator_overload_mixed_with_builtin(self):
    """Test operator overloading alongside built-in operators."""
    source = """struct Value:
    n: i64

impl Value:
    fn __add__(self: Value, other: Value) -> Value:
        Value { n: self.n + other.n }

fn main() -> i64:
    let v1: Value = Value { n: 5 }
    let v2: Value = Value { n: 3 }
    let sum: Value = v1 + v2
    sum.n + 10 + 20
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 38  # (5+3) + 10 + 20 = 8 + 30 = 38

  # === Option? Operator Tests ===

  def test_option_try_some(self):
    """Test ? operator on Option unwraps Some."""
    source = """enum Option<T>:
    Some(T)
    None

fn get_value(opt: Option<i64>) -> Option<i64>:
    let value: i64 = opt?
    Option<i64>::Some(value + 10)

fn main() -> i64:
    let some_val: Option<i64> = Option<i64>::Some(32)
    let result: Option<i64> = get_value(some_val)
    match result:
        Option<i64>::Some(v):
            v
        Option<i64>::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42  # 32 + 10

  def test_option_try_none_propagates(self):
    """Test ? operator on None returns None early."""
    source = """enum Option<T>:
    Some(T)
    None

fn get_value(opt: Option<i64>) -> Option<i64>:
    let value: i64 = opt?
    Option<i64>::Some(value + 10)

fn main() -> i64:
    let none_val: Option<i64> = Option<i64>::None
    let result: Option<i64> = get_value(none_val)
    match result:
        Option<i64>::Some(v):
            v
        Option<i64>::None:
            99
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 99  # None propagated

  def test_option_try_chained(self):
    """Test chained ? operators on Option."""
    source = """enum Option<T>:
    Some(T)
    None

fn get_opt(x: i64) -> Option<i64>:
    if x > 0:
        Option<i64>::Some(x)
    else:
        Option<i64>::None

fn chain(a: i64, b: i64) -> Option<i64>:
    let x: i64 = get_opt(a)?
    let y: i64 = get_opt(b)?
    Option<i64>::Some(x + y)

fn main() -> i64:
    let r1: Option<i64> = chain(10, 20)
    match r1:
        Option<i64>::Some(v):
            v
        Option<i64>::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 30  # 10 + 20

  def test_option_try_chain_fails_first(self):
    """Test chained ? where first fails."""
    source = """enum Option<T>:
    Some(T)
    None

fn get_opt(x: i64) -> Option<i64>:
    if x > 0:
        Option<i64>::Some(x)
    else:
        Option<i64>::None

fn chain(a: i64, b: i64) -> Option<i64>:
    let x: i64 = get_opt(a)?
    let y: i64 = get_opt(b)?
    Option<i64>::Some(x + y)

fn main() -> i64:
    let r1: Option<i64> = chain(-5, 20)
    match r1:
        Option<i64>::Some(v):
            v
        Option<i64>::None:
            88
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 88  # First ? failed

  def test_option_try_chain_fails_second(self):
    """Test chained ? where second fails."""
    source = """enum Option<T>:
    Some(T)
    None

fn get_opt(x: i64) -> Option<i64>:
    if x > 0:
        Option<i64>::Some(x)
    else:
        Option<i64>::None

fn chain(a: i64, b: i64) -> Option<i64>:
    let x: i64 = get_opt(a)?
    let y: i64 = get_opt(b)?
    Option<i64>::Some(x + y)

fn main() -> i64:
    let r1: Option<i64> = chain(10, -5)
    match r1:
        Option<i64>::Some(v):
            v
        Option<i64>::None:
            77
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 77  # Second ? failed

  def test_option_try_with_variable(self):
    """Test ? operator on Option stored in a variable."""
    source = """enum Option<T>:
    Some(T)
    None

fn process(opt: Option<i64>) -> Option<i64>:
    let val: i64 = opt?
    Option<i64>::Some(val * 2)

fn main() -> i64:
    let opt: Option<i64> = Option<i64>::Some(21)
    let res: Option<i64> = process(opt)
    match res:
        Option<i64>::Some(v):
            v
        Option<i64>::None:
            0
"""
    exit_code, _ = self._compile_and_run(source)
    assert exit_code == 42  # 21 * 2

  def test_option_try_type_mismatch_error(self):
    """Test ? operator error when function doesn't return Option."""
    source = """enum Option<T>:
    Some(T)
    None

fn bad(opt: Option<i64>) -> i64:
    let val: i64 = opt?
    val

fn main() -> i64:
    0
"""
    from oxide.checker import TypeError as CheckerTypeError

    tokens = tokenize(source)
    ast = parse(tokens)
    with pytest.raises(CheckerTypeError, match="requires function to return Option"):
      check(ast)

  def test_option_try_inner_type_mismatch_error(self):
    """Test ? operator error when Option inner types don't match."""
    source = """enum Option<T>:
    Some(T)
    None

fn bad(opt: Option<i64>) -> Option<bool>:
    let val: i64 = opt?
    Option<bool>::Some(true)

fn main() -> i64:
    0
"""
    from oxide.checker import TypeError as CheckerTypeError

    tokens = tokenize(source)
    ast = parse(tokens)
    with pytest.raises(CheckerTypeError, match="mismatch"):
      check(ast)
