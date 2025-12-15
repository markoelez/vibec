"""AST node definitions for the Vibec language."""

from dataclasses import dataclass

# === Type Annotations ===


@dataclass(frozen=True, slots=True)
class TypeAnnotation:
  """Represents a type annotation like 'i64' or 'bool'."""

  name: str


# === Expressions ===


@dataclass(frozen=True, slots=True)
class IntLiteral:
  """Integer literal like 42."""

  value: int


@dataclass(frozen=True, slots=True)
class BoolLiteral:
  """Boolean literal: true or false."""

  value: bool


@dataclass(frozen=True, slots=True)
class StringLiteral:
  """String literal like "hello"."""

  value: str


@dataclass(frozen=True, slots=True)
class VarExpr:
  """Variable reference."""

  name: str


@dataclass(frozen=True, slots=True)
class BinaryExpr:
  """Binary expression like a + b or x < y."""

  left: "Expr"
  op: str
  right: "Expr"


@dataclass(frozen=True, slots=True)
class UnaryExpr:
  """Unary expression like -x or not b."""

  op: str
  operand: "Expr"


@dataclass(frozen=True, slots=True)
class CallExpr:
  """Function call like foo(1, 2)."""

  name: str
  args: tuple["Expr", ...]


# Expression union type
Expr = IntLiteral | BoolLiteral | StringLiteral | VarExpr | BinaryExpr | UnaryExpr | CallExpr


# === Statements ===


@dataclass(frozen=True, slots=True)
class LetStmt:
  """Variable declaration: let x: i64 = 42"""

  name: str
  type_ann: TypeAnnotation
  value: Expr


@dataclass(frozen=True, slots=True)
class ReturnStmt:
  """Return statement: return expr"""

  value: Expr


@dataclass(frozen=True, slots=True)
class ExprStmt:
  """Expression statement (expression used as statement)."""

  expr: Expr


@dataclass(frozen=True, slots=True)
class IfStmt:
  """If statement with optional else block."""

  condition: Expr
  then_body: tuple["Stmt", ...]
  else_body: tuple["Stmt", ...] | None


@dataclass(frozen=True, slots=True)
class WhileStmt:
  """While loop."""

  condition: Expr
  body: tuple["Stmt", ...]


@dataclass(frozen=True, slots=True)
class AssignStmt:
  """Variable assignment: x = 42"""

  name: str
  value: Expr


# Statement union type
Stmt = LetStmt | AssignStmt | ReturnStmt | ExprStmt | IfStmt | WhileStmt


# === Top-level Definitions ===


@dataclass(frozen=True, slots=True)
class Parameter:
  """Function parameter with name and type."""

  name: str
  type_ann: TypeAnnotation


@dataclass(frozen=True, slots=True)
class Function:
  """Function definition."""

  name: str
  params: tuple[Parameter, ...]
  return_type: TypeAnnotation
  body: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class Program:
  """Root node containing all functions."""

  functions: tuple[Function, ...]
