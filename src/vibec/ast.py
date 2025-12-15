"""AST node definitions for the Vibec language."""

from dataclasses import dataclass

# === Type Annotations ===


@dataclass(frozen=True, slots=True)
class SimpleType:
  """Simple type like 'i64', 'bool', 'str'."""

  name: str


@dataclass(frozen=True, slots=True)
class ArrayType:
  """Fixed-size array type like [i64; 5]."""

  element_type: "TypeAnnotation"
  size: int


@dataclass(frozen=True, slots=True)
class VecType:
  """Dynamic vector type like vec[i64]."""

  element_type: "TypeAnnotation"


# Type annotation union
TypeAnnotation = SimpleType | ArrayType | VecType


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


@dataclass(frozen=True, slots=True)
class ArrayLiteral:
  """Array literal like [1, 2, 3]."""

  elements: tuple["Expr", ...]


@dataclass(frozen=True, slots=True)
class IndexExpr:
  """Index expression like arr[0]."""

  target: "Expr"
  index: "Expr"


@dataclass(frozen=True, slots=True)
class MethodCallExpr:
  """Method call like list.push(x) or list.len()."""

  target: "Expr"
  method: str
  args: tuple["Expr", ...]


@dataclass(frozen=True, slots=True)
class StructLiteral:
  """Struct literal like Point { x: 10, y: 20 }."""

  name: str
  fields: tuple[tuple[str, "Expr"], ...]  # (field_name, value) pairs


@dataclass(frozen=True, slots=True)
class FieldAccessExpr:
  """Field access like p.x."""

  target: "Expr"
  field: str


# Expression union type
Expr = (
  IntLiteral
  | BoolLiteral
  | StringLiteral
  | VarExpr
  | BinaryExpr
  | UnaryExpr
  | CallExpr
  | ArrayLiteral
  | IndexExpr
  | MethodCallExpr
  | StructLiteral
  | FieldAccessExpr
)


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


@dataclass(frozen=True, slots=True)
class IndexAssignStmt:
  """Index assignment: arr[0] = 42"""

  target: Expr
  index: Expr
  value: Expr


@dataclass(frozen=True, slots=True)
class ForStmt:
  """For loop: for i in range(start, end): body"""

  var: str
  start: Expr
  end: Expr
  body: tuple["Stmt", ...]


@dataclass(frozen=True, slots=True)
class FieldAssignStmt:
  """Field assignment: p.x = 10"""

  target: Expr
  field: str
  value: Expr


# Statement union type
Stmt = LetStmt | AssignStmt | IndexAssignStmt | FieldAssignStmt | ReturnStmt | ExprStmt | IfStmt | WhileStmt | ForStmt


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
class StructField:
  """Field in a struct definition."""

  name: str
  type_ann: TypeAnnotation


@dataclass(frozen=True, slots=True)
class StructDef:
  """Struct definition: struct Point: x: i64, y: i64"""

  name: str
  fields: tuple[StructField, ...]


@dataclass(frozen=True, slots=True)
class Program:
  """Root node containing structs and functions."""

  structs: tuple[StructDef, ...]
  functions: tuple[Function, ...]
