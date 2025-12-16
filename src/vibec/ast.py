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


@dataclass(frozen=True, slots=True)
class TupleType:
  """Tuple type like (i64, bool, str)."""

  element_types: tuple["TypeAnnotation", ...]


@dataclass(frozen=True, slots=True)
class RefType:
  """Reference type: &T or &mut T."""

  inner: "TypeAnnotation"
  mutable: bool


@dataclass(frozen=True, slots=True)
class FnType:
  """Function/closure type: Fn(i64, i64) -> i64."""

  param_types: tuple["TypeAnnotation", ...]
  return_type: "TypeAnnotation"


@dataclass(frozen=True, slots=True)
class ResultType:
  """Result type: Result[T, E]."""

  ok_type: "TypeAnnotation"
  err_type: "TypeAnnotation"


@dataclass(frozen=True, slots=True)
class DictType:
  """Dictionary/hashmap type: dict[K, V]."""

  key_type: "TypeAnnotation"
  value_type: "TypeAnnotation"


# Type annotation union
TypeAnnotation = SimpleType | ArrayType | VecType | TupleType | RefType | FnType | ResultType | DictType


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
  """Function call like foo(1, 2) or foo(x=1, y=2)."""

  name: str
  args: tuple["Expr", ...]
  kwargs: tuple[tuple[str, "Expr"], ...] = ()  # (name, value) pairs


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


@dataclass(frozen=True, slots=True)
class TupleLiteral:
  """Tuple literal like (10, 20, 30)."""

  elements: tuple["Expr", ...]


@dataclass(frozen=True, slots=True)
class TupleIndexExpr:
  """Tuple index access like t.0 or t.1."""

  target: "Expr"
  index: int


@dataclass(frozen=True, slots=True)
class EnumLiteral:
  """Enum variant instantiation like Option::Some(42) or Option::None."""

  enum_name: str
  variant_name: str
  payload: "Expr | None"  # None for unit variants


@dataclass(frozen=True, slots=True)
class MatchArm:
  """A single arm in a match expression."""

  enum_name: str
  variant_name: str
  binding: str | None  # Variable name to bind payload (None for unit variants)
  body: tuple["Stmt", ...]


@dataclass(frozen=True, slots=True)
class MatchExpr:
  """Match expression: match expr: arms..."""

  target: "Expr"
  arms: tuple[MatchArm, ...]


@dataclass(frozen=True, slots=True)
class RefExpr:
  """Create reference: &x or &mut x."""

  target: "Expr"
  mutable: bool


@dataclass(frozen=True, slots=True)
class DerefExpr:
  """Dereference: *ptr."""

  target: "Expr"


@dataclass(frozen=True, slots=True)
class ClosureExpr:
  """Closure expression: |a: i64, b: i64| -> i64: a + b."""

  params: tuple["Parameter", ...]
  return_type: "TypeAnnotation"
  body: "Expr"


@dataclass(frozen=True, slots=True)
class ClosureCallExpr:
  """Call a closure: closure_var(arg1, arg2)."""

  target: "Expr"
  args: tuple["Expr", ...]


@dataclass(frozen=True, slots=True)
class OkExpr:
  """Ok(value) expression for Result type."""

  value: "Expr"


@dataclass(frozen=True, slots=True)
class ErrExpr:
  """Err(error) expression for Result type."""

  value: "Expr"


@dataclass(frozen=True, slots=True)
class TryExpr:
  """Try expression: expr? - unwraps Ok or returns Err early."""

  target: "Expr"


@dataclass(frozen=True, slots=True)
class DictLiteral:
  """Dictionary literal: {key: value, ...}."""

  entries: tuple[tuple["Expr", "Expr"], ...]  # (key, value) pairs


@dataclass(frozen=True, slots=True)
class ListComprehension:
  """List comprehension: [expr for var in iterable] or [expr for var in iterable if cond]."""

  element_expr: "Expr"  # Expression to evaluate for each element
  var_name: str  # Loop variable name
  start: "Expr"  # Range start (for now, only range() is supported)
  end: "Expr"  # Range end
  condition: "Expr | None"  # Optional filter condition


@dataclass(frozen=True, slots=True)
class DictComprehension:
  """Dict comprehension: {key: value for var in range(start, end) if cond}."""

  key_expr: "Expr"  # Expression for key
  value_expr: "Expr"  # Expression for value
  var_name: str  # Loop variable name
  start: "Expr"  # Range start
  end: "Expr"  # Range end
  condition: "Expr | None"  # Optional filter condition


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
  | TupleLiteral
  | TupleIndexExpr
  | EnumLiteral
  | MatchExpr
  | RefExpr
  | DerefExpr
  | ClosureExpr
  | ClosureCallExpr
  | OkExpr
  | ErrExpr
  | DictLiteral
  | TryExpr
  | ListComprehension
  | DictComprehension
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


@dataclass(frozen=True, slots=True)
class DerefAssignStmt:
  """Assignment through dereference: *ptr = value"""

  target: Expr
  value: Expr


# Statement union type
Stmt = LetStmt | AssignStmt | IndexAssignStmt | FieldAssignStmt | DerefAssignStmt | ReturnStmt | ExprStmt | IfStmt | WhileStmt | ForStmt


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
class EnumVariant:
  """A variant in an enum definition."""

  name: str
  payload_type: TypeAnnotation | None  # None for unit variants


@dataclass(frozen=True, slots=True)
class EnumDef:
  """Enum definition: enum Option: Some(i64), None"""

  name: str
  variants: tuple[EnumVariant, ...]


@dataclass(frozen=True, slots=True)
class ImplBlock:
  """Implementation block: impl StructName: fn method(self, ...) -> ..."""

  struct_name: str
  methods: tuple[Function, ...]


@dataclass(frozen=True, slots=True)
class Program:
  """Root node containing structs, enums, impl blocks, and functions."""

  structs: tuple[StructDef, ...]
  enums: tuple[EnumDef, ...]
  impls: tuple[ImplBlock, ...]
  functions: tuple[Function, ...]
