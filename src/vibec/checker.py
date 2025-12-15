"""Type checker for the Vibec language."""

from dataclasses import dataclass

from .ast import (
  Expr,
  Stmt,
  IfStmt,
  LetStmt,
  Program,
  VarExpr,
  CallExpr,
  ExprStmt,
  Function,
  UnaryExpr,
  WhileStmt,
  AssignStmt,
  BinaryExpr,
  IntLiteral,
  ReturnStmt,
  BoolLiteral,
  StringLiteral,
)


class TypeError(Exception):
  """Raised when a type error is detected."""

  pass


@dataclass
class FunctionSignature:
  """Stores a function's type signature."""

  param_types: tuple[str, ...]
  return_type: str


class TypeChecker:
  """Single-pass type checker with scoped symbol tables."""

  # Built-in function signatures
  BUILTINS: dict[str, FunctionSignature] = {
    "print": FunctionSignature(("i64",), "i64"),  # print returns 0
  }

  def __init__(self) -> None:
    # Function signatures: name -> signature
    self.functions: dict[str, FunctionSignature] = dict(self.BUILTINS)
    # Variable scopes: list of (name -> type) dicts
    self.scopes: list[dict[str, str]] = []
    # Current function's return type (for checking return statements)
    self.current_return_type: str | None = None

  def _enter_scope(self) -> None:
    self.scopes.append({})

  def _exit_scope(self) -> None:
    self.scopes.pop()

  def _define_var(self, name: str, type_name: str) -> None:
    if name in self.scopes[-1]:
      raise TypeError(f"Variable '{name}' already defined in this scope")
    self.scopes[-1][name] = type_name

  def _lookup_var(self, name: str) -> str:
    for scope in reversed(self.scopes):
      if name in scope:
        return scope[name]
    raise TypeError(f"Undefined variable '{name}'")

  def _check_type(self, name: str) -> None:
    """Verify that a type name is valid."""
    if name not in ("i64", "bool", "str"):
      raise TypeError(f"Unknown type '{name}'")

  def check(self, program: Program) -> None:
    """Type check an entire program."""
    # First pass: register all function signatures
    for func in program.functions:
      self._check_type(func.return_type.name)
      param_types: list[str] = []
      for param in func.params:
        self._check_type(param.type_ann.name)
        param_types.append(param.type_ann.name)

      if func.name in self.functions:
        raise TypeError(f"Function '{func.name}' already defined")

      self.functions[func.name] = FunctionSignature(tuple(param_types), func.return_type.name)

    # Check for main function
    if "main" not in self.functions:
      raise TypeError("No 'main' function defined")

    # Second pass: check function bodies
    for func in program.functions:
      self._check_function(func)

  def _check_function(self, func: Function) -> None:
    """Type check a function body."""
    self._enter_scope()
    self.current_return_type = func.return_type.name

    # Add parameters to scope
    for param in func.params:
      self._define_var(param.name, param.type_ann.name)

    # Check statements
    for stmt in func.body:
      self._check_stmt(stmt)

    self.current_return_type = None
    self._exit_scope()

  def _check_stmt(self, stmt: Stmt) -> None:
    """Type check a statement."""
    match stmt:
      case LetStmt(name, type_ann, value):
        self._check_type(type_ann.name)
        value_type = self._check_expr(value)
        if value_type != type_ann.name:
          raise TypeError(f"Cannot assign {value_type} to variable of type {type_ann.name}")
        self._define_var(name, type_ann.name)

      case AssignStmt(name, value):
        var_type = self._lookup_var(name)
        value_type = self._check_expr(value)
        if value_type != var_type:
          raise TypeError(f"Cannot assign {value_type} to variable of type {var_type}")

      case ReturnStmt(value):
        value_type = self._check_expr(value)
        if value_type != self.current_return_type:
          raise TypeError(f"Cannot return {value_type} from function returning {self.current_return_type}")

      case ExprStmt(expr):
        self._check_expr(expr)

      case IfStmt(condition, then_body, else_body):
        cond_type = self._check_expr(condition)
        if cond_type != "bool":
          raise TypeError(f"If condition must be bool, got {cond_type}")

        self._enter_scope()
        for s in then_body:
          self._check_stmt(s)
        self._exit_scope()

        if else_body:
          self._enter_scope()
          for s in else_body:
            self._check_stmt(s)
          self._exit_scope()

      case WhileStmt(condition, body):
        cond_type = self._check_expr(condition)
        if cond_type != "bool":
          raise TypeError(f"While condition must be bool, got {cond_type}")

        self._enter_scope()
        for s in body:
          self._check_stmt(s)
        self._exit_scope()

  def _check_expr(self, expr: Expr) -> str:
    """Type check an expression and return its type."""
    match expr:
      case IntLiteral(_):
        return "i64"

      case BoolLiteral(_):
        return "bool"

      case StringLiteral(_):
        return "str"

      case VarExpr(name):
        return self._lookup_var(name)

      case BinaryExpr(left, op, right):
        left_type = self._check_expr(left)
        right_type = self._check_expr(right)

        # Arithmetic operators: i64 -> i64
        if op in ("+", "-", "*", "/", "%"):
          if left_type != "i64" or right_type != "i64":
            raise TypeError(f"Arithmetic operator '{op}' requires i64 operands")
          return "i64"

        # Comparison operators: i64 -> bool
        if op in ("<", ">", "<=", ">="):
          if left_type != "i64" or right_type != "i64":
            raise TypeError(f"Comparison operator '{op}' requires i64 operands")
          return "bool"

        # Equality operators: same type -> bool
        if op in ("==", "!="):
          if left_type != right_type:
            raise TypeError(f"Cannot compare {left_type} with {right_type}")
          return "bool"

        # Logical operators: bool -> bool
        if op in ("and", "or"):
          if left_type != "bool" or right_type != "bool":
            raise TypeError(f"Logical operator '{op}' requires bool operands")
          return "bool"

        raise TypeError(f"Unknown operator '{op}'")

      case UnaryExpr(op, operand):
        operand_type = self._check_expr(operand)

        if op == "-":
          if operand_type != "i64":
            raise TypeError("Unary '-' requires i64 operand")
          return "i64"

        if op == "not":
          if operand_type != "bool":
            raise TypeError("'not' requires bool operand")
          return "bool"

        raise TypeError(f"Unknown unary operator '{op}'")

      case CallExpr(name, args):
        # Special case: print accepts i64 or str
        if name == "print":
          if len(args) != 1:
            raise TypeError(f"print() expects 1 argument, got {len(args)}")
          arg_type = self._check_expr(args[0])
          if arg_type not in ("i64", "str"):
            raise TypeError(f"print() expects i64 or str, got {arg_type}")
          return "i64"

        if name not in self.functions:
          raise TypeError(f"Undefined function '{name}'")

        sig = self.functions[name]

        if len(args) != len(sig.param_types):
          raise TypeError(f"Function '{name}' expects {len(sig.param_types)} arguments, got {len(args)}")

        for i, (arg, expected_type) in enumerate(zip(args, sig.param_types)):
          arg_type = self._check_expr(arg)
          if arg_type != expected_type:
            raise TypeError(f"Argument {i + 1} of '{name}' expects {expected_type}, got {arg_type}")

        return sig.return_type

    raise TypeError(f"Unknown expression type: {type(expr)}")


def check(program: Program) -> None:
  """Convenience function to type check a program."""
  TypeChecker().check(program)
