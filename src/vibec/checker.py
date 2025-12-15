"""Type checker for the Vibec language."""

from dataclasses import dataclass

from .ast import (
  Expr,
  Stmt,
  IfStmt,
  ForStmt,
  LetStmt,
  Program,
  VarExpr,
  VecType,
  CallExpr,
  ExprStmt,
  Function,
  ArrayType,
  IndexExpr,
  TupleType,
  UnaryExpr,
  WhileStmt,
  AssignStmt,
  BinaryExpr,
  IntLiteral,
  ReturnStmt,
  SimpleType,
  BoolLiteral,
  ArrayLiteral,
  TupleLiteral,
  StringLiteral,
  StructLiteral,
  MethodCallExpr,
  TupleIndexExpr,
  TypeAnnotation,
  FieldAccessExpr,
  FieldAssignStmt,
  IndexAssignStmt,
)


class TypeError(Exception):
  """Raised when a type error is detected."""

  pass


@dataclass
class FunctionSignature:
  """Stores a function's type signature."""

  param_types: tuple[str, ...]
  return_type: str


def type_to_str(t: TypeAnnotation) -> str:
  """Convert a type annotation to a canonical string representation."""
  match t:
    case SimpleType(name):
      return name
    case ArrayType(elem, size):
      return f"[{type_to_str(elem)};{size}]"
    case VecType(elem):
      return f"vec[{type_to_str(elem)}]"
    case TupleType(elems):
      return f"({','.join(type_to_str(e) for e in elems)})"
  raise TypeError(f"Unknown type annotation: {t}")


def get_element_type(type_str: str) -> str | None:
  """Extract element type from array or vec type string."""
  if type_str.startswith("[") and ";" in type_str:
    # Array type: [i64;5] -> i64
    return type_str[1 : type_str.index(";")]
  elif type_str.startswith("vec["):
    # Vec type: vec[i64] -> i64
    return type_str[4:-1]
  return None


def is_array_type(type_str: str) -> bool:
  """Check if type string represents an array."""
  return type_str.startswith("[") and ";" in type_str


def is_vec_type(type_str: str) -> bool:
  """Check if type string represents a vec."""
  return type_str.startswith("vec[")


def is_tuple_type(type_str: str) -> bool:
  """Check if type string represents a tuple."""
  return type_str.startswith("(") and type_str.endswith(")")


def get_tuple_element_types(type_str: str) -> list[str]:
  """Extract element types from tuple type string like (i64,bool) -> ['i64', 'bool']."""
  if not is_tuple_type(type_str):
    return []
  inner = type_str[1:-1]  # Remove parens
  if not inner:
    return []
  # Simple split - works for non-nested types
  return [t.strip() for t in inner.split(",")]


@dataclass
class StructInfo:
  """Stores a struct's field information."""

  fields: dict[str, str]  # field_name -> field_type


class TypeChecker:
  """Single-pass type checker with scoped symbol tables."""

  # Built-in function signatures
  BUILTINS: dict[str, FunctionSignature] = {
    "print": FunctionSignature(("i64",), "i64"),  # print returns 0
  }

  def __init__(self) -> None:
    # Struct definitions: name -> StructInfo
    self.structs: dict[str, StructInfo] = {}
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

  def _check_type_ann(self, t: TypeAnnotation) -> str:
    """Verify type annotation is valid and return canonical string."""
    match t:
      case SimpleType(name):
        if name in ("i64", "bool", "str"):
          return name
        if name in self.structs:
          return name  # Struct type
        raise TypeError(f"Unknown type '{name}'")
      case ArrayType(elem, size):
        if size <= 0:
          raise TypeError("Array size must be positive")
        elem_str = self._check_type_ann(elem)
        return f"[{elem_str};{size}]"
      case VecType(elem):
        elem_str = self._check_type_ann(elem)
        return f"vec[{elem_str}]"
      case TupleType(elems):
        elem_strs = [self._check_type_ann(e) for e in elems]
        return f"({','.join(elem_strs)})"
    raise TypeError(f"Unknown type annotation: {t}")

  def check(self, program: Program) -> None:
    """Type check an entire program."""
    # First pass: register all struct definitions
    for struct in program.structs:
      if struct.name in self.structs:
        raise TypeError(f"Struct '{struct.name}' already defined")
      self.structs[struct.name] = StructInfo({})  # Placeholder

    # Second pass: check struct field types (allows recursive/mutual refs)
    for struct in program.structs:
      fields: dict[str, str] = {}
      for field in struct.fields:
        field_type = self._check_type_ann(field.type_ann)
        if field.name in fields:
          raise TypeError(f"Duplicate field '{field.name}' in struct '{struct.name}'")
        fields[field.name] = field_type
      self.structs[struct.name] = StructInfo(fields)

    # Third pass: register all function signatures
    for func in program.functions:
      ret_type = self._check_type_ann(func.return_type)
      param_types: list[str] = []
      for param in func.params:
        param_types.append(self._check_type_ann(param.type_ann))

      if func.name in self.functions:
        raise TypeError(f"Function '{func.name}' already defined")

      self.functions[func.name] = FunctionSignature(tuple(param_types), ret_type)

    # Check for main function
    if "main" not in self.functions:
      raise TypeError("No 'main' function defined")

    # Fourth pass: check function bodies
    for func in program.functions:
      self._check_function(func)

  def _check_function(self, func: Function) -> None:
    """Type check a function body."""
    self._enter_scope()
    self.current_return_type = self._check_type_ann(func.return_type)

    # Add parameters to scope
    for param in func.params:
      self._define_var(param.name, self._check_type_ann(param.type_ann))

    # Check statements
    for stmt in func.body:
      self._check_stmt(stmt)

    self.current_return_type = None
    self._exit_scope()

  def _check_stmt(self, stmt: Stmt) -> None:
    """Type check a statement."""
    match stmt:
      case LetStmt(name, type_ann, value):
        declared_type = self._check_type_ann(type_ann)
        value_type = self._check_expr(value)
        # Allow empty array literal to match any array type
        if value_type == "[]" and is_array_type(declared_type):
          pass  # OK: empty array assigned to array type
        elif value_type == "[]" and is_vec_type(declared_type):
          pass  # OK: empty array assigned to vec type
        elif value_type != declared_type:
          raise TypeError(f"Cannot assign {value_type} to variable of type {declared_type}")
        self._define_var(name, declared_type)

      case AssignStmt(name, value):
        var_type = self._lookup_var(name)
        value_type = self._check_expr(value)
        if value_type != var_type:
          raise TypeError(f"Cannot assign {value_type} to variable of type {var_type}")

      case IndexAssignStmt(target, index, value):
        target_type = self._check_expr(target)
        index_type = self._check_expr(index)
        if index_type != "i64":
          raise TypeError(f"Index must be i64, got {index_type}")

        elem_type = get_element_type(target_type)
        if elem_type is None:
          raise TypeError(f"Cannot index into type {target_type}")

        value_type = self._check_expr(value)
        if value_type != elem_type:
          raise TypeError(f"Cannot assign {value_type} to element of type {elem_type}")

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

      case ForStmt(var, start, end, body):
        start_type = self._check_expr(start)
        if start_type != "i64":
          raise TypeError(f"For loop start must be i64, got {start_type}")

        end_type = self._check_expr(end)
        if end_type != "i64":
          raise TypeError(f"For loop end must be i64, got {end_type}")

        self._enter_scope()
        self._define_var(var, "i64")
        for s in body:
          self._check_stmt(s)
        self._exit_scope()

      case FieldAssignStmt(target, field, value):
        target_type = self._check_expr(target)
        if target_type not in self.structs:
          raise TypeError(f"Cannot access field of non-struct type {target_type}")
        struct_info = self.structs[target_type]
        if field not in struct_info.fields:
          raise TypeError(f"Struct '{target_type}' has no field '{field}'")
        field_type = struct_info.fields[field]
        value_type = self._check_expr(value)
        if value_type != field_type:
          raise TypeError(f"Cannot assign {value_type} to field of type {field_type}")

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

      case ArrayLiteral(elements):
        if not elements:
          return "[]"  # Empty array, type determined by context
        first_type = self._check_expr(elements[0])
        for i, elem in enumerate(elements[1:], 2):
          elem_type = self._check_expr(elem)
          if elem_type != first_type:
            raise TypeError(f"Array element {i} has type {elem_type}, expected {first_type}")
        return f"[{first_type};{len(elements)}]"

      case IndexExpr(target, index):
        target_type = self._check_expr(target)
        index_type = self._check_expr(index)
        if index_type != "i64":
          raise TypeError(f"Index must be i64, got {index_type}")
        elem_type = get_element_type(target_type)
        if elem_type is None:
          raise TypeError(f"Cannot index into type {target_type}")
        return elem_type

      case MethodCallExpr(target, method, args):
        target_type = self._check_expr(target)
        return self._check_method_call(target_type, method, args)

      case StructLiteral(name, fields):
        if name not in self.structs:
          raise TypeError(f"Unknown struct '{name}'")
        struct_info = self.structs[name]
        provided_fields: set[str] = set()
        for field_name, field_value in fields:
          if field_name in provided_fields:
            raise TypeError(f"Duplicate field '{field_name}' in struct literal")
          provided_fields.add(field_name)
          if field_name not in struct_info.fields:
            raise TypeError(f"Struct '{name}' has no field '{field_name}'")
          expected_type = struct_info.fields[field_name]
          actual_type = self._check_expr(field_value)
          if actual_type != expected_type:
            raise TypeError(f"Field '{field_name}' expects {expected_type}, got {actual_type}")
        # Check all required fields are provided
        missing = set(struct_info.fields.keys()) - provided_fields
        if missing:
          raise TypeError(f"Missing fields in struct literal: {', '.join(sorted(missing))}")
        return name

      case FieldAccessExpr(target, field):
        target_type = self._check_expr(target)
        if target_type not in self.structs:
          raise TypeError(f"Cannot access field of non-struct type {target_type}")
        struct_info = self.structs[target_type]
        if field not in struct_info.fields:
          raise TypeError(f"Struct '{target_type}' has no field '{field}'")
        return struct_info.fields[field]

      case TupleLiteral(elements):
        elem_types = [self._check_expr(e) for e in elements]
        return f"({','.join(elem_types)})"

      case TupleIndexExpr(target, index):
        target_type = self._check_expr(target)
        if not is_tuple_type(target_type):
          raise TypeError(f"Cannot index non-tuple type {target_type}")
        elem_types = get_tuple_element_types(target_type)
        if index < 0 or index >= len(elem_types):
          raise TypeError(f"Tuple index {index} out of bounds for tuple with {len(elem_types)} elements")
        return elem_types[index]

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

  def _check_method_call(self, target_type: str, method: str, args: tuple[Expr, ...]) -> str:
    """Type check a method call and return its return type."""
    if is_vec_type(target_type):
      elem_type = get_element_type(target_type)
      assert elem_type is not None

      if method == "push":
        if len(args) != 1:
          raise TypeError(f"push() expects 1 argument, got {len(args)}")
        arg_type = self._check_expr(args[0])
        if arg_type != elem_type:
          raise TypeError(f"push() expects {elem_type}, got {arg_type}")
        return "i64"  # push returns nothing useful, use i64

      elif method == "pop":
        if len(args) != 0:
          raise TypeError(f"pop() expects 0 arguments, got {len(args)}")
        return elem_type

      elif method == "len":
        if len(args) != 0:
          raise TypeError(f"len() expects 0 arguments, got {len(args)}")
        return "i64"

      else:
        raise TypeError(f"Unknown vec method '{method}'")

    elif is_array_type(target_type):
      if method == "len":
        if len(args) != 0:
          raise TypeError(f"len() expects 0 arguments, got {len(args)}")
        return "i64"
      else:
        raise TypeError(f"Unknown array method '{method}'")

    else:
      raise TypeError(f"Cannot call method on type {target_type}")


def check(program: Program) -> None:
  """Convenience function to type check a program."""
  TypeChecker().check(program)
