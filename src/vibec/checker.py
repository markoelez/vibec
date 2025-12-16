"""Type checker for the Vibec language."""

from dataclasses import dataclass

from .ast import (
  Expr,
  Stmt,
  FnType,
  IfStmt,
  OkExpr,
  ErrExpr,
  ForStmt,
  LetStmt,
  Program,
  RefExpr,
  RefType,
  TryExpr,
  VarExpr,
  VecType,
  CallExpr,
  DictType,
  ExprStmt,
  Function,
  ArrayType,
  DerefExpr,
  IndexExpr,
  MatchExpr,
  TupleType,
  UnaryExpr,
  WhileStmt,
  AssignStmt,
  BinaryExpr,
  IntLiteral,
  ResultType,
  ReturnStmt,
  SimpleType,
  BoolLiteral,
  ClosureExpr,
  DictLiteral,
  EnumLiteral,
  ArrayLiteral,
  TupleLiteral,
  StringLiteral,
  StructLiteral,
  MethodCallExpr,
  TupleIndexExpr,
  TypeAnnotation,
  ClosureCallExpr,
  DerefAssignStmt,
  FieldAccessExpr,
  FieldAssignStmt,
  IndexAssignStmt,
  DictComprehension,
  ListComprehension,
)


class TypeError(Exception):
  """Raised when a type error is detected."""

  pass


@dataclass
class FunctionSignature:
  """Stores a function's type signature."""

  param_names: tuple[str, ...]  # Parameter names for keyword argument resolution
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
    case RefType(inner, mutable):
      prefix = "&mut " if mutable else "&"
      return f"{prefix}{type_to_str(inner)}"
    case FnType(params, ret):
      param_strs = ",".join(type_to_str(p) for p in params)
      return f"Fn({param_strs})->{type_to_str(ret)}"
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


def is_dict_type(type_str: str) -> bool:
  """Check if type string represents a dict."""
  return type_str.startswith("dict[")


def get_dict_key_type(type_str: str) -> str | None:
  """Extract key type from dict type string like dict[i64,str] -> i64."""
  if not is_dict_type(type_str):
    return None
  inner = type_str[5:-1]  # Remove "dict[" and "]"
  # Find the comma that separates key and value types (handle nested types)
  depth = 0
  for i, c in enumerate(inner):
    if c in "([":
      depth += 1
    elif c in ")]":
      depth -= 1
    elif c == "," and depth == 0:
      return inner[:i]
  return None


def get_dict_value_type(type_str: str) -> str | None:
  """Extract value type from dict type string like dict[i64,str] -> str."""
  if not is_dict_type(type_str):
    return None
  inner = type_str[5:-1]  # Remove "dict[" and "]"
  # Find the comma that separates key and value types (handle nested types)
  depth = 0
  for i, c in enumerate(inner):
    if c in "([":
      depth += 1
    elif c in ")]":
      depth -= 1
    elif c == "," and depth == 0:
      return inner[i + 1 :]
  return None


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


def is_ref_type(type_str: str) -> bool:
  """Check if type string represents a reference."""
  return type_str.startswith("&")


def is_copy_type(type_str: str) -> bool:
  """Check if a type is Copy (implicitly cloned on use).

  Copy types:
  - Primitives: i64, bool
  - References: &T, &mut T (copying the pointer)
  - str (string literals are immutable)
  """
  if type_str in ("i64", "bool", "str"):
    return True
  if is_ref_type(type_str):
    return True
  return False


def is_mut_ref_type(type_str: str) -> bool:
  """Check if type string represents a mutable reference."""
  return type_str.startswith("&mut ")


def get_ref_inner_type(type_str: str) -> str | None:
  """Extract inner type from reference type string."""
  if type_str.startswith("&mut "):
    return type_str[5:]
  elif type_str.startswith("&"):
    return type_str[1:]
  return None


def is_fn_type(type_str: str) -> bool:
  """Check if type string represents a function/closure type."""
  return type_str.startswith("Fn(")


def parse_fn_type(type_str: str) -> tuple[list[str], str] | None:
  """Parse function type string like Fn(i64,i64)->i64 into (param_types, return_type)."""
  if not is_fn_type(type_str):
    return None
  # Find the closing paren
  paren_depth = 0
  arrow_pos = -1
  for i, c in enumerate(type_str):
    if c == "(":
      paren_depth += 1
    elif c == ")":
      paren_depth -= 1
      if paren_depth == 0:
        arrow_pos = i + 1
        break
  if arrow_pos == -1 or not type_str[arrow_pos:].startswith("->"):
    return None
  # Extract param types and return type
  params_str = type_str[3 : arrow_pos - 1]  # Skip "Fn(" and ")"
  return_type = type_str[arrow_pos + 2 :]  # Skip "->"
  if not params_str:
    return ([], return_type)
  # Simple comma split (doesn't handle nested types with commas)
  param_types = [p.strip() for p in params_str.split(",")]
  return (param_types, return_type)


def is_result_type(type_str: str) -> bool:
  """Check if type string represents a Result type."""
  return type_str.startswith("Result[")


def get_result_ok_type(type_str: str) -> str | None:
  """Extract Ok type from Result type string like Result[i64,str] -> i64."""
  if not is_result_type(type_str):
    return None
  inner = type_str[7:-1]  # Remove "Result[" and "]"
  # Find the comma that separates Ok and Err types (handle nested types)
  depth = 0
  for i, c in enumerate(inner):
    if c in "([":
      depth += 1
    elif c in ")]":
      depth -= 1
    elif c == "," and depth == 0:
      return inner[:i]
  return None


def get_result_err_type(type_str: str) -> str | None:
  """Extract Err type from Result type string like Result[i64,str] -> str."""
  if not is_result_type(type_str):
    return None
  inner = type_str[7:-1]  # Remove "Result[" and "]"
  # Find the comma that separates Ok and Err types (handle nested types)
  depth = 0
  for i, c in enumerate(inner):
    if c in "([":
      depth += 1
    elif c in ")]":
      depth -= 1
    elif c == "," and depth == 0:
      return inner[i + 1 :]
  return None


def result_types_compatible(actual: str, expected: str) -> bool:
  """Check if actual Result type is compatible with expected Result type.

  This handles partial types like Result[i64,?] and Result[?,i64] which can match
  any concrete Result type with matching known parts.
  """
  if not is_result_type(actual) or not is_result_type(expected):
    return False

  actual_ok = get_result_ok_type(actual)
  actual_err = get_result_err_type(actual)
  expected_ok = get_result_ok_type(expected)
  expected_err = get_result_err_type(expected)

  # Check Ok type: must match, or actual can be '?' (unknown)
  ok_match = actual_ok == expected_ok or actual_ok == "?"
  # Check Err type: must match, or actual can be '?' (unknown)
  err_match = actual_err == expected_err or actual_err == "?"

  return ok_match and err_match


@dataclass
class StructInfo:
  """Stores a struct's field information."""

  fields: dict[str, str]  # field_name -> field_type


@dataclass
class EnumInfo:
  """Stores an enum's variant information."""

  variants: dict[str, str | None]  # variant_name -> payload_type (None for unit variants)


@dataclass
class VarState:
  """Tracks the ownership state of a variable."""

  type_str: str
  ownership: str  # "owned", "moved", "borrowed", "mut_borrowed"
  scope_depth: int  # Scope level where variable was defined


@dataclass
class Borrow:
  """Tracks an active borrow of a variable."""

  var_name: str  # The variable being borrowed
  mutable: bool  # Is this a mutable borrow?
  scope_depth: int  # Scope level where borrow was created


class TypeChecker:
  """Single-pass type checker with scoped symbol tables."""

  # Built-in function signatures
  BUILTINS: dict[str, FunctionSignature] = {
    "print": FunctionSignature(("value",), ("i64",), "i64"),  # print returns 0
  }

  def __init__(self) -> None:
    # Struct definitions: name -> StructInfo
    self.structs: dict[str, StructInfo] = {}
    # Enum definitions: name -> EnumInfo
    self.enums: dict[str, EnumInfo] = {}
    # Struct methods: struct_name -> method_name -> FunctionSignature
    self.struct_methods: dict[str, dict[str, FunctionSignature]] = {}
    # Function signatures: name -> signature
    self.functions: dict[str, FunctionSignature] = dict(self.BUILTINS)
    # Variable scopes: list of (name -> VarState) dicts
    self.scopes: list[dict[str, VarState]] = []
    # Current function's return type (for checking return statements)
    self.current_return_type: str | None = None
    # Current struct type when checking impl methods (for resolving 'Self')
    self.current_impl_type: str | None = None
    # Current scope depth (for tracking borrow lifetimes)
    self.scope_depth: int = 0
    # Active borrows: list of Borrow tracking current borrows
    self.active_borrows: list[Borrow] = []
    # Loop depth (for detecting moves inside loops)
    self.loop_depth: int = 0

  def _enter_scope(self) -> None:
    self.scopes.append({})
    self.scope_depth += 1

  def _exit_scope(self) -> None:
    # End all borrows that were created in this scope
    self.active_borrows = [b for b in self.active_borrows if b.scope_depth < self.scope_depth]
    # Restore ownership for variables that were borrowed in this scope
    for borrow in list(self.active_borrows):
      if borrow.scope_depth == self.scope_depth:
        self._set_var_ownership(borrow.var_name, "owned")
    self.scopes.pop()
    self.scope_depth -= 1

  def _define_var(self, name: str, type_name: str) -> None:
    if name in self.scopes[-1]:
      raise TypeError(f"Variable '{name}' already defined in this scope")
    self.scopes[-1][name] = VarState(type_name, "owned", self.scope_depth)

  def _lookup_var(self, name: str) -> str:
    """Look up variable and return its type string."""
    for scope in reversed(self.scopes):
      if name in scope:
        return scope[name].type_str
    raise TypeError(f"Undefined variable '{name}'")

  def _lookup_var_state(self, name: str) -> VarState | None:
    """Look up variable and return its full state."""
    for scope in reversed(self.scopes):
      if name in scope:
        return scope[name]
    return None

  def _set_var_ownership(self, name: str, ownership: str) -> None:
    """Update the ownership state of a variable."""
    for scope in reversed(self.scopes):
      if name in scope:
        scope[name] = VarState(scope[name].type_str, ownership, scope[name].scope_depth)
        return

  def _check_not_moved(self, name: str) -> None:
    """Check that a variable hasn't been moved."""
    state = self._lookup_var_state(name)
    if state is not None and state.ownership == "moved":
      raise TypeError(f"Use of moved variable '{name}'")

  def _resolve_kwargs(
    self,
    func_name: str,
    sig: FunctionSignature,
    args: tuple[Expr, ...],
    kwargs: tuple[tuple[str, Expr], ...],
  ) -> list[Expr]:
    """Resolve positional and keyword arguments to parameter order.

    Returns a list of expressions in the order of the function's parameters.
    """
    num_params = len(sig.param_names)
    total_args = len(args) + len(kwargs)

    if total_args != num_params:
      raise TypeError(f"Function '{func_name}' expects {num_params} arguments, got {total_args}")

    # Start with positional arguments
    resolved: list[Expr | None] = list(args) + [None] * len(kwargs)

    # Track which parameters have been filled
    filled: set[int] = set(range(len(args)))

    # Process keyword arguments
    for kwarg_name, kwarg_value in kwargs:
      # Find the parameter index
      try:
        param_idx = sig.param_names.index(kwarg_name)
      except ValueError:
        raise TypeError(f"Unknown keyword argument '{kwarg_name}' for function '{func_name}'")

      # Check for duplicate
      if param_idx in filled:
        raise TypeError(f"Duplicate argument for parameter '{kwarg_name}' in call to '{func_name}'")

      resolved[param_idx] = kwarg_value
      filled.add(param_idx)

    # Verify all parameters are filled (should be guaranteed by count check, but be safe)
    for i, expr in enumerate(resolved):
      if expr is None:
        raise TypeError(f"Missing argument for parameter '{sig.param_names[i]}' in call to '{func_name}'")

    return resolved  # type: ignore

  def _maybe_move_var(self, name: str) -> None:
    """Mark a variable as moved if its type is not Copy."""
    state = self._lookup_var_state(name)
    if state is not None and not is_copy_type(state.type_str):
      # Check for move inside loop
      if self.loop_depth > 0:
        raise TypeError(f"Cannot move '{name}' inside a loop")
      self._set_var_ownership(name, "moved")

  def _has_active_borrow(self, name: str) -> bool:
    """Check if variable has any active borrow."""
    return any(b.var_name == name for b in self.active_borrows)

  def _has_mut_borrow(self, name: str) -> bool:
    """Check if variable has an active mutable borrow."""
    return any(b.var_name == name and b.mutable for b in self.active_borrows)

  def _has_shared_borrow(self, name: str) -> bool:
    """Check if variable has an active shared (immutable) borrow."""
    return any(b.var_name == name and not b.mutable for b in self.active_borrows)

  def _create_borrow(self, name: str, mutable: bool) -> None:
    """Create a borrow of a variable, checking for conflicts."""
    state = self._lookup_var_state(name)
    if state is None:
      return  # Variable not found, let _lookup_var handle the error

    # Check for use-after-move
    if state.ownership == "moved":
      raise TypeError(f"Cannot borrow moved variable '{name}'")

    # Check for borrow conflicts
    if mutable:
      # Mutable borrow requires no existing borrows
      if self._has_active_borrow(name):
        if self._has_mut_borrow(name):
          raise TypeError(f"Cannot borrow '{name}' as mutable: already borrowed as mutable")
        else:
          raise TypeError(f"Cannot borrow '{name}' as mutable: already borrowed as immutable")
      self._set_var_ownership(name, "mut_borrowed")
    else:
      # Shared borrow requires no mutable borrows
      if self._has_mut_borrow(name):
        raise TypeError(f"Cannot borrow '{name}' as immutable: already borrowed as mutable")
      self._set_var_ownership(name, "borrowed")

    # Record the borrow
    self.active_borrows.append(Borrow(name, mutable, self.scope_depth))

  def _check_not_borrowed(self, name: str) -> None:
    """Check that a variable is not currently borrowed (for mutation/move)."""
    if self._has_active_borrow(name):
      if self._has_mut_borrow(name):
        raise TypeError(f"Cannot use '{name}': it is currently mutably borrowed")
      else:
        raise TypeError(f"Cannot mutate '{name}': it is currently borrowed")

  def _check_type_ann(self, t: TypeAnnotation) -> str:
    """Verify type annotation is valid and return canonical string."""
    match t:
      case SimpleType(name):
        if name in ("i64", "bool", "str"):
          return name
        # Handle 'Self' type in impl blocks
        if name == "Self" and self.current_impl_type is not None:
          return self.current_impl_type
        if name in self.structs:
          return name  # Struct type
        if name in self.enums:
          return name  # Enum type
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
      case RefType(inner, mutable):
        inner_str = self._check_type_ann(inner)
        prefix = "&mut " if mutable else "&"
        return f"{prefix}{inner_str}"
      case FnType(params, ret):
        param_strs = [self._check_type_ann(p) for p in params]
        ret_str = self._check_type_ann(ret)
        return f"Fn({','.join(param_strs)})->{ret_str}"
      case ResultType(ok_type, err_type):
        ok_str = self._check_type_ann(ok_type)
        err_str = self._check_type_ann(err_type)
        return f"Result[{ok_str},{err_str}]"
      case DictType(key_type, value_type):
        key_str = self._check_type_ann(key_type)
        value_str = self._check_type_ann(value_type)
        # Only allow i64 and str as key types
        if key_str not in ("i64", "str"):
          raise TypeError(f"Dict keys must be i64 or str, got {key_str}")
        return f"dict[{key_str},{value_str}]"
    raise TypeError(f"Unknown type annotation: {t}")

  def check(self, program: Program) -> None:
    """Type check an entire program."""
    # First pass: register all struct definitions
    for struct in program.structs:
      if struct.name in self.structs:
        raise TypeError(f"Struct '{struct.name}' already defined")
      self.structs[struct.name] = StructInfo({})  # Placeholder

    # Register all enum definitions (before checking types)
    for enum in program.enums:
      if enum.name in self.enums:
        raise TypeError(f"Enum '{enum.name}' already defined")
      if enum.name in self.structs:
        raise TypeError(f"'{enum.name}' already defined as a struct")
      self.enums[enum.name] = EnumInfo({})  # Placeholder

    # Second pass: check struct field types (allows recursive/mutual refs)
    for struct in program.structs:
      fields: dict[str, str] = {}
      for field in struct.fields:
        field_type = self._check_type_ann(field.type_ann)
        if field.name in fields:
          raise TypeError(f"Duplicate field '{field.name}' in struct '{struct.name}'")
        fields[field.name] = field_type
      self.structs[struct.name] = StructInfo(fields)

    # Check enum variant types
    for enum in program.enums:
      variants: dict[str, str | None] = {}
      for variant in enum.variants:
        if variant.name in variants:
          raise TypeError(f"Duplicate variant '{variant.name}' in enum '{enum.name}'")
        payload_type: str | None = None
        if variant.payload_type is not None:
          payload_type = self._check_type_ann(variant.payload_type)
        variants[variant.name] = payload_type
      self.enums[enum.name] = EnumInfo(variants)

    # Third pass: register impl block methods
    for impl in program.impls:
      if impl.struct_name not in self.structs:
        raise TypeError(f"Cannot implement methods for unknown type '{impl.struct_name}'")
      if impl.struct_name not in self.struct_methods:
        self.struct_methods[impl.struct_name] = {}

      self.current_impl_type = impl.struct_name
      for method in impl.methods:
        ret_type = self._check_type_ann(method.return_type)
        param_names: list[str] = []
        param_types: list[str] = []
        for param in method.params:
          param_names.append(param.name)
          param_types.append(self._check_type_ann(param.type_ann))

        if method.name in self.struct_methods[impl.struct_name]:
          raise TypeError(f"Method '{method.name}' already defined for '{impl.struct_name}'")

        self.struct_methods[impl.struct_name][method.name] = FunctionSignature(tuple(param_names), tuple(param_types), ret_type)
      self.current_impl_type = None

    # Fourth pass: register all function signatures
    for func in program.functions:
      ret_type = self._check_type_ann(func.return_type)
      param_names: list[str] = []
      param_types: list[str] = []
      for param in func.params:
        param_names.append(param.name)
        param_types.append(self._check_type_ann(param.type_ann))

      if func.name in self.functions:
        raise TypeError(f"Function '{func.name}' already defined")

      self.functions[func.name] = FunctionSignature(tuple(param_names), tuple(param_types), ret_type)

    # Check for main function
    if "main" not in self.functions:
      raise TypeError("No 'main' function defined")

    # Fifth pass: check impl method bodies
    for impl in program.impls:
      self.current_impl_type = impl.struct_name
      for method in impl.methods:
        self._check_function(method)
      self.current_impl_type = None

    # Sixth pass: check function bodies
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

    # Check for implicit return: if last statement is ExprStmt, verify type matches
    if func.body:
      last_stmt = func.body[-1]
      if isinstance(last_stmt, ExprStmt):
        expr_type = self._check_expr(last_stmt.expr)
        # Handle Result type compatibility (partial types with '?')
        if is_result_type(expr_type) and is_result_type(self.current_return_type):
          if not result_types_compatible(expr_type, self.current_return_type):
            raise TypeError(f"Implicit return type {expr_type} doesn't match function return type {self.current_return_type}")
        elif expr_type != self.current_return_type:
          raise TypeError(f"Implicit return type {expr_type} doesn't match function return type {self.current_return_type}")

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
        # Allow empty dict literal to match any dict type
        elif value_type == "dict[?,?]" and is_dict_type(declared_type):
          pass  # OK: empty dict assigned to dict type
        # Handle Result type unification for Ok/Err
        elif is_result_type(declared_type) and value_type.startswith("Result["):
          # Unify partial Result types (Ok or Err with ?)
          declared_ok = get_result_ok_type(declared_type)
          declared_err = get_result_err_type(declared_type)
          value_ok = get_result_ok_type(value_type)
          value_err = get_result_err_type(value_type)
          # Check Ok types match (allow ? as wildcard)
          if value_ok != "?" and value_ok != declared_ok:
            raise TypeError(f"Result Ok type mismatch: expected {declared_ok}, got {value_ok}")
          # Check Err types match (allow ? as wildcard)
          if value_err != "?" and value_err != declared_err:
            raise TypeError(f"Result Err type mismatch: expected {declared_err}, got {value_err}")
        elif value_type != declared_type:
          raise TypeError(f"Cannot assign {value_type} to variable of type {declared_type}")
        # Mark source variable as moved if applicable
        match value:
          case VarExpr(src_name):
            self._maybe_move_var(src_name)
          case _:
            pass
        self._define_var(name, declared_type)

      case AssignStmt(name, value):
        # Check variable isn't borrowed before mutation
        self._check_not_borrowed(name)
        var_type = self._lookup_var(name)
        value_type = self._check_expr(value)
        if value_type != var_type:
          raise TypeError(f"Cannot assign {value_type} to variable of type {var_type}")

      case IndexAssignStmt(target, index, value):
        target_type = self._check_expr(target)
        index_type = self._check_expr(index)

        # Handle dict assignment: d[key] = value
        if is_dict_type(target_type):
          key_type = get_dict_key_type(target_type)
          val_type = get_dict_value_type(target_type)
          if key_type is None or val_type is None:
            raise TypeError(f"Invalid dict type: {target_type}")
          if index_type != key_type:
            raise TypeError(f"Dict key must be {key_type}, got {index_type}")
          value_type = self._check_expr(value)
          if value_type != val_type:
            raise TypeError(f"Dict value must be {val_type}, got {value_type}")
          return

        # Handle array/vec assignment
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
        # Handle Result type unification for Ok/Err returns
        if self.current_return_type and is_result_type(self.current_return_type) and value_type.startswith("Result["):
          declared_ok = get_result_ok_type(self.current_return_type)
          declared_err = get_result_err_type(self.current_return_type)
          value_ok = get_result_ok_type(value_type)
          value_err = get_result_err_type(value_type)
          # Check Ok types match (allow ? as wildcard)
          if value_ok != "?" and value_ok != declared_ok:
            raise TypeError(f"Result Ok type mismatch: expected {declared_ok}, got {value_ok}")
          # Check Err types match (allow ? as wildcard)
          if value_err != "?" and value_err != declared_err:
            raise TypeError(f"Result Err type mismatch: expected {declared_err}, got {value_err}")
        elif value_type != self.current_return_type:
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

        self.loop_depth += 1
        self._enter_scope()
        for s in body:
          self._check_stmt(s)
        self._exit_scope()
        self.loop_depth -= 1

      case ForStmt(var, start, end, body):
        start_type = self._check_expr(start)
        if start_type != "i64":
          raise TypeError(f"For loop start must be i64, got {start_type}")

        end_type = self._check_expr(end)
        if end_type != "i64":
          raise TypeError(f"For loop end must be i64, got {end_type}")

        self.loop_depth += 1
        self._enter_scope()
        self._define_var(var, "i64")
        for s in body:
          self._check_stmt(s)
        self._exit_scope()
        self.loop_depth -= 1

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

      case DerefAssignStmt(target, value):
        # *ptr = value - assign through a mutable reference
        target_type = self._check_expr(target)
        if not is_mut_ref_type(target_type):
          raise TypeError(f"Cannot assign through non-mutable reference '{target_type}'")
        inner_type = get_ref_inner_type(target_type)
        if inner_type is None:
          raise TypeError(f"Invalid reference type '{target_type}'")
        value_type = self._check_expr(value)
        if value_type != inner_type:
          raise TypeError(f"Cannot assign {value_type} to dereferenced {target_type}")

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
        self._check_not_moved(name)
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

        # Handle dict indexing
        if is_dict_type(target_type):
          key_type = get_dict_key_type(target_type)
          value_type = get_dict_value_type(target_type)
          if key_type is None or value_type is None:
            raise TypeError(f"Invalid dict type: {target_type}")
          if index_type != key_type:
            raise TypeError(f"Dict key must be {key_type}, got {index_type}")
          return value_type

        # Handle array/vec indexing
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

      case EnumLiteral(enum_name, variant_name, payload):
        if enum_name not in self.enums:
          raise TypeError(f"Unknown enum '{enum_name}'")
        enum_info = self.enums[enum_name]
        if variant_name not in enum_info.variants:
          raise TypeError(f"Enum '{enum_name}' has no variant '{variant_name}'")
        expected_payload = enum_info.variants[variant_name]
        if expected_payload is None:
          # Unit variant - no payload allowed
          if payload is not None:
            raise TypeError(f"Variant '{variant_name}' does not take a payload")
        else:
          # Payload variant - payload required
          if payload is None:
            raise TypeError(f"Variant '{variant_name}' requires a payload of type {expected_payload}")
          payload_type = self._check_expr(payload)
          if payload_type != expected_payload:
            raise TypeError(f"Variant '{variant_name}' expects {expected_payload}, got {payload_type}")
        return enum_name

      case MatchExpr(target, arms):
        target_type = self._check_expr(target)

        # Handle Result types
        if is_result_type(target_type):
          ok_type = get_result_ok_type(target_type)
          err_type = get_result_err_type(target_type)

          # Check each arm
          covered_variants: set[str] = set()
          for arm in arms:
            if arm.enum_name != "Result":
              raise TypeError(f"Match arm pattern uses wrong type '{arm.enum_name}', expected 'Result'")
            if arm.variant_name not in ("Ok", "Err"):
              raise TypeError(f"Result has no variant '{arm.variant_name}'")
            if arm.variant_name in covered_variants:
              raise TypeError(f"Duplicate match arm for variant '{arm.variant_name}'")
            covered_variants.add(arm.variant_name)

            # Check binding
            self._enter_scope()
            if arm.binding is not None:
              binding_type = ok_type if arm.variant_name == "Ok" else err_type
              if binding_type is not None:
                self._define_var(arm.binding, binding_type)

            # Check arm body
            for stmt in arm.body:
              self._check_stmt(stmt)
            self._exit_scope()

          # Check exhaustiveness
          missing = {"Ok", "Err"} - covered_variants
          if missing:
            raise TypeError(f"Non-exhaustive match: missing variants {', '.join(sorted(missing))}")

          # Match expressions always return i64 (from return statements in arms)
          return "i64"

        # Handle regular enums
        if target_type not in self.enums:
          raise TypeError(f"Cannot match on non-enum type {target_type}")
        enum_info = self.enums[target_type]

        # Check each arm
        covered_variants = set()
        for arm in arms:
          if arm.enum_name != target_type:
            raise TypeError(f"Match arm pattern uses wrong enum '{arm.enum_name}', expected '{target_type}'")
          if arm.variant_name not in enum_info.variants:
            raise TypeError(f"Enum '{target_type}' has no variant '{arm.variant_name}'")
          if arm.variant_name in covered_variants:
            raise TypeError(f"Duplicate match arm for variant '{arm.variant_name}'")
          covered_variants.add(arm.variant_name)

          # Check binding
          variant_payload = enum_info.variants[arm.variant_name]
          self._enter_scope()
          if variant_payload is not None and arm.binding is not None:
            self._define_var(arm.binding, variant_payload)
          elif variant_payload is None and arm.binding is not None:
            raise TypeError(f"Variant '{arm.variant_name}' has no payload to bind")

          # Check arm body - get result type from last statement if it's a return
          for stmt in arm.body:
            self._check_stmt(stmt)
          self._exit_scope()

        # Check exhaustiveness
        missing = set(enum_info.variants.keys()) - covered_variants
        if missing:
          raise TypeError(f"Non-exhaustive match: missing variants {', '.join(sorted(missing))}")

        # Match expressions always return i64 (from return statements in arms)
        return "i64"

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

      case CallExpr(name, args, kwargs):
        # Special case: print accepts i64 or str
        if name == "print":
          total_args = len(args) + len(kwargs)
          if total_args != 1:
            raise TypeError(f"print() expects 1 argument, got {total_args}")
          if kwargs:
            _, arg = kwargs[0]
          else:
            arg = args[0]
          arg_type = self._check_expr(arg)
          if arg_type not in ("i64", "str"):
            raise TypeError(f"print() expects i64 or str, got {arg_type}")
          return "i64"

        # Check if it's a closure variable being called
        var_state = self._lookup_var_state(name)
        if var_state is not None and is_fn_type(var_state.type_str):
          # It's a closure call - closures don't support kwargs for simplicity
          if kwargs:
            raise TypeError(f"Closure '{name}' does not support keyword arguments")
          self._check_not_moved(name)
          parsed = parse_fn_type(var_state.type_str)
          if parsed is None:
            raise TypeError(f"Invalid function type '{var_state.type_str}'")
          param_types, return_type = parsed

          if len(args) != len(param_types):
            raise TypeError(f"Closure '{name}' expects {len(param_types)} arguments, got {len(args)}")

          for i, (arg, expected_type) in enumerate(zip(args, param_types)):
            arg_type = self._check_expr(arg)
            if arg_type != expected_type:
              raise TypeError(f"Argument {i + 1} of closure '{name}' expects {expected_type}, got {arg_type}")

          return return_type

        if name not in self.functions:
          raise TypeError(f"Undefined function '{name}'")

        sig = self.functions[name]

        # Resolve keyword arguments to positional order
        resolved_args = self._resolve_kwargs(name, sig, args, kwargs)

        for i, (arg, expected_type) in enumerate(zip(resolved_args, sig.param_types)):
          arg_type = self._check_expr(arg)
          if arg_type != expected_type:
            param_name = sig.param_names[i]
            raise TypeError(f"Argument '{param_name}' of '{name}' expects {expected_type}, got {arg_type}")
          # Mark variable as moved if passed by value (not a reference type param)
          if not is_ref_type(expected_type):
            match arg:
              case VarExpr(var_name):
                self._maybe_move_var(var_name)
              case _:
                pass

        return sig.return_type

      case ClosureExpr(params, return_type, body):
        # Type check a closure expression
        # Create a new scope for the closure's parameters
        self._enter_scope()

        # Add parameters to scope
        param_types: list[str] = []
        for param in params:
          param_type = self._check_type_ann(param.type_ann)
          param_types.append(param_type)
          self._define_var(param.name, param_type)

        # Type check the body expression
        ret_type = self._check_type_ann(return_type)
        body_type = self._check_expr(body)
        if body_type != ret_type:
          raise TypeError(f"Closure body has type {body_type}, expected {ret_type}")

        self._exit_scope()

        # Return the function type
        return f"Fn({','.join(param_types)})->{ret_type}"

      case ClosureCallExpr(target, args):
        # Call a closure expression (not a named variable)
        target_type = self._check_expr(target)
        if not is_fn_type(target_type):
          raise TypeError(f"Cannot call non-function type '{target_type}'")

        parsed = parse_fn_type(target_type)
        if parsed is None:
          raise TypeError(f"Invalid function type '{target_type}'")
        param_types, return_type = parsed

        if len(args) != len(param_types):
          raise TypeError(f"Closure expects {len(param_types)} arguments, got {len(args)}")

        for i, (arg, expected_type) in enumerate(zip(args, param_types)):
          arg_type = self._check_expr(arg)
          if arg_type != expected_type:
            raise TypeError(f"Argument {i + 1} expects {expected_type}, got {arg_type}")

        return return_type

      case RefExpr(target, mutable):
        # &x or &mut x - create a reference to the target
        target_type = self._check_expr(target)
        # Create the borrow (checks for conflicts)
        match target:
          case VarExpr(name):
            self._create_borrow(name, mutable)
          case _:
            pass  # Complex expressions - limited borrow tracking
        prefix = "&mut " if mutable else "&"
        return f"{prefix}{target_type}"

      case DerefExpr(target):
        # *ptr - dereference a pointer
        target_type = self._check_expr(target)
        if not is_ref_type(target_type):
          raise TypeError(f"Cannot dereference non-reference type '{target_type}'")
        inner_type = get_ref_inner_type(target_type)
        if inner_type is None:
          raise TypeError(f"Invalid reference type '{target_type}'")
        return inner_type

      case OkExpr(value):
        # Ok(value) - creates a Result, but we need context to know the Err type
        # For now, we mark this as a partial Result that will be unified with context
        value_type = self._check_expr(value)
        # Return a marker type that will be unified with expected Result type
        return f"Result[{value_type},?]"

      case ErrExpr(value):
        # Err(error) - creates a Result, but we need context to know the Ok type
        value_type = self._check_expr(value)
        return f"Result[?,{value_type}]"

      case TryExpr(target):
        # expr? - unwraps Ok or returns early with Err
        target_type = self._check_expr(target)
        if not is_result_type(target_type):
          raise TypeError(f"The '?' operator can only be used on Result types, got {target_type}")
        # Verify current function returns a compatible Result type
        if self.current_return_type is None:
          raise TypeError("The '?' operator can only be used inside a function")
        if not is_result_type(self.current_return_type):
          raise TypeError(f"The '?' operator requires function to return Result, but returns {self.current_return_type}")
        # Check error types are compatible
        expr_err_type = get_result_err_type(target_type)
        func_err_type = get_result_err_type(self.current_return_type)
        if expr_err_type != func_err_type:
          raise TypeError(f"Error type mismatch: expression has {expr_err_type}, function returns {func_err_type}")
        # Return the Ok type (unwrapped)
        ok_type = get_result_ok_type(target_type)
        if ok_type is None:
          raise TypeError(f"Invalid Result type: {target_type}")
        return ok_type

      case ListComprehension(element_expr, var_name, start, end, condition):
        # [expr for var in range(start, end) if condition]
        # Check range bounds
        start_type = self._check_expr(start)
        end_type = self._check_expr(end)
        if start_type != "i64":
          raise TypeError(f"Range start must be i64, got {start_type}")
        if end_type != "i64":
          raise TypeError(f"Range end must be i64, got {end_type}")

        # Create scope with loop variable
        self._enter_scope()
        self._define_var(var_name, "i64")

        # Check element expression
        elem_type = self._check_expr(element_expr)

        # Check condition if present
        if condition is not None:
          cond_type = self._check_expr(condition)
          if cond_type != "bool":
            raise TypeError(f"List comprehension condition must be bool, got {cond_type}")

        self._exit_scope()
        return f"vec[{elem_type}]"

      case DictLiteral(entries):
        # {key: value, ...}
        if not entries:
          # Empty dict - type will be inferred from context
          return "dict[?,?]"

        # Check all entries have consistent types
        first_key_type = self._check_expr(entries[0][0])
        first_value_type = self._check_expr(entries[0][1])

        # Validate key type is hashable (i64 or str)
        if first_key_type not in ("i64", "str"):
          raise TypeError(f"Dict keys must be i64 or str, got {first_key_type}")

        for key, value in entries[1:]:
          key_type = self._check_expr(key)
          value_type = self._check_expr(value)
          if key_type != first_key_type:
            raise TypeError(f"Dict key type mismatch: expected {first_key_type}, got {key_type}")
          if value_type != first_value_type:
            raise TypeError(f"Dict value type mismatch: expected {first_value_type}, got {value_type}")

        return f"dict[{first_key_type},{first_value_type}]"

      case DictComprehension(key_expr, value_expr, var_name, start, end, condition):
        # {k: v for var in range(start, end) if condition}
        # Check range bounds
        start_type = self._check_expr(start)
        end_type = self._check_expr(end)
        if start_type != "i64":
          raise TypeError(f"Range start must be i64, got {start_type}")
        if end_type != "i64":
          raise TypeError(f"Range end must be i64, got {end_type}")

        # Create scope with loop variable
        self._enter_scope()
        self._define_var(var_name, "i64")

        # Check key expression
        key_type = self._check_expr(key_expr)
        if key_type not in ("i64", "str"):
          raise TypeError(f"Dict keys must be i64 or str, got {key_type}")

        # Check value expression
        value_type = self._check_expr(value_expr)

        # Check condition if present
        if condition is not None:
          cond_type = self._check_expr(condition)
          if cond_type != "bool":
            raise TypeError(f"Dict comprehension condition must be bool, got {cond_type}")

        self._exit_scope()
        return f"dict[{key_type},{value_type}]"

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

      # Functional iterator methods
      elif method == "into_iter":
        if len(args) != 0:
          raise TypeError(f"into_iter() expects 0 arguments, got {len(args)}")
        return target_type  # Returns same vec type (eager evaluation)

      elif method == "iter":
        if len(args) != 0:
          raise TypeError(f"iter() expects 0 arguments, got {len(args)}")
        return target_type  # Returns same vec type (eager evaluation)

      elif method == "skip":
        if len(args) != 1:
          raise TypeError(f"skip() expects 1 argument, got {len(args)}")
        arg_type = self._check_expr(args[0])
        if arg_type != "i64":
          raise TypeError(f"skip() expects i64, got {arg_type}")
        return target_type  # Returns new vec of same type

      elif method == "take":
        if len(args) != 1:
          raise TypeError(f"take() expects 1 argument, got {len(args)}")
        arg_type = self._check_expr(args[0])
        if arg_type != "i64":
          raise TypeError(f"take() expects i64, got {arg_type}")
        return target_type  # Returns new vec of same type

      elif method == "map":
        if len(args) != 1:
          raise TypeError(f"map() expects 1 argument (closure), got {len(args)}")
        closure_type = self._check_expr(args[0])
        if not is_fn_type(closure_type):
          raise TypeError(f"map() expects a closure, got {closure_type}")
        parsed = parse_fn_type(closure_type)
        if parsed is None:
          raise TypeError(f"Invalid closure type '{closure_type}'")
        param_types, return_type = parsed
        if len(param_types) != 1:
          raise TypeError(f"map() closure must take 1 argument, got {len(param_types)}")
        if param_types[0] != elem_type:
          raise TypeError(f"map() closure expects {elem_type}, got {param_types[0]}")
        return f"vec[{return_type}]"  # Returns vec of closure return type

      elif method == "filter":
        if len(args) != 1:
          raise TypeError(f"filter() expects 1 argument (closure), got {len(args)}")
        closure_type = self._check_expr(args[0])
        if not is_fn_type(closure_type):
          raise TypeError(f"filter() expects a closure, got {closure_type}")
        parsed = parse_fn_type(closure_type)
        if parsed is None:
          raise TypeError(f"Invalid closure type '{closure_type}'")
        param_types, return_type = parsed
        if len(param_types) != 1:
          raise TypeError(f"filter() closure must take 1 argument, got {len(param_types)}")
        if param_types[0] != elem_type:
          raise TypeError(f"filter() closure expects {elem_type}, got {param_types[0]}")
        if return_type != "bool":
          raise TypeError(f"filter() closure must return bool, got {return_type}")
        return target_type  # Returns vec of same type

      elif method == "collect":
        if len(args) != 0:
          raise TypeError(f"collect() expects 0 arguments, got {len(args)}")
        return target_type  # Returns the vec (no-op for eager evaluation)

      elif method == "sum":
        if len(args) != 0:
          raise TypeError(f"sum() expects 0 arguments, got {len(args)}")
        if elem_type != "i64":
          raise TypeError(f"sum() requires vec[i64], got {target_type}")
        return "i64"

      elif method == "fold":
        if len(args) != 2:
          raise TypeError(f"fold() expects 2 arguments (init, closure), got {len(args)}")
        init_type = self._check_expr(args[0])
        closure_type = self._check_expr(args[1])
        if not is_fn_type(closure_type):
          raise TypeError(f"fold() expects a closure as second argument, got {closure_type}")
        parsed = parse_fn_type(closure_type)
        if parsed is None:
          raise TypeError(f"Invalid closure type '{closure_type}'")
        param_types, return_type = parsed
        if len(param_types) != 2:
          raise TypeError(f"fold() closure must take 2 arguments (acc, elem), got {len(param_types)}")
        if param_types[0] != init_type:
          raise TypeError(f"fold() closure accumulator type {param_types[0]} doesn't match init type {init_type}")
        if param_types[1] != elem_type:
          raise TypeError(f"fold() closure element type {param_types[1]} doesn't match vec element type {elem_type}")
        if return_type != init_type:
          raise TypeError(f"fold() closure return type {return_type} doesn't match accumulator type {init_type}")
        return init_type

      else:
        raise TypeError(f"Unknown vec method '{method}'")

    elif is_array_type(target_type):
      if method == "len":
        if len(args) != 0:
          raise TypeError(f"len() expects 0 arguments, got {len(args)}")
        return "i64"
      else:
        raise TypeError(f"Unknown array method '{method}'")

    elif is_dict_type(target_type):
      key_type = get_dict_key_type(target_type)
      value_type = get_dict_value_type(target_type)
      assert key_type is not None and value_type is not None

      if method == "len":
        if len(args) != 0:
          raise TypeError(f"len() expects 0 arguments, got {len(args)}")
        return "i64"

      elif method == "contains":
        if len(args) != 1:
          raise TypeError(f"contains() expects 1 argument, got {len(args)}")
        arg_type = self._check_expr(args[0])
        if arg_type != key_type:
          raise TypeError(f"contains() expects key of type {key_type}, got {arg_type}")
        return "bool"

      elif method == "get":
        if len(args) != 1:
          raise TypeError(f"get() expects 1 argument, got {len(args)}")
        arg_type = self._check_expr(args[0])
        if arg_type != key_type:
          raise TypeError(f"get() expects key of type {key_type}, got {arg_type}")
        return value_type

      elif method == "insert":
        if len(args) != 2:
          raise TypeError(f"insert() expects 2 arguments, got {len(args)}")
        k_type = self._check_expr(args[0])
        v_type = self._check_expr(args[1])
        if k_type != key_type:
          raise TypeError(f"insert() key must be {key_type}, got {k_type}")
        if v_type != value_type:
          raise TypeError(f"insert() value must be {value_type}, got {v_type}")
        return "i64"  # Returns nothing useful

      elif method == "remove":
        if len(args) != 1:
          raise TypeError(f"remove() expects 1 argument, got {len(args)}")
        arg_type = self._check_expr(args[0])
        if arg_type != key_type:
          raise TypeError(f"remove() expects key of type {key_type}, got {arg_type}")
        return "bool"  # Returns whether key existed

      else:
        raise TypeError(f"Unknown dict method '{method}'")

    elif target_type in self.struct_methods:
      # Struct method call
      methods = self.struct_methods[target_type]
      if method not in methods:
        raise TypeError(f"Struct '{target_type}' has no method '{method}'")
      sig = methods[method]
      # First parameter should be 'self' (the struct type)
      if not sig.param_types or sig.param_types[0] != target_type:
        raise TypeError(f"Method '{method}' does not take self parameter")
      # Check remaining arguments (skip self)
      expected_params = sig.param_types[1:]
      if len(args) != len(expected_params):
        raise TypeError(f"Method '{method}' expects {len(expected_params)} arguments, got {len(args)}")
      for i, (arg, expected_type) in enumerate(zip(args, expected_params)):
        arg_type = self._check_expr(arg)
        if arg_type != expected_type:
          raise TypeError(f"Argument {i + 1} of '{method}' expects {expected_type}, got {arg_type}")
      return sig.return_type

    else:
      raise TypeError(f"Cannot call method on type {target_type}")


def check(program: Program) -> None:
  """Convenience function to type check a program."""
  TypeChecker().check(program)
