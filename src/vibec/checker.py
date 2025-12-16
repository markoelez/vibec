"""Type checker for the Vibec language."""

from dataclasses import dataclass

from .ast import (
  Expr,
  Stmt,
  FnType,
  IfStmt,
  OkExpr,
  EnumDef,
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
  MatchArm,
  ArrayType,
  DerefExpr,
  ImplBlock,
  IndexExpr,
  MatchExpr,
  SliceExpr,
  StructDef,
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
  GenericType,
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
    case GenericType(name, type_args):
      args_str = ",".join(type_to_str(a) for a in type_args)
      return f"{name}<{args_str}>"
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
  mutable: bool = True  # Can the variable be reassigned?


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
    # Type aliases: alias_name -> resolved_type_string
    self.type_aliases: dict[str, str] = {}
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
    # Generic struct definitions: name -> list of type param names
    self.generic_structs: dict[str, list[str]] = {}
    # Generic enum definitions: name -> list of type param names
    self.generic_enums: dict[str, list[str]] = {}
    # Generic function definitions: name -> list of type param names
    self.generic_functions: dict[str, list[str]] = {}
    # Monomorphized (instantiated) types we've seen: mangled_name -> original_name
    self.instantiated_types: set[str] = set()
    # Current type parameters in scope (for checking generic function/struct/enum bodies)
    self.current_type_params: set[str] = set()
    # Generic struct definitions (original AST): name -> StructDef
    self.generic_struct_defs: dict[str, "StructDef"] = {}
    # Generic enum definitions (original AST): name -> EnumDef
    self.generic_enum_defs: dict[str, "EnumDef"] = {}
    # Generic function definitions (original AST): name -> Function
    self.generic_function_defs: dict[str, "Function"] = {}
    # Generic impl block definitions (original AST): struct_name -> ImplBlock
    self.generic_impl_defs: dict[str, "ImplBlock"] = {}
    # Instantiated generic functions: mangled_name -> InstantiatedFunction
    self.instantiated_functions: dict[str, "InstantiatedFunction"] = {}
    # Instantiated generic methods: mangled_name -> InstantiatedMethod
    self.instantiated_methods: dict[str, "InstantiatedMethod"] = {}
    # Current type substitution (for checking generic function/method bodies)
    self.current_type_subst: dict[str, str] = {}
    # Inferred generic calls: id(CallExpr) -> mangled_name (for AST transformation)
    self.inferred_calls: dict[int, str] = {}

  def _resolve_generic_struct(self, name: str, type_args: tuple[TypeAnnotation, ...]) -> str:
    """Resolve a generic struct instantiation to its monomorphized name."""
    if not type_args:
      return name
    # For generic structs, mangle the name: Pair<i64, str> -> Pair<i64,str>
    # Apply type substitution if we're in a generic context
    resolved_args: list[str] = []
    for arg in type_args:
      arg_str = type_to_str(arg)
      if arg_str in self.current_type_subst:
        arg_str = self.current_type_subst[arg_str]
      resolved_args.append(arg_str)
    return f"{name}<{','.join(resolved_args)}>"

  def _resolve_generic_enum(self, name: str, type_args: tuple[TypeAnnotation, ...]) -> str:
    """Resolve a generic enum instantiation to its monomorphized name."""
    if not type_args:
      return name
    # For generic enums, mangle the name: Option<i64> -> Option<i64>
    # Apply type substitution if we're in a generic context
    resolved_args: list[str] = []
    for arg in type_args:
      arg_str = type_to_str(arg)
      if arg_str in self.current_type_subst:
        arg_str = self.current_type_subst[arg_str]
      resolved_args.append(arg_str)
    return f"{name}<{','.join(resolved_args)}>"

  def _ensure_generic_struct_instantiated(self, name: str, type_args: list[str]) -> None:
    """Ensure a generic struct has been instantiated with the given type args."""
    mangled = f"{name}<{','.join(type_args)}>"
    if mangled in self.instantiated_types:
      return  # Already instantiated

    # Get the generic struct definition
    if name not in self.generic_struct_defs:
      raise TypeError(f"Internal error: no definition for generic struct '{name}'")

    struct_def = self.generic_struct_defs[name]
    type_params = self.generic_structs[name]

    # Create a substitution map: T -> i64, U -> str, etc.
    subst = dict(zip(type_params, type_args))

    # Instantiate the struct with substituted types
    fields: dict[str, str] = {}
    old_type_params = self.current_type_params
    self.current_type_params = set()  # No type params when instantiating
    for field in struct_def.fields:
      field_type_str = self._substitute_type(field.type_ann, subst)
      fields[field.name] = field_type_str
    self.current_type_params = old_type_params

    # Register the instantiated struct
    self.structs[mangled] = StructInfo(fields)
    self.instantiated_types.add(mangled)

  def _ensure_generic_enum_instantiated(self, name: str, type_args: list[str]) -> None:
    """Ensure a generic enum has been instantiated with the given type args."""
    mangled = f"{name}<{','.join(type_args)}>"
    if mangled in self.instantiated_types:
      return  # Already instantiated

    # Get the generic enum definition
    if name not in self.generic_enum_defs:
      raise TypeError(f"Internal error: no definition for generic enum '{name}'")

    enum_def = self.generic_enum_defs[name]
    type_params = self.generic_enums[name]

    # Create a substitution map: T -> i64, etc.
    subst = dict(zip(type_params, type_args))

    # Instantiate the enum with substituted types
    variants: dict[str, str | None] = {}
    old_type_params = self.current_type_params
    self.current_type_params = set()  # No type params when instantiating
    for variant in enum_def.variants:
      if variant.payload_type is not None:
        payload_type_str = self._substitute_type(variant.payload_type, subst)
        variants[variant.name] = payload_type_str
      else:
        variants[variant.name] = None
    self.current_type_params = old_type_params

    # Register the instantiated enum
    self.enums[mangled] = EnumInfo(variants)
    self.instantiated_types.add(mangled)

  def _ensure_generic_function_instantiated(self, name: str, type_args: list[str]) -> FunctionSignature:
    """Ensure a generic function has been instantiated with the given type args.

    Returns the function signature for the instantiated function.
    """
    mangled = f"{name}<{','.join(type_args)}>"
    if mangled in self.instantiated_functions:
      inst = self.instantiated_functions[mangled]
      return FunctionSignature(
        tuple(p.name for p in inst.original_func.params),
        tuple(inst.param_types),
        inst.return_type,
      )

    # Get the generic function definition
    if name not in self.generic_function_defs:
      raise TypeError(f"Internal error: no definition for generic function '{name}'")

    func_def = self.generic_function_defs[name]
    type_params = self.generic_functions[name]

    if len(type_args) != len(type_params):
      raise TypeError(f"Generic function '{name}' expects {len(type_params)} type args, got {len(type_args)}")

    # Create a substitution map: T -> i64, U -> str, etc.
    subst = dict(zip(type_params, type_args))

    # Instantiate parameter types
    param_types: list[str] = []
    old_type_params = self.current_type_params
    self.current_type_params = set()  # No type params when instantiating
    for param in func_def.params:
      param_types.append(self._substitute_type(param.type_ann, subst))

    # Instantiate return type
    return_type = self._substitute_type(func_def.return_type, subst)
    self.current_type_params = old_type_params

    # Create and store the instantiated function
    inst_func = InstantiatedFunction(
      mangled_name=mangled,
      original_func=func_def,
      type_subst=subst,
      param_types=param_types,
      return_type=return_type,
    )
    self.instantiated_functions[mangled] = inst_func

    # Also register the signature so it can be found by normal lookup
    param_names = tuple(p.name for p in func_def.params)
    sig = FunctionSignature(param_names, tuple(param_types), return_type)
    self.functions[mangled] = sig

    # Type check the function body with the substituted types
    self._check_instantiated_function(func_def, subst)

    return sig

  def _ensure_generic_method_instantiated(self, struct_name: str, struct_type_args: list[str], method_name: str) -> FunctionSignature:
    """Ensure a generic method has been instantiated for a specific struct instance.

    Returns the method signature for the instantiated method.
    """
    struct_mangled = f"{struct_name}<{','.join(struct_type_args)}>"
    method_mangled = f"{struct_mangled}_{method_name}"

    if method_mangled in self.instantiated_methods:
      inst = self.instantiated_methods[method_mangled]
      return FunctionSignature(
        tuple(p.name for p in inst.original_method.params),
        tuple(inst.param_types),
        inst.return_type,
      )

    # Get the generic impl block definition
    if struct_name not in self.generic_impl_defs:
      raise TypeError(f"No methods defined for generic struct '{struct_name}'")

    impl_def = self.generic_impl_defs[struct_name]
    type_params = self.generic_structs.get(struct_name, impl_def.type_params)

    if len(struct_type_args) != len(type_params):
      raise TypeError(f"Generic struct '{struct_name}' expects {len(type_params)} type args, got {len(struct_type_args)}")

    # Find the method in the impl block
    method_def: Function | None = None
    for method in impl_def.methods:
      if method.name == method_name:
        method_def = method
        break

    if method_def is None:
      raise TypeError(f"Method '{method_name}' not found for struct '{struct_name}'")

    # Create a substitution map: T -> i64, U -> str, etc.
    subst = dict(zip(type_params, struct_type_args))

    # Instantiate parameter types
    param_types: list[str] = []
    old_type_params = self.current_type_params
    old_impl_type = self.current_impl_type
    self.current_type_params = set()  # No type params when instantiating
    self.current_impl_type = struct_mangled  # For resolving Self

    for param in method_def.params:
      param_types.append(self._substitute_type(param.type_ann, subst))

    # Instantiate return type
    return_type = self._substitute_type(method_def.return_type, subst)

    self.current_type_params = old_type_params
    self.current_impl_type = old_impl_type

    # Create and store the instantiated method
    inst_method = InstantiatedMethod(
      mangled_name=method_mangled,
      struct_mangled_name=struct_mangled,
      original_method=method_def,
      type_subst=subst,
      param_types=param_types,
      return_type=return_type,
    )
    self.instantiated_methods[method_mangled] = inst_method

    # Register the method signature for this instantiated struct
    if struct_mangled not in self.struct_methods:
      self.struct_methods[struct_mangled] = {}

    param_names = tuple(p.name for p in method_def.params)
    sig = FunctionSignature(param_names, tuple(param_types), return_type)
    self.struct_methods[struct_mangled][method_name] = sig

    # Type check the method body with the substituted types
    self._check_instantiated_method(method_def, subst, struct_mangled)

    return sig

  def _check_instantiated_function(self, func: "Function", subst: dict[str, str]) -> None:
    """Type check an instantiated generic function body."""
    # Save current state
    old_return_type = self.current_return_type
    old_type_subst = self.current_type_subst

    self._enter_scope()
    self.current_return_type = self._substitute_type(func.return_type, subst)
    self.current_type_subst = subst

    # Add parameters to scope with substituted types
    for param in func.params:
      param_type = self._substitute_type(param.type_ann, subst)
      self._define_var(param.name, param_type)

    # Check statements
    for stmt in func.body:
      self._check_stmt(stmt)

    # Check for implicit return
    if func.body:
      last_stmt = func.body[-1]
      if isinstance(last_stmt, ExprStmt):
        expr_type = self._check_expr(last_stmt.expr)
        if is_result_type(expr_type) and is_result_type(self.current_return_type):
          if not result_types_compatible(expr_type, self.current_return_type):
            raise TypeError(f"Implicit return type {expr_type} doesn't match function return type {self.current_return_type}")
        elif expr_type != self.current_return_type:
          raise TypeError(f"Implicit return type {expr_type} doesn't match function return type {self.current_return_type}")

    self._exit_scope()

    # Restore state
    self.current_return_type = old_return_type
    self.current_type_subst = old_type_subst

  def _check_instantiated_method(self, method: "Function", subst: dict[str, str], struct_mangled: str) -> None:
    """Type check an instantiated generic method body."""
    # Save current state
    old_impl_type = self.current_impl_type
    old_return_type = self.current_return_type
    old_type_subst = self.current_type_subst

    self.current_impl_type = struct_mangled
    self.current_type_subst = subst

    self._enter_scope()
    self.current_return_type = self._substitute_type(method.return_type, subst)

    # Add parameters to scope with substituted types
    for param in method.params:
      param_type = self._substitute_type(param.type_ann, subst)
      self._define_var(param.name, param_type)

    # Check statements
    for stmt in method.body:
      self._check_stmt(stmt)

    # Check for implicit return
    if method.body:
      last_stmt = method.body[-1]
      if isinstance(last_stmt, ExprStmt):
        expr_type = self._check_expr(last_stmt.expr)
        if is_result_type(expr_type) and is_result_type(self.current_return_type):
          if not result_types_compatible(expr_type, self.current_return_type):
            raise TypeError(f"Implicit return type {expr_type} doesn't match function return type {self.current_return_type}")
        elif expr_type != self.current_return_type:
          raise TypeError(f"Implicit return type {expr_type} doesn't match function return type {self.current_return_type}")

    self._exit_scope()

    # Restore state
    self.current_impl_type = old_impl_type
    self.current_return_type = old_return_type
    self.current_type_subst = old_type_subst

  def _infer_type_args(self, type_params: list[str], param_types: list[TypeAnnotation], arg_types: list[str]) -> dict[str, str]:
    """Infer type arguments from actual argument types.

    Returns a mapping from type parameter names to concrete types.
    """
    inferred: dict[str, str] = {}

    for param_type, arg_type in zip(param_types, arg_types):
      self._unify_types(param_type, arg_type, inferred, type_params)

    # Check all type params were inferred
    for param in type_params:
      if param not in inferred:
        raise TypeError(f"Cannot infer type for type parameter '{param}'")

    return inferred

  def _unify_types(self, param_type: TypeAnnotation, arg_type: str, inferred: dict[str, str], type_params: list[str]) -> None:
    """Unify a parameter type with an argument type to infer type parameters."""
    match param_type:
      case SimpleType(name):
        if name in type_params:
          # This is a type parameter - infer it
          if name in inferred:
            if inferred[name] != arg_type:
              raise TypeError(f"Conflicting types for type parameter '{name}': {inferred[name]} vs {arg_type}")
          else:
            inferred[name] = arg_type
        # Otherwise it's a concrete type - no inference needed
      case ArrayType(elem, _):
        # Extract element type from arg_type string like "[i64;5]"
        if arg_type.startswith("[") and ";" in arg_type:
          elem_str = arg_type[1 : arg_type.index(";")]
          self._unify_types(elem, elem_str, inferred, type_params)
      case VecType(elem):
        # Extract element type from arg_type string like "vec[i64]"
        if arg_type.startswith("vec[") and arg_type.endswith("]"):
          elem_str = arg_type[4:-1]
          self._unify_types(elem, elem_str, inferred, type_params)
      case TupleType(elems):
        # Extract element types from arg_type string like "(i64,str)"
        if arg_type.startswith("(") and arg_type.endswith(")"):
          inner = arg_type[1:-1]
          if inner:
            arg_elem_strs = self._split_type_args(inner)
            for param_elem, arg_elem in zip(elems, arg_elem_strs):
              self._unify_types(param_elem, arg_elem, inferred, type_params)
      case RefType(inner, _):
        # Extract inner type from arg_type string like "&i64" or "&mut i64"
        if arg_type.startswith("&mut "):
          inner_str = arg_type[5:]
        elif arg_type.startswith("&"):
          inner_str = arg_type[1:]
        else:
          return
        self._unify_types(inner, inner_str, inferred, type_params)
      case GenericType(name, type_args):
        # Extract type args from arg_type string like "Box<i64>"
        if "<" in arg_type:
          idx = arg_type.index("<")
          type_name = arg_type[:idx]
          if type_name == name:
            args_str = arg_type[idx + 1 : -1]
            arg_strs = self._split_type_args(args_str)
            for param_arg, arg_str in zip(type_args, arg_strs):
              self._unify_types(param_arg, arg_str, inferred, type_params)
      case ResultType(ok_type, err_type):
        # Extract ok/err types from arg_type string like "Result[i64,str]"
        if arg_type.startswith("Result[") and arg_type.endswith("]"):
          inner = arg_type[7:-1]
          parts = self._split_type_args(inner)
          if len(parts) == 2:
            self._unify_types(ok_type, parts[0], inferred, type_params)
            self._unify_types(err_type, parts[1], inferred, type_params)
      case DictType(key_type, value_type):
        # Extract key/value types from arg_type string like "dict[i64,str]"
        if arg_type.startswith("dict[") and arg_type.endswith("]"):
          inner = arg_type[5:-1]
          parts = self._split_type_args(inner)
          if len(parts) == 2:
            self._unify_types(key_type, parts[0], inferred, type_params)
            self._unify_types(value_type, parts[1], inferred, type_params)
      case FnType(params, ret):
        # Extract param/return types from arg_type string like "Fn(i64,str)->bool"
        if arg_type.startswith("Fn("):
          arrow_idx = arg_type.find(")->")
          if arrow_idx != -1:
            params_str = arg_type[3:arrow_idx]
            ret_str = arg_type[arrow_idx + 3 :]
            param_strs = self._split_type_args(params_str) if params_str else []
            for param_type_ann, param_str in zip(params, param_strs):
              self._unify_types(param_type_ann, param_str, inferred, type_params)
            self._unify_types(ret, ret_str, inferred, type_params)

  def _split_type_args(self, s: str) -> list[str]:
    """Split a comma-separated type string, respecting nested brackets."""
    result: list[str] = []
    depth = 0
    current = ""
    for c in s:
      if c in "<[(":
        depth += 1
        current += c
      elif c in ">])":
        depth -= 1
        current += c
      elif c == "," and depth == 0:
        result.append(current.strip())
        current = ""
      else:
        current += c
    if current.strip():
      result.append(current.strip())
    return result

  def _substitute_type(self, t: TypeAnnotation, subst: dict[str, str]) -> str:
    """Substitute type parameters with concrete types in a type annotation."""
    match t:
      case SimpleType(name):
        if name in subst:
          return subst[name]
        return self._check_type_ann(t)
      case ArrayType(elem, size):
        elem_str = self._substitute_type(elem, subst)
        return f"[{elem_str};{size}]"
      case VecType(elem):
        elem_str = self._substitute_type(elem, subst)
        return f"vec[{elem_str}]"
      case TupleType(elems):
        elem_strs = [self._substitute_type(e, subst) for e in elems]
        return f"({','.join(elem_strs)})"
      case RefType(inner, mutable):
        inner_str = self._substitute_type(inner, subst)
        prefix = "&mut " if mutable else "&"
        return f"{prefix}{inner_str}"
      case FnType(params, ret):
        param_strs = [self._substitute_type(p, subst) for p in params]
        ret_str = self._substitute_type(ret, subst)
        return f"Fn({','.join(param_strs)})->{ret_str}"
      case ResultType(ok_type, err_type):
        ok_str = self._substitute_type(ok_type, subst)
        err_str = self._substitute_type(err_type, subst)
        return f"Result[{ok_str},{err_str}]"
      case DictType(key_type, value_type):
        key_str = self._substitute_type(key_type, subst)
        value_str = self._substitute_type(value_type, subst)
        return f"dict[{key_str},{value_str}]"
      case GenericType(name, type_args):
        # Substitute in the type args
        arg_strs = [self._substitute_type(arg, subst) for arg in type_args]
        mangled = f"{name}<{','.join(arg_strs)}>"
        # Ensure the instantiation exists
        if name in self.generic_structs:
          self._ensure_generic_struct_instantiated(name, arg_strs)
        elif name in self.generic_enums:
          self._ensure_generic_enum_instantiated(name, arg_strs)
        return mangled
    raise TypeError(f"Unknown type annotation for substitution: {t}")

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

  def _define_var(self, name: str, type_name: str, mutable: bool = True) -> None:
    if name in self.scopes[-1]:
      raise TypeError(f"Variable '{name}' already defined in this scope")
    self.scopes[-1][name] = VarState(type_name, "owned", self.scope_depth, mutable)

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

  def _check_mutable(self, name: str) -> None:
    """Check that a variable is mutable (not const)."""
    state = self._lookup_var_state(name)
    if state is not None and not state.mutable:
      raise TypeError(f"Cannot assign to const variable '{name}'")

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
        # Check if it's a type parameter in current generic context
        if name in self.current_type_params:
          return name  # Type parameter used in generic context
        # Check type aliases first (returns the resolved type)
        if name in self.type_aliases:
          return self.type_aliases[name]
        if name in self.structs:
          return name  # Struct type
        if name in self.enums:
          return name  # Enum type
        # Check if it's a generic struct/enum without type args (error)
        if name in self.generic_structs:
          params = self.generic_structs[name]
          raise TypeError(f"Generic struct '{name}' requires type arguments: {name}<{', '.join(params)}>")
        if name in self.generic_enums:
          params = self.generic_enums[name]
          raise TypeError(f"Generic enum '{name}' requires type arguments: {name}<{', '.join(params)}>")
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
      case GenericType(name, type_args):
        # Handle generic type instantiation like Pair<i64, str>
        if name in self.generic_structs:
          expected_params = self.generic_structs[name]
          if len(type_args) != len(expected_params):
            raise TypeError(f"Generic struct '{name}' expects {len(expected_params)} type args, got {len(type_args)}")
          # Check all type args are valid
          arg_strs = [self._check_type_ann(arg) for arg in type_args]
          # Monomorphize this instantiation
          mangled = f"{name}<{','.join(arg_strs)}>"
          self._ensure_generic_struct_instantiated(name, arg_strs)
          return mangled
        elif name in self.generic_enums:
          expected_params = self.generic_enums[name]
          if len(type_args) != len(expected_params):
            raise TypeError(f"Generic enum '{name}' expects {len(expected_params)} type args, got {len(type_args)}")
          # Check all type args are valid
          arg_strs = [self._check_type_ann(arg) for arg in type_args]
          # Monomorphize this instantiation
          mangled = f"{name}<{','.join(arg_strs)}>"
          self._ensure_generic_enum_instantiated(name, arg_strs)
          return mangled
        else:
          raise TypeError(f"'{name}' is not a generic type")
    raise TypeError(f"Unknown type annotation: {t}")

  def check(self, program: Program) -> None:
    """Type check an entire program."""
    # First: register struct and enum placeholders (so type aliases can reference them)
    # Generic types are stored separately and only instantiated when used
    for struct in program.structs:
      if struct.name in self.structs or struct.name in self.generic_structs:
        raise TypeError(f"Struct '{struct.name}' already defined")
      if struct.type_params:
        # Generic struct - store for later instantiation
        self.generic_structs[struct.name] = list(struct.type_params)
        self.generic_struct_defs[struct.name] = struct
      else:
        # Non-generic struct - register placeholder
        self.structs[struct.name] = StructInfo({})

    for enum in program.enums:
      if enum.name in self.enums or enum.name in self.generic_enums:
        raise TypeError(f"Enum '{enum.name}' already defined")
      if enum.name in self.structs or enum.name in self.generic_structs:
        raise TypeError(f"'{enum.name}' already defined as a struct")
      if enum.type_params:
        # Generic enum - store for later instantiation
        self.generic_enums[enum.name] = list(enum.type_params)
        self.generic_enum_defs[enum.name] = enum
      else:
        # Non-generic enum - register placeholder
        self.enums[enum.name] = EnumInfo({})

    # Register type aliases (can now reference structs/enums)
    for alias in program.type_aliases:
      if alias.name in self.type_aliases:
        raise TypeError(f"Type alias '{alias.name}' already defined")
      if alias.name in ("i64", "bool", "str"):
        raise TypeError(f"Cannot create type alias for built-in type '{alias.name}'")
      if alias.name in self.structs or alias.name in self.generic_structs:
        raise TypeError(f"'{alias.name}' already defined as a struct")
      if alias.name in self.enums or alias.name in self.generic_enums:
        raise TypeError(f"'{alias.name}' already defined as an enum")
      # Resolve the target type (may use other aliases, structs, or enums)
      resolved_type = self._check_type_ann(alias.target)
      self.type_aliases[alias.name] = resolved_type

    # Second pass: check struct field types (allows recursive/mutual refs)
    # Only for non-generic structs; generic structs are checked when instantiated
    for struct in program.structs:
      if struct.type_params:
        continue  # Skip generic structs - they're checked during instantiation
      fields: dict[str, str] = {}
      for field in struct.fields:
        field_type = self._check_type_ann(field.type_ann)
        if field.name in fields:
          raise TypeError(f"Duplicate field '{field.name}' in struct '{struct.name}'")
        fields[field.name] = field_type
      self.structs[struct.name] = StructInfo(fields)

    # Check enum variant types
    # Only for non-generic enums; generic enums are checked when instantiated
    for enum in program.enums:
      if enum.type_params:
        continue  # Skip generic enums - they're checked during instantiation
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
    # Skip generic impl blocks for now (they're checked when instantiated)
    for impl in program.impls:
      if impl.type_params:
        continue  # Skip generic impl blocks - checked during instantiation
      if impl.struct_name not in self.structs:
        # Check if it's a generic struct
        if impl.struct_name in self.generic_structs:
          continue  # Generic struct impl - will be handled during instantiation
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

    # Store generic impl blocks for later instantiation
    for impl in program.impls:
      if impl.type_params or impl.struct_name in self.generic_structs:
        # Generic impl block - store for later instantiation
        self.generic_impl_defs[impl.struct_name] = impl

    # Fourth pass: register all function signatures
    # Skip generic functions for now (they're checked when instantiated)
    for func in program.functions:
      if func.type_params:
        # Generic function - store for later instantiation
        self.generic_functions[func.name] = list(func.type_params)
        self.generic_function_defs[func.name] = func
        continue
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
    # Skip generic impl blocks (they're checked when instantiated)
    for impl in program.impls:
      if impl.type_params:
        continue  # Skip generic impl blocks
      if impl.struct_name in self.generic_structs:
        continue  # Skip impl blocks for generic structs
      self.current_impl_type = impl.struct_name
      for method in impl.methods:
        self._check_function(method)
      self.current_impl_type = None

    # Sixth pass: check function bodies
    # Skip generic functions (they're checked when instantiated)
    for func in program.functions:
      if func.type_params:
        continue  # Skip generic functions
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
      case LetStmt(name, type_ann, value, mutable):
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
        self._define_var(name, declared_type, mutable)

      case AssignStmt(name, value):
        # Check variable is mutable (not const)
        self._check_mutable(name)
        # Check variable isn't borrowed before mutation
        self._check_not_borrowed(name)
        var_type = self._lookup_var(name)
        value_type = self._check_expr(value)
        if value_type != var_type:
          raise TypeError(f"Cannot assign {value_type} to variable of type {var_type}")

      case IndexAssignStmt(target, index, value):
        # Check mutability if target is a variable
        match target:
          case VarExpr(name):
            self._check_mutable(name)
          case _:
            pass
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
        # Check mutability if target is a variable
        match target:
          case VarExpr(name):
            self._check_mutable(name)
          case _:
            pass
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

      case SliceExpr(target, start, stop, step):
        target_type = self._check_expr(target)

        # Check slice indices are i64 (if provided)
        if start is not None:
          start_type = self._check_expr(start)
          if start_type != "i64":
            raise TypeError(f"Slice start must be i64, got {start_type}")
        if stop is not None:
          stop_type = self._check_expr(stop)
          if stop_type != "i64":
            raise TypeError(f"Slice stop must be i64, got {stop_type}")
        if step is not None:
          step_type = self._check_expr(step)
          if step_type != "i64":
            raise TypeError(f"Slice step must be i64, got {step_type}")

        # Slicing returns a vec of the element type
        if is_array_type(target_type):
          elem_type = get_element_type(target_type)
          return f"vec[{elem_type}]"
        elif is_vec_type(target_type):
          elem_type = get_element_type(target_type)
          return f"vec[{elem_type}]"
        elif target_type == "str":
          return "str"
        else:
          raise TypeError(f"Cannot slice type {target_type}")

      case MethodCallExpr(target, method, args):
        target_type = self._check_expr(target)
        return self._check_method_call(target_type, method, args)

      case StructLiteral(name, type_args, fields):
        # For now, we handle non-generic structs (type_args empty) and will add generic support later
        struct_type = self._resolve_generic_struct(name, type_args)
        if struct_type not in self.structs:
          raise TypeError(f"Unknown struct '{struct_type}'")
        struct_info = self.structs[struct_type]
        provided_fields: set[str] = set()
        for field_name, field_value in fields:
          if field_name in provided_fields:
            raise TypeError(f"Duplicate field '{field_name}' in struct literal")
          provided_fields.add(field_name)
          if field_name not in struct_info.fields:
            raise TypeError(f"Struct '{struct_type}' has no field '{field_name}'")
          expected_type = struct_info.fields[field_name]
          actual_type = self._check_expr(field_value)
          if actual_type != expected_type:
            raise TypeError(f"Field '{field_name}' expects {expected_type}, got {actual_type}")
        # Check all required fields are provided
        missing = set(struct_info.fields.keys()) - provided_fields
        if missing:
          raise TypeError(f"Missing fields in struct literal: {', '.join(sorted(missing))}")
        return struct_type

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

      case EnumLiteral(enum_name, type_args, variant_name, payload):
        # For now, we handle non-generic enums (type_args empty) and will add generic support later
        enum_type = self._resolve_generic_enum(enum_name, type_args)
        if enum_type not in self.enums:
          raise TypeError(f"Unknown enum '{enum_type}'")
        enum_info = self.enums[enum_type]
        if variant_name not in enum_info.variants:
          raise TypeError(f"Enum '{enum_type}' has no variant '{variant_name}'")
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
        return enum_type

      case MatchExpr(target, arms):
        target_type = self._check_expr(target)

        # Handle Result types
        if is_result_type(target_type):
          ok_type = get_result_ok_type(target_type)
          err_type = get_result_err_type(target_type)

          # Check each arm
          covered_variants: set[str] = set()
          # Track which variants have an unguarded arm (for exhaustiveness)
          unguarded_variants: set[str] = set()
          for arm in arms:
            if arm.enum_name != "Result":
              raise TypeError(f"Match arm pattern uses wrong type '{arm.enum_name}', expected 'Result'")
            if arm.variant_name not in ("Ok", "Err"):
              raise TypeError(f"Result has no variant '{arm.variant_name}'")
            # Allow duplicate variants if they have guards
            if arm.variant_name in covered_variants and arm.guard is None and arm.variant_name in unguarded_variants:
              raise TypeError(f"Duplicate match arm for variant '{arm.variant_name}'")
            covered_variants.add(arm.variant_name)
            if arm.guard is None:
              unguarded_variants.add(arm.variant_name)

            # Check binding and guard
            self._enter_scope()
            if arm.binding is not None:
              binding_type = ok_type if arm.variant_name == "Ok" else err_type
              if binding_type is not None:
                self._define_var(arm.binding, binding_type)

            # Check pattern guard (if present)
            if arm.guard is not None:
              guard_type = self._check_expr(arm.guard)
              if guard_type != "bool":
                raise TypeError(f"Pattern guard must be bool, got {guard_type}")

            # Check arm body
            for stmt in arm.body:
              self._check_stmt(stmt)
            self._exit_scope()

          # Check exhaustiveness (guards make this complex - for now, require all variants)
          # Note: In Rust, guards can make a match non-exhaustive, but we require all variants covered
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
        covered_variants: set[str] = set()
        # Track which variants have an unguarded arm (for exhaustiveness)
        unguarded_variants: set[str] = set()
        for arm in arms:
          if arm.enum_name != target_type:
            raise TypeError(f"Match arm pattern uses wrong enum '{arm.enum_name}', expected '{target_type}'")
          if arm.variant_name not in enum_info.variants:
            raise TypeError(f"Enum '{target_type}' has no variant '{arm.variant_name}'")
          # Allow duplicate variants if they have guards
          if arm.variant_name in covered_variants and arm.guard is None and arm.variant_name in unguarded_variants:
            raise TypeError(f"Duplicate match arm for variant '{arm.variant_name}'")
          covered_variants.add(arm.variant_name)
          if arm.guard is None:
            unguarded_variants.add(arm.variant_name)

          # Check binding and guard
          variant_payload = enum_info.variants[arm.variant_name]
          self._enter_scope()
          if variant_payload is not None and arm.binding is not None:
            self._define_var(arm.binding, variant_payload)
          elif variant_payload is None and arm.binding is not None:
            raise TypeError(f"Variant '{arm.variant_name}' has no payload to bind")

          # Check pattern guard (if present)
          if arm.guard is not None:
            guard_type = self._check_expr(arm.guard)
            if guard_type != "bool":
              raise TypeError(f"Pattern guard must be bool, got {guard_type}")

          # Check arm body - get result type from last statement if it's a return
          for stmt in arm.body:
            self._check_stmt(stmt)
          self._exit_scope()

        # Check exhaustiveness (guards make this complex - for now, require all variants)
        # Note: In Rust, guards can make a match non-exhaustive, but we require all variants covered
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

        # Check if it's a generic function with explicit type args (e.g., identity<i64>)
        if "<" in name and ">" in name:
          # Parse the mangled name to extract base name and type args
          idx = name.index("<")
          base_name = name[:idx]
          type_args_str = name[idx + 1 : -1]  # Remove < and >
          type_args = self._split_type_args(type_args_str)

          if base_name not in self.generic_functions:
            raise TypeError(f"'{base_name}' is not a generic function")

          func_def = self.generic_function_defs[base_name]
          type_params = self.generic_functions[base_name]

          if len(type_args) != len(type_params):
            raise TypeError(f"Generic function '{base_name}' expects {len(type_params)} type args, got {len(type_args)}")

          # Validate type arguments
          for type_arg in type_args:
            # Check it's a valid type (this will raise if invalid)
            if type_arg not in ("i64", "bool", "str") and type_arg not in self.structs and type_arg not in self.enums:
              if type_arg not in self.type_aliases and not ("<" in type_arg):
                raise TypeError(f"Unknown type '{type_arg}'")

          # First, check all argument types (without kwargs resolution for now)
          if kwargs:
            raise TypeError(f"Generic function '{base_name}' does not support keyword arguments yet")

          if len(args) != len(func_def.params):
            raise TypeError(f"Generic function '{base_name}' expects {len(func_def.params)} arguments, got {len(args)}")

          # Instantiate the generic function
          sig = self._ensure_generic_function_instantiated(base_name, type_args)

          # Check argument types against instantiated signature
          for i, (arg, expected_type) in enumerate(zip(args, sig.param_types)):
            arg_type = self._check_expr(arg)
            if arg_type != expected_type:
              param_name = sig.param_names[i]
              raise TypeError(f"Argument '{param_name}' of '{name}' expects {expected_type}, got {arg_type}")
            # Mark variable as moved if passed by value
            if not is_ref_type(expected_type):
              match arg:
                case VarExpr(var_name):
                  self._maybe_move_var(var_name)
                case _:
                  pass

          return sig.return_type

        # Check if it's a generic function without explicit type args - try to infer
        if name in self.generic_functions:
          type_params = self.generic_functions[name]
          func_def = self.generic_function_defs[name]

          # Check argument count
          if len(args) != len(func_def.params):
            raise TypeError(f"Generic function '{name}' expects {len(func_def.params)} arguments, got {len(args)}")

          # kwargs not supported for generic functions (for now)
          if kwargs:
            raise TypeError(f"Generic function '{name}' does not support keyword arguments yet")

          # Type check all arguments first to get their types
          arg_types: list[str] = []
          for arg in args:
            arg_types.append(self._check_expr(arg))

          # Try to infer type parameters from argument types
          param_type_anns = [p.type_ann for p in func_def.params]
          try:
            inferred = self._infer_type_args(type_params, param_type_anns, arg_types)
          except TypeError as e:
            # Inference failed - give a helpful error
            raise TypeError(f"Cannot infer type arguments for '{name}': {e}")

          # Convert inferred dict to ordered list of type args
          inferred_type_args = [inferred[tp] for tp in type_params]

          # Instantiate the generic function with inferred types
          sig = self._ensure_generic_function_instantiated(name, inferred_type_args)
          mangled_name = f"{name}<{','.join(inferred_type_args)}>"

          # Store the inferred call for AST transformation
          self.inferred_calls[id(expr)] = mangled_name

          # Verify argument types match (should always pass since we inferred from them)
          for i, (arg_type, expected_type) in enumerate(zip(arg_types, sig.param_types)):
            if arg_type != expected_type:
              param_name = sig.param_names[i]
              raise TypeError(f"Argument '{param_name}' of '{name}' expects {expected_type}, got {arg_type}")

          # Mark variables as moved if passed by value
          for arg, expected_type in zip(args, sig.param_types):
            if not is_ref_type(expected_type):
              match arg:
                case VarExpr(var_name):
                  self._maybe_move_var(var_name)
                case _:
                  pass

          return sig.return_type

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

  # === AST Transformation for Type Inference ===

  def transform_program(self, program: Program) -> Program:
    """Walk AST and create new CallExpr nodes with resolved_name where needed."""
    if not self.inferred_calls:
      return program  # No transformation needed

    # Transform functions
    new_functions = tuple(self._transform_function(f) for f in program.functions)

    # Transform impl blocks
    new_impls = tuple(self._transform_impl(impl) for impl in program.impls)

    return Program(
      program.type_aliases,
      program.structs,
      program.enums,
      new_impls,
      new_functions,
    )

  def _transform_function(self, func: Function) -> Function:
    """Transform a function, creating new CallExpr nodes where needed."""
    new_body = tuple(self._transform_stmt(stmt) for stmt in func.body)
    if new_body == func.body:
      return func  # No changes
    return Function(func.name, func.type_params, func.params, func.return_type, new_body)

  def _transform_impl(self, impl: ImplBlock) -> ImplBlock:
    """Transform an impl block, creating new method bodies where needed."""
    new_methods = tuple(self._transform_function(m) for m in impl.methods)
    if new_methods == impl.methods:
      return impl  # No changes
    return ImplBlock(impl.struct_name, impl.type_params, new_methods)

  def _transform_stmt(self, stmt: Stmt) -> Stmt:
    """Transform a statement, recursively transforming contained expressions."""
    match stmt:
      case LetStmt(name, type_ann, value, mutable):
        new_value = self._transform_expr(value)
        if new_value is value:
          return stmt
        return LetStmt(name, type_ann, new_value, mutable)

      case AssignStmt(name, value):
        new_value = self._transform_expr(value)
        if new_value is value:
          return stmt
        return AssignStmt(name, new_value)

      case IndexAssignStmt(target, index, value):
        new_target = self._transform_expr(target)
        new_index = self._transform_expr(index)
        new_value = self._transform_expr(value)
        if new_target is target and new_index is index and new_value is value:
          return stmt
        return IndexAssignStmt(new_target, new_index, new_value)

      case FieldAssignStmt(target, field, value):
        new_target = self._transform_expr(target)
        new_value = self._transform_expr(value)
        if new_target is target and new_value is value:
          return stmt
        return FieldAssignStmt(new_target, field, new_value)

      case DerefAssignStmt(target, value):
        new_target = self._transform_expr(target)
        new_value = self._transform_expr(value)
        if new_target is target and new_value is value:
          return stmt
        return DerefAssignStmt(new_target, new_value)

      case ReturnStmt(value):
        new_value = self._transform_expr(value)
        if new_value is value:
          return stmt
        return ReturnStmt(new_value)

      case ExprStmt(expr):
        new_expr = self._transform_expr(expr)
        if new_expr is expr:
          return stmt
        return ExprStmt(new_expr)

      case IfStmt(condition, then_body, else_body):
        new_condition = self._transform_expr(condition)
        new_then = tuple(self._transform_stmt(s) for s in then_body)
        new_else = tuple(self._transform_stmt(s) for s in else_body) if else_body else None
        if new_condition is condition and new_then == then_body and new_else == else_body:
          return stmt
        return IfStmt(new_condition, new_then, new_else)

      case WhileStmt(condition, body):
        new_condition = self._transform_expr(condition)
        new_body = tuple(self._transform_stmt(s) for s in body)
        if new_condition is condition and new_body == body:
          return stmt
        return WhileStmt(new_condition, new_body)

      case ForStmt(var, start, end, body):
        new_start = self._transform_expr(start)
        new_end = self._transform_expr(end)
        new_body = tuple(self._transform_stmt(s) for s in body)
        if new_start is start and new_end is end and new_body == body:
          return stmt
        return ForStmt(var, new_start, new_end, new_body)

      case _:
        return stmt

  def _transform_expr(self, expr: Expr) -> Expr:
    """Transform an expression, creating new CallExpr nodes where needed."""
    match expr:
      case CallExpr(name, args, kwargs, resolved_name):
        # Check if this call needs the resolved_name filled in
        expr_id = id(expr)
        new_resolved = self.inferred_calls.get(expr_id, resolved_name)

        # Transform arguments
        new_args = tuple(self._transform_expr(a) for a in args)
        new_kwargs = tuple((k, self._transform_expr(v)) for k, v in kwargs)

        if new_args == args and new_kwargs == kwargs and new_resolved == resolved_name:
          return expr
        return CallExpr(name, new_args, new_kwargs, new_resolved)

      case BinaryExpr(left, op, right):
        new_left = self._transform_expr(left)
        new_right = self._transform_expr(right)
        if new_left is left and new_right is right:
          return expr
        return BinaryExpr(new_left, op, new_right)

      case UnaryExpr(op, operand):
        new_operand = self._transform_expr(operand)
        if new_operand is operand:
          return expr
        return UnaryExpr(op, new_operand)

      case ArrayLiteral(elements):
        new_elements = tuple(self._transform_expr(e) for e in elements)
        if new_elements == elements:
          return expr
        return ArrayLiteral(new_elements)

      case IndexExpr(target, index):
        new_target = self._transform_expr(target)
        new_index = self._transform_expr(index)
        if new_target is target and new_index is index:
          return expr
        return IndexExpr(new_target, new_index)

      case SliceExpr(target, start, stop, step):
        new_target = self._transform_expr(target)
        new_start = self._transform_expr(start) if start else None
        new_stop = self._transform_expr(stop) if stop else None
        new_step = self._transform_expr(step) if step else None
        if new_target is target and new_start is start and new_stop is stop and new_step is step:
          return expr
        return SliceExpr(new_target, new_start, new_stop, new_step)

      case MethodCallExpr(target, method, args):
        new_target = self._transform_expr(target)
        new_args = tuple(self._transform_expr(a) for a in args)
        if new_target is target and new_args == args:
          return expr
        return MethodCallExpr(new_target, method, new_args)

      case StructLiteral(name, type_args, fields):
        new_fields = tuple((k, self._transform_expr(v)) for k, v in fields)
        if new_fields == fields:
          return expr
        return StructLiteral(name, type_args, new_fields)

      case FieldAccessExpr(target, field):
        new_target = self._transform_expr(target)
        if new_target is target:
          return expr
        return FieldAccessExpr(new_target, field)

      case TupleLiteral(elements):
        new_elements = tuple(self._transform_expr(e) for e in elements)
        if new_elements == elements:
          return expr
        return TupleLiteral(new_elements)

      case TupleIndexExpr(target, index):
        new_target = self._transform_expr(target)
        if new_target is target:
          return expr
        return TupleIndexExpr(new_target, index)

      case EnumLiteral(enum_name, type_args, variant_name, payload):
        if payload is None:
          return expr
        new_payload = self._transform_expr(payload)
        if new_payload is payload:
          return expr
        return EnumLiteral(enum_name, type_args, variant_name, new_payload)

      case MatchExpr(target, arms):
        new_target = self._transform_expr(target)
        new_arms = tuple(
          MatchArm(
            arm.enum_name,
            arm.variant_name,
            arm.binding,
            tuple(self._transform_stmt(s) for s in arm.body),
            self._transform_expr(arm.guard) if arm.guard else None,
          )
          for arm in arms
        )
        if new_target is target and all(
          new_arm.body == old_arm.body and new_arm.guard is old_arm.guard for new_arm, old_arm in zip(new_arms, arms)
        ):
          return expr
        return MatchExpr(new_target, new_arms)

      case RefExpr(target, mutable):
        new_target = self._transform_expr(target)
        if new_target is target:
          return expr
        return RefExpr(new_target, mutable)

      case DerefExpr(target):
        new_target = self._transform_expr(target)
        if new_target is target:
          return expr
        return DerefExpr(new_target)

      case ClosureExpr(params, return_type, body):
        new_body = self._transform_expr(body)
        if new_body is body:
          return expr
        return ClosureExpr(params, return_type, new_body)

      case ClosureCallExpr(target, args):
        new_target = self._transform_expr(target)
        new_args = tuple(self._transform_expr(a) for a in args)
        if new_target is target and new_args == args:
          return expr
        return ClosureCallExpr(new_target, new_args)

      case OkExpr(value):
        new_value = self._transform_expr(value)
        if new_value is value:
          return expr
        return OkExpr(new_value)

      case ErrExpr(value):
        new_value = self._transform_expr(value)
        if new_value is value:
          return expr
        return ErrExpr(new_value)

      case TryExpr(target):
        new_target = self._transform_expr(target)
        if new_target is target:
          return expr
        return TryExpr(new_target)

      case DictLiteral(entries):
        new_entries = tuple((self._transform_expr(k), self._transform_expr(v)) for k, v in entries)
        if new_entries == entries:
          return expr
        return DictLiteral(new_entries)

      case ListComprehension(element_expr, var_name, start, end, condition):
        new_element = self._transform_expr(element_expr)
        new_start = self._transform_expr(start)
        new_end = self._transform_expr(end)
        new_condition = self._transform_expr(condition) if condition else None
        if new_element is element_expr and new_start is start and new_end is end and new_condition is condition:
          return expr
        return ListComprehension(new_element, var_name, new_start, new_end, new_condition)

      case DictComprehension(key_expr, value_expr, var_name, start, end, condition):
        new_key = self._transform_expr(key_expr)
        new_value = self._transform_expr(value_expr)
        new_start = self._transform_expr(start)
        new_end = self._transform_expr(end)
        new_condition = self._transform_expr(condition) if condition else None
        if new_key is key_expr and new_value is value_expr and new_start is start and new_end is end and new_condition is condition:
          return expr
        return DictComprehension(new_key, new_value, var_name, new_start, new_end, new_condition)

      case _:
        # Literals and other leaf nodes don't need transformation
        return expr

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
        # Check if this is a generic struct that might have more methods to instantiate
        if "<" in target_type:
          idx = target_type.index("<")
          base_name = target_type[:idx]
          type_args_str = target_type[idx + 1 : -1]
          type_args = self._split_type_args(type_args_str)
          if base_name in self.generic_impl_defs:
            # Instantiate the method
            sig = self._ensure_generic_method_instantiated(base_name, type_args, method)
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

    elif "<" in target_type:
      # Generic struct instance method call (e.g., Box<i64>)
      # Extract the base struct name and type args
      idx = target_type.index("<")
      base_name = target_type[:idx]
      type_args_str = target_type[idx + 1 : -1]
      type_args = self._split_type_args(type_args_str)

      # Check if there's a generic impl block for this struct
      if base_name in self.generic_impl_defs:
        # Instantiate the method
        sig = self._ensure_generic_method_instantiated(base_name, type_args, method)

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
        raise TypeError(f"No methods defined for type {target_type}")

    else:
      raise TypeError(f"Cannot call method on type {target_type}")


@dataclass
class InstantiatedFunction:
  """An instantiated generic function with concrete types."""

  mangled_name: str
  original_func: "Function"
  type_subst: dict[str, str]  # T -> i64, U -> str, etc.
  param_types: list[str]  # Concrete parameter types
  return_type: str  # Concrete return type


@dataclass
class InstantiatedMethod:
  """An instantiated generic method with concrete types."""

  mangled_name: str  # e.g., "Box<i64>_get"
  struct_mangled_name: str  # e.g., "Box<i64>"
  original_method: "Function"
  type_subst: dict[str, str]  # T -> i64, etc.
  param_types: list[str]  # Concrete parameter types
  return_type: str  # Concrete return type


@dataclass
class TypeCheckResult:
  """Result of type checking, including monomorphized type information."""

  # Instantiated generic structs: mangled_name -> list of (field_name, field_type)
  instantiated_structs: dict[str, list[tuple[str, str]]]
  # Instantiated generic enums: mangled_name -> dict of variant_name -> (tag, has_payload, payload_type)
  instantiated_enums: dict[str, dict[str, tuple[int, bool, str | None]]]
  # Instantiated generic functions: mangled_name -> InstantiatedFunction
  instantiated_functions: dict[str, InstantiatedFunction]
  # Instantiated generic methods: mangled_name -> InstantiatedMethod
  instantiated_methods: dict[str, InstantiatedMethod]


def check(program: Program) -> tuple[Program, TypeCheckResult]:
  """Type check a program and transform AST for inferred generic calls.

  Returns:
    tuple: (transformed_program, type_check_result)
      - transformed_program: AST with resolved_name filled in for inferred generic calls
      - type_check_result: Monomorphized type information for codegen
  """
  checker = TypeChecker()
  checker.check(program)

  # Transform AST to fill in resolved_name for inferred generic calls
  transformed_program = checker.transform_program(program)

  # Extract instantiated generic types for codegen
  instantiated_structs: dict[str, list[tuple[str, str]]] = {}
  instantiated_enums: dict[str, dict[str, tuple[int, bool, str | None]]] = {}

  for mangled_name in checker.instantiated_types:
    if mangled_name in checker.structs:
      struct_info = checker.structs[mangled_name]
      instantiated_structs[mangled_name] = list(struct_info.fields.items())
    if mangled_name in checker.enums:
      enum_info = checker.enums[mangled_name]
      variants: dict[str, tuple[int, bool, str | None]] = {}
      for i, (var_name, payload_type) in enumerate(enum_info.variants.items()):
        variants[var_name] = (i, payload_type is not None, payload_type)
      instantiated_enums[mangled_name] = variants

  type_check_result = TypeCheckResult(
    instantiated_structs,
    instantiated_enums,
    checker.instantiated_functions,
    checker.instantiated_methods,
  )

  return transformed_program, type_check_result
