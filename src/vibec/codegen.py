"""ARM64 code generator for macOS."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
  MatchArm,
  ArrayType,
  DerefExpr,
  IndexExpr,
  MatchExpr,
  Parameter,
  SliceExpr,
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

if TYPE_CHECKING:
  from .checker import TypeCheckResult, InstantiatedMethod, InstantiatedFunction


def type_to_str(t: TypeAnnotation, type_aliases: dict[str, str] | None = None) -> str:
  """Convert type annotation to string, resolving type aliases if provided."""
  aliases = type_aliases or {}
  match t:
    case SimpleType(name):
      # Resolve type alias if it exists
      if name in aliases:
        return aliases[name]
      return name
    case ArrayType(elem, size):
      return f"[{type_to_str(elem, aliases)};{size}]"
    case VecType(elem):
      return f"vec[{type_to_str(elem, aliases)}]"
    case TupleType(elems):
      return f"({','.join(type_to_str(e, aliases) for e in elems)})"
    case ResultType(ok_type, err_type):
      return f"Result[{type_to_str(ok_type, aliases)},{type_to_str(err_type, aliases)}]"
    case DictType(key_type, value_type):
      return f"dict[{type_to_str(key_type, aliases)},{type_to_str(value_type, aliases)}]"
    case RefType(inner, mutable):
      prefix = "&mut " if mutable else "&"
      return f"{prefix}{type_to_str(inner, aliases)}"
    case FnType(params, ret):
      param_strs = ",".join(type_to_str(p, aliases) for p in params)
      return f"Fn({param_strs})->{type_to_str(ret, aliases)}"
    case GenericType(name, type_args):
      args_str = ",".join(type_to_str(a, aliases) for a in type_args)
      return f"{name}<{args_str}>"
  return "unknown"


def is_ref_type(type_str: str) -> bool:
  """Check if type string represents a reference."""
  return type_str.startswith("&")


def is_dict_type(type_str: str) -> bool:
  """Check if type string represents a dict."""
  return type_str.startswith("dict[")


def get_array_size(type_str: str) -> int:
  """Extract size from array type string like [i64;5]. Returns 0 if not an array."""
  if type_str.startswith("[") and ";" in type_str:
    return int(type_str[type_str.index(";") + 1 : -1])
  return 0


def is_vec_type(type_str: str) -> bool:
  """Check if type is a vec."""
  return type_str.startswith("vec[")


def is_array_type(type_str: str) -> bool:
  """Check if type is an array like [i64;5]."""
  return type_str.startswith("[") and ";" in type_str


def is_tuple_type(type_str: str) -> bool:
  """Check if type is a tuple."""
  return type_str.startswith("(") and type_str.endswith(")")


def get_tuple_size(type_str: str) -> int:
  """Get number of elements in tuple type like (i64,i64) -> 2."""
  if not is_tuple_type(type_str):
    return 0
  inner = type_str[1:-1]  # Remove parentheses
  if not inner:
    return 0
  # Count elements by tracking depth
  depth = 0
  count = 1
  for c in inner:
    if c in "([":
      depth += 1
    elif c in ")]":
      depth -= 1
    elif c == "," and depth == 0:
      count += 1
  return count


def is_enum_type(type_str: str, enums: dict[str, dict[str, tuple[int, bool]]]) -> bool:
  """Check if type is an enum."""
  return type_str in enums


def is_result_type(type_str: str) -> bool:
  """Check if type is a Result."""
  return type_str.startswith("Result[")


class CodeGenerator:
  """Generates ARM64 assembly for macOS.

  Stack frame layout (growing downward):
      [x29]      -> saved x29 (frame pointer)
      [x29 - 8]  -> saved x30 (link register)
      [x29 - 16] -> local slot 0 (first param or local)
      [x29 - 24] -> local slot 1
      ...
      [sp]       -> bottom of frame (16-byte aligned)

  During expression evaluation, temps are pushed below sp.
  """

  def __init__(self) -> None:
    self.output: list[str] = []
    self.label_counter = 0
    # Maps variable name -> (offset from x29, type)
    self.locals: dict[str, tuple[int, str]] = {}
    self.frame_size = 0
    self.current_func_name = ""
    self.next_slot = 0
    # String literals: value -> label
    self.strings: dict[str, str] = {}
    self.string_counter = 0
    # Type aliases: alias_name -> resolved_type_string
    self.type_aliases: dict[str, str] = {}
    # Struct definitions: name -> list of (field_name, field_type)
    self.structs: dict[str, list[tuple[str, str]]] = {}
    # Enum definitions: name -> dict of variant_name -> (tag, has_payload)
    self.enums: dict[str, dict[str, tuple[int, bool]]] = {}
    # Struct methods: struct_name -> method_name -> mangled_function_name
    self.struct_methods: dict[str, dict[str, str]] = {}
    # Closures: list of (label, params, return_type, body) to generate at end
    self.closures: list[tuple[str, tuple[Parameter, ...], TypeAnnotation, Expr]] = []
    self.closure_counter = 0
    # Function parameter names: function_name -> list of param names (for kwargs resolution)
    self.function_params: dict[str, list[str]] = {}
    # Operator overloads: id(BinaryExpr) -> (left_type, right_type, method_name, return_type)
    self.operator_overloads: dict[int, tuple[str, str, str, str]] = {}
    # Try expression types: id(TryExpr) -> "option" or "result"
    self.try_expr_types: dict[int, str] = {}

  def _emit(self, line: str) -> None:
    self.output.append(line)

  def _new_label(self, prefix: str = "L") -> str:
    label = f"{prefix}{self.label_counter}"
    self.label_counter += 1
    return label

  def _get_string_label(self, value: str) -> str:
    """Get or create a label for a string literal."""
    if value not in self.strings:
      label = f"_str{self.string_counter}"
      self.strings[value] = label
      self.string_counter += 1
    return self.strings[value]

  def _escape_string(self, s: str) -> str:
    """Escape a string for assembly .asciz directive."""
    result = []
    for c in s:
      if c == "\n":
        result.append("\\n")
      elif c == "\t":
        result.append("\\t")
      elif c == "\\":
        result.append("\\\\")
      elif c == '"':
        result.append('\\"')
      elif ord(c) < 32 or ord(c) > 126:
        result.append(f"\\x{ord(c):02x}")
      else:
        result.append(c)
    return "".join(result)

  def _collect_strings(self, program: Program) -> None:
    """Collect all string literals from the program."""
    for func in program.functions:
      self._collect_strings_from_stmts(func.body)

  def _collect_strings_from_stmts(self, stmts: tuple[Stmt, ...]) -> None:
    """Collect strings from a list of statements."""
    for stmt in stmts:
      match stmt:
        case LetStmt(_, _, value):
          self._collect_strings_from_expr(value)
        case AssignStmt(_, value):
          self._collect_strings_from_expr(value)
        case IndexAssignStmt(target, index, value):
          self._collect_strings_from_expr(target)
          self._collect_strings_from_expr(index)
          self._collect_strings_from_expr(value)
        case ReturnStmt(value):
          self._collect_strings_from_expr(value)
        case ExprStmt(expr):
          self._collect_strings_from_expr(expr)
        case IfStmt(condition, then_body, else_body):
          self._collect_strings_from_expr(condition)
          self._collect_strings_from_stmts(then_body)
          if else_body:
            self._collect_strings_from_stmts(else_body)
        case WhileStmt(condition, body):
          self._collect_strings_from_expr(condition)
          self._collect_strings_from_stmts(body)
        case ForStmt(_, start, end, body):
          self._collect_strings_from_expr(start)
          self._collect_strings_from_expr(end)
          self._collect_strings_from_stmts(body)
        case FieldAssignStmt(target, _, value):
          self._collect_strings_from_expr(target)
          self._collect_strings_from_expr(value)
        case DerefAssignStmt(target, value):
          self._collect_strings_from_expr(target)
          self._collect_strings_from_expr(value)

  def _collect_strings_from_expr(self, expr: Expr) -> None:
    """Collect strings from an expression."""
    match expr:
      case StringLiteral(value):
        self._get_string_label(value)
      case BinaryExpr(left, _, right):
        self._collect_strings_from_expr(left)
        self._collect_strings_from_expr(right)
      case UnaryExpr(_, operand):
        self._collect_strings_from_expr(operand)
      case CallExpr(_, args, kwargs):
        for arg in args:
          self._collect_strings_from_expr(arg)
        for _, kwarg_value in kwargs:
          self._collect_strings_from_expr(kwarg_value)
      case ArrayLiteral(elements):
        for elem in elements:
          self._collect_strings_from_expr(elem)
      case IndexExpr(target, index):
        self._collect_strings_from_expr(target)
        self._collect_strings_from_expr(index)
      case SliceExpr(target, start, stop, step):
        self._collect_strings_from_expr(target)
        if start is not None:
          self._collect_strings_from_expr(start)
        if stop is not None:
          self._collect_strings_from_expr(stop)
        if step is not None:
          self._collect_strings_from_expr(step)
      case MethodCallExpr(target, _, args):
        self._collect_strings_from_expr(target)
        for arg in args:
          self._collect_strings_from_expr(arg)
      case StructLiteral(_, _, fields):
        for _, value in fields:
          self._collect_strings_from_expr(value)
      case FieldAccessExpr(target, _):
        self._collect_strings_from_expr(target)
      case TupleLiteral(elements):
        for elem in elements:
          self._collect_strings_from_expr(elem)
      case TupleIndexExpr(target, _):
        self._collect_strings_from_expr(target)
      case EnumLiteral(_, _, _, payload):
        if payload is not None:
          self._collect_strings_from_expr(payload)
      case MatchExpr(target, arms):
        self._collect_strings_from_expr(target)
        for arm in arms:
          if arm.guard is not None:
            self._collect_strings_from_expr(arm.guard)
          self._collect_strings_from_stmts(arm.body)
      case RefExpr(target, _):
        self._collect_strings_from_expr(target)
      case DerefExpr(target):
        self._collect_strings_from_expr(target)
      case ClosureExpr(_, _, body):
        self._collect_strings_from_expr(body)
      case ClosureCallExpr(target, args):
        self._collect_strings_from_expr(target)
        for arg in args:
          self._collect_strings_from_expr(arg)
      case OkExpr(value):
        self._collect_strings_from_expr(value)
      case ErrExpr(value):
        self._collect_strings_from_expr(value)
      case TryExpr(target):
        self._collect_strings_from_expr(target)
      case ListComprehension(element_expr, _, start, end, condition):
        self._collect_strings_from_expr(element_expr)
        self._collect_strings_from_expr(start)
        self._collect_strings_from_expr(end)
        if condition is not None:
          self._collect_strings_from_expr(condition)
      case DictLiteral(entries):
        for key, value in entries:
          self._collect_strings_from_expr(key)
          self._collect_strings_from_expr(value)
      case DictComprehension(key_expr, value_expr, _, start, end, condition):
        self._collect_strings_from_expr(key_expr)
        self._collect_strings_from_expr(value_expr)
        self._collect_strings_from_expr(start)
        self._collect_strings_from_expr(end)
        if condition is not None:
          self._collect_strings_from_expr(condition)
      case _:
        pass

  def generate(self, program: Program, type_check_result: "TypeCheckResult | None" = None) -> str:
    """Generate assembly for the entire program."""
    # Register type aliases first (so they can be used in other definitions)
    for alias in program.type_aliases:
      # Resolve the target type (may use other aliases registered earlier)
      resolved = type_to_str(alias.target, self.type_aliases)
      self.type_aliases[alias.name] = resolved

    # Register struct definitions (skip generic structs)
    for struct in program.structs:
      if struct.type_params:
        continue  # Skip generic structs - they're instantiated below
      fields = [(f.name, type_to_str(f.type_ann, self.type_aliases)) for f in struct.fields]
      self.structs[struct.name] = fields

    # Register enum definitions (skip generic enums)
    for enum in program.enums:
      if enum.type_params:
        continue  # Skip generic enums - they're instantiated below
      variants: dict[str, tuple[int, bool]] = {}
      for i, variant in enumerate(enum.variants):
        has_payload = variant.payload_type is not None
        variants[variant.name] = (i, has_payload)
      self.enums[enum.name] = variants

    # Register instantiated generic types from type checker
    if type_check_result:
      for mangled_name, fields in type_check_result.instantiated_structs.items():
        self.structs[mangled_name] = fields
      for mangled_name, inst_variants in type_check_result.instantiated_enums.items():
        # Convert from (tag, has_payload, payload_type) to (tag, has_payload)
        converted: dict[str, tuple[int, bool]] = {}
        for name, (tag, has_payload, _) in inst_variants.items():
          converted[name] = (tag, has_payload)
        self.enums[mangled_name] = converted

    # Register struct methods (with mangled names) - skip generic impl blocks
    for impl in program.impls:
      if impl.type_params:
        continue  # Skip generic impl blocks
      # Skip if this is an impl for a generic struct
      if impl.struct_name in [s.name for s in program.structs if s.type_params]:
        continue
      if impl.struct_name not in self.struct_methods:
        self.struct_methods[impl.struct_name] = {}
      for method in impl.methods:
        mangled_name = f"{impl.struct_name}_{method.name}"
        self.struct_methods[impl.struct_name][method.name] = mangled_name

    # Register instantiated generic struct methods
    if type_check_result:
      for mangled_name, inst_method in type_check_result.instantiated_methods.items():
        struct_mangled = inst_method.struct_mangled_name
        if struct_mangled not in self.struct_methods:
          self.struct_methods[struct_mangled] = {}
        # The method's assembly label
        asm_label = self._mangle_generic_name(mangled_name)
        # Remove leading _ since _gen_method_call adds it
        self.struct_methods[struct_mangled][inst_method.original_method.name] = asm_label[1:]

    # Register operator overloads from type checker
    if type_check_result:
      self.operator_overloads = type_check_result.operator_overloads

    # Register try expression types from type checker
    if type_check_result:
      self.try_expr_types = type_check_result.try_expr_types

    # Register function parameter names (for kwargs resolution) - skip generic functions
    for func in program.functions:
      if func.type_params:
        continue  # Skip generic functions
      self.function_params[func.name] = [p.name for p in func.params]

    # First pass: collect all string literals
    self._collect_strings(program)
    # Also collect from impl methods
    for impl in program.impls:
      for method in impl.methods:
        self._collect_strings_from_stmts(method.body)

    # Data section
    self._emit(".section __DATA,__data")
    self._emit("_fmt_int:")
    self._emit('    .asciz "%lld\\n"')
    self._emit("_fmt_str:")
    self._emit('    .asciz "%s\\n"')

    # Emit all string literals
    for value, label in self.strings.items():
      self._emit(f"{label}:")
      self._emit(f'    .asciz "{self._escape_string(value)}"')
    self._emit("")

    # Text section
    self._emit(".section __TEXT,__text")
    self._emit(".globl _main")
    self._emit("")

    # Generate impl block methods (skip generic impl blocks)
    for impl in program.impls:
      if impl.type_params:
        continue  # Skip generic impl blocks
      # Skip if this is an impl for a generic struct (handled separately)
      if impl.struct_name in [s.name for s in program.structs if s.type_params]:
        continue
      for method in impl.methods:
        mangled_name = f"{impl.struct_name}_{method.name}"
        self._gen_function(method, mangled_name, impl.struct_name)

    # Generate instantiated generic methods
    if type_check_result:
      for mangled_name, inst_method in type_check_result.instantiated_methods.items():
        self._gen_instantiated_method(inst_method)

    # Generate functions (skip generic functions)
    for func in program.functions:
      if func.type_params:
        continue  # Skip generic functions
      self._gen_function(func)

    # Generate instantiated generic functions
    if type_check_result:
      for mangled_name, inst_func in type_check_result.instantiated_functions.items():
        self._gen_instantiated_function(inst_func)

    # Generate closure functions (collected during code generation)
    for label, params, return_type, body in self.closures:
      self._gen_closure_function(label, params, return_type, body)

    return "\n".join(self.output)

  def _gen_closure_function(self, label: str, params: tuple[Parameter, ...], return_type: TypeAnnotation, body: Expr) -> None:
    """Generate a closure as a standalone function."""
    # Save current state
    old_locals = self.locals
    old_next_slot = self.next_slot
    old_func_name = self.current_func_name

    self.locals = {}
    self.next_slot = 0
    self.current_func_name = label

    # Calculate frame size: params + some space for temps
    frame_size = max(48, (32 + len(params) * 8 + 15) & ~15)
    self.frame_size = frame_size

    # Emit function prologue
    self._emit(f"{label}:")
    self._emit(f"    sub sp, sp, #{frame_size}")
    self._emit(f"    stp x29, x30, [sp, #{frame_size - 16}]")
    self._emit(f"    add x29, sp, #{frame_size - 16}")

    # Store parameters
    for i, param in enumerate(params):
      offset = -16 - (self.next_slot * 8)
      type_str = type_to_str(param.type_ann, self.type_aliases)
      self.locals[param.name] = (offset, type_str)
      if i < 8:
        self._emit(f"    str x{i}, [x29, #{offset}]")
      self.next_slot += 1

    # Generate body expression
    self._gen_expr(body)

    # Epilogue - result is in x0
    self._emit(f"{label}_epilogue:")
    self._emit(f"    ldp x29, x30, [sp, #{frame_size - 16}]")
    self._emit(f"    add sp, sp, #{frame_size}")
    self._emit("    ret")
    self._emit("")

    # Restore state
    self.locals = old_locals
    self.next_slot = old_next_slot
    self.current_func_name = old_func_name

  def _gen_instantiated_function(self, inst_func: "InstantiatedFunction") -> None:
    """Generate code for an instantiated generic function."""
    # The mangled name uses <> which is not valid in assembly labels
    # Convert to a valid label: identity<i64> -> _identity_i64
    label = self._mangle_generic_name(inst_func.mangled_name)

    # Save current state
    old_locals = self.locals
    old_next_slot = self.next_slot
    old_func_name = self.current_func_name

    self.locals = {}
    self.next_slot = 0
    self.current_func_name = label

    func = inst_func.original_func
    subst = inst_func.type_subst

    # Count locals needed (with substituted types)
    local_count = len(func.params) + self._count_locals_with_subst(func.body, subst)
    frame_size = max(48, (32 + local_count * 8 + 15) & ~15)
    self.frame_size = frame_size

    # Emit function prologue
    self._emit(f"{label}:")
    self._emit(f"    sub sp, sp, #{frame_size}")
    self._emit(f"    stp x29, x30, [sp, #{frame_size - 16}]")
    self._emit(f"    add x29, sp, #{frame_size - 16}")

    # Store parameters with substituted types
    for i, (param, param_type) in enumerate(zip(func.params, inst_func.param_types)):
      slots = self._slots_for_type_str(param_type)
      offset = -16 - (self.next_slot * 8)
      self.locals[param.name] = (offset, param_type)
      if i < 8:
        if slots == 1:
          self._emit(f"    str x{i}, [x29, #{offset}]")
        else:
          # Multi-slot parameter (struct, tuple, etc.)
          for j in range(slots):
            self._emit(f"    str x{i + j}, [x29, #{offset - j * 8}]")
      self.next_slot += slots

    # Store the substitution map for use during code generation
    old_type_subst = getattr(self, "current_type_subst", None)
    self.current_type_subst = subst

    # Generate body
    for stmt in func.body:
      self._gen_stmt(stmt)

    self.current_type_subst = old_type_subst

    # Emit epilogue
    self._emit(f"{label}_epilogue:")
    self._emit(f"    ldp x29, x30, [sp, #{frame_size - 16}]")
    self._emit(f"    add sp, sp, #{frame_size}")
    self._emit("    ret")
    self._emit("")

    # Restore state
    self.locals = old_locals
    self.next_slot = old_next_slot
    self.current_func_name = old_func_name

  def _gen_instantiated_method(self, inst_method: "InstantiatedMethod") -> None:
    """Generate code for an instantiated generic method."""
    # The mangled name uses <> which is not valid in assembly labels
    # Convert to a valid label: Box<i64>_get -> _Box_i64__get
    label = self._mangle_generic_name(inst_method.mangled_name)

    # Save current state
    old_locals = self.locals
    old_next_slot = self.next_slot
    old_func_name = self.current_func_name

    self.locals = {}
    self.next_slot = 0
    self.current_func_name = label

    method = inst_method.original_method
    subst = inst_method.type_subst

    # Count locals needed (with substituted types)
    local_count = len(method.params) + self._count_locals_with_subst(method.body, subst)
    frame_size = max(48, (32 + local_count * 8 + 15) & ~15)
    self.frame_size = frame_size

    # Emit function prologue
    self._emit(f"{label}:")
    self._emit(f"    sub sp, sp, #{frame_size}")
    self._emit(f"    stp x29, x30, [sp, #{frame_size - 16}]")
    self._emit(f"    add x29, sp, #{frame_size - 16}")

    # Store parameters with substituted types
    for i, (param, param_type) in enumerate(zip(method.params, inst_method.param_types)):
      slots = self._slots_for_type_str(param_type)
      offset = -16 - (self.next_slot * 8)
      self.locals[param.name] = (offset, param_type)
      if i < 8:
        if slots == 1:
          self._emit(f"    str x{i}, [x29, #{offset}]")
        else:
          # Multi-slot parameter (struct, tuple, etc.)
          for j in range(slots):
            self._emit(f"    str x{i + j}, [x29, #{offset - j * 8}]")
      self.next_slot += slots

    # Store the substitution map and struct name for use during code generation
    old_type_subst = getattr(self, "current_type_subst", None)
    old_current_struct = getattr(self, "current_struct_name", None)
    self.current_type_subst = subst
    self.current_struct_name = inst_method.struct_mangled_name

    # Generate body
    for stmt in method.body:
      self._gen_stmt(stmt)

    self.current_type_subst = old_type_subst
    self.current_struct_name = old_current_struct

    # Emit epilogue
    self._emit(f"{label}_epilogue:")
    self._emit(f"    ldp x29, x30, [sp, #{frame_size - 16}]")
    self._emit(f"    add sp, sp, #{frame_size}")
    self._emit("    ret")
    self._emit("")

    # Restore state
    self.locals = old_locals
    self.next_slot = old_next_slot
    self.current_func_name = old_func_name

  def _mangle_generic_name(self, name: str) -> str:
    """Convert a generic name like 'identity<i64>' to a valid assembly label."""
    # Replace < with _, > with empty, and add underscore prefix
    result = name.replace("<", "_").replace(">", "").replace(",", "_")
    return f"_{result}"

  def _count_locals_with_subst(self, stmts: tuple[Stmt, ...], subst: dict[str, str]) -> int:
    """Count local variable slots needed, using type substitution for generics."""
    count = 0
    for stmt in stmts:
      match stmt:
        case LetStmt(_, type_ann, _):
          type_str = self._substitute_type_str(type_to_str(type_ann, self.type_aliases), subst)
          count += self._slots_for_type_str(type_str)
        case IfStmt(_, then_body, else_body):
          count += self._count_locals_with_subst(then_body, subst)
          if else_body:
            count += self._count_locals_with_subst(else_body, subst)
        case WhileStmt(_, body):
          count += self._count_locals_with_subst(body, subst)
        case ForStmt(_, _, _, body):
          count += 2  # Loop variable + end value temp
          count += self._count_locals_with_subst(body, subst)
        case ExprStmt(MatchExpr(_, arms)):
          for arm in arms:
            if arm.binding is not None:
              count += 1  # Binding variable slot
            count += self._count_locals_with_subst(arm.body, subst)
    return count

  def _substitute_type_str(self, type_str: str, subst: dict[str, str]) -> str:
    """Apply type substitution to a type string."""
    for param, concrete in subst.items():
      type_str = type_str.replace(param, concrete)
    return type_str

  def _count_locals(self, stmts: tuple[Stmt, ...]) -> int:
    """Count local variable slots needed (arrays need multiple slots)."""
    count = 0
    for stmt in stmts:
      match stmt:
        case LetStmt(_, type_ann, _):
          count += self._slots_for_type(type_ann)
        case IfStmt(_, then_body, else_body):
          count += self._count_locals(then_body)
          if else_body:
            count += self._count_locals(else_body)
        case WhileStmt(_, body):
          count += self._count_locals(body)
        case ForStmt(_, _, _, body):
          count += 2  # Loop variable + end value temp
          count += self._count_locals(body)
        case ExprStmt(MatchExpr(_, arms)):
          for arm in arms:
            if arm.binding is not None:
              count += 1  # Binding variable slot
            count += self._count_locals(arm.body)
    return count

  def _slots_for_type(self, t: TypeAnnotation) -> int:
    """Return number of 8-byte slots needed for a type."""
    match t:
      case ArrayType(_, size):
        return size  # Each element is 8 bytes
      case VecType(_):
        return 1  # List is a pointer
      case DictType(_, _):
        return 1  # Dict is a pointer
      case TupleType(elems):
        return len(elems)  # One slot per element
      case ResultType(_, _):
        return 2  # Result needs 2 slots: tag (0=Ok, 1=Err) + payload
      case SimpleType(name):
        if name in self.structs:
          return len(self.structs[name])  # One slot per field
        if name in self.enums:
          return 2  # Enums need 2 slots: tag + payload
        return 1  # Simple types are 8 bytes
      case _:
        return 1

  def _slots_for_type_str(self, type_str: str) -> int:
    """Return number of 8-byte slots needed for a type string (supports type aliases)."""
    if is_array_type(type_str):
      return get_array_size(type_str)
    if is_vec_type(type_str):
      return 1  # Vec is a pointer
    if is_dict_type(type_str):
      return 1  # Dict is a pointer
    if is_tuple_type(type_str):
      return get_tuple_size(type_str)
    if is_result_type(type_str):
      return 2  # Result needs 2 slots: tag + payload
    if type_str in self.structs:
      return len(self.structs[type_str])
    if type_str in self.enums:
      return 2  # Enums need 2 slots: tag + payload
    return 1  # Simple types are 8 bytes

  def _gen_function(self, func: Function, mangled_name: str | None = None, impl_struct: str | None = None) -> None:
    """Generate assembly for a function or method."""
    self.locals = {}
    func_label = mangled_name if mangled_name else func.name
    self.current_func_name = func_label
    self.next_slot = 0

    # Calculate frame size: params + local variables
    # Each slot is 8 bytes, plus 16 for saved fp/lr
    # Note: enums take 2 slots each for both storage and registers
    # For methods, 'self' is treated as a struct (multiple slots)
    num_param_slots = 0
    for p in func.params:
      if p.name == "self" and impl_struct and impl_struct in self.structs:
        num_param_slots += len(self.structs[impl_struct])
      else:
        num_param_slots += self._slots_for_type(p.type_ann)
    num_locals = self._count_locals(func.body)
    total_slots = num_param_slots + num_locals

    # Frame size: 16 (saved fp/lr) + 16 (base offset) + slots, 16-byte aligned
    # Locals are at [x29-16], [x29-24], etc., growing downward
    # With x29 = sp + frame_size - 16, we need frame_size >= 32 + (n-1)*8
    slots_size = total_slots * 8
    self.frame_size = (32 + slots_size + 15) & ~15
    if self.frame_size < 48:
      self.frame_size = 48  # Minimum for temp storage and proper alignment

    # Function label (prefix with _ for macOS)
    self._emit(".align 4")
    self._emit(f"_{func_label}:")

    # Prologue: allocate full frame and save fp/lr at top
    self._emit(f"    sub sp, sp, #{self.frame_size}")
    self._emit(f"    stp x29, x30, [sp, #{self.frame_size - 16}]")
    self._emit(f"    add x29, sp, #{self.frame_size - 16}")

    # Store parameters (first 8 in x0-x7)
    # Parameters go at [x29 - 16], [x29 - 24], etc.
    # Enum parameters use 2 registers (tag, payload)
    # Self parameter (struct) uses multiple registers
    reg_idx = 0
    for param in func.params:
      offset = -16 - (self.next_slot * 8)

      # Handle 'self' parameter specially in methods
      if param.name == "self" and impl_struct and impl_struct in self.structs:
        type_str = impl_struct
        self.locals[param.name] = (offset, type_str)
        num_fields = len(self.structs[impl_struct])
        for i in range(num_fields):
          if reg_idx < 8:
            self._emit(f"    str x{reg_idx}, [x29, #{offset - i * 8}]")
          reg_idx += 1
        self.next_slot += num_fields
        continue

      type_str = type_to_str(param.type_ann, self.type_aliases)
      self.locals[param.name] = (offset, type_str)
      slots_needed = self._slots_for_type(param.type_ann)

      if type_str in self.enums or is_result_type(type_str):
        # Enum/Result: store tag from x{reg_idx}, payload from x{reg_idx+1}
        if reg_idx < 8:
          self._emit(f"    str x{reg_idx}, [x29, #{offset}]")  # Tag
        if reg_idx + 1 < 8:
          self._emit(f"    str x{reg_idx + 1}, [x29, #{offset - 8}]")  # Payload
        reg_idx += 2
      elif type_str in self.structs:
        # Struct parameter: store each field from separate registers
        num_fields = len(self.structs[type_str])
        for i in range(num_fields):
          if reg_idx < 8:
            self._emit(f"    str x{reg_idx}, [x29, #{offset - i * 8}]")
          reg_idx += 1
      else:
        if reg_idx < 8:
          self._emit(f"    str x{reg_idx}, [x29, #{offset}]")
        reg_idx += 1

      self.next_slot += slots_needed

    # Generate body
    # Handle implicit return: if last statement is ExprStmt, treat as return
    for i, stmt in enumerate(func.body):
      is_last = i == len(func.body) - 1
      if is_last and isinstance(stmt, ExprStmt):
        # Generate expression and return its value
        self._gen_expr(stmt.expr)
        self._emit(f"    b _{func_label}_epilogue")
      else:
        self._gen_stmt(stmt)

    # Epilogue
    self._emit(f"_{func_label}_epilogue:")
    self._emit(f"    ldp x29, x30, [sp, #{self.frame_size - 16}]")
    self._emit(f"    add sp, sp, #{self.frame_size}")
    self._emit("    ret")
    self._emit("")

  def _gen_stmt(self, stmt: Stmt) -> None:
    """Generate assembly for a statement."""
    match stmt:
      case LetStmt(name, type_ann, value, _):
        type_str = type_to_str(type_ann, self.type_aliases)
        offset = -16 - (self.next_slot * 8)
        self.locals[name] = (offset, type_str)
        slots_needed = self._slots_for_type_str(type_str)

        # Use type_str to determine handling (supports type aliases)
        if is_array_type(type_str):
          # Array: initialize elements in place
          size = get_array_size(type_str)
          match value:
            case ArrayLiteral(elements):
              for i, elem in enumerate(elements):
                self._gen_expr(elem)
                self._emit(f"    str x0, [x29, #{offset - i * 8}]")
              # Zero-fill remaining slots if literal is smaller
              for i in range(len(elements), size):
                self._emit(f"    str xzr, [x29, #{offset - i * 8}]")
            case _:
              # Initialize all to zero
              for i in range(size):
                self._emit(f"    str xzr, [x29, #{offset - i * 8}]")
          self.next_slot += slots_needed

        elif is_vec_type(type_str):
          # Check if value is empty array literal -> allocate new vec
          # Otherwise, evaluate expression (returns vec pointer) and store it
          match value:
            case ArrayLiteral(elements) if not elements:
              # Empty array: allocate initial buffer (16 elements * 8 bytes + 16 header)
              # Header: [capacity, length] at base, data follows
              self._emit("    mov x0, #144")  # 16 + 16*8 = 144 bytes
              self._emit("    bl _malloc")
              self._emit("    mov x1, #16")  # Initial capacity
              self._emit("    str x1, [x0]")  # Store capacity
              self._emit("    str xzr, [x0, #8]")  # Length = 0
              self._emit(f"    str x0, [x29, #{offset}]")
            case _:
              # Expression returns vec pointer (from skip, take, map, etc.)
              self._gen_expr(value)
              self._emit(f"    str x0, [x29, #{offset}]")
          self.next_slot += 1

        elif type_str in self.structs:
          # Struct type: initialize fields in place
          fields = self.structs[type_str]
          match value:
            case StructLiteral(_, _, field_values):
              # Map field name -> value for lookup
              value_map = dict(field_values)
              for i, (field_name, _) in enumerate(fields):
                if field_name in value_map:
                  self._gen_expr(value_map[field_name])
                  self._emit(f"    str x0, [x29, #{offset - i * 8}]")
                else:
                  self._emit(f"    str xzr, [x29, #{offset - i * 8}]")
            case VarExpr(src_name):
              # Copy struct from another variable
              src_offset, _ = self.locals[src_name]
              for i in range(len(fields)):
                self._emit(f"    ldr x0, [x29, #{src_offset - i * 8}]")
                self._emit(f"    str x0, [x29, #{offset - i * 8}]")
            case CallExpr(_, _, _, _) | MethodCallExpr(_, _, _) | BinaryExpr(_, _, _):
              # Function/method call or binary expression returning a struct
              # For multi-field structs, values are returned in x0, x1, x2, ...
              self._gen_expr(value)
              # Store returned value(s) from registers x0, x1, x2, ...
              for i in range(len(fields)):
                self._emit(f"    str x{i}, [x29, #{offset - i * 8}]")
            case _:
              # Other expressions - evaluate and store
              # Struct values are returned in multiple registers
              self._gen_expr(value)
              for i in range(len(fields)):
                self._emit(f"    str x{i}, [x29, #{offset - i * 8}]")
          self.next_slot += len(fields)

        elif is_tuple_type(type_str):
          # Tuple type: initialize elements in place
          tuple_size = get_tuple_size(type_str)
          match value:
            case TupleLiteral(elements):
              for i, elem in enumerate(elements):
                self._gen_expr(elem)
                self._emit(f"    str x0, [x29, #{offset - i * 8}]")
            case _:
              # Initialize all to zero
              for i in range(tuple_size):
                self._emit(f"    str xzr, [x29, #{offset - i * 8}]")
          self.next_slot += tuple_size

        elif type_str in self.enums:
          # Enum type: store tag and payload (2 slots)
          match value:
            case EnumLiteral(_, _, variant_name, payload):
              variants = self.enums[type_str]
              tag, _ = variants[variant_name]
              self._emit(f"    mov x0, #{tag}")
              self._emit(f"    str x0, [x29, #{offset}]")  # Tag at first slot
              if payload is not None:
                self._gen_expr(payload)
                self._emit(f"    str x0, [x29, #{offset - 8}]")  # Payload at second slot
              else:
                self._emit(f"    str xzr, [x29, #{offset - 8}]")  # Zero payload
            case _:
              # Expression that returns an enum (e.g., function call)
              self._gen_expr(value)
              # x0 has tag, x1 has payload (if enum was returned from function)
              self._emit(f"    str x0, [x29, #{offset}]")  # Tag at first slot
              self._emit(f"    str x1, [x29, #{offset - 8}]")  # Payload at second slot
          self.next_slot += 2

        elif is_result_type(type_str):
          # Result type: tag (0=Ok, 1=Err) + payload (2 slots)
          match value:
            case OkExpr(ok_value):
              self._emit("    mov x0, #0")  # Ok tag = 0
              self._emit(f"    str x0, [x29, #{offset}]")
              self._gen_expr(ok_value)
              self._emit(f"    str x0, [x29, #{offset - 8}]")
            case ErrExpr(err_value):
              self._emit("    mov x0, #1")  # Err tag = 1
              self._emit(f"    str x0, [x29, #{offset}]")
              self._gen_expr(err_value)
              self._emit(f"    str x0, [x29, #{offset - 8}]")
            case _:
              # Expression that returns a Result (e.g., function call)
              self._gen_expr(value)
              # x0 has tag, x1 has payload (if Result was returned from function)
              self._emit(f"    str x0, [x29, #{offset}]")
              self._emit(f"    str x1, [x29, #{offset - 8}]")
          self.next_slot += 2

        elif is_dict_type(type_str):
          # Dict type: store pointer
          match value:
            case DictLiteral(entries):
              # Create dict from literal
              self._gen_dict_literal(entries)
              self._emit(f"    str x0, [x29, #{offset}]")
            case _:
              # Empty dict or expression returning dict
              self._gen_expr(value)
              self._emit(f"    str x0, [x29, #{offset}]")
          self.next_slot += 1

        else:
          # Simple type: evaluate and store
          self._gen_expr(value)
          self._emit(f"    str x0, [x29, #{offset}]")
          self.next_slot += 1

      case AssignStmt(name, value):
        # Evaluate value to x0
        self._gen_expr(value)
        # Store to existing variable slot
        offset, _ = self.locals[name]
        self._emit(f"    str x0, [x29, #{offset}]")

      case IndexAssignStmt(target, index, value):
        # Get target address
        match target:
          case VarExpr(name):
            offset, type_str = self.locals[name]
            # Evaluate index/key
            self._gen_expr(index)
            self._emit("    str x0, [sp, #-16]!")  # Save index/key

            # Evaluate value
            self._gen_expr(value)
            self._emit("    mov x2, x0")  # Value in x2

            self._emit("    ldr x1, [sp], #16")  # Restore index/key to x1

            if is_dict_type(type_str):
              # Dict: insert/update key-value pair (may grow and reallocate)
              self._emit(f"    ldr x0, [x29, #{offset}]")  # Dict pointer
              self._gen_dict_insert_inline()
              # Update local variable with potentially new dict pointer
              self._emit(f"    str x0, [x29, #{offset}]")
            elif is_vec_type(type_str):
              # List: load base pointer, access data[index]
              self._emit(f"    ldr x0, [x29, #{offset}]")  # Base pointer
              self._emit("    add x0, x0, #16")  # Skip header
              self._emit("    str x2, [x0, x1, lsl #3]")
            else:
              # Array: direct stack access
              self._emit(f"    add x0, x29, #{offset}")
              self._emit("    neg x1, x1")  # Negate for downward growth
              self._emit("    str x2, [x0, x1, lsl #3]")

      case ReturnStmt(value):
        # Evaluate return value to x0
        self._gen_expr(value)
        # Jump to epilogue
        self._emit(f"    b _{self.current_func_name}_epilogue")

      case ExprStmt(expr):
        # Evaluate expression (result discarded)
        self._gen_expr(expr)

      case IfStmt(condition, then_body, else_body):
        else_label = self._new_label("else")
        end_label = self._new_label("endif")

        # Evaluate condition
        self._gen_expr(condition)
        self._emit(f"    cbz x0, {else_label}")

        # Then branch
        for s in then_body:
          self._gen_stmt(s)
        self._emit(f"    b {end_label}")

        # Else branch
        self._emit(f"{else_label}:")
        if else_body:
          for s in else_body:
            self._gen_stmt(s)

        self._emit(f"{end_label}:")

      case WhileStmt(condition, body):
        loop_label = self._new_label("while")
        end_label = self._new_label("endwhile")

        self._emit(f"{loop_label}:")
        # Evaluate condition
        self._gen_expr(condition)
        self._emit(f"    cbz x0, {end_label}")

        # Loop body
        for s in body:
          self._gen_stmt(s)
        self._emit(f"    b {loop_label}")

        self._emit(f"{end_label}:")

      case ForStmt(var, start, end, body):
        loop_label = self._new_label("for")
        end_label = self._new_label("endfor")

        # Allocate loop variable slot
        offset = -16 - (self.next_slot * 8)
        self.locals[var] = (offset, "i64")
        self.next_slot += 1

        # Initialize loop variable with start value
        self._gen_expr(start)
        self._emit(f"    str x0, [x29, #{offset}]")

        # Store end value in a temp slot
        end_offset = -16 - (self.next_slot * 8)
        self.next_slot += 1
        self._gen_expr(end)
        self._emit(f"    str x0, [x29, #{end_offset}]")

        self._emit(f"{loop_label}:")
        # Check: loop_var < end
        self._emit(f"    ldr x0, [x29, #{offset}]")
        self._emit(f"    ldr x1, [x29, #{end_offset}]")
        self._emit("    cmp x0, x1")
        self._emit(f"    b.ge {end_label}")

        # Loop body
        for s in body:
          self._gen_stmt(s)

        # Increment loop variable
        self._emit(f"    ldr x0, [x29, #{offset}]")
        self._emit("    add x0, x0, #1")
        self._emit(f"    str x0, [x29, #{offset}]")
        self._emit(f"    b {loop_label}")

        self._emit(f"{end_label}:")

      case FieldAssignStmt(target, field, value):
        # Get struct variable
        match target:
          case VarExpr(name):
            var_offset, type_str = self.locals[name]
            if type_str in self.structs:
              fields = self.structs[type_str]
              field_idx = next(i for i, (f, _) in enumerate(fields) if f == field)
              # Evaluate value
              self._gen_expr(value)
              # Store to field slot
              self._emit(f"    str x0, [x29, #{var_offset - field_idx * 8}]")
          case _:
            pass  # Nested field access would need more work

      case DerefAssignStmt(target, value):
        # *ptr = value - store through pointer
        # First evaluate the value and save it
        self._gen_expr(value)
        self._emit("    str x0, [sp, #-16]!")  # Push value
        # Then evaluate the pointer
        self._gen_expr(target)
        self._emit("    mov x1, x0")  # Address in x1
        # Pop value and store through pointer
        self._emit("    ldr x0, [sp], #16")
        self._emit("    str x0, [x1]")

  def _gen_expr(self, expr: Expr) -> None:
    """Generate assembly for an expression, result in x0."""
    match expr:
      case IntLiteral(value):
        if 0 <= value <= 65535:
          self._emit(f"    mov x0, #{value}")
        elif -65536 <= value < 0:
          self._emit(f"    mov x0, #{value}")
        else:
          # For larger values, use movz/movk
          self._emit(f"    movz x0, #{value & 0xFFFF}")
          if value > 0xFFFF or value < 0:
            self._emit(f"    movk x0, #{(value >> 16) & 0xFFFF}, lsl #16")
          if abs(value) > 0xFFFFFFFF:
            self._emit(f"    movk x0, #{(value >> 32) & 0xFFFF}, lsl #32")
          if abs(value) > 0xFFFFFFFFFFFF:
            self._emit(f"    movk x0, #{(value >> 48) & 0xFFFF}, lsl #48")

      case BoolLiteral(value):
        self._emit(f"    mov x0, #{1 if value else 0}")

      case StringLiteral(value):
        label = self.strings[value]
        self._emit(f"    adrp x0, {label}@PAGE")
        self._emit(f"    add x0, x0, {label}@PAGEOFF")

      case VarExpr(name):
        offset, type_str = self.locals[name]
        self._emit(f"    ldr x0, [x29, #{offset}]")
        # For Result and enum types, also load payload into x1
        if is_result_type(type_str) or type_str in self.enums:
          self._emit(f"    ldr x1, [x29, #{offset - 8}]")

      case BinaryExpr(left, op, right):
        # Check for operator overloading
        expr_id = id(expr)
        if expr_id in self.operator_overloads:
          left_type, right_type, method_name, _ = self.operator_overloads[expr_id]
          self._gen_operator_call(left, right, left_type, right_type, method_name)
        else:
          # Evaluate left to x0, push to stack
          self._gen_expr(left)
          self._emit("    str x0, [sp, #-16]!")

          # Evaluate right to x0
          self._gen_expr(right)
          self._emit("    mov x1, x0")

          # Pop left to x0
          self._emit("    ldr x0, [sp], #16")

          # Apply operator
          match op:
            case "+":
              self._emit("    add x0, x0, x1")
            case "-":
              self._emit("    sub x0, x0, x1")
            case "*":
              self._emit("    mul x0, x0, x1")
            case "/":
              self._emit("    sdiv x0, x0, x1")
            case "%":
              # x0 = x0 - (x0 / x1) * x1
              self._emit("    sdiv x2, x0, x1")
              self._emit("    msub x0, x2, x1, x0")
            case "<":
              self._emit("    cmp x0, x1")
              self._emit("    cset x0, lt")
            case ">":
              self._emit("    cmp x0, x1")
              self._emit("    cset x0, gt")
            case "<=":
              self._emit("    cmp x0, x1")
              self._emit("    cset x0, le")
            case ">=":
              self._emit("    cmp x0, x1")
              self._emit("    cset x0, ge")
            case "==":
              self._emit("    cmp x0, x1")
              self._emit("    cset x0, eq")
            case "!=":
              self._emit("    cmp x0, x1")
              self._emit("    cset x0, ne")
            case "and":
              self._emit("    and x0, x0, x1")
            case "or":
              self._emit("    orr x0, x0, x1")

      case UnaryExpr(op, operand):
        self._gen_expr(operand)
        match op:
          case "-":
            self._emit("    neg x0, x0")
          case "not":
            # Logical not: 0 -> 1, non-zero -> 0
            self._emit("    cmp x0, #0")
            self._emit("    cset x0, eq")

      case CallExpr(name, args, kwargs, resolved_name):
        # Use resolved_name if available (from type inference), otherwise use name
        func_name = resolved_name if resolved_name else name
        # Resolve kwargs to positional order
        resolved_args = self._resolve_kwargs_codegen(func_name, args, kwargs)
        if func_name == "print":
          self._gen_print(resolved_args)
        elif func_name in self.locals:
          # Closure call - variable holds a function pointer
          self._gen_closure_call(func_name, resolved_args)
        else:
          self._gen_call(func_name, resolved_args)

      case ArrayLiteral(elements):
        # Array literal outside of let: shouldn't happen after type checking
        # But we can handle it by returning address of first element
        if elements:
          self._gen_expr(elements[0])

      case IndexExpr(target, index):
        match target:
          case VarExpr(name):
            offset, type_str = self.locals[name]
            # Evaluate index
            self._gen_expr(index)
            self._emit("    mov x1, x0")  # Index/key in x1

            if is_dict_type(type_str):
              # Dict lookup
              self._emit(f"    ldr x0, [x29, #{offset}]")  # Dict pointer
              self._gen_dict_lookup_inline()
            elif is_vec_type(type_str):
              # List: load base pointer, access data[index]
              self._emit(f"    ldr x0, [x29, #{offset}]")  # Base pointer
              self._emit("    add x0, x0, #16")  # Skip header
              self._emit("    ldr x0, [x0, x1, lsl #3]")
            else:
              # Array: direct stack access
              self._emit(f"    add x0, x29, #{offset}")
              self._emit("    neg x1, x1")  # Negate for downward growth
              self._emit("    ldr x0, [x0, x1, lsl #3]")
          case _:
            # Handle nested index expressions
            self._gen_expr(target)
            self._emit("    str x0, [sp, #-16]!")  # Save target address
            self._gen_expr(index)
            self._emit("    mov x1, x0")  # Index in x1
            self._emit("    ldr x0, [sp], #16")  # Restore target
            self._emit("    ldr x0, [x0, x1, lsl #3]")

      case SliceExpr(target, start_expr, stop_expr, step_expr):
        self._gen_slice_expr(target, start_expr, stop_expr, step_expr)

      case MethodCallExpr(target, method, args):
        self._gen_method_call(target, method, args)

      case StructLiteral(name, type_args, fields):
        # Struct literal outside of let: evaluate fields and return in registers
        # Resolve mangled struct name for generic structs
        struct_name = name
        if type_args:
          # Apply type substitution if we're in a generic context
          resolved_type_args: list[str] = []
          for a in type_args:
            arg_str = type_to_str(a, self.type_aliases)
            current_subst = getattr(self, "current_type_subst", None)
            if current_subst is not None and arg_str in current_subst:
              arg_str = current_subst[arg_str]
            resolved_type_args.append(arg_str)
          struct_name = f"{name}<{','.join(resolved_type_args)}>"
        struct_fields = self.structs[struct_name]
        value_map = dict(fields)
        # Store fields on stack temporarily (in order)
        for i, (field_name, _) in enumerate(struct_fields):
          if field_name in value_map:
            self._gen_expr(value_map[field_name])
            self._emit("    str x0, [sp, #-16]!")
          else:
            self._emit("    str xzr, [sp, #-16]!")
        # Load fields into registers x0, x1, x2, ... for return
        # Note: fields are pushed in order, so the last field is at [sp], first at [sp + (n-1)*16]
        for i in range(len(struct_fields)):
          self._emit(f"    ldr x{i}, [sp, #{(len(struct_fields) - 1 - i) * 16}]")
        # Clean up stack
        self._emit(f"    add sp, sp, #{len(struct_fields) * 16}")

      case FieldAccessExpr(target, field):
        match target:
          case VarExpr(name):
            var_offset, type_str = self.locals[name]
            if type_str in self.structs:
              fields = self.structs[type_str]
              field_idx = next(i for i, (f, _) in enumerate(fields) if f == field)
              # Load field value
              self._emit(f"    ldr x0, [x29, #{var_offset - field_idx * 8}]")
          case _:
            # Handle nested field access
            self._gen_expr(target)
            # Would need type info to know field offset - limited support
            pass

      case TupleLiteral(elements):
        # Tuple literal outside let: allocate temp on stack and return first element
        # This is uncommon but we handle it
        for elem in elements:
          self._gen_expr(elem)
          self._emit("    str x0, [sp, #-16]!")
        # Load first element as result (or 0 for empty)
        if elements:
          self._emit(f"    ldr x0, [sp, #{(len(elements) - 1) * 16}]")
        else:
          self._emit("    mov x0, #0")
        # Clean up stack
        self._emit(f"    add sp, sp, #{len(elements) * 16}")

      case TupleIndexExpr(target, index):
        match target:
          case VarExpr(name):
            var_offset, type_str = self.locals[name]
            if is_tuple_type(type_str):
              # Load tuple element by index
              self._emit(f"    ldr x0, [x29, #{var_offset - index * 8}]")
          case _:
            # Nested tuple access would need type tracking
            self._gen_expr(target)
            pass

      case EnumLiteral(enum_name, type_args, variant_name, payload):
        # Enum literal: return tag in x0 and payload in x1

        # Construct the mangled enum name for generic enums
        mangled_enum_name = enum_name
        if type_args:
          type_arg_strs = [type_to_str(a, self.type_aliases) for a in type_args]
          mangled_enum_name = f"{enum_name}<{','.join(type_arg_strs)}>"

        variants = self.enums[mangled_enum_name]
        tag, _ = variants[variant_name]

        if payload is not None:
          # Evaluate payload first, then set tag
          self._gen_expr(payload)
          self._emit("    mov x1, x0")  # Payload in x1
          self._emit(f"    mov x0, #{tag}")  # Tag in x0
        else:
          # No payload
          self._emit(f"    mov x0, #{tag}")  # Tag in x0
          self._emit("    mov x1, #0")  # No payload

      case MatchExpr(target, arms):
        self._gen_match(target, arms)

      case RefExpr(target, _mutable):
        # &x or &mut x - compute address of target
        match target:
          case VarExpr(name):
            offset, _ = self.locals[name]
            # Load address of variable into x0
            self._emit(f"    add x0, x29, #{offset}")
          case _:
            # For complex expressions, evaluate and the result IS the address
            # This handles things like &arr[i] etc.
            self._gen_expr(target)

      case DerefExpr(target):
        # *ptr - load value from address
        self._gen_expr(target)  # Get address into x0
        self._emit("    ldr x0, [x0]")  # Load value at that address

      case ClosureExpr(params, return_type, body):
        # Generate a closure: create a function and return its address
        label = f"_closure{self.closure_counter}"
        self.closure_counter += 1
        # Store closure info for later generation
        self.closures.append((label, params, return_type, body))
        # Return address of the closure function
        self._emit(f"    adrp x0, {label}@PAGE")
        self._emit(f"    add x0, x0, {label}@PAGEOFF")

      case ClosureCallExpr(target, args):
        # Call a closure expression result
        # First, push all args to stack
        for i, arg in enumerate(reversed(args)):
          self._gen_expr(arg)
          self._emit("    str x0, [sp, #-16]!")
        # Evaluate target to get function pointer
        self._gen_expr(target)
        self._emit("    mov x9, x0")  # Function pointer in x9
        # Pop args into registers x0-x7
        for i in range(len(args)):
          self._emit(f"    ldr x{i}, [sp], #16")
        # Call through function pointer
        self._emit("    blr x9")

      case OkExpr(value):
        # Ok(value) - create Result with tag=0 (Ok) and payload=value
        self._gen_expr(value)
        self._emit("    mov x1, x0")  # Payload in x1
        self._emit("    mov x0, #0")  # Tag in x0 (Ok = 0)

      case ErrExpr(value):
        # Err(error) - create Result with tag=1 (Err) and payload=error
        self._gen_expr(value)
        self._emit("    mov x1, x0")  # Payload in x1
        self._emit("    mov x0, #1")  # Tag in x0 (Err = 1)

      case TryExpr(target):
        # expr? - check if Option is None or Result is Err, if so return early
        self._gen_expr(target)
        # x0 = tag, x1 = payload

        # Check if this is Option or Result type
        expr_id = id(expr)
        try_type = self.try_expr_types.get(expr_id, "result")  # Default to Result for backwards compatibility

        if try_type == "option":
          # Option: Some has tag 0, None has tag 1
          none_label = self._new_label("try_none")
          some_label = self._new_label("try_some")
          # Check if tag is None (1)
          self._emit("    cmp x0, #1")
          self._emit(f"    b.eq {none_label}")
          # Some path: result is in x1 (payload), move to x0
          self._emit("    mov x0, x1")
          self._emit(f"    b {some_label}")
          # None path: return early with None
          self._emit(f"{none_label}:")
          self._emit("    mov x0, #1")  # Tag = None
          self._emit("    mov x1, #0")  # No payload for None
          self._emit(f"    b _{self.current_func_name}_epilogue")
          self._emit(f"{some_label}:")
        else:
          # Result: Ok has tag 0, Err has tag 1
          err_label = self._new_label("try_err")
          ok_label = self._new_label("try_ok")
          # Check if tag is Err (1)
          self._emit("    cmp x0, #1")
          self._emit(f"    b.eq {err_label}")
          # Ok path: result is in x1 (payload), move to x0
          self._emit("    mov x0, x1")
          self._emit(f"    b {ok_label}")
          # Err path: return early with Err
          self._emit(f"{err_label}:")
          self._emit("    mov x0, #1")  # Tag = Err
          # x1 already has error payload
          self._emit(f"    b _{self.current_func_name}_epilogue")
          self._emit(f"{ok_label}:")

      case ListComprehension(element_expr, var_name, start, end, condition):
        # [expr for var in range(start, end) if condition]
        # This generates: let result = []; for var in range(start, end): if condition: result.push(expr)
        self._gen_list_comprehension(element_expr, var_name, start, end, condition)

      case DictLiteral(entries):
        # {key: value, ...}
        self._gen_dict_literal(entries)

      case DictComprehension(key_expr, value_expr, var_name, start, end, condition):
        # {k: v for var in range(start, end) if condition}
        self._gen_dict_comprehension(key_expr, value_expr, var_name, start, end, condition)

  def _gen_list_comprehension(self, element_expr: Expr, var_name: str, start: Expr, end: Expr, condition: Expr | None) -> None:
    """Generate code for list comprehension: [expr for var in range(start, end) if condition]."""
    # Allocate a temporary vec on stack
    # We need: vec_ptr, loop_var, start_val, end_val
    # Use stack offsets relative to sp for temporaries

    # Save any existing binding for var_name (for nested comprehensions or shadowing)
    old_binding = self.locals.get(var_name)
    saved_next_slot = self.next_slot

    # Save start and end values to stack
    self._gen_expr(start)
    self._emit("    str x0, [sp, #-16]!")  # start at [sp]
    self._gen_expr(end)
    self._emit("    str x0, [sp, #-16]!")  # end at [sp], start at [sp+16]

    # Allocate initial vec: malloc(24) for header (cap=8, len=0) + space
    # Initial capacity 8 elements = 8 * 8 = 64 bytes
    self._emit("    mov x0, #88")  # 24 (header) + 64 (8 elements)
    self._emit("    bl _malloc")
    # Store capacity and length in header
    self._emit("    mov x1, #8")
    self._emit("    str x1, [x0]")  # capacity = 8
    self._emit("    str xzr, [x0, #8]")  # length = 0
    # Save vec ptr to stack
    self._emit("    str x0, [sp, #-16]!")  # vec_ptr at [sp], end at [sp+16], start at [sp+32]

    # Allocate loop variable
    loop_var_offset = -16 - (self.next_slot * 8)
    self.locals[var_name] = (loop_var_offset, "i64")
    self.next_slot += 1

    # Initialize loop variable with start
    self._emit("    ldr x0, [sp, #32]")  # Load start
    self._emit(f"    str x0, [x29, #{loop_var_offset}]")

    # Loop labels
    loop_start = self._new_label("lc_loop")
    loop_end = self._new_label("lc_end")
    cond_skip = self._new_label("lc_skip") if condition else None

    self._emit(f"{loop_start}:")

    # Check loop condition: var < end
    self._emit(f"    ldr x0, [x29, #{loop_var_offset}]")  # Load var
    self._emit("    ldr x1, [sp, #16]")  # Load end
    self._emit("    cmp x0, x1")
    self._emit(f"    b.ge {loop_end}")

    # If there's a condition, check it
    if condition is not None:
      self._gen_expr(condition)
      self._emit("    cmp x0, #0")
      self._emit(f"    b.eq {cond_skip}")

    # Evaluate element expression
    self._gen_expr(element_expr)
    self._emit("    mov x2, x0")  # Save element value in x2

    # Push to vec: load vec ptr, check capacity, maybe grow, store element
    self._emit("    ldr x0, [sp]")  # Load vec_ptr
    self._emit("    ldr x3, [x0]")  # capacity
    self._emit("    ldr x4, [x0, #8]")  # length

    # Check if we need to grow (length >= capacity)
    grow_label = self._new_label("lc_grow")
    no_grow_label = self._new_label("lc_no_grow")
    self._emit("    cmp x4, x3")
    self._emit(f"    b.ge {grow_label}")
    self._emit(f"    b {no_grow_label}")

    # Grow the vec
    self._emit(f"{grow_label}:")
    # New capacity = old * 2
    self._emit("    lsl x3, x3, #1")  # capacity * 2
    # New size = 24 + capacity * 8
    self._emit("    lsl x5, x3, #3")  # capacity * 8
    self._emit("    add x5, x5, #24")  # + header size
    # Save registers we need
    self._emit("    str x2, [sp, #-16]!")  # Save element value
    self._emit("    str x3, [sp, #-16]!")  # Save new capacity
    # realloc(vec_ptr, new_size)
    self._emit("    ldr x0, [sp, #32]")  # Load vec_ptr (now at sp+32 due to pushes)
    self._emit("    mov x1, x5")
    self._emit("    bl _realloc")
    # Restore and update
    self._emit("    ldr x3, [sp], #16")  # Restore new capacity
    self._emit("    ldr x2, [sp], #16")  # Restore element value
    self._emit("    str x0, [sp]")  # Update vec_ptr on stack
    self._emit("    str x3, [x0]")  # Update capacity in header
    self._emit("    ldr x4, [x0, #8]")  # Reload length

    self._emit(f"{no_grow_label}:")
    # Store element at data[length]
    self._emit("    ldr x0, [sp]")  # Load vec_ptr
    self._emit("    ldr x4, [x0, #8]")  # Load current length
    self._emit("    add x5, x0, #16")  # data pointer
    self._emit("    str x2, [x5, x4, lsl #3]")  # data[length] = element
    # Increment length
    self._emit("    add x4, x4, #1")
    self._emit("    str x4, [x0, #8]")

    # Skip label for condition
    if cond_skip is not None:
      self._emit(f"{cond_skip}:")

    # Increment loop variable
    self._emit(f"    ldr x0, [x29, #{loop_var_offset}]")
    self._emit("    add x0, x0, #1")
    self._emit(f"    str x0, [x29, #{loop_var_offset}]")
    self._emit(f"    b {loop_start}")

    self._emit(f"{loop_end}:")

    # Return vec pointer
    self._emit("    ldr x0, [sp]")
    # Clean up stack (vec_ptr, end, start)
    self._emit("    add sp, sp, #48")

    # Restore old binding if any
    if old_binding is not None:
      self.locals[var_name] = old_binding
    else:
      del self.locals[var_name]
    self.next_slot = saved_next_slot

  def _gen_dict_literal(self, entries: tuple[tuple[Expr, Expr], ...]) -> None:
    """Generate code for dict literal: {key: value, ...}.

    Dict structure:
      - 8 bytes: capacity
      - 8 bytes: length (number of entries)
      - entries: [key (8), value (8), occupied (8)] * capacity
    Entry size = 24 bytes
    """
    # Initial capacity (at least 16, or 2x entries) - use 16 minimum to reduce resizing
    initial_capacity = max(16, len(entries) * 2)
    # Total size: 16 (header) + capacity * 24 (entries)
    total_size = 16 + initial_capacity * 24

    # Allocate dict
    self._emit(f"    mov x0, #{total_size}")
    self._emit("    bl _malloc")
    self._emit("    str x0, [sp, #-16]!")  # Save dict ptr

    # Initialize header
    self._emit(f"    mov x1, #{initial_capacity}")
    self._emit("    str x1, [x0]")  # capacity
    self._emit("    str xzr, [x0, #8]")  # length = 0

    # Zero out all occupied flags
    self._emit("    add x1, x0, #16")  # entries start
    for i in range(initial_capacity):
      self._emit(f"    str xzr, [x1, #{i * 24 + 16}]")  # occupied = 0

    # Insert each entry
    for key, value in entries:
      # Save key and value on stack
      self._gen_expr(key)
      self._emit("    str x0, [sp, #-16]!")  # Save key
      self._gen_expr(value)
      self._emit("    str x0, [sp, #-16]!")  # Save value

      # Load dict ptr, key, value
      self._emit("    ldr x0, [sp, #32]")  # dict ptr
      self._emit("    ldr x1, [sp, #16]")  # key
      self._emit("    ldr x2, [sp]")  # value

      # Call dict_insert (inline)
      self._gen_dict_insert_inline()

      # Clean up key/value from stack
      self._emit("    add sp, sp, #32")

    # Return dict pointer
    self._emit("    ldr x0, [sp], #16")

  def _gen_dict_comprehension(
    self, key_expr: Expr, value_expr: Expr, var_name: str, start: Expr, end: Expr, condition: Expr | None
  ) -> None:
    """Generate code for dict comprehension: {k: v for var in range(start, end) if condition}."""
    # Save any existing binding for var_name (for nested comprehensions or shadowing)
    old_binding = self.locals.get(var_name)
    saved_next_slot = self.next_slot

    # Save start and end values to stack
    self._gen_expr(start)
    self._emit("    str x0, [sp, #-16]!")  # start at [sp]
    self._gen_expr(end)
    self._emit("    str x0, [sp, #-16]!")  # end at [sp], start at [sp+16]

    # Allocate initial dict: malloc(16 + 16*24) for header + 16 entries
    initial_capacity = 16
    total_size = 16 + initial_capacity * 24
    self._emit(f"    mov x0, #{total_size}")
    self._emit("    bl _malloc")
    # Store capacity and length in header
    self._emit(f"    mov x1, #{initial_capacity}")
    self._emit("    str x1, [x0]")  # capacity = 16
    self._emit("    str xzr, [x0, #8]")  # length = 0
    # Zero out occupied flags
    for i in range(initial_capacity):
      self._emit(f"    str xzr, [x0, #{16 + i * 24 + 16}]")  # occupied = 0
    # Save dict ptr to stack
    self._emit("    str x0, [sp, #-16]!")  # dict_ptr at [sp], end at [sp+16], start at [sp+32]

    # Allocate loop variable
    loop_var_offset = -16 - (self.next_slot * 8)
    self.locals[var_name] = (loop_var_offset, "i64")
    self.next_slot += 1

    # Initialize loop variable with start
    self._emit("    ldr x0, [sp, #32]")  # Load start
    self._emit(f"    str x0, [x29, #{loop_var_offset}]")

    # Loop labels
    loop_start = self._new_label("dc_loop")
    loop_end = self._new_label("dc_end")
    cond_skip = self._new_label("dc_skip") if condition else None

    self._emit(f"{loop_start}:")

    # Check loop condition: var < end
    self._emit(f"    ldr x0, [x29, #{loop_var_offset}]")  # Load var
    self._emit("    ldr x1, [sp, #16]")  # Load end
    self._emit("    cmp x0, x1")
    self._emit(f"    b.ge {loop_end}")

    # If there's a condition, check it
    if condition is not None:
      self._gen_expr(condition)
      self._emit("    cmp x0, #0")
      self._emit(f"    b.eq {cond_skip}")

    # Evaluate key expression
    self._gen_expr(key_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save key

    # Evaluate value expression
    self._gen_expr(value_expr)
    self._emit("    mov x2, x0")  # Value in x2

    # Load key and dict ptr
    self._emit("    ldr x1, [sp], #16")  # Key in x1
    self._emit("    ldr x0, [sp]")  # Dict ptr in x0

    # Insert into dict (handles growth)
    self._gen_dict_insert_inline()

    # Update dict pointer on stack (may have changed due to growth)
    self._emit("    str x0, [sp]")

    # Skip label for condition
    if cond_skip is not None:
      self._emit(f"{cond_skip}:")

    # Increment loop variable
    self._emit(f"    ldr x0, [x29, #{loop_var_offset}]")
    self._emit("    add x0, x0, #1")
    self._emit(f"    str x0, [x29, #{loop_var_offset}]")
    self._emit(f"    b {loop_start}")

    self._emit(f"{loop_end}:")

    # Return dict pointer
    self._emit("    ldr x0, [sp]")
    # Clean up stack (dict_ptr, end, start)
    self._emit("    add sp, sp, #48")

    # Restore old binding if any
    if old_binding is not None:
      self.locals[var_name] = old_binding
    else:
      del self.locals[var_name]
    self.next_slot = saved_next_slot

  def _gen_dict_insert_inline(self) -> None:
    """Generate inline dict insert: x0=dict_ptr, x1=key, x2=value.

    Uses simple linear probing with modulo hashing.
    Grows dict if load factor > 70%.
    """
    # First check if we need to grow (length >= capacity * 0.7)
    # Save registers
    self._emit("    str x0, [sp, #-16]!")  # dict_ptr
    self._emit("    str x1, [sp, #-16]!")  # key
    self._emit("    str x2, [sp, #-16]!")  # value

    grow_check_label = self._new_label("dict_grow_check")
    no_grow_label = self._new_label("dict_no_grow")
    grow_label = self._new_label("dict_grow")

    self._emit(f"{grow_check_label}:")
    self._emit("    ldr x0, [sp, #32]")  # dict_ptr
    self._emit("    ldr x3, [x0]")  # capacity
    self._emit("    ldr x4, [x0, #8]")  # length
    # Check if length * 10 >= capacity * 7 (70% load)
    self._emit("    mov x5, #10")
    self._emit("    mul x5, x4, x5")  # length * 10
    self._emit("    mov x6, #7")
    self._emit("    mul x6, x3, x6")  # capacity * 7
    self._emit("    cmp x5, x6")
    self._emit(f"    b.lt {no_grow_label}")

    # Need to grow - double capacity
    self._emit(f"{grow_label}:")
    self._emit("    ldr x0, [sp, #32]")  # old dict_ptr
    self._emit("    ldr x3, [x0]")  # old capacity
    self._emit("    lsl x7, x3, #1")  # new capacity = old * 2
    # Allocate new dict: 16 + new_capacity * 24
    self._emit("    mov x8, #24")
    self._emit("    mul x8, x7, x8")
    self._emit("    add x0, x8, #16")
    self._emit("    bl _malloc")  # new dict in x0
    self._emit("    str x0, [sp, #-16]!")  # save new dict ptr

    # Initialize new dict header
    self._emit("    ldr x7, [sp, #48]")  # old dict
    self._emit("    ldr x3, [x7]")  # old capacity
    self._emit("    lsl x3, x3, #1")  # new capacity
    self._emit("    str x3, [x0]")  # store new capacity
    self._emit("    str xzr, [x0, #8]")  # length = 0

    # Zero out occupied flags in new dict
    zero_loop = self._new_label("dict_zero")
    zero_done = self._new_label("dict_zero_done")
    self._emit("    mov x4, #0")  # counter
    self._emit(f"{zero_loop}:")
    self._emit("    cmp x4, x3")
    self._emit(f"    b.ge {zero_done}")
    self._emit("    mov x5, #24")
    self._emit("    mul x5, x4, x5")
    self._emit("    add x5, x0, x5")
    self._emit("    add x5, x5, #16")
    self._emit("    str xzr, [x5, #16]")  # occupied = 0
    self._emit("    add x4, x4, #1")
    self._emit(f"    b {zero_loop}")
    self._emit(f"{zero_done}:")

    # Rehash all entries from old dict to new dict
    rehash_loop = self._new_label("dict_rehash")
    rehash_next = self._new_label("dict_rehash_next")
    rehash_done = self._new_label("dict_rehash_done")
    self._emit("    ldr x7, [sp, #48]")  # old dict
    self._emit("    ldr x8, [x7]")  # old capacity
    self._emit("    mov x9, #0")  # counter

    self._emit(f"{rehash_loop}:")
    self._emit("    cmp x9, x8")
    self._emit(f"    b.ge {rehash_done}")

    # Check if old entry is occupied
    self._emit("    mov x10, #24")
    self._emit("    mul x10, x9, x10")
    self._emit("    add x10, x7, x10")
    self._emit("    add x10, x10, #16")  # old entry addr
    self._emit("    ldr x11, [x10, #16]")  # occupied?
    self._emit(f"    cbz x11, {rehash_next}")

    # Copy to new dict using inline insert
    self._emit("    ldr x1, [x10]")  # key
    self._emit("    ldr x2, [x10, #8]")  # value
    self._emit("    ldr x0, [sp]")  # new dict
    # Inline simple insert (no growth check needed during rehash)
    self._gen_dict_simple_insert()

    self._emit(f"{rehash_next}:")
    self._emit("    add x9, x9, #1")
    self._emit(f"    b {rehash_loop}")
    self._emit(f"{rehash_done}:")

    # Free old dict
    self._emit("    ldr x0, [sp, #48]")  # old dict
    self._emit("    bl _free")

    # Update stack with new dict ptr
    self._emit("    ldr x0, [sp], #16")  # pop new dict
    self._emit("    str x0, [sp, #32]")  # update dict_ptr on stack

    self._emit(f"{no_grow_label}:")
    # Restore registers and do the actual insert
    self._emit("    ldr x2, [sp], #16")  # value
    self._emit("    ldr x1, [sp], #16")  # key
    self._emit("    ldr x0, [sp], #16")  # dict_ptr
    self._gen_dict_simple_insert()

  def _gen_dict_simple_insert(self) -> None:
    """Simple dict insert without growth check: x0=dict_ptr, x1=key, x2=value."""
    # Load capacity
    self._emit("    ldr x3, [x0]")  # capacity
    # Hash: key % capacity (handle negative by using unsigned)
    self._emit("    udiv x4, x1, x3")
    self._emit("    msub x4, x4, x3, x1")  # x4 = key % capacity

    # Linear probe loop
    probe_loop = self._new_label("dict_probe")
    probe_found = self._new_label("dict_found")
    probe_empty = self._new_label("dict_empty")

    self._emit(f"{probe_loop}:")
    # Calculate entry address: dict_ptr + 16 + index * 24
    self._emit("    mov x5, #24")
    self._emit("    mul x5, x4, x5")
    self._emit("    add x5, x0, x5")
    self._emit("    add x5, x5, #16")  # x5 = entry address

    # Check if occupied
    self._emit("    ldr x6, [x5, #16]")  # occupied flag
    self._emit(f"    cbz x6, {probe_empty}")  # If not occupied, insert here

    # Check if key matches
    self._emit("    ldr x6, [x5]")  # stored key
    self._emit("    cmp x6, x1")
    self._emit(f"    b.eq {probe_found}")

    # Linear probe: next slot
    self._emit("    add x4, x4, #1")
    self._emit("    udiv x6, x4, x3")
    self._emit("    msub x4, x6, x3, x4")  # x4 = (x4 + 1) % capacity
    self._emit(f"    b {probe_loop}")

    # Empty slot found - insert new entry
    self._emit(f"{probe_empty}:")
    self._emit("    str x1, [x5]")  # key
    self._emit("    str x2, [x5, #8]")  # value
    self._emit("    mov x6, #1")
    self._emit("    str x6, [x5, #16]")  # occupied = 1
    # Increment length
    self._emit("    ldr x6, [x0, #8]")
    self._emit("    add x6, x6, #1")
    self._emit("    str x6, [x0, #8]")
    done_label = self._new_label("dict_insert_done")
    self._emit(f"    b {done_label}")

    # Existing key found - update value
    self._emit(f"{probe_found}:")
    self._emit("    str x2, [x5, #8]")  # update value

    self._emit(f"{done_label}:")

  def _gen_dict_lookup_inline(self) -> None:
    """Generate inline dict lookup: x0=dict_ptr, x1=key. Result in x0."""
    # x0 = dict_ptr, x1 = key
    # Load capacity
    self._emit("    ldr x3, [x0]")  # capacity
    # Hash: key % capacity
    self._emit("    udiv x4, x1, x3")
    self._emit("    msub x4, x4, x3, x1")  # x4 = key % capacity

    # Linear probe loop
    probe_loop = self._new_label("dict_get_probe")
    probe_found = self._new_label("dict_get_found")
    probe_not_found = self._new_label("dict_get_not_found")

    self._emit(f"{probe_loop}:")
    # Calculate entry address: dict_ptr + 16 + index * 24
    self._emit("    mov x5, #24")
    self._emit("    mul x5, x4, x5")
    self._emit("    add x5, x0, x5")
    self._emit("    add x5, x5, #16")  # x5 = entry address

    # Check if occupied
    self._emit("    ldr x6, [x5, #16]")  # occupied flag
    self._emit(f"    cbz x6, {probe_not_found}")  # If not occupied, key not found

    # Check if key matches
    self._emit("    ldr x6, [x5]")  # stored key
    self._emit("    cmp x6, x1")
    self._emit(f"    b.eq {probe_found}")

    # Linear probe: next slot
    self._emit("    add x4, x4, #1")
    self._emit("    udiv x6, x4, x3")
    self._emit("    msub x4, x6, x3, x4")  # x4 = (x4 + 1) % capacity
    self._emit(f"    b {probe_loop}")

    # Key found - return value
    self._emit(f"{probe_found}:")
    self._emit("    ldr x0, [x5, #8]")  # value
    done_label = self._new_label("dict_get_done")
    self._emit(f"    b {done_label}")

    # Key not found - return 0 (or could panic)
    self._emit(f"{probe_not_found}:")
    self._emit("    mov x0, #0")

    self._emit(f"{done_label}:")

  def _gen_dict_contains_inline(self) -> None:
    """Generate inline dict contains check: x0=dict_ptr, x1=key. Result (0 or 1) in x0."""
    # x0 = dict_ptr, x1 = key
    # Load capacity
    self._emit("    ldr x3, [x0]")  # capacity
    # Hash: key % capacity
    self._emit("    udiv x4, x1, x3")
    self._emit("    msub x4, x4, x3, x1")  # x4 = key % capacity

    # Linear probe loop
    probe_loop = self._new_label("dict_contains_probe")
    probe_found = self._new_label("dict_contains_found")
    probe_not_found = self._new_label("dict_contains_not_found")

    self._emit(f"{probe_loop}:")
    # Calculate entry address: dict_ptr + 16 + index * 24
    self._emit("    mov x5, #24")
    self._emit("    mul x5, x4, x5")
    self._emit("    add x5, x0, x5")
    self._emit("    add x5, x5, #16")  # x5 = entry address

    # Check if occupied
    self._emit("    ldr x6, [x5, #16]")  # occupied flag
    self._emit(f"    cbz x6, {probe_not_found}")  # If not occupied, key not found

    # Check if key matches
    self._emit("    ldr x6, [x5]")  # stored key
    self._emit("    cmp x6, x1")
    self._emit(f"    b.eq {probe_found}")

    # Linear probe: next slot
    self._emit("    add x4, x4, #1")
    self._emit("    udiv x6, x4, x3")
    self._emit("    msub x4, x6, x3, x4")  # x4 = (x4 + 1) % capacity
    self._emit(f"    b {probe_loop}")

    # Key found
    self._emit(f"{probe_found}:")
    self._emit("    mov x0, #1")
    done_label = self._new_label("dict_contains_done")
    self._emit(f"    b {done_label}")

    # Key not found
    self._emit(f"{probe_not_found}:")
    self._emit("    mov x0, #0")

    self._emit(f"{done_label}:")

  def _gen_dict_remove_inline(self) -> None:
    """Generate inline dict remove: x0=dict_ptr, x1=key. Result (0 or 1) in x0."""
    # x0 = dict_ptr, x1 = key
    # Load capacity
    self._emit("    ldr x3, [x0]")  # capacity
    # Hash: key % capacity
    self._emit("    udiv x4, x1, x3")
    self._emit("    msub x4, x4, x3, x1")  # x4 = key % capacity

    # Linear probe loop
    probe_loop = self._new_label("dict_remove_probe")
    probe_found = self._new_label("dict_remove_found")
    probe_not_found = self._new_label("dict_remove_not_found")

    self._emit(f"{probe_loop}:")
    # Calculate entry address: dict_ptr + 16 + index * 24
    self._emit("    mov x5, #24")
    self._emit("    mul x5, x4, x5")
    self._emit("    add x5, x0, x5")
    self._emit("    add x5, x5, #16")  # x5 = entry address

    # Check if occupied
    self._emit("    ldr x6, [x5, #16]")  # occupied flag
    self._emit(f"    cbz x6, {probe_not_found}")  # If not occupied, key not found

    # Check if key matches
    self._emit("    ldr x6, [x5]")  # stored key
    self._emit("    cmp x6, x1")
    self._emit(f"    b.eq {probe_found}")

    # Linear probe: next slot
    self._emit("    add x4, x4, #1")
    self._emit("    udiv x6, x4, x3")
    self._emit("    msub x4, x6, x3, x4")  # x4 = (x4 + 1) % capacity
    self._emit(f"    b {probe_loop}")

    # Key found - remove it
    self._emit(f"{probe_found}:")
    self._emit("    str xzr, [x5, #16]")  # Mark as unoccupied
    # Decrement length
    self._emit("    ldr x6, [x0, #8]")
    self._emit("    sub x6, x6, #1")
    self._emit("    str x6, [x0, #8]")
    self._emit("    mov x0, #1")  # Return true
    done_label = self._new_label("dict_remove_done")
    self._emit(f"    b {done_label}")

    # Key not found
    self._emit(f"{probe_not_found}:")
    self._emit("    mov x0, #0")  # Return false

    self._emit(f"{done_label}:")

  def _gen_print(self, args: tuple[Expr, ...]) -> None:
    """Generate code for print() builtin.

    On ARM64 macOS, variadic arguments (like printf's format args)
    are passed on the stack, not in registers.
    """
    arg = args[0]
    is_string = self._is_string_expr(arg)

    # Evaluate the argument
    self._gen_expr(arg)
    # Store the value on the stack for variadic argument
    self._emit("    str x0, [sp, #-16]!")
    # Load appropriate format string
    if is_string:
      self._emit("    adrp x0, _fmt_str@PAGE")
      self._emit("    add x0, x0, _fmt_str@PAGEOFF")
    else:
      self._emit("    adrp x0, _fmt_int@PAGE")
      self._emit("    add x0, x0, _fmt_int@PAGEOFF")
    # Call printf (variadic arg is at [sp])
    self._emit("    bl _printf")
    # Clean up stack
    self._emit("    add sp, sp, #16")
    # Return 0 (print returns i64)
    self._emit("    mov x0, #0")

  def _is_string_expr(self, expr: Expr) -> bool:
    """Check if an expression evaluates to a string."""
    match expr:
      case StringLiteral(_):
        return True
      case VarExpr(name):
        _, type_name = self.locals[name]
        return type_name == "str"
      case _:
        return False

  def _resolve_kwargs_codegen(self, name: str, args: tuple[Expr, ...], kwargs: tuple[tuple[str, Expr], ...]) -> tuple[Expr, ...]:
    """Resolve kwargs to positional argument order for codegen."""
    if not kwargs:
      return args

    # Get param names for this function
    param_names = self.function_params.get(name)
    if param_names is None:
      # Built-in function like print - kwargs validated by checker, extract values in order
      return args + tuple(value for _, value in kwargs)

    # Build resolved list
    resolved: list[Expr | None] = list(args) + [None] * len(kwargs)

    for kwarg_name, kwarg_value in kwargs:
      param_idx = param_names.index(kwarg_name)  # Checker already validated
      resolved[param_idx] = kwarg_value

    return tuple(resolved)  # type: ignore

  def _gen_call(self, name: str, args: tuple[Expr, ...]) -> None:
    """Generate code for a function call."""
    # Count total register slots needed (enums and Results use 2)
    slot_info: list[tuple[Expr, bool]] = []  # (arg, is_two_slot)
    for arg in args:
      is_two_slot = False
      match arg:
        case VarExpr(var_name):
          if var_name in self.locals:
            _, type_str = self.locals[var_name]
            is_two_slot = type_str in self.enums or is_result_type(type_str)
      slot_info.append((arg, is_two_slot))

    # Evaluate arguments and push to stack (in reverse order)
    for arg, is_two_slot in reversed(slot_info):
      if is_two_slot:
        # Push both tag and payload for enums/Results
        match arg:
          case VarExpr(var_name):
            offset, _ = self.locals[var_name]
            self._emit(f"    ldr x0, [x29, #{offset - 8}]")  # Payload first (will be popped second)
            self._emit("    str x0, [sp, #-16]!")
            self._emit(f"    ldr x0, [x29, #{offset}]")  # Tag (will be popped first)
            self._emit("    str x0, [sp, #-16]!")
          case _:
            # Expression that returns enum/Result (x0=tag, x1=payload)
            self._gen_expr(arg)
            self._emit("    str x1, [sp, #-16]!")  # Payload first
            self._emit("    str x0, [sp, #-16]!")  # Tag second
      else:
        self._gen_expr(arg)
        self._emit("    str x0, [sp, #-16]!")

    # Pop arguments into registers
    reg_idx = 0
    for _, is_two_slot in slot_info:
      if is_two_slot:
        self._emit(f"    ldr x{reg_idx}, [sp], #16")  # Tag
        reg_idx += 1
        self._emit(f"    ldr x{reg_idx}, [sp], #16")  # Payload
        reg_idx += 1
      else:
        self._emit(f"    ldr x{reg_idx}, [sp], #16")
        reg_idx += 1

    # Call function (handle generic names)
    if "<" in name:
      # Generic function - use mangled name
      label = self._mangle_generic_name(name)
      self._emit(f"    bl {label}")
    else:
      self._emit(f"    bl _{name}")

  def _gen_closure_call(self, name: str, args: tuple[Expr, ...]) -> None:
    """Generate code for calling a closure stored in a variable."""
    # Push args to stack in reverse order
    for arg in reversed(args):
      self._gen_expr(arg)
      self._emit("    str x0, [sp, #-16]!")

    # Load function pointer from variable
    offset, _ = self.locals[name]
    self._emit(f"    ldr x9, [x29, #{offset}]")

    # Pop args into registers
    for i in range(len(args)):
      self._emit(f"    ldr x{i}, [sp], #16")

    # Call through function pointer
    self._emit("    blr x9")

  def _gen_slice_expr(self, target: Expr, start_expr: Expr | None, stop_expr: Expr | None, step_expr: Expr | None) -> None:
    """Generate code for slice expressions like arr[1:3] or vec[::2].

    Returns a new vec containing the sliced elements.
    Stack layout during computation:
      [sp+40]: step (or 1 if not provided)
      [sp+32]: stop (or len if not provided)
      [sp+24]: start (or 0 if not provided)
      [sp+16]: source length
      [sp+8]:  source data pointer
      [sp+0]:  scratch space
    """
    slice_label = self._new_label("slice")
    slice_loop = f"{slice_label}_loop"
    slice_done = f"{slice_label}_done"

    # Evaluate target and get type info
    match target:
      case VarExpr(name):
        offset, type_str = self.locals[name]

        if is_vec_type(type_str):
          # Vec: load base pointer and length from header
          self._emit(f"    ldr x0, [x29, #{offset}]")  # Vec pointer
          self._emit("    ldr x2, [x0, #8]")  # Length
          self._emit("    add x1, x0, #16")  # Data pointer (skip header)
          is_vec = True
        elif is_array_type(type_str):
          # Array: compute stack address and use compile-time length
          array_size = get_array_size(type_str)
          self._emit(f"    add x1, x29, #{offset}")  # Data pointer (stack address)
          self._emit(f"    mov x2, #{array_size}")  # Length (compile-time)
          is_vec = False
        else:
          # String slicing not yet supported in codegen
          raise NotImplementedError(f"Slice codegen for type {type_str} not implemented")
      case _:
        # For complex targets, evaluate and assume vec
        self._gen_expr(target)
        self._emit("    ldr x2, [x0, #8]")  # Length
        self._emit("    add x1, x0, #16")  # Data pointer
        is_vec = True

    # Save source data ptr (x1) and length (x2) on stack
    self._emit("    str x1, [sp, #-48]!")  # Reserve 48 bytes, store data ptr at [sp]
    self._emit("    str x2, [sp, #8]")  # Length at [sp+8]

    # Compute start (default 0)
    if start_expr is not None:
      self._gen_expr(start_expr)
      # Handle negative indices
      self._emit("    cmp x0, #0")
      self._emit(f"    b.ge {slice_label}_start_ok")
      self._emit("    ldr x1, [sp, #8]")  # Load length
      self._emit("    add x0, x0, x1")  # start = start + len
      self._emit(f"{slice_label}_start_ok:")
    else:
      self._emit("    mov x0, #0")
    self._emit("    str x0, [sp, #16]")  # Store start at [sp+16]

    # Compute stop (default length)
    if stop_expr is not None:
      self._gen_expr(stop_expr)
      # Handle negative indices
      self._emit("    cmp x0, #0")
      self._emit(f"    b.ge {slice_label}_stop_ok")
      self._emit("    ldr x1, [sp, #8]")  # Load length
      self._emit("    add x0, x0, x1")  # stop = stop + len
      self._emit(f"{slice_label}_stop_ok:")
    else:
      self._emit("    ldr x0, [sp, #8]")  # Default to length
    self._emit("    str x0, [sp, #24]")  # Store stop at [sp+24]

    # Compute step (default 1)
    if step_expr is not None:
      self._gen_expr(step_expr)
    else:
      self._emit("    mov x0, #1")
    self._emit("    str x0, [sp, #32]")  # Store step at [sp+32]

    # Calculate result length: max(0, (stop - start + step - 1) / step) for positive step
    # For simplicity, we only support positive step for now
    self._emit("    ldr x1, [sp, #16]")  # start
    self._emit("    ldr x2, [sp, #24]")  # stop
    self._emit("    ldr x3, [sp, #32]")  # step
    self._emit("    sub x0, x2, x1")  # stop - start
    self._emit("    cmp x0, #0")
    self._emit(f"    b.le {slice_label}_empty")
    # Ceiling division: (stop - start + step - 1) / step
    self._emit("    add x0, x0, x3")
    self._emit("    sub x0, x0, #1")
    self._emit("    sdiv x0, x0, x3")  # Result length
    self._emit(f"    b {slice_label}_alloc")

    self._emit(f"{slice_label}_empty:")
    self._emit("    mov x0, #0")  # Empty slice

    # Allocate new vec: 16 (header) + length * 8 (data)
    self._emit(f"{slice_label}_alloc:")
    self._emit("    str x0, [sp, #40]")  # Save result length at [sp+40]
    self._emit("    lsl x0, x0, #3")  # length * 8
    self._emit("    add x0, x0, #16")  # + header
    self._emit("    bl _malloc")
    self._emit("    str x0, [sp, #-16]!")  # Push new vec ptr (now at [sp])

    # Initialize vec header
    self._emit("    ldr x1, [sp, #56]")  # Result length from [sp+40+16]
    self._emit("    str x1, [x0]")  # capacity = length
    self._emit("    str x1, [x0, #8]")  # length

    # Copy loop: iterate with step
    self._emit("    ldr x4, [sp]")  # New vec ptr
    self._emit("    add x4, x4, #16")  # New data ptr
    self._emit("    ldr x5, [sp, #16]")  # Source data ptr
    self._emit("    ldr x6, [sp, #32]")  # start index
    self._emit("    ldr x7, [sp, #40]")  # stop index
    self._emit("    ldr x8, [sp, #48]")  # step
    self._emit("    mov x9, #0")  # dest index

    self._emit(f"{slice_loop}:")
    self._emit("    cmp x6, x7")  # Compare current index with stop
    self._emit(f"    b.ge {slice_done}")

    # For arrays, we need to negate the index (stack grows downward)
    if not is_vec:
      self._emit("    neg x10, x6")
      self._emit("    ldr x10, [x5, x10, lsl #3]")  # Load source[index]
    else:
      self._emit("    ldr x10, [x5, x6, lsl #3]")  # Load source[index]
    self._emit("    str x10, [x4, x9, lsl #3]")  # Store to dest[dest_index]

    self._emit("    add x6, x6, x8")  # index += step
    self._emit("    add x9, x9, #1")  # dest_index++
    self._emit(f"    b {slice_loop}")

    self._emit(f"{slice_done}:")
    self._emit("    ldr x0, [sp]")  # Return new vec ptr
    self._emit("    add sp, sp, #64")  # Clean up stack (48 + 16)

  def _gen_operator_call(self, left: Expr, right: Expr, left_type: str, right_type: str, method_name: str) -> None:
    """Generate code for an overloaded binary operator (calls a struct method)."""
    # Get struct fields and method
    left_fields = self.structs[left_type]
    mangled_name = self.struct_methods[left_type][method_name]

    # Determine if right is a struct type
    right_is_struct = right_type in self.structs
    right_fields = self.structs.get(right_type, [])
    right_num_slots = len(right_fields) if right_is_struct else 1

    # For operator overload, left operand is 'self' (the struct), right is 'other'
    # We need to pass all fields of self, then all fields of other (or just the value for non-structs)
    #
    # Strategy: Evaluate left first (may use stack), save result, then evaluate right,
    # then push everything onto stack in correct order, then pop into registers.

    # Step 1: Evaluate left operand and save to a temp area on stack
    left_is_var = isinstance(left, VarExpr)
    left_offset = 0
    if left_is_var:
      left_offset, _ = self.locals[left.name]
    else:
      # Evaluate left expression - result is in x0, x1, ... for multi-field structs
      self._gen_expr(left)
      # Save left result to stack (we'll copy it later)
      for i in range(len(left_fields)):
        self._emit(f"    str x{i}, [sp, #-16]!")

    # Step 2: Evaluate right operand and push onto stack (in reverse field order for final layout)
    right_is_var = isinstance(right, VarExpr)
    if right_is_var:
      right_offset, _ = self.locals[right.name]
      if right_is_struct:
        for i in range(len(right_fields) - 1, -1, -1):
          self._emit(f"    ldr x0, [x29, #{right_offset - i * 8}]")
          self._emit("    str x0, [sp, #-16]!")
      else:
        self._emit(f"    ldr x0, [x29, #{right_offset}]")
        self._emit("    str x0, [sp, #-16]!")
    else:
      # Evaluate right expression - result is in x0, x1, ... for multi-field structs
      self._gen_expr(right)
      if right_is_struct:
        # Push all struct fields (in reverse order)
        for i in range(len(right_fields) - 1, -1, -1):
          self._emit(f"    str x{i}, [sp, #-16]!")
      else:
        self._emit("    str x0, [sp, #-16]!")

    # Step 3: Push left operand onto stack (in reverse field order)
    if left_is_var:
      for i in range(len(left_fields) - 1, -1, -1):
        self._emit(f"    ldr x0, [x29, #{left_offset - i * 8}]")
        self._emit("    str x0, [sp, #-16]!")
    else:
      # Left result is saved on stack, reload and push in reverse order
      # After pushing x0, x1 (left result) and then pushing right operand fields:
      #   x0 was pushed first, so it's at the highest address (furthest from sp)
      #   x1 was pushed second, etc.
      # After pushing right_num_slots values, the layout from current sp is:
      #   [sp + right_num_slots * 16 + (left_num - 1) * 16] = first pushed = field 0
      #   [sp + right_num_slots * 16 + (left_num - 2) * 16] = second pushed = field 1
      #   ...
      #   [sp + right_num_slots * 16] = last pushed = field (left_num - 1)
      # We need to push in reverse field order: push field (left_num-1), ..., field 1, field 0
      # So load all first into temp registers, then push in reverse order
      base_offset = right_num_slots * 16
      # Load all left fields into temp registers x9, x10, ... (up to 8 fields max)
      for i in range(len(left_fields)):
        # field i is at base_offset + (left_num - 1 - i) * 16
        offset = base_offset + (len(left_fields) - 1 - i) * 16
        self._emit(f"    ldr x{9 + i}, [sp, #{offset}]")
      # Push in reverse order: field (left_num-1) first, then field (left_num-2), ..., field 0
      for i in range(len(left_fields) - 1, -1, -1):
        self._emit(f"    str x{9 + i}, [sp, #-16]!")

    # Step 4: Pop into registers: left fields first, then right fields
    reg_idx = 0
    for _ in left_fields:
      self._emit(f"    ldr x{reg_idx}, [sp], #16")
      reg_idx += 1
    for _ in range(right_num_slots):
      self._emit(f"    ldr x{reg_idx}, [sp], #16")
      reg_idx += 1

    # Step 5: Clean up the saved left result if we evaluated left expression
    if not left_is_var:
      self._emit(f"    add sp, sp, #{len(left_fields) * 16}")

    # Call the method
    self._emit(f"    bl _{mangled_name}")

  def _gen_method_call(self, target: Expr, method: str, args: tuple[Expr, ...]) -> None:
    """Generate code for method calls on lists/arrays."""
    # For chained method calls (non-VarExpr target), evaluate and use _from_ptr methods
    if not isinstance(target, VarExpr):
      self._gen_expr(target)
      self._emit("    str x0, [sp, #-16]!")  # Push vec ptr

      match method:
        case "into_iter" | "iter" | "collect":
          self._emit("    ldr x0, [sp], #16")
        case "skip":
          self._gen_vec_skip_from_ptr(args[0])
        case "take":
          self._gen_vec_take_from_ptr(args[0])
        case "map":
          self._gen_vec_map_from_ptr(args[0])
        case "filter":
          self._gen_vec_filter_from_ptr(args[0])
        case "sum":
          self._gen_vec_sum_from_ptr()
        case "fold":
          self._gen_vec_fold_from_ptr(args[0], args[1])
        case "len":
          self._emit("    ldr x9, [sp], #16")
          self._emit("    ldr x0, [x9, #8]")
        case "pop":
          self._emit("    ldr x9, [sp], #16")
          self._emit("    ldr x10, [x9, #8]")
          self._emit("    sub x10, x10, #1")
          self._emit("    str x10, [x9, #8]")
          self._emit("    add x9, x9, #16")
          self._emit("    ldr x0, [x9, x10, lsl #3]")
        case _:
          self._emit("    add sp, sp, #16")  # Clean up
      return

    # VarExpr target - original variable-based dispatch
    name = target.name
    offset, type_str = self.locals[name]

    if is_vec_type(type_str):
      match method:
        case "push":
          # push(value): list.data[list.len++] = value
          # First check if we need to grow
          grow_label = self._new_label("grow")
          done_label = self._new_label("push_done")

          self._emit(f"    ldr x9, [x29, #{offset}]")  # List base
          self._emit("    ldr x10, [x9]")  # Capacity
          self._emit("    ldr x11, [x9, #8]")  # Length

          # Check if len >= capacity
          self._emit("    cmp x11, x10")
          self._emit(f"    b.ge {grow_label}")

          # Store the value
          self._gen_expr(args[0])
          self._emit("    mov x12, x0")  # Value in x12
          self._emit(f"    ldr x9, [x29, #{offset}]")  # Reload base
          self._emit("    ldr x11, [x9, #8]")  # Length
          self._emit("    add x13, x9, #16")  # Data start
          self._emit("    str x12, [x13, x11, lsl #3]")

          # Increment length
          self._emit("    add x11, x11, #1")
          self._emit("    str x11, [x9, #8]")
          self._emit("    mov x0, #0")  # Return 0
          self._emit(f"    b {done_label}")

          # Grow the list (double capacity)
          self._emit(f"{grow_label}:")
          self._emit(f"    ldr x9, [x29, #{offset}]")
          self._emit("    ldr x10, [x9]")  # Old capacity
          self._emit("    lsl x10, x10, #1")  # Double it
          self._emit("    add x0, x10, #2")  # new_cap + 2 (header)
          self._emit("    lsl x0, x0, #3")  # * 8 bytes
          self._emit("    bl _malloc")  # New buffer in x0

          # Copy header and data
          self._emit(f"    ldr x9, [x29, #{offset}]")  # Old base
          self._emit("    ldr x1, [x9]")  # Old capacity
          self._emit("    lsl x1, x1, #1")  # New capacity
          self._emit("    str x1, [x0]")  # Store new capacity
          self._emit("    ldr x2, [x9, #8]")  # Length
          self._emit("    str x2, [x0, #8]")  # Copy length

          # Copy data elements
          copy_loop = self._new_label("copy")
          copy_done = self._new_label("copy_done")
          self._emit("    mov x3, #0")  # Counter
          self._emit(f"{copy_loop}:")
          self._emit("    cmp x3, x2")
          self._emit(f"    b.ge {copy_done}")
          self._emit("    add x4, x9, #16")  # Old data
          self._emit("    ldr x5, [x4, x3, lsl #3]")
          self._emit("    add x4, x0, #16")  # New data
          self._emit("    str x5, [x4, x3, lsl #3]")
          self._emit("    add x3, x3, #1")
          self._emit(f"    b {copy_loop}")
          self._emit(f"{copy_done}:")

          # Free old buffer
          self._emit("    str x0, [sp, #-16]!")  # Save new pointer
          self._emit("    mov x0, x9")
          self._emit("    bl _free")
          self._emit("    ldr x9, [sp], #16")  # Restore new pointer

          # Update local variable
          self._emit(f"    str x9, [x29, #{offset}]")

          # Now do the push on new buffer
          self._emit("    ldr x11, [x9, #8]")  # Length
          self._gen_expr(args[0])
          self._emit("    mov x12, x0")
          self._emit(f"    ldr x9, [x29, #{offset}]")
          self._emit("    ldr x11, [x9, #8]")
          self._emit("    add x13, x9, #16")
          self._emit("    str x12, [x13, x11, lsl #3]")
          self._emit("    add x11, x11, #1")
          self._emit("    str x11, [x9, #8]")
          self._emit("    mov x0, #0")

          self._emit(f"{done_label}:")

        case "pop":
          # pop(): return list.data[--list.len]
          self._emit(f"    ldr x9, [x29, #{offset}]")  # List base
          self._emit("    ldr x10, [x9, #8]")  # Length
          self._emit("    sub x10, x10, #1")  # Decrement
          self._emit("    str x10, [x9, #8]")  # Store new length
          self._emit("    add x9, x9, #16")  # Data start
          self._emit("    ldr x0, [x9, x10, lsl #3]")  # Return value

        case "len":
          # len(): return list.len
          self._emit(f"    ldr x9, [x29, #{offset}]")  # List base
          self._emit("    ldr x0, [x9, #8]")  # Length

        case "into_iter" | "iter" | "collect":
          # These are no-ops for eager evaluation - just return the vec pointer
          self._emit(f"    ldr x0, [x29, #{offset}]")

        case "skip":
          # skip(n): create new vec without first n elements
          self._gen_vec_skip(offset, args[0])

        case "take":
          # take(n): create new vec with only first n elements
          self._gen_vec_take(offset, args[0])

        case "map":
          # map(closure): create new vec with closure applied to each element
          self._gen_vec_map(offset, args[0])

        case "filter":
          # filter(closure): create new vec with elements matching predicate
          self._gen_vec_filter(offset, args[0])

        case "sum":
          # sum(): return sum of all elements (for vec[i64])
          self._gen_vec_sum(offset)

        case "fold":
          # fold(init, closure): reduce vec to single value
          self._gen_vec_fold(offset, args[0], args[1])

    elif is_dict_type(type_str):
      match method:
        case "len":
          # len(): return dict.len
          self._emit(f"    ldr x9, [x29, #{offset}]")  # Dict base
          self._emit("    ldr x0, [x9, #8]")  # Length

        case "contains":
          # contains(key): return true if key exists
          self._gen_expr(args[0])  # Key in x0
          self._emit("    mov x1, x0")  # Key in x1
          self._emit(f"    ldr x0, [x29, #{offset}]")  # Dict pointer
          self._gen_dict_contains_inline()

        case "get":
          # get(key): return value for key (or 0 if not found)
          self._gen_expr(args[0])  # Key in x0
          self._emit("    mov x1, x0")  # Key in x1
          self._emit(f"    ldr x0, [x29, #{offset}]")  # Dict pointer
          self._gen_dict_lookup_inline()

        case "insert":
          # insert(key, value): insert key-value pair
          self._gen_expr(args[0])  # Key
          self._emit("    str x0, [sp, #-16]!")  # Save key
          self._gen_expr(args[1])  # Value
          self._emit("    mov x2, x0")  # Value in x2
          self._emit("    ldr x1, [sp], #16")  # Key in x1
          self._emit(f"    ldr x0, [x29, #{offset}]")  # Dict pointer
          self._gen_dict_insert_inline()
          self._emit("    mov x0, #0")  # Return 0

        case "remove":
          # remove(key): remove key and return whether it existed
          self._gen_expr(args[0])  # Key in x0
          self._emit("    mov x1, x0")  # Key in x1
          self._emit(f"    ldr x0, [x29, #{offset}]")  # Dict pointer
          self._gen_dict_remove_inline()

    elif type_str in self.struct_methods and method in self.struct_methods[type_str]:
      # Struct method call
      mangled_name = self.struct_methods[type_str][method]
      struct_fields = self.structs[type_str]

      # Push all arguments (including self) to stack in reverse order
      # First, push the explicit arguments in reverse
      for arg in reversed(args):
        self._gen_expr(arg)
        self._emit("    str x0, [sp, #-16]!")

      # Then push self (struct fields in reverse order)
      for i in range(len(struct_fields) - 1, -1, -1):
        self._emit(f"    ldr x0, [x29, #{offset - i * 8}]")
        self._emit("    str x0, [sp, #-16]!")

      # Pop into registers: self fields first, then args
      reg_idx = 0
      for _ in struct_fields:
        self._emit(f"    ldr x{reg_idx}, [sp], #16")
        reg_idx += 1
      for _ in args:
        self._emit(f"    ldr x{reg_idx}, [sp], #16")
        reg_idx += 1

      # Call the method
      self._emit(f"    bl _{mangled_name}")

    else:
      # Array methods
      match method:
        case "len":
          # Array length is compile-time constant
          size = get_array_size(type_str)
          if size > 0:
            self._emit(f"    mov x0, #{size}")

  def _gen_vec_skip(self, src_offset: int, skip_expr: Expr) -> None:
    """Generate code for vec.skip(n) - creates new vec without first n elements."""
    # Evaluate skip count
    self._gen_expr(skip_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save skip count

    # Load source vec info
    self._emit(f"    ldr x9, [x29, #{src_offset}]")  # Source base
    self._emit("    ldr x10, [x9, #8]")  # Source length

    # Calculate new length = max(0, len - skip)
    self._emit("    ldr x11, [sp]")  # Skip count
    self._emit("    subs x12, x10, x11")  # new_len = len - skip
    skip_negative = self._new_label("skip_neg")
    skip_continue = self._new_label("skip_cont")
    self._emit(f"    b.lt {skip_negative}")
    self._emit(f"    b {skip_continue}")
    self._emit(f"{skip_negative}:")
    self._emit("    mov x12, #0")  # new_len = 0 if negative
    self._emit(f"{skip_continue}:")

    # Allocate new vec: 16 (header) + new_len * 8
    self._emit("    add x0, x12, #2")  # capacity = new_len
    self._emit("    lsl x0, x0, #3")  # * 8
    self._emit("    str x12, [sp, #-16]!")  # Save new_len
    self._emit("    bl _malloc")

    # Set up new vec header
    self._emit("    ldr x12, [sp], #16")  # Restore new_len
    self._emit("    str x12, [x0]")  # capacity = new_len
    self._emit("    str x12, [x0, #8]")  # length = new_len

    # Copy elements: src[skip..] -> dest[0..]
    self._emit("    ldr x11, [sp], #16")  # Restore skip count
    self._emit(f"    ldr x9, [x29, #{src_offset}]")  # Source base
    self._emit("    add x9, x9, #16")  # Source data start
    self._emit("    add x9, x9, x11, lsl #3")  # Offset by skip count
    self._emit("    add x13, x0, #16")  # Dest data start
    self._emit("    mov x14, #0")  # Counter

    copy_loop = self._new_label("skip_copy")
    copy_done = self._new_label("skip_copy_done")
    self._emit(f"{copy_loop}:")
    self._emit("    cmp x14, x12")  # Compare counter with new_len
    self._emit(f"    b.ge {copy_done}")
    self._emit("    ldr x15, [x9, x14, lsl #3]")  # Load source[i + skip]
    self._emit("    str x15, [x13, x14, lsl #3]")  # Store dest[i]
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {copy_loop}")
    self._emit(f"{copy_done}:")

    # x0 already has the new vec pointer

  def _gen_vec_take(self, src_offset: int, take_expr: Expr) -> None:
    """Generate code for vec.take(n) - creates new vec with only first n elements."""
    # Evaluate take count
    self._gen_expr(take_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save take count

    # Load source vec info
    self._emit(f"    ldr x9, [x29, #{src_offset}]")  # Source base
    self._emit("    ldr x10, [x9, #8]")  # Source length

    # Calculate new length = min(len, take)
    self._emit("    ldr x11, [sp]")  # Take count
    self._emit("    cmp x10, x11")
    take_min = self._new_label("take_min")
    take_use_take = self._new_label("take_use")
    self._emit(f"    b.lt {take_min}")
    self._emit("    mov x12, x11")  # new_len = take
    self._emit(f"    b {take_use_take}")
    self._emit(f"{take_min}:")
    self._emit("    mov x12, x10")  # new_len = len
    self._emit(f"{take_use_take}:")

    # Allocate new vec
    self._emit("    add x0, x12, #2")  # capacity = new_len
    self._emit("    lsl x0, x0, #3")
    self._emit("    str x12, [sp, #-16]!")  # Save new_len
    self._emit("    bl _malloc")

    # Set up new vec header
    self._emit("    ldr x12, [sp], #16")  # Restore new_len
    self._emit("    add sp, sp, #16")  # Pop take count (not needed anymore)
    self._emit("    str x12, [x0]")  # capacity
    self._emit("    str x12, [x0, #8]")  # length

    # Copy first new_len elements
    self._emit(f"    ldr x9, [x29, #{src_offset}]")  # Source base
    self._emit("    add x9, x9, #16")  # Source data
    self._emit("    add x13, x0, #16")  # Dest data
    self._emit("    mov x14, #0")

    copy_loop = self._new_label("take_copy")
    copy_done = self._new_label("take_copy_done")
    self._emit(f"{copy_loop}:")
    self._emit("    cmp x14, x12")
    self._emit(f"    b.ge {copy_done}")
    self._emit("    ldr x15, [x9, x14, lsl #3]")
    self._emit("    str x15, [x13, x14, lsl #3]")
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {copy_loop}")
    self._emit(f"{copy_done}:")

  def _gen_vec_map(self, src_offset: int, closure_expr: Expr) -> None:
    """Generate code for vec.map(closure) - creates new vec with closure applied."""
    # Evaluate closure to get function pointer
    self._gen_expr(closure_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save closure pointer

    # Load source vec info
    self._emit(f"    ldr x9, [x29, #{src_offset}]")
    self._emit("    ldr x10, [x9, #8]")  # Length
    self._emit("    str x10, [sp, #-16]!")  # Save length

    # Allocate new vec of same size
    self._emit("    add x0, x10, #2")
    self._emit("    lsl x0, x0, #3")
    self._emit("    bl _malloc")
    self._emit("    str x0, [sp, #-16]!")  # Save new vec pointer

    # Set up header
    self._emit("    ldr x10, [sp, #16]")  # Get length
    self._emit("    str x10, [x0]")  # capacity
    self._emit("    str x10, [x0, #8]")  # length

    # Map each element
    self._emit("    mov x14, #0")  # Counter
    map_loop = self._new_label("map_loop")
    map_done = self._new_label("map_done")
    self._emit(f"{map_loop}:")
    self._emit("    ldr x10, [sp, #16]")  # Length
    self._emit("    cmp x14, x10")
    self._emit(f"    b.ge {map_done}")

    # Save counter
    self._emit("    str x14, [sp, #-16]!")

    # Load element from source
    self._emit(f"    ldr x9, [x29, #{src_offset}]")
    self._emit("    add x9, x9, #16")
    self._emit("    ldr x0, [x9, x14, lsl #3]")  # x0 = source[i]

    # Call closure
    # Stack: [index] [new_vec] [len] [closure] = sp+0, sp+16, sp+32, sp+48
    self._emit("    ldr x9, [sp, #48]")  # Closure pointer
    self._emit("    blr x9")

    # Store result in dest
    self._emit("    ldr x14, [sp], #16")  # Restore counter
    self._emit("    ldr x13, [sp]")  # New vec pointer
    self._emit("    add x13, x13, #16")
    self._emit("    str x0, [x13, x14, lsl #3]")

    self._emit("    add x14, x14, #1")
    self._emit(f"    b {map_loop}")
    self._emit(f"{map_done}:")

    # Return new vec pointer
    self._emit("    ldr x0, [sp], #16")  # Pop new vec ptr
    self._emit("    add sp, sp, #16")  # Pop length
    self._emit("    add sp, sp, #16")  # Pop closure ptr

  def _gen_vec_filter(self, src_offset: int, closure_expr: Expr) -> None:
    """Generate code for vec.filter(closure) - creates new vec with matching elements."""
    # Evaluate closure
    self._gen_expr(closure_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save closure

    # Load source info
    self._emit(f"    ldr x9, [x29, #{src_offset}]")
    self._emit("    ldr x10, [x9, #8]")  # Length

    # Allocate new vec (same capacity as source, will have len <= original)
    self._emit("    add x0, x10, #2")
    self._emit("    lsl x0, x0, #3")
    self._emit("    str x10, [sp, #-16]!")  # Save original length
    self._emit("    bl _malloc")
    self._emit("    str x0, [sp, #-16]!")  # Save new vec

    # Initialize: capacity = original_len, length = 0
    self._emit("    ldr x10, [sp, #16]")  # Original length
    self._emit("    str x10, [x0]")  # capacity
    self._emit("    str xzr, [x0, #8]")  # length = 0

    # Filter loop
    self._emit("    mov x14, #0")  # Source index
    filter_loop = self._new_label("filter_loop")
    filter_done = self._new_label("filter_done")
    filter_skip = self._new_label("filter_skip")
    self._emit(f"{filter_loop}:")
    self._emit("    ldr x10, [sp, #16]")  # Original length
    self._emit("    cmp x14, x10")
    self._emit(f"    b.ge {filter_done}")

    # Save index
    self._emit("    str x14, [sp, #-16]!")

    # Load element
    self._emit(f"    ldr x9, [x29, #{src_offset}]")
    self._emit("    add x9, x9, #16")
    self._emit("    ldr x0, [x9, x14, lsl #3]")
    self._emit("    str x0, [sp, #-16]!")  # Save element for potential copy

    # Call predicate
    # Stack: [elem] [idx] [new_vec] [len] [closure] = sp+0, sp+16, sp+32, sp+48, sp+64
    self._emit("    ldr x9, [sp, #64]")  # Closure pointer
    self._emit("    blr x9")

    # Check result
    self._emit("    cmp x0, #0")
    self._emit(f"    b.eq {filter_skip}")

    # Element matches - add to dest
    # Stack: [elem] [idx] [new_vec] [len] [closure] = sp+0, sp+16, sp+32, sp+48, sp+64
    self._emit("    ldr x0, [sp]")  # Element value (sp+0)
    self._emit("    ldr x13, [sp, #32]")  # New vec (sp+32)
    self._emit("    ldr x11, [x13, #8]")  # Current dest length
    self._emit("    add x12, x13, #16")  # Dest data
    self._emit("    str x0, [x12, x11, lsl #3]")
    self._emit("    add x11, x11, #1")
    self._emit("    str x11, [x13, #8]")  # Update length

    self._emit(f"{filter_skip}:")
    self._emit("    add sp, sp, #16")  # Pop element
    self._emit("    ldr x14, [sp], #16")  # Restore index
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {filter_loop}")

    self._emit(f"{filter_done}:")
    self._emit("    ldr x0, [sp], #16")  # Return new vec
    self._emit("    add sp, sp, #16")  # Pop length
    self._emit("    add sp, sp, #16")  # Pop closure

  def _gen_vec_sum(self, src_offset: int) -> None:
    """Generate code for vec.sum() - returns sum of all elements."""
    self._emit(f"    ldr x9, [x29, #{src_offset}]")  # Vec base
    self._emit("    ldr x10, [x9, #8]")  # Length
    self._emit("    add x9, x9, #16")  # Data start
    self._emit("    mov x0, #0")  # Accumulator
    self._emit("    mov x11, #0")  # Counter

    sum_loop = self._new_label("sum_loop")
    sum_done = self._new_label("sum_done")
    self._emit(f"{sum_loop}:")
    self._emit("    cmp x11, x10")
    self._emit(f"    b.ge {sum_done}")
    self._emit("    ldr x12, [x9, x11, lsl #3]")
    self._emit("    add x0, x0, x12")
    self._emit("    add x11, x11, #1")
    self._emit(f"    b {sum_loop}")
    self._emit(f"{sum_done}:")

  def _gen_vec_fold(self, src_offset: int, init_expr: Expr, closure_expr: Expr) -> None:
    """Generate code for vec.fold(init, closure) - reduces to single value."""
    # Evaluate initial value
    self._gen_expr(init_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save accumulator

    # Evaluate closure
    self._gen_expr(closure_expr)
    self._emit("    str x0, [sp, #-16]!")  # Save closure

    # Load source info
    self._emit(f"    ldr x9, [x29, #{src_offset}]")
    self._emit("    ldr x10, [x9, #8]")  # Length
    self._emit("    str x10, [sp, #-16]!")  # Save length

    # Fold loop
    self._emit("    mov x14, #0")  # Index
    fold_loop = self._new_label("fold_loop")
    fold_done = self._new_label("fold_done")
    self._emit(f"{fold_loop}:")
    self._emit("    ldr x10, [sp]")  # Length
    self._emit("    cmp x14, x10")
    self._emit(f"    b.ge {fold_done}")

    # Save index
    self._emit("    str x14, [sp, #-16]!")

    # Call closure(acc, elem)
    # Stack: [idx] [len] [closure] [acc] = sp+0, sp+16, sp+32, sp+48
    self._emit("    ldr x0, [sp, #48]")  # Accumulator
    self._emit(f"    ldr x9, [x29, #{src_offset}]")
    self._emit("    add x9, x9, #16")
    self._emit("    ldr x1, [x9, x14, lsl #3]")  # Element
    self._emit("    ldr x9, [sp, #32]")  # Closure
    self._emit("    blr x9")

    # Update accumulator
    self._emit("    str x0, [sp, #48]")

    # Restore index and continue
    self._emit("    ldr x14, [sp], #16")
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {fold_loop}")

    self._emit(f"{fold_done}:")
    self._emit("    add sp, sp, #16")  # Pop length
    self._emit("    add sp, sp, #16")  # Pop closure
    self._emit("    ldr x0, [sp], #16")  # Return accumulator

  # === Vec methods that work with pointer on stack (for chaining) ===

  def _gen_vec_skip_from_ptr(self, skip_expr: Expr) -> None:
    """Generate skip() when vec pointer is on stack."""
    # Stack: [vec_ptr]
    self._gen_expr(skip_expr)
    self._emit("    mov x11, x0")  # Skip count in x11

    # Load source vec info
    self._emit("    ldr x9, [sp]")  # Source base (keep on stack for now)
    self._emit("    ldr x10, [x9, #8]")  # Source length

    # Calculate new length = max(0, len - skip)
    self._emit("    subs x12, x10, x11")
    skip_neg = self._new_label("skip_neg")
    skip_cont = self._new_label("skip_cont")
    self._emit(f"    b.lt {skip_neg}")
    self._emit(f"    b {skip_cont}")
    self._emit(f"{skip_neg}:")
    self._emit("    mov x12, #0")
    self._emit(f"{skip_cont}:")

    # Save new_len and skip
    self._emit("    str x12, [sp, #-16]!")  # new_len
    self._emit("    str x11, [sp, #-16]!")  # skip

    # Allocate new vec
    self._emit("    add x0, x12, #2")
    self._emit("    lsl x0, x0, #3")
    self._emit("    bl _malloc")

    # Set up header
    self._emit("    ldr x11, [sp], #16")  # skip
    self._emit("    ldr x12, [sp], #16")  # new_len
    self._emit("    str x12, [x0]")  # capacity
    self._emit("    str x12, [x0, #8]")  # length

    # Copy elements
    self._emit("    ldr x9, [sp], #16")  # Source vec (and remove from stack)
    self._emit("    add x9, x9, #16")  # Source data
    self._emit("    add x9, x9, x11, lsl #3")  # + skip offset
    self._emit("    add x13, x0, #16")  # Dest data
    self._emit("    mov x14, #0")

    copy_loop = self._new_label("skip_copy")
    copy_done = self._new_label("skip_done")
    self._emit(f"{copy_loop}:")
    self._emit("    cmp x14, x12")
    self._emit(f"    b.ge {copy_done}")
    self._emit("    ldr x15, [x9, x14, lsl #3]")
    self._emit("    str x15, [x13, x14, lsl #3]")
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {copy_loop}")
    self._emit(f"{copy_done}:")

  def _gen_vec_take_from_ptr(self, take_expr: Expr) -> None:
    """Generate take() when vec pointer is on stack."""
    self._gen_expr(take_expr)
    self._emit("    mov x11, x0")  # Take count

    self._emit("    ldr x9, [sp]")  # Source vec
    self._emit("    ldr x10, [x9, #8]")  # Length

    # new_len = min(len, take)
    self._emit("    cmp x10, x11")
    take_min = self._new_label("take_min")
    take_done2 = self._new_label("take_d")
    self._emit(f"    b.lt {take_min}")
    self._emit("    mov x12, x11")
    self._emit(f"    b {take_done2}")
    self._emit(f"{take_min}:")
    self._emit("    mov x12, x10")
    self._emit(f"{take_done2}:")

    # Save new_len
    self._emit("    str x12, [sp, #-16]!")

    # Allocate
    self._emit("    add x0, x12, #2")
    self._emit("    lsl x0, x0, #3")
    self._emit("    bl _malloc")

    # Set header
    self._emit("    ldr x12, [sp], #16")
    self._emit("    str x12, [x0]")
    self._emit("    str x12, [x0, #8]")

    # Copy
    self._emit("    ldr x9, [sp], #16")  # Source vec
    self._emit("    add x9, x9, #16")
    self._emit("    add x13, x0, #16")
    self._emit("    mov x14, #0")

    copy_loop = self._new_label("take_copy")
    copy_done = self._new_label("take_done")
    self._emit(f"{copy_loop}:")
    self._emit("    cmp x14, x12")
    self._emit(f"    b.ge {copy_done}")
    self._emit("    ldr x15, [x9, x14, lsl #3]")
    self._emit("    str x15, [x13, x14, lsl #3]")
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {copy_loop}")
    self._emit(f"{copy_done}:")

  def _gen_vec_map_from_ptr(self, closure_expr: Expr) -> None:
    """Generate map() when vec pointer is on stack."""
    # Stack: [vec_ptr]
    self._gen_expr(closure_expr)
    self._emit("    str x0, [sp, #-16]!")  # Stack: [closure, vec_ptr]

    self._emit("    ldr x9, [sp, #16]")  # Vec ptr
    self._emit("    ldr x10, [x9, #8]")  # Length
    self._emit("    str x10, [sp, #-16]!")  # Stack: [len, closure, vec_ptr]

    # Allocate
    self._emit("    add x0, x10, #2")
    self._emit("    lsl x0, x0, #3")
    self._emit("    bl _malloc")
    self._emit("    str x0, [sp, #-16]!")  # Stack: [new_vec, len, closure, vec_ptr]

    # Header
    self._emit("    ldr x10, [sp, #16]")  # len
    self._emit("    str x10, [x0]")
    self._emit("    str x10, [x0, #8]")

    # Map loop
    self._emit("    mov x14, #0")
    map_loop = self._new_label("map_loop")
    map_done = self._new_label("map_done")
    self._emit(f"{map_loop}:")
    self._emit("    ldr x10, [sp, #16]")
    self._emit("    cmp x14, x10")
    self._emit(f"    b.ge {map_done}")

    self._emit("    str x14, [sp, #-16]!")  # Save index

    # Load element
    # Stack: [idx] [new_vec] [len] [closure] [vec_ptr] = sp+0, sp+16, sp+32, sp+48, sp+64
    self._emit("    ldr x9, [sp, #64]")  # vec_ptr
    self._emit("    add x9, x9, #16")
    self._emit("    ldr x0, [x9, x14, lsl #3]")

    # Call closure
    self._emit("    ldr x9, [sp, #48]")  # closure
    self._emit("    blr x9")

    # Store result
    self._emit("    ldr x14, [sp], #16")
    self._emit("    ldr x13, [sp]")  # new_vec
    self._emit("    add x13, x13, #16")
    self._emit("    str x0, [x13, x14, lsl #3]")

    self._emit("    add x14, x14, #1")
    self._emit(f"    b {map_loop}")
    self._emit(f"{map_done}:")

    self._emit("    ldr x0, [sp], #16")  # new_vec
    self._emit("    add sp, sp, #16")  # len
    self._emit("    add sp, sp, #16")  # closure
    self._emit("    add sp, sp, #16")  # vec_ptr

  def _gen_vec_filter_from_ptr(self, closure_expr: Expr) -> None:
    """Generate filter() when vec pointer is on stack."""
    self._gen_expr(closure_expr)
    self._emit("    str x0, [sp, #-16]!")  # Stack: [closure, vec_ptr]

    self._emit("    ldr x9, [sp, #16]")
    self._emit("    ldr x10, [x9, #8]")  # len
    self._emit("    str x10, [sp, #-16]!")  # Stack: [len, closure, vec_ptr]

    # Allocate (max capacity = original len)
    self._emit("    add x0, x10, #2")
    self._emit("    lsl x0, x0, #3")
    self._emit("    bl _malloc")
    self._emit("    str x0, [sp, #-16]!")  # Stack: [new_vec, len, closure, vec_ptr]

    # Init header
    self._emit("    ldr x10, [sp, #16]")
    self._emit("    str x10, [x0]")  # capacity
    self._emit("    str xzr, [x0, #8]")  # length = 0

    # Filter loop
    self._emit("    mov x14, #0")
    filter_loop = self._new_label("filter_loop")
    filter_done = self._new_label("filter_done")
    filter_skip = self._new_label("filter_skip")
    self._emit(f"{filter_loop}:")
    self._emit("    ldr x10, [sp, #16]")
    self._emit("    cmp x14, x10")
    self._emit(f"    b.ge {filter_done}")

    self._emit("    str x14, [sp, #-16]!")  # Save index

    # Load element
    # Stack after index push: [idx] [new_vec] [len] [closure] [vec_ptr] = sp+0, sp+16, sp+32, sp+48, sp+64
    self._emit("    ldr x9, [sp, #64]")  # vec_ptr
    self._emit("    add x9, x9, #16")
    self._emit("    ldr x0, [x9, x14, lsl #3]")
    self._emit("    str x0, [sp, #-16]!")  # Save element

    # Call predicate
    # Stack: [elem] [idx] [new_vec] [len] [closure] [vec_ptr] = sp+0, sp+16, sp+32, sp+48, sp+64, sp+80
    self._emit("    ldr x9, [sp, #64]")  # closure
    self._emit("    blr x9")

    # Check
    self._emit("    cmp x0, #0")
    self._emit(f"    b.eq {filter_skip}")

    # Add to result
    # Stack: [elem] [idx] [new_vec] [len] [closure] [vec_ptr] = sp+0, sp+16, sp+32, sp+48, sp+64, sp+80
    self._emit("    ldr x0, [sp]")  # element (sp+0)
    self._emit("    ldr x13, [sp, #32]")  # new_vec (sp+32)
    self._emit("    ldr x11, [x13, #8]")  # current len
    self._emit("    add x12, x13, #16")
    self._emit("    str x0, [x12, x11, lsl #3]")
    self._emit("    add x11, x11, #1")
    self._emit("    str x11, [x13, #8]")

    self._emit(f"{filter_skip}:")
    self._emit("    add sp, sp, #16")  # element
    self._emit("    ldr x14, [sp], #16")  # index
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {filter_loop}")

    self._emit(f"{filter_done}:")
    self._emit("    ldr x0, [sp], #16")  # new_vec
    self._emit("    add sp, sp, #16")  # len
    self._emit("    add sp, sp, #16")  # closure
    self._emit("    add sp, sp, #16")  # vec_ptr

  def _gen_vec_sum_from_ptr(self) -> None:
    """Generate sum() when vec pointer is on stack."""
    self._emit("    ldr x9, [sp], #16")  # Vec ptr
    self._emit("    ldr x10, [x9, #8]")  # Length
    self._emit("    add x9, x9, #16")  # Data
    self._emit("    mov x0, #0")  # Sum
    self._emit("    mov x11, #0")  # Counter

    sum_loop = self._new_label("sum_loop")
    sum_done = self._new_label("sum_done")
    self._emit(f"{sum_loop}:")
    self._emit("    cmp x11, x10")
    self._emit(f"    b.ge {sum_done}")
    self._emit("    ldr x12, [x9, x11, lsl #3]")
    self._emit("    add x0, x0, x12")
    self._emit("    add x11, x11, #1")
    self._emit(f"    b {sum_loop}")
    self._emit(f"{sum_done}:")

  def _gen_vec_fold_from_ptr(self, init_expr: Expr, closure_expr: Expr) -> None:
    """Generate fold() when vec pointer is on stack."""
    # Stack: [vec_ptr]
    self._gen_expr(init_expr)
    self._emit("    str x0, [sp, #-16]!")  # Stack: [acc, vec_ptr]

    self._gen_expr(closure_expr)
    self._emit("    str x0, [sp, #-16]!")  # Stack: [closure, acc, vec_ptr]

    self._emit("    ldr x9, [sp, #32]")  # vec_ptr
    self._emit("    ldr x10, [x9, #8]")  # len
    self._emit("    str x10, [sp, #-16]!")  # Stack: [len, closure, acc, vec_ptr]

    # Fold loop
    self._emit("    mov x14, #0")
    fold_loop = self._new_label("fold_loop")
    fold_done = self._new_label("fold_done")
    self._emit(f"{fold_loop}:")
    self._emit("    ldr x10, [sp]")
    self._emit("    cmp x14, x10")
    self._emit(f"    b.ge {fold_done}")

    self._emit("    str x14, [sp, #-16]!")

    # closure(acc, elem)
    # Stack: [idx] [len] [closure] [acc] [vec_ptr] = sp+0, sp+16, sp+32, sp+48, sp+64
    self._emit("    ldr x0, [sp, #48]")  # acc
    self._emit("    ldr x9, [sp, #64]")  # vec_ptr
    self._emit("    add x9, x9, #16")
    self._emit("    ldr x1, [x9, x14, lsl #3]")  # elem
    self._emit("    ldr x9, [sp, #32]")  # closure
    self._emit("    blr x9")

    # Update acc
    self._emit("    str x0, [sp, #48]")

    self._emit("    ldr x14, [sp], #16")
    self._emit("    add x14, x14, #1")
    self._emit(f"    b {fold_loop}")

    self._emit(f"{fold_done}:")
    self._emit("    add sp, sp, #16")  # len
    self._emit("    add sp, sp, #16")  # closure
    self._emit("    ldr x0, [sp], #16")  # acc
    self._emit("    add sp, sp, #16")  # vec_ptr

  def _gen_match(self, target: Expr, arms: tuple[MatchArm, ...]) -> None:
    """Generate code for a match expression."""
    end_label = self._new_label("match_end")

    # Check if this is a Result type match
    is_result_match = arms[0].enum_name == "Result" if arms else False

    # Get the enum/Result variable's tag and payload offset
    var_offset: int | None = None
    match target:
      case VarExpr(name):
        var_offset, type_str = self.locals[name]
        # Load tag from first slot
        self._emit(f"    ldr x8, [x29, #{var_offset}]")  # Tag in x8
        # For Result, also save payload location for binding
        if is_result_match or is_result_type(type_str):
          is_result_match = True
      case _:
        # For complex expressions, evaluate - for Result x0=tag, x1=payload
        self._gen_expr(target)
        self._emit("    mov x8, x0")  # Tag in x8
        if is_result_match:
          self._emit("    mov x9, x1")  # Payload in x9

    # Generate labels for each arm and for "next arm" checks (for guards)
    arm_labels: list[str] = []
    next_arm_labels: list[str] = []
    for i, arm in enumerate(arms):
      arm_labels.append(self._new_label("match_arm"))
      next_arm_labels.append(self._new_label("match_next"))

    # Generate comparison chain for each arm
    for i, arm in enumerate(arms):
      if is_result_match:
        # Result: Ok=0, Err=1
        tag = 0 if arm.variant_name == "Ok" else 1
      else:
        variants = self.enums[arm.enum_name]
        tag, _ = variants[arm.variant_name]

      # Compare tag
      self._emit(f"    cmp x8, #{tag}")
      self._emit(f"    b.eq {arm_labels[i]}")

    # Fall through to end (should never happen with exhaustive match)
    self._emit(f"    b {end_label}")

    # Generate code for each arm
    for i, arm in enumerate(arms):
      self._emit(f"{arm_labels[i]}:")

      if is_result_match:
        # Result always has payload
        has_payload = True
      else:
        variants = self.enums[arm.enum_name]
        _, has_payload = variants[arm.variant_name]

      # If there's a binding, allocate slot and store payload
      binding_offset: int | None = None
      if arm.binding is not None and has_payload:
        binding_offset = -16 - (self.next_slot * 8)
        self.locals[arm.binding] = (binding_offset, "i64")  # Assume i64 for now
        self.next_slot += 1
        # Load payload from enum/Result variable
        match target:
          case VarExpr(name):
            var_offset_local, _ = self.locals[name]
            self._emit(f"    ldr x0, [x29, #{var_offset_local - 8}]")  # Payload
            self._emit(f"    str x0, [x29, #{binding_offset}]")
          case _:
            # Payload was saved in x9
            if is_result_match:
              self._emit(f"    str x9, [x29, #{binding_offset}]")

      # Check pattern guard (if present)
      if arm.guard is not None:
        # Evaluate guard expression - result in x0 (0=false, 1=true)
        self._gen_expr(arm.guard)
        self._emit("    cmp x0, #0")
        # If guard is false, jump to try next arm
        self._emit(f"    b.eq {next_arm_labels[i]}")

      # Generate arm body
      for stmt in arm.body:
        self._gen_stmt(stmt)

      self._emit(f"    b {end_label}")

      # Label for failed guard - try to find another matching arm
      self._emit(f"{next_arm_labels[i]}:")
      # Look for next arm with same variant (for multiple guards on same variant)
      found_next = False
      for j in range(i + 1, len(arms)):
        next_arm = arms[j]
        if next_arm.variant_name == arm.variant_name and next_arm.enum_name == arm.enum_name:
          # Jump to the next arm's code (it will set up its own binding if needed)
          self._emit(f"    b {arm_labels[j]}")
          found_next = True
          break
      if not found_next:
        # No more arms for this variant - this shouldn't happen with exhaustive match
        # But if it does, fall through to end
        self._emit(f"    b {end_label}")

    self._emit(f"{end_label}:")
    # Match result is in x0 from the last executed arm's return


def generate(program: Program, type_check_result: "TypeCheckResult | None" = None) -> str:
  """Convenience function to generate assembly for a program."""
  return CodeGenerator().generate(program, type_check_result)
