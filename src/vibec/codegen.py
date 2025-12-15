"""ARM64 code generator for macOS."""

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
  UnaryExpr,
  WhileStmt,
  AssignStmt,
  BinaryExpr,
  IntLiteral,
  ReturnStmt,
  SimpleType,
  BoolLiteral,
  ArrayLiteral,
  StringLiteral,
  StructLiteral,
  MethodCallExpr,
  TypeAnnotation,
  FieldAccessExpr,
  FieldAssignStmt,
  IndexAssignStmt,
)


def type_to_str(t: TypeAnnotation) -> str:
  """Convert type annotation to string."""
  match t:
    case SimpleType(name):
      return name
    case ArrayType(elem, size):
      return f"[{type_to_str(elem)};{size}]"
    case VecType(elem):
      return f"vec[{type_to_str(elem)}]"
  return "unknown"


def get_array_size(type_str: str) -> int | None:
  """Extract size from array type string like [i64;5]."""
  if type_str.startswith("[") and ";" in type_str:
    return int(type_str[type_str.index(";") + 1 : -1])
  return None


def is_vec_type(type_str: str) -> bool:
  """Check if type is a vec."""
  return type_str.startswith("vec[")


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
    # Struct definitions: name -> list of (field_name, field_type)
    self.structs: dict[str, list[tuple[str, str]]] = {}

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
      case CallExpr(_, args):
        for arg in args:
          self._collect_strings_from_expr(arg)
      case ArrayLiteral(elements):
        for elem in elements:
          self._collect_strings_from_expr(elem)
      case IndexExpr(target, index):
        self._collect_strings_from_expr(target)
        self._collect_strings_from_expr(index)
      case MethodCallExpr(target, _, args):
        self._collect_strings_from_expr(target)
        for arg in args:
          self._collect_strings_from_expr(arg)
      case StructLiteral(_, fields):
        for _, value in fields:
          self._collect_strings_from_expr(value)
      case FieldAccessExpr(target, _):
        self._collect_strings_from_expr(target)
      case _:
        pass

  def generate(self, program: Program) -> str:
    """Generate assembly for the entire program."""
    # Register struct definitions
    for struct in program.structs:
      fields = [(f.name, type_to_str(f.type_ann)) for f in struct.fields]
      self.structs[struct.name] = fields

    # First pass: collect all string literals
    self._collect_strings(program)

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

    for func in program.functions:
      self._gen_function(func)

    return "\n".join(self.output)

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
    return count

  def _slots_for_type(self, t: TypeAnnotation) -> int:
    """Return number of 8-byte slots needed for a type."""
    match t:
      case ArrayType(_, size):
        return size  # Each element is 8 bytes
      case VecType(_):
        return 1  # List is a pointer
      case SimpleType(name):
        if name in self.structs:
          return len(self.structs[name])  # One slot per field
        return 1  # Simple types are 8 bytes
      case _:
        return 1

  def _gen_function(self, func: Function) -> None:
    """Generate assembly for a function."""
    self.locals = {}
    self.current_func_name = func.name
    self.next_slot = 0

    # Calculate frame size: params + local variables
    # Each slot is 8 bytes, plus 16 for saved fp/lr
    num_params = len(func.params)
    num_locals = self._count_locals(func.body)
    total_slots = num_params + num_locals

    # Frame size: 16 (saved fp/lr) + slots, 16-byte aligned
    slots_size = total_slots * 8
    self.frame_size = 16 + ((slots_size + 15) & ~15)
    if self.frame_size < 32:
      self.frame_size = 32  # Minimum for some temp storage

    # Function label (prefix with _ for macOS)
    self._emit(".align 4")
    self._emit(f"_{func.name}:")

    # Prologue: allocate full frame and save fp/lr at top
    self._emit(f"    sub sp, sp, #{self.frame_size}")
    self._emit(f"    stp x29, x30, [sp, #{self.frame_size - 16}]")
    self._emit(f"    add x29, sp, #{self.frame_size - 16}")

    # Store parameters (first 8 in x0-x7)
    # Parameters go at [x29 - 16], [x29 - 24], etc.
    for i, param in enumerate(func.params):
      offset = -16 - (self.next_slot * 8)
      type_str = type_to_str(param.type_ann)
      self.locals[param.name] = (offset, type_str)
      if i < 8:
        self._emit(f"    str x{i}, [x29, #{offset}]")
      self.next_slot += self._slots_for_type(param.type_ann)

    # Generate body
    for stmt in func.body:
      self._gen_stmt(stmt)

    # Epilogue
    self._emit(f"_{func.name}_epilogue:")
    self._emit(f"    ldp x29, x30, [sp, #{self.frame_size - 16}]")
    self._emit(f"    add sp, sp, #{self.frame_size}")
    self._emit("    ret")
    self._emit("")

  def _gen_stmt(self, stmt: Stmt) -> None:
    """Generate assembly for a statement."""
    match stmt:
      case LetStmt(name, type_ann, value):
        type_str = type_to_str(type_ann)
        offset = -16 - (self.next_slot * 8)
        self.locals[name] = (offset, type_str)
        slots_needed = self._slots_for_type(type_ann)

        match type_ann:
          case ArrayType(_, size):
            # Array: initialize elements in place
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

          case VecType(_):
            # List: allocate initial buffer (16 elements * 8 bytes + 16 header)
            # Header: [capacity, length] at base, data follows
            self._emit("    mov x0, #144")  # 16 + 16*8 = 144 bytes
            self._emit("    bl _malloc")
            self._emit("    mov x1, #16")  # Initial capacity
            self._emit("    str x1, [x0]")  # Store capacity
            self._emit("    str xzr, [x0, #8]")  # Length = 0
            self._emit(f"    str x0, [x29, #{offset}]")
            self.next_slot += 1

          case SimpleType(name) if name in self.structs:
            # Struct type: initialize fields in place
            fields = self.structs[name]
            match value:
              case StructLiteral(_, field_values):
                # Map field name -> value for lookup
                value_map = dict(field_values)
                for i, (field_name, _) in enumerate(fields):
                  if field_name in value_map:
                    self._gen_expr(value_map[field_name])
                    self._emit(f"    str x0, [x29, #{offset - i * 8}]")
                  else:
                    self._emit(f"    str xzr, [x29, #{offset - i * 8}]")
              case _:
                # Initialize all to zero
                for i in range(len(fields)):
                  self._emit(f"    str xzr, [x29, #{offset - i * 8}]")
            self.next_slot += len(fields)

          case _:
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
            # Evaluate index
            self._gen_expr(index)
            self._emit("    str x0, [sp, #-16]!")  # Save index

            # Evaluate value
            self._gen_expr(value)
            self._emit("    mov x2, x0")  # Value in x2

            self._emit("    ldr x1, [sp], #16")  # Restore index to x1

            if is_vec_type(type_str):
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
        offset, _ = self.locals[name]
        self._emit(f"    ldr x0, [x29, #{offset}]")

      case BinaryExpr(left, op, right):
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

      case CallExpr(name, args):
        if name == "print":
          self._gen_print(args)
        else:
          self._gen_call(name, args)

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
            self._emit("    mov x1, x0")  # Index in x1

            if is_vec_type(type_str):
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

      case MethodCallExpr(target, method, args):
        self._gen_method_call(target, method, args)

      case StructLiteral(name, fields):
        # Struct literal outside of let: allocate temp and return address
        # This is uncommon, but we can handle it
        struct_fields = self.structs[name]
        value_map = dict(fields)
        # Store fields on stack temporarily
        for i, (field_name, _) in enumerate(struct_fields):
          if field_name in value_map:
            self._gen_expr(value_map[field_name])
            self._emit("    str x0, [sp, #-16]!")
          else:
            self._emit("    str xzr, [sp, #-16]!")
        # Load first field's value as result (or return 0)
        if struct_fields:
          self._emit(f"    ldr x0, [sp, #{(len(struct_fields) - 1) * 16}]")
        else:
          self._emit("    mov x0, #0")
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

  def _gen_call(self, name: str, args: tuple[Expr, ...]) -> None:
    """Generate code for a function call."""
    # Evaluate arguments and push to stack (in reverse order)
    for arg in reversed(args):
      self._gen_expr(arg)
      self._emit("    str x0, [sp, #-16]!")

    # Pop arguments into registers x0-x7
    for i in range(len(args)):
      self._emit(f"    ldr x{i}, [sp], #16")

    # Call function
    self._emit(f"    bl _{name}")

  def _gen_method_call(self, target: Expr, method: str, args: tuple[Expr, ...]) -> None:
    """Generate code for method calls on lists/arrays."""
    match target:
      case VarExpr(name):
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

        else:
          # Array methods
          match method:
            case "len":
              # Array length is compile-time constant
              size = get_array_size(type_str)
              if size is not None:
                self._emit(f"    mov x0, #{size}")


def generate(program: Program) -> str:
  """Convenience function to generate assembly for a program."""
  return CodeGenerator().generate(program)
