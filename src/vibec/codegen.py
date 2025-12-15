"""ARM64 code generator for macOS."""

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
      case _:
        pass

  def generate(self, program: Program) -> str:
    """Generate assembly for the entire program."""
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
    """Count local variables in a statement list (including nested blocks)."""
    count = 0
    for stmt in stmts:
      match stmt:
        case LetStmt():
          count += 1
        case IfStmt(_, then_body, else_body):
          count += self._count_locals(then_body)
          if else_body:
            count += self._count_locals(else_body)
        case WhileStmt(_, body):
          count += self._count_locals(body)
    return count

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
      self.locals[param.name] = (offset, param.type_ann.name)
      if i < 8:
        self._emit(f"    str x{i}, [x29, #{offset}]")
      self.next_slot += 1

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
        # Evaluate value to x0
        self._gen_expr(value)
        # Allocate slot
        offset = -16 - (self.next_slot * 8)
        self.locals[name] = (offset, type_ann.name)
        self.next_slot += 1
        # Store value
        self._emit(f"    str x0, [x29, #{offset}]")

      case AssignStmt(name, value):
        # Evaluate value to x0
        self._gen_expr(value)
        # Store to existing variable slot
        offset, _ = self.locals[name]
        self._emit(f"    str x0, [x29, #{offset}]")

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


def generate(program: Program) -> str:
  """Convenience function to generate assembly for a program."""
  return CodeGenerator().generate(program)
