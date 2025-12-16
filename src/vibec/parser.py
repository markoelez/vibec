"""Recursive descent parser for the Vibec language."""

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
  Parameter,
  SliceExpr,
  StructDef,
  TupleType,
  TypeAlias,
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
  EnumVariant,
  GenericType,
  StructField,
  ArrayLiteral,
  TupleLiteral,
  StringLiteral,
  StructLiteral,
  MethodCallExpr,
  TupleIndexExpr,
  TypeAnnotation,
  DerefAssignStmt,
  FieldAccessExpr,
  FieldAssignStmt,
  IndexAssignStmt,
  DictComprehension,
  ListComprehension,
)
from .tokens import Token, TokenType


class ParseError(Exception):
  """Raised when the parser encounters a syntax error."""

  def __init__(self, message: str, token: Token) -> None:
    super().__init__(f"{message} at line {token.line}, column {token.column}")
    self.token = token


# Operator precedence (higher = binds tighter)
PRECEDENCE: dict[TokenType, int] = {
  TokenType.OR: 1,
  TokenType.AND: 2,
  TokenType.EQ: 3,
  TokenType.NE: 3,
  TokenType.LT: 4,
  TokenType.GT: 4,
  TokenType.LE: 4,
  TokenType.GE: 4,
  TokenType.PLUS: 5,
  TokenType.MINUS: 5,
  TokenType.STAR: 6,
  TokenType.SLASH: 6,
  TokenType.PERCENT: 6,
}

# Map token types to operator strings
OP_STRINGS: dict[TokenType, str] = {
  TokenType.PLUS: "+",
  TokenType.MINUS: "-",
  TokenType.STAR: "*",
  TokenType.SLASH: "/",
  TokenType.PERCENT: "%",
  TokenType.EQ: "==",
  TokenType.NE: "!=",
  TokenType.LT: "<",
  TokenType.GT: ">",
  TokenType.LE: "<=",
  TokenType.GE: ">=",
  TokenType.AND: "and",
  TokenType.OR: "or",
}

# Comparison operators (for chained comparisons like 0 < x < 10)
COMPARISON_OPS: set[TokenType] = {
  TokenType.LT,
  TokenType.GT,
  TokenType.LE,
  TokenType.GE,
  TokenType.EQ,
  TokenType.NE,
}


class Parser:
  """Parses tokens into an AST."""

  def __init__(self, tokens: list[Token]) -> None:
    self.tokens = tokens
    self.pos = 0

  def _current(self) -> Token:
    return self.tokens[self.pos]

  def _peek(self, offset: int = 1) -> Token:
    pos = self.pos + offset
    if pos >= len(self.tokens):
      return self.tokens[-1]
    return self.tokens[pos]

  def _at_end(self) -> bool:
    return self._current().type == TokenType.EOF

  def _check(self, *types: TokenType) -> bool:
    return self._current().type in types

  def _advance(self) -> Token:
    token = self._current()
    if not self._at_end():
      self.pos += 1
    return token

  def _expect(self, type: TokenType, message: str) -> Token:
    if not self._check(type):
      raise ParseError(message, self._current())
    return self._advance()

  def _skip_newlines(self) -> None:
    while self._check(TokenType.NEWLINE):
      self._advance()

  # === Parsing Functions ===

  def parse(self) -> Program:
    """Parse the entire program."""
    type_aliases: list[TypeAlias] = []
    structs: list[StructDef] = []
    enums: list[EnumDef] = []
    impls: list[ImplBlock] = []
    functions: list[Function] = []
    self._skip_newlines()

    while not self._at_end():
      if self._check(TokenType.TYPE):
        type_aliases.append(self._parse_type_alias())
      elif self._check(TokenType.STRUCT):
        structs.append(self._parse_struct())
      elif self._check(TokenType.ENUM):
        enums.append(self._parse_enum())
      elif self._check(TokenType.IMPL):
        impls.append(self._parse_impl())
      elif self._check(TokenType.FN):
        functions.append(self._parse_function())
      else:
        raise ParseError("Expected 'type', 'struct', 'enum', 'impl', or 'fn'", self._current())
      self._skip_newlines()

    return Program(tuple(type_aliases), tuple(structs), tuple(enums), tuple(impls), tuple(functions))

  def _parse_type_alias(self) -> TypeAlias:
    """Parse: type Name = ExistingType"""
    self._expect(TokenType.TYPE, "Expected 'type'")
    name_token = self._expect(TokenType.IDENT, "Expected type alias name")
    self._expect(TokenType.ASSIGN, "Expected '='")
    target = self._parse_type()
    self._expect(TokenType.NEWLINE, "Expected newline after type alias")
    return TypeAlias(name_token.value, target)

  def _parse_struct(self) -> StructDef:
    """Parse: struct Name<T, U>: INDENT field: type ... DEDENT"""
    self._expect(TokenType.STRUCT, "Expected 'struct'")
    name_token = self._expect(TokenType.IDENT, "Expected struct name")

    # Parse optional type parameters: <T, U>
    type_params = self._parse_type_params()

    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    self._expect(TokenType.INDENT, "Expected indented block")

    fields: list[StructField] = []
    while not self._check(TokenType.DEDENT, TokenType.EOF):
      self._skip_newlines()
      if self._check(TokenType.DEDENT, TokenType.EOF):
        break
      field_name = self._expect(TokenType.IDENT, "Expected field name")
      self._expect(TokenType.COLON, "Expected ':'")
      field_type = self._parse_type()
      self._expect(TokenType.NEWLINE, "Expected newline after field")
      fields.append(StructField(field_name.value, field_type))

    self._expect(TokenType.DEDENT, "Expected dedent")
    return StructDef(name_token.value, tuple(type_params), tuple(fields))

  def _parse_enum(self) -> EnumDef:
    """Parse: enum Name<T>: INDENT Variant1(Type) Variant2 ... DEDENT"""
    self._expect(TokenType.ENUM, "Expected 'enum'")
    name_token = self._expect(TokenType.IDENT, "Expected enum name")

    # Parse optional type parameters: <T>
    type_params = self._parse_type_params()

    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    self._expect(TokenType.INDENT, "Expected indented block")

    variants: list[EnumVariant] = []
    while not self._check(TokenType.DEDENT, TokenType.EOF):
      self._skip_newlines()
      if self._check(TokenType.DEDENT, TokenType.EOF):
        break
      variant_name = self._expect(TokenType.IDENT, "Expected variant name")
      payload_type: TypeAnnotation | None = None
      if self._check(TokenType.LPAREN):
        self._advance()
        payload_type = self._parse_type()
        self._expect(TokenType.RPAREN, "Expected ')' after payload type")
      self._expect(TokenType.NEWLINE, "Expected newline after variant")
      variants.append(EnumVariant(variant_name.value, payload_type))

    self._expect(TokenType.DEDENT, "Expected dedent")
    return EnumDef(name_token.value, tuple(type_params), tuple(variants))

  def _parse_impl(self) -> ImplBlock:
    """Parse: impl StructName<T, U>: INDENT fn method(self, ...) -> type: ... DEDENT"""
    self._expect(TokenType.IMPL, "Expected 'impl'")
    struct_name = self._expect(TokenType.IDENT, "Expected struct name")

    # Parse optional type parameters: <T, U>
    type_params = self._parse_type_params()

    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    self._expect(TokenType.INDENT, "Expected indented block")

    methods: list[Function] = []
    while not self._check(TokenType.DEDENT, TokenType.EOF):
      self._skip_newlines()
      if self._check(TokenType.DEDENT, TokenType.EOF):
        break
      methods.append(self._parse_function())

    self._expect(TokenType.DEDENT, "Expected dedent")
    return ImplBlock(struct_name.value, tuple(type_params), tuple(methods))

  def _parse_type_params(self) -> list[str]:
    """Parse type parameters: <T, U, V> - returns empty list if no type params."""
    type_params: list[str] = []
    if self._check(TokenType.LT):
      self._advance()  # consume '<'
      type_params.append(self._expect(TokenType.IDENT, "Expected type parameter name").value)
      while self._check(TokenType.COMMA):
        self._advance()
        if self._check(TokenType.GT):
          break  # Allow trailing comma
        type_params.append(self._expect(TokenType.IDENT, "Expected type parameter name").value)
      self._expect(TokenType.GT, "Expected '>'")
    return type_params

  def _type_to_str(self, t: TypeAnnotation) -> str:
    """Convert type annotation to string for match arm enum names."""
    match t:
      case SimpleType(name):
        return name
      case ArrayType(elem, size):
        return f"[{self._type_to_str(elem)};{size}]"
      case VecType(elem):
        return f"vec[{self._type_to_str(elem)}]"
      case TupleType(elems):
        return f"({','.join(self._type_to_str(e) for e in elems)})"
      case RefType(inner, mutable):
        prefix = "&mut " if mutable else "&"
        return f"{prefix}{self._type_to_str(inner)}"
      case FnType(params, ret):
        param_strs = ",".join(self._type_to_str(p) for p in params)
        return f"Fn({param_strs})->{self._type_to_str(ret)}"
      case ResultType(ok_type, err_type):
        return f"Result[{self._type_to_str(ok_type)},{self._type_to_str(err_type)}]"
      case DictType(key_type, value_type):
        return f"dict[{self._type_to_str(key_type)},{self._type_to_str(value_type)}]"
      case GenericType(name, type_args):
        args_str = ",".join(self._type_to_str(a) for a in type_args)
        return f"{name}<{args_str}>"
    return "unknown"

  def _looks_like_generic_instantiation(self) -> bool:
    """Check if current position looks like a generic type instantiation.

    This helps disambiguate `Name<T>` (generic) from `name < value` (comparison).
    We scan ahead to find matching `>` and check if followed by `{`, `::`, or `(`.
    """
    if not self._check(TokenType.LT):
      return False

    # Save position
    saved_pos = self.pos
    depth = 0
    self._advance()  # consume '<'
    depth = 1

    while not self._at_end() and depth > 0:
      if self._check(TokenType.LT):
        depth += 1
      elif self._check(TokenType.GT):
        depth -= 1
        if depth == 0:
          # Found matching '>', check what follows
          self._advance()  # consume '>'
          # Now check for struct literal, enum literal, or generic function call
          result = self._check(TokenType.LBRACE, TokenType.COLONCOLON, TokenType.LPAREN)
          self.pos = saved_pos  # restore position
          return result
      elif self._check(TokenType.NEWLINE, TokenType.EOF, TokenType.SEMICOLON):
        # Hit end of expression - not a generic
        self.pos = saved_pos
        return False
      self._advance()

    # Didn't find matching '>'
    self.pos = saved_pos
    return False

  def _parse_function(self) -> Function:
    """Parse: fn name<T, U>(params) -> type: INDENT body DEDENT"""
    self._expect(TokenType.FN, "Expected 'fn'")

    name_token = self._expect(TokenType.IDENT, "Expected function name")
    name = name_token.value

    # Parse optional type parameters: <T, U>
    type_params = self._parse_type_params()

    self._expect(TokenType.LPAREN, "Expected '('")
    params = self._parse_parameters()
    self._expect(TokenType.RPAREN, "Expected ')'")

    self._expect(TokenType.ARROW, "Expected '->'")
    return_type = self._parse_type()

    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    body = self._parse_block()

    return Function(name, tuple(type_params), tuple(params), return_type, tuple(body))

  def _parse_parameters(self) -> list[Parameter]:
    """Parse comma-separated parameters."""
    params: list[Parameter] = []

    if self._check(TokenType.RPAREN):
      return params

    params.append(self._parse_parameter())

    while self._check(TokenType.COMMA):
      self._advance()
      params.append(self._parse_parameter())

    return params

  def _parse_parameter(self) -> Parameter:
    """Parse: name: type or 'self' or 'self: Type' for methods."""
    # Handle 'self' parameter - may or may not have type annotation
    if self._check(TokenType.SELF):
      self._advance()
      # Check if there's an explicit type annotation
      if self._check(TokenType.COLON):
        self._advance()
        type_ann = self._parse_type()
        return Parameter("self", type_ann)
      # Use placeholder type "Self" - type checker will resolve it
      return Parameter("self", SimpleType("Self"))

    name_token = self._expect(TokenType.IDENT, "Expected parameter name")
    self._expect(TokenType.COLON, "Expected ':'")
    type_ann = self._parse_type()
    return Parameter(name_token.value, type_ann)

  def _parse_type(self) -> TypeAnnotation:
    """Parse a type annotation: simple, [T; N], vec[T], (T1, T2, ...), &T, or &mut T."""
    # Reference type: &T or &mut T
    if self._check(TokenType.AMP):
      self._advance()
      mutable = False
      if self._check(TokenType.MUT):
        self._advance()
        mutable = True
      inner_type = self._parse_type()
      return RefType(inner_type, mutable)

    # Array type: [T; N]
    if self._check(TokenType.LBRACKET):
      self._advance()
      element_type = self._parse_type()
      self._expect(TokenType.SEMICOLON, "Expected ';' in array type")
      size_token = self._expect(TokenType.INT, "Expected array size")
      self._expect(TokenType.RBRACKET, "Expected ']'")
      return ArrayType(element_type, int(size_token.value))

    # Tuple type: (T1, T2, ...)
    if self._check(TokenType.LPAREN):
      self._advance()
      element_types: list[TypeAnnotation] = []
      if not self._check(TokenType.RPAREN):
        element_types.append(self._parse_type())
        while self._check(TokenType.COMMA):
          self._advance()
          if self._check(TokenType.RPAREN):
            break  # Allow trailing comma
          element_types.append(self._parse_type())
      self._expect(TokenType.RPAREN, "Expected ')'")
      return TupleType(tuple(element_types))

    # Simple type or vec[T] or Fn(T1, T2) -> T
    token = self._expect(TokenType.IDENT, "Expected type name")

    # Check for vec[T]
    if token.value == "vec" and self._check(TokenType.LBRACKET):
      self._advance()
      element_type = self._parse_type()
      self._expect(TokenType.RBRACKET, "Expected ']'")
      return VecType(element_type)

    # Check for Fn(T1, T2) -> T (function/closure type)
    if token.value == "Fn" and self._check(TokenType.LPAREN):
      self._advance()
      param_types: list[TypeAnnotation] = []
      if not self._check(TokenType.RPAREN):
        param_types.append(self._parse_type())
        while self._check(TokenType.COMMA):
          self._advance()
          if self._check(TokenType.RPAREN):
            break
          param_types.append(self._parse_type())
      self._expect(TokenType.RPAREN, "Expected ')' in Fn type")
      self._expect(TokenType.ARROW, "Expected '->' in Fn type")
      return_type = self._parse_type()
      return FnType(tuple(param_types), return_type)

    # Check for Result[T, E]
    if token.value == "Result" and self._check(TokenType.LBRACKET):
      self._advance()
      ok_type = self._parse_type()
      self._expect(TokenType.COMMA, "Expected ',' in Result type")
      err_type = self._parse_type()
      self._expect(TokenType.RBRACKET, "Expected ']' in Result type")
      return ResultType(ok_type, err_type)

    # Check for dict[K, V]
    if token.value == "dict" and self._check(TokenType.LBRACKET):
      self._advance()
      key_type = self._parse_type()
      self._expect(TokenType.COMMA, "Expected ',' in dict type")
      value_type = self._parse_type()
      self._expect(TokenType.RBRACKET, "Expected ']' in dict type")
      return DictType(key_type, value_type)

    # Check for generic type instantiation: Name<T1, T2>
    if self._check(TokenType.LT):
      self._advance()  # consume '<'
      type_args: list[TypeAnnotation] = []
      type_args.append(self._parse_type())
      while self._check(TokenType.COMMA):
        self._advance()
        if self._check(TokenType.GT):
          break  # Allow trailing comma
        type_args.append(self._parse_type())
      self._expect(TokenType.GT, "Expected '>'")
      return GenericType(token.value, tuple(type_args))

    return SimpleType(token.value)

  def _parse_block(self) -> list[Stmt]:
    """Parse: INDENT stmt* DEDENT"""
    self._expect(TokenType.INDENT, "Expected indented block")
    stmts: list[Stmt] = []

    while not self._check(TokenType.DEDENT, TokenType.EOF):
      self._skip_newlines()
      if self._check(TokenType.DEDENT, TokenType.EOF):
        break
      stmts.append(self._parse_statement())

    self._expect(TokenType.DEDENT, "Expected dedent")
    return stmts

  def _parse_statement(self) -> Stmt:
    """Parse a single statement."""
    if self._check(TokenType.LET):
      return self._parse_let(mutable=True)
    elif self._check(TokenType.CONST):
      return self._parse_let(mutable=False)
    elif self._check(TokenType.RETURN):
      return self._parse_return()
    elif self._check(TokenType.IF):
      return self._parse_if()
    elif self._check(TokenType.WHILE):
      return self._parse_while()
    elif self._check(TokenType.FOR):
      return self._parse_for()
    elif self._check(TokenType.STAR):
      # Could be: dereference assignment (*ptr = value) or expression statement
      return self._parse_deref_assign_or_expr()
    elif self._check(TokenType.IDENT):
      # Could be: assignment, index/field assignment, or expression statement
      if self._peek().type == TokenType.ASSIGN:
        return self._parse_assign()
      elif self._peek().type in (TokenType.LBRACKET, TokenType.DOT):
        return self._parse_complex_assign_or_expr()
      else:
        return self._parse_expr_stmt()
    else:
      return self._parse_expr_stmt()

  def _parse_complex_assign_or_expr(self) -> Stmt:
    """Parse index/field assignment or expression statement."""
    expr = self._parse_expression()
    if self._check(TokenType.ASSIGN):
      self._advance()  # consume '='
      value = self._parse_expression()
      self._expect(TokenType.NEWLINE, "Expected newline after assignment")
      if isinstance(expr, IndexExpr):
        return IndexAssignStmt(expr.target, expr.index, value)
      elif isinstance(expr, FieldAccessExpr):
        return FieldAssignStmt(expr.target, expr.field, value)
      else:
        raise ParseError("Invalid assignment target", self._current())
    self._expect(TokenType.NEWLINE, "Expected newline after expression")
    return ExprStmt(expr)

  def _parse_deref_assign_or_expr(self) -> Stmt:
    """Parse dereference assignment (*ptr = value) or expression statement."""
    expr = self._parse_expression()
    if self._check(TokenType.ASSIGN):
      self._advance()  # consume '='
      value = self._parse_expression()
      self._expect(TokenType.NEWLINE, "Expected newline after assignment")
      if isinstance(expr, DerefExpr):
        return DerefAssignStmt(expr.target, value)
      else:
        raise ParseError("Invalid dereference assignment target", self._current())
    self._expect(TokenType.NEWLINE, "Expected newline after expression")
    return ExprStmt(expr)

  def _parse_assign(self) -> AssignStmt:
    """Parse: name = expr"""
    name_token = self._advance()  # consume identifier
    self._advance()  # consume '='
    value = self._parse_expression()
    self._expect(TokenType.NEWLINE, "Expected newline after assignment")
    return AssignStmt(name_token.value, value)

  def _parse_let(self, mutable: bool = True) -> LetStmt:
    """Parse: let name: type = expr OR const name: type = expr"""
    self._advance()  # consume 'let' or 'const'
    name_token = self._expect(TokenType.IDENT, "Expected variable name")
    self._expect(TokenType.COLON, "Expected ':'")
    type_ann = self._parse_type()
    self._expect(TokenType.ASSIGN, "Expected '='")
    value = self._parse_expression()
    self._expect(TokenType.NEWLINE, "Expected newline after let/const statement")
    return LetStmt(name_token.value, type_ann, value, mutable)

  def _parse_return(self) -> ReturnStmt:
    """Parse: return expr"""
    self._advance()  # consume 'return'
    value = self._parse_expression()
    self._expect(TokenType.NEWLINE, "Expected newline after return statement")
    return ReturnStmt(value)

  def _parse_if(self) -> IfStmt:
    """Parse: if cond: INDENT body DEDENT [else: INDENT body DEDENT]"""
    self._advance()  # consume 'if'
    condition = self._parse_expression()
    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    then_body = self._parse_block()

    else_body: tuple[Stmt, ...] | None = None
    self._skip_newlines()

    if self._check(TokenType.ELSE):
      self._advance()
      self._expect(TokenType.COLON, "Expected ':' after else")
      self._expect(TokenType.NEWLINE, "Expected newline after ':'")
      else_body = tuple(self._parse_block())

    return IfStmt(condition, tuple(then_body), else_body)

  def _parse_while(self) -> WhileStmt:
    """Parse: while cond: INDENT body DEDENT"""
    self._advance()  # consume 'while'
    condition = self._parse_expression()
    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    body = self._parse_block()
    return WhileStmt(condition, tuple(body))

  def _parse_for(self) -> ForStmt:
    """Parse: for var in range(end): or for var in range(start, end):"""
    self._advance()  # consume 'for'
    var_token = self._expect(TokenType.IDENT, "Expected loop variable")
    self._expect(TokenType.IN, "Expected 'in'")
    self._expect(TokenType.RANGE, "Expected 'range'")
    self._expect(TokenType.LPAREN, "Expected '('")

    # Parse range arguments: range(end) or range(start, end)
    first = self._parse_expression()
    if self._check(TokenType.COMMA):
      self._advance()
      start = first
      end = self._parse_expression()
    else:
      start = IntLiteral(0)
      end = first

    self._expect(TokenType.RPAREN, "Expected ')'")
    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    body = self._parse_block()
    return ForStmt(var_token.value, start, end, tuple(body))

  def _parse_expr_stmt(self) -> ExprStmt:
    """Parse an expression statement."""
    expr = self._parse_expression()
    # Match expressions already consume their block, no trailing newline needed
    if not isinstance(expr, MatchExpr):
      self._expect(TokenType.NEWLINE, "Expected newline after expression")
    return ExprStmt(expr)

  # === Expression Parsing with Precedence Climbing ===

  def _parse_expression(self) -> Expr:
    """Parse an expression."""
    return self._parse_binary(0)

  def _parse_binary(self, min_prec: int) -> Expr:
    """Parse binary expression with precedence climbing.

    Supports Python-style chained comparisons (e.g., 0 < x < 10).
    """
    left = self._parse_unary()

    while True:
      token = self._current()
      prec = PRECEDENCE.get(token.type)

      if prec is None or prec < min_prec:
        break

      # Check for chained comparisons (e.g., 0 < x < 10 -> (0 < x) and (x < 10))
      if token.type in COMPARISON_OPS:
        left = self._parse_chained_comparison(left, min_prec)
        continue

      op = OP_STRINGS[token.type]
      self._advance()
      right = self._parse_binary(prec + 1)
      left = BinaryExpr(left, op, right)

    return left

  def _parse_chained_comparison(self, first: Expr, min_prec: int) -> Expr:
    """Parse chained comparisons like 0 < x < 10.

    Transforms into: (0 < x) and (x < 10)
    Each operand is evaluated only once in the AST (though the middle operand
    appears in multiple comparisons, it's the same AST node).
    """
    # Collect all operands and operators in the chain
    operands: list[Expr] = [first]
    operators: list[str] = []

    # Arithmetic operators have higher precedence than comparisons
    # Parse operands at arithmetic level (precedence 5) to allow: 0 < x + 1 < 10
    arithmetic_prec = 5  # PLUS/MINUS precedence

    while True:
      token = self._current()
      if token.type not in COMPARISON_OPS:
        break
      prec = PRECEDENCE[token.type]
      if prec < min_prec:
        break

      operators.append(OP_STRINGS[token.type])
      self._advance()
      # Parse at arithmetic precedence to capture expressions like x + 1
      operand = self._parse_binary(arithmetic_prec)
      operands.append(operand)

    # Build the result
    if len(operators) == 1:
      # Single comparison, not actually a chain
      return BinaryExpr(operands[0], operators[0], operands[1])

    # Multiple comparisons: chain them with 'and'
    # a < b < c < d -> (a < b) and (b < c) and (c < d)
    comparisons: list[Expr] = []
    for i, op in enumerate(operators):
      comparisons.append(BinaryExpr(operands[i], op, operands[i + 1]))

    result = comparisons[0]
    for comp in comparisons[1:]:
      result = BinaryExpr(result, "and", comp)

    return result

  def _parse_unary(self) -> Expr:
    """Parse unary expression."""
    if self._check(TokenType.MINUS):
      self._advance()
      operand = self._parse_unary()
      return UnaryExpr("-", operand)
    elif self._check(TokenType.NOT):
      self._advance()
      operand = self._parse_unary()
      return UnaryExpr("not", operand)

    # Reference creation: &expr or &mut expr
    elif self._check(TokenType.AMP):
      self._advance()
      mutable = False
      if self._check(TokenType.MUT):
        self._advance()
        mutable = True
      operand = self._parse_unary()
      return RefExpr(operand, mutable)

    # Dereference: *expr
    elif self._check(TokenType.STAR):
      self._advance()
      operand = self._parse_unary()
      return DerefExpr(operand)

    return self._parse_primary()

  def _parse_primary(self) -> Expr:
    """Parse primary expression (literals, variables, calls, parenthesized)."""
    token = self._current()

    if token.type == TokenType.INT:
      self._advance()
      return IntLiteral(int(token.value))

    elif token.type == TokenType.TRUE:
      self._advance()
      return BoolLiteral(True)

    elif token.type == TokenType.FALSE:
      self._advance()
      return BoolLiteral(False)

    elif token.type == TokenType.STRING:
      self._advance()
      return StringLiteral(token.value)

    elif token.type == TokenType.LBRACKET:
      # Array literal: [1, 2, 3]
      return self._parse_array_literal()

    elif token.type == TokenType.MATCH:
      # Match expression
      return self._parse_match()

    elif token.type == TokenType.PIPE:
      # Closure expression: |a: i64, b: i64| -> i64: a + b
      return self._parse_closure()

    elif token.type == TokenType.SELF:
      # Self reference in method
      self._advance()
      return self._parse_postfix(VarExpr("self"))

    elif token.type == TokenType.IDENT:
      name = token.value
      self._advance()

      # Check for generic type arguments: Name<T1, T2>
      # Only parse as generics if followed by `{` (struct literal) or `::` (enum literal)
      # to avoid ambiguity with less-than comparison operator
      type_args: tuple[TypeAnnotation, ...] = ()
      if self._check(TokenType.LT) and self._looks_like_generic_instantiation():
        self._advance()  # consume '<'
        type_args_list: list[TypeAnnotation] = []
        type_args_list.append(self._parse_type())
        while self._check(TokenType.COMMA):
          self._advance()
          if self._check(TokenType.GT):
            break  # Allow trailing comma
          type_args_list.append(self._parse_type())
        self._expect(TokenType.GT, "Expected '>'")
        type_args = tuple(type_args_list)

      # Check if it's an enum literal: EnumName::Variant or EnumName<T>::Variant(payload)
      if self._check(TokenType.COLONCOLON):
        self._advance()  # consume '::'
        variant_name = self._expect(TokenType.IDENT, "Expected variant name")
        payload: Expr | None = None
        if self._check(TokenType.LPAREN):
          self._advance()
          payload = self._parse_expression()
          self._expect(TokenType.RPAREN, "Expected ')'")
        return self._parse_postfix(EnumLiteral(name, type_args, variant_name.value, payload))
      # Check for Ok(value) and Err(value) - Result constructors
      elif name == "Ok" and self._check(TokenType.LPAREN) and not type_args:
        self._advance()
        value = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')' after Ok value")
        expr: Expr = OkExpr(value)
      elif name == "Err" and self._check(TokenType.LPAREN) and not type_args:
        self._advance()
        value = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')' after Err value")
        expr = ErrExpr(value)
      # Check if it's a function call (with or without type arguments)
      elif self._check(TokenType.LPAREN):
        self._advance()
        args, kwargs = self._parse_arguments()
        self._expect(TokenType.RPAREN, "Expected ')'")
        # For generic function calls like identity<i64>(42), mangle the name
        if type_args:
          mangled_name = f"{name}<{','.join(self._type_to_str(t) for t in type_args)}>"
          expr = CallExpr(mangled_name, tuple(args), tuple(kwargs))
        else:
          expr = CallExpr(name, tuple(args), tuple(kwargs))
      # Check if it's a struct literal: Name { field: value, ... } or Name<T, U> { ... }
      elif self._check(TokenType.LBRACE):
        expr = self._parse_struct_literal(name, type_args)
      else:
        expr = VarExpr(name)

      # Handle postfix operations: indexing, field access, method calls
      return self._parse_postfix(expr)

    elif token.type == TokenType.LPAREN:
      self._advance()
      # Could be parenthesized expression or tuple literal
      if self._check(TokenType.RPAREN):
        # Empty tuple ()
        self._advance()
        return self._parse_postfix(TupleLiteral(()))

      first = self._parse_expression()

      if self._check(TokenType.COMMA):
        # Tuple literal: (expr, expr, ...)
        elements: list[Expr] = [first]
        while self._check(TokenType.COMMA):
          self._advance()
          if self._check(TokenType.RPAREN):
            break  # Allow trailing comma
          elements.append(self._parse_expression())
        self._expect(TokenType.RPAREN, "Expected ')'")
        return self._parse_postfix(TupleLiteral(tuple(elements)))
      else:
        # Parenthesized expression
        self._expect(TokenType.RPAREN, "Expected ')'")
        return self._parse_postfix(first)

    elif token.type == TokenType.LBRACE:
      # Dict literal: {key: value, ...}
      return self._parse_dict_literal()

    else:
      raise ParseError(f"Unexpected token '{token.value}'", token)

  def _parse_dict_literal(self) -> DictLiteral | DictComprehension:
    """Parse dict literal {key: value, ...} or dict comprehension {k: v for x in range(...)}."""
    self._expect(TokenType.LBRACE, "Expected '{'")

    if self._check(TokenType.RBRACE):
      # Empty dict
      self._advance()
      return DictLiteral(())

    # Parse first key: value
    key_expr = self._parse_expression()
    self._expect(TokenType.COLON, "Expected ':' in dict literal")
    value_expr = self._parse_expression()

    # Check if this is a dict comprehension: {k: v for var in range(...)}
    if self._check(TokenType.FOR):
      return self._parse_dict_comprehension(key_expr, value_expr)

    # Regular dict literal
    entries: list[tuple[Expr, Expr]] = [(key_expr, value_expr)]

    while self._check(TokenType.COMMA):
      self._advance()
      if self._check(TokenType.RBRACE):
        break  # Allow trailing comma
      key = self._parse_expression()
      self._expect(TokenType.COLON, "Expected ':' in dict literal")
      value = self._parse_expression()
      entries.append((key, value))

    self._expect(TokenType.RBRACE, "Expected '}'")
    return DictLiteral(tuple(entries))

  def _parse_dict_comprehension(self, key_expr: Expr, value_expr: Expr) -> DictComprehension:
    """Parse remainder of dict comprehension after key: value: for var in range(start, end) [if cond]}."""
    self._expect(TokenType.FOR, "Expected 'for'")
    var_token = self._expect(TokenType.IDENT, "Expected loop variable")
    self._expect(TokenType.IN, "Expected 'in'")

    # Expect range(start, end)
    self._expect(TokenType.RANGE, "Expected 'range'")
    self._expect(TokenType.LPAREN, "Expected '(' after range")
    start_expr = self._parse_expression()
    self._expect(TokenType.COMMA, "Expected ',' in range()")
    end_expr = self._parse_expression()
    self._expect(TokenType.RPAREN, "Expected ')' after range arguments")

    # Optional condition: if cond
    condition: Expr | None = None
    if self._check(TokenType.IF):
      self._advance()
      condition = self._parse_expression()

    self._expect(TokenType.RBRACE, "Expected '}'")
    return DictComprehension(key_expr, value_expr, var_token.value, start_expr, end_expr, condition)

  def _parse_postfix(self, expr: Expr) -> Expr:
    """Parse postfix operations: indexing [i], slicing [i:j], field access .field, tuple index .0, method calls .method(), try ?."""
    while True:
      if self._check(TokenType.LBRACKET):
        # Index or slice expression: expr[index] or expr[start:stop:step]
        self._advance()
        expr = self._parse_index_or_slice(expr)
      elif self._check(TokenType.DOT):
        # Field access, tuple index, or method call
        self._advance()
        if self._check(TokenType.INT):
          # Tuple index: expr.0, expr.1
          index_token = self._advance()
          expr = TupleIndexExpr(expr, int(index_token.value))
        elif self._check(TokenType.IDENT):
          member_token = self._advance()
          if self._check(TokenType.LPAREN):
            # Method call: expr.method(args) - only positional args for methods
            self._advance()
            args, kwargs = self._parse_arguments()
            if kwargs:
              raise ParseError("Method calls do not support keyword arguments", self._current())
            self._expect(TokenType.RPAREN, "Expected ')'")
            expr = MethodCallExpr(expr, member_token.value, tuple(args))
          else:
            # Field access: expr.field
            expr = FieldAccessExpr(expr, member_token.value)
        else:
          raise ParseError("Expected field name or tuple index", self._current())
      elif self._check(TokenType.QUESTION):
        # Try expression: expr?
        self._advance()
        expr = TryExpr(expr)
      else:
        break
    return expr

  def _parse_index_or_slice(self, target: Expr) -> Expr:
    """Parse index [i] or slice [start:stop:step] expression after opening bracket.

    Supports Python-style slice syntax:
      - arr[i]       -> IndexExpr (regular index)
      - arr[:]       -> SliceExpr (full slice)
      - arr[start:]  -> SliceExpr (from start to end)
      - arr[:stop]   -> SliceExpr (from beginning to stop)
      - arr[start:stop] -> SliceExpr (from start to stop)
      - arr[::step]  -> SliceExpr (full slice with step)
      - arr[start:stop:step] -> SliceExpr (full slice with step)

    Note: '::' is lexed as COLONCOLON, so we need to handle that specially.
    """
    # Check for [::step] - full slice with step (COLONCOLON is lexed as single token)
    if self._check(TokenType.COLONCOLON):
      self._advance()  # consume '::'
      start: Expr | None = None
      stop: Expr | None = None
      step: Expr | None = None

      # Check for step value
      if not self._check(TokenType.RBRACKET):
        step = self._parse_expression()

      self._expect(TokenType.RBRACKET, "Expected ']'")
      return SliceExpr(target, start, stop, step)

    # Check for empty slice [:...]
    if self._check(TokenType.COLON):
      # It's a slice starting with :
      self._advance()  # consume first ':'
      start = None
      stop = None
      step = None

      # Check for :step (was written as ::step but already handled COLONCOLON above)
      # This handles [:stop] or [:stop:step] or [:]
      if not self._check(TokenType.COLON, TokenType.RBRACKET):
        stop = self._parse_expression()

      # Check for step
      if self._check(TokenType.COLON):
        self._advance()
        if not self._check(TokenType.RBRACKET):
          step = self._parse_expression()

      self._expect(TokenType.RBRACKET, "Expected ']'")
      return SliceExpr(target, start, stop, step)

    # Parse first expression (could be index or slice start)
    first = self._parse_expression()

    # Check if this is a slice (colon or coloncolon follows)
    if self._check(TokenType.COLONCOLON):
      # [start::step]
      self._advance()  # consume '::'
      start = first
      stop = None
      step = None

      if not self._check(TokenType.RBRACKET):
        step = self._parse_expression()

      self._expect(TokenType.RBRACKET, "Expected ']'")
      return SliceExpr(target, start, stop, step)

    if self._check(TokenType.COLON):
      self._advance()  # consume ':'
      start = first
      stop = None
      step = None

      # Check for stop value
      if not self._check(TokenType.COLON, TokenType.RBRACKET):
        stop = self._parse_expression()

      # Check for step
      if self._check(TokenType.COLON):
        self._advance()
        if not self._check(TokenType.RBRACKET):
          step = self._parse_expression()

      self._expect(TokenType.RBRACKET, "Expected ']'")
      return SliceExpr(target, start, stop, step)

    # Regular index expression
    self._expect(TokenType.RBRACKET, "Expected ']'")
    return IndexExpr(target, first)

  def _parse_struct_literal(self, name: str, type_args: tuple[TypeAnnotation, ...] = ()) -> StructLiteral:
    """Parse struct literal: Name { field: value, ... } or Name<T, U> { field: value, ... }."""
    self._expect(TokenType.LBRACE, "Expected '{'")
    fields: list[tuple[str, Expr]] = []

    if not self._check(TokenType.RBRACE):
      field_name = self._expect(TokenType.IDENT, "Expected field name")
      self._expect(TokenType.COLON, "Expected ':'")
      field_value = self._parse_expression()
      fields.append((field_name.value, field_value))

      while self._check(TokenType.COMMA):
        self._advance()
        if self._check(TokenType.RBRACE):
          break  # Allow trailing comma
        field_name = self._expect(TokenType.IDENT, "Expected field name")
        self._expect(TokenType.COLON, "Expected ':'")
        field_value = self._parse_expression()
        fields.append((field_name.value, field_value))

    self._expect(TokenType.RBRACE, "Expected '}'")
    return StructLiteral(name, type_args, tuple(fields))

  def _parse_array_literal(self) -> ArrayLiteral | ListComprehension:
    """Parse array literal [expr, expr, ...] or list comprehension [expr for var in range(start, end) if cond]."""
    self._expect(TokenType.LBRACKET, "Expected '['")

    if self._check(TokenType.RBRACKET):
      # Empty array
      self._advance()
      return ArrayLiteral(())

    # Parse first expression
    first_expr = self._parse_expression()

    # Check if this is a list comprehension: [expr for var in range(...)]
    if self._check(TokenType.FOR):
      return self._parse_list_comprehension(first_expr)

    # Regular array literal
    elements: list[Expr] = [first_expr]
    while self._check(TokenType.COMMA):
      self._advance()
      if self._check(TokenType.RBRACKET):
        break  # Allow trailing comma
      elements.append(self._parse_expression())

    self._expect(TokenType.RBRACKET, "Expected ']'")
    return ArrayLiteral(tuple(elements))

  def _parse_list_comprehension(self, element_expr: Expr) -> ListComprehension:
    """Parse remainder of list comprehension after element expression: for var in range(start, end) [if cond]]."""
    self._expect(TokenType.FOR, "Expected 'for'")
    var_token = self._expect(TokenType.IDENT, "Expected loop variable")
    self._expect(TokenType.IN, "Expected 'in'")

    # Expect range(start, end)
    self._expect(TokenType.RANGE, "Expected 'range'")
    self._expect(TokenType.LPAREN, "Expected '(' after range")
    start_expr = self._parse_expression()
    self._expect(TokenType.COMMA, "Expected ',' in range()")
    end_expr = self._parse_expression()
    self._expect(TokenType.RPAREN, "Expected ')' after range arguments")

    # Optional condition: if cond
    condition: Expr | None = None
    if self._check(TokenType.IF):
      self._advance()
      condition = self._parse_expression()

    self._expect(TokenType.RBRACKET, "Expected ']'")
    return ListComprehension(element_expr, var_token.value, start_expr, end_expr, condition)

  def _parse_match(self) -> MatchExpr:
    """Parse match expression: match expr: INDENT arms... DEDENT"""
    self._expect(TokenType.MATCH, "Expected 'match'")
    target = self._parse_expression()
    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    self._expect(TokenType.INDENT, "Expected indented block")

    arms: list[MatchArm] = []
    while not self._check(TokenType.DEDENT, TokenType.EOF):
      self._skip_newlines()
      if self._check(TokenType.DEDENT, TokenType.EOF):
        break

      # Parse arm pattern: EnumName::Variant(binding), EnumName<T>::Variant(binding),
      # or Ok(binding)/Err(binding) for Result
      first_ident = self._expect(TokenType.IDENT, "Expected enum name or variant")

      # Check for generic type arguments in enum name: EnumName<T, U>
      enum_name = first_ident.value
      if self._check(TokenType.LT):
        # Parse generic type args: <T1, T2, ...>
        self._advance()  # consume '<'
        type_arg_strs: list[str] = []
        type_arg_strs.append(self._type_to_str(self._parse_type()))
        while self._check(TokenType.COMMA):
          self._advance()
          if self._check(TokenType.GT):
            break
          type_arg_strs.append(self._type_to_str(self._parse_type()))
        self._expect(TokenType.GT, "Expected '>'")
        # Mangle the enum name with type args: EnumName<i64> -> EnumName<i64>
        enum_name = f"{first_ident.value}<{','.join(type_arg_strs)}>"

      # Check if this is Ok/Err (Result shorthand) or EnumName::Variant
      binding: str | None = None
      if first_ident.value in ("Ok", "Err") and self._check(TokenType.LPAREN):
        # Result shorthand: Ok(binding) or Err(binding)
        enum_name = "Result"
        variant_name = first_ident.value
        self._advance()  # consume '('
        binding_token = self._expect(TokenType.IDENT, "Expected binding name")
        binding = binding_token.value
        self._expect(TokenType.RPAREN, "Expected ')'")
      else:
        # Standard enum: EnumName::Variant or EnumName<T>::Variant(binding)
        self._expect(TokenType.COLONCOLON, "Expected '::'")
        variant_token = self._expect(TokenType.IDENT, "Expected variant name")
        variant_name = variant_token.value

        if self._check(TokenType.LPAREN):
          self._advance()
          binding_token = self._expect(TokenType.IDENT, "Expected binding name")
          binding = binding_token.value
          self._expect(TokenType.RPAREN, "Expected ')'")

      self._expect(TokenType.COLON, "Expected ':'")
      self._expect(TokenType.NEWLINE, "Expected newline after ':'")
      body = self._parse_block()
      arms.append(MatchArm(enum_name, variant_name, binding, tuple(body)))

    self._expect(TokenType.DEDENT, "Expected dedent")
    return MatchExpr(target, tuple(arms))

  def _parse_closure(self) -> ClosureExpr:
    """Parse closure: |a: i64, b: i64| -> i64: expr."""
    self._expect(TokenType.PIPE, "Expected '|'")

    # Parse parameters
    params: list[Parameter] = []
    if not self._check(TokenType.PIPE):
      params.append(self._parse_parameter())
      while self._check(TokenType.COMMA):
        self._advance()
        if self._check(TokenType.PIPE):
          break
        params.append(self._parse_parameter())

    self._expect(TokenType.PIPE, "Expected '|' after closure parameters")

    # Parse return type
    self._expect(TokenType.ARROW, "Expected '->' after closure parameters")
    return_type = self._parse_type()

    # Parse body (single expression after colon)
    self._expect(TokenType.COLON, "Expected ':' before closure body")
    body = self._parse_expression()

    return ClosureExpr(tuple(params), return_type, body)

  def _parse_arguments(self) -> tuple[list[Expr], list[tuple[str, Expr]]]:
    """Parse comma-separated arguments, including keyword arguments.

    Returns (positional_args, keyword_args) where keyword_args is [(name, value), ...].
    Keyword arguments must come after all positional arguments.
    """
    args: list[Expr] = []
    kwargs: list[tuple[str, Expr]] = []
    seen_kwarg = False

    if self._check(TokenType.RPAREN):
      return args, kwargs

    # Parse first argument
    first_arg, first_kwarg = self._parse_argument()
    if first_kwarg is not None:
      kwargs.append(first_kwarg)
      seen_kwarg = True
    else:
      args.append(first_arg)  # type: ignore

    while self._check(TokenType.COMMA):
      self._advance()
      if self._check(TokenType.RPAREN):
        break  # Allow trailing comma
      arg, kwarg = self._parse_argument()
      if kwarg is not None:
        kwargs.append(kwarg)
        seen_kwarg = True
      else:
        if seen_kwarg:
          raise ParseError("Positional argument cannot follow keyword argument", self._current())
        args.append(arg)  # type: ignore

    return args, kwargs

  def _parse_argument(self) -> tuple[Expr | None, tuple[str, Expr] | None]:
    """Parse a single argument, which may be positional or keyword.

    Returns (expr, None) for positional, (None, (name, expr)) for keyword.
    """
    # Check if this is a keyword argument: IDENT = expr
    # Use look-ahead to check if next token is '=' without consuming IDENT
    if self._check(TokenType.IDENT) and self._peek().type == TokenType.ASSIGN:
      name_token = self._advance()  # consume IDENT
      self._advance()  # consume '='
      value = self._parse_expression()
      return None, (name_token.value, value)

    # Regular positional argument
    expr = self._parse_expression()
    return expr, None


def parse(tokens: list[Token]) -> Program:
  """Convenience function to parse tokens into a Program."""
  return Parser(tokens).parse()
