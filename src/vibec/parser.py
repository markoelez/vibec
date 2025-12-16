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
  ExprStmt,
  Function,
  MatchArm,
  ArrayType,
  DerefExpr,
  ImplBlock,
  IndexExpr,
  MatchExpr,
  Parameter,
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
  EnumLiteral,
  EnumVariant,
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
    structs: list[StructDef] = []
    enums: list[EnumDef] = []
    impls: list[ImplBlock] = []
    functions: list[Function] = []
    self._skip_newlines()

    while not self._at_end():
      if self._check(TokenType.STRUCT):
        structs.append(self._parse_struct())
      elif self._check(TokenType.ENUM):
        enums.append(self._parse_enum())
      elif self._check(TokenType.IMPL):
        impls.append(self._parse_impl())
      elif self._check(TokenType.FN):
        functions.append(self._parse_function())
      else:
        raise ParseError("Expected 'struct', 'enum', 'impl', or 'fn'", self._current())
      self._skip_newlines()

    return Program(tuple(structs), tuple(enums), tuple(impls), tuple(functions))

  def _parse_struct(self) -> StructDef:
    """Parse: struct Name: INDENT field: type ... DEDENT"""
    self._expect(TokenType.STRUCT, "Expected 'struct'")
    name_token = self._expect(TokenType.IDENT, "Expected struct name")
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
    return StructDef(name_token.value, tuple(fields))

  def _parse_enum(self) -> EnumDef:
    """Parse: enum Name: INDENT Variant1(Type) Variant2 ... DEDENT"""
    self._expect(TokenType.ENUM, "Expected 'enum'")
    name_token = self._expect(TokenType.IDENT, "Expected enum name")
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
    return EnumDef(name_token.value, tuple(variants))

  def _parse_impl(self) -> ImplBlock:
    """Parse: impl StructName: INDENT fn method(self, ...) -> type: ... DEDENT"""
    self._expect(TokenType.IMPL, "Expected 'impl'")
    struct_name = self._expect(TokenType.IDENT, "Expected struct name")
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
    return ImplBlock(struct_name.value, tuple(methods))

  def _parse_function(self) -> Function:
    """Parse: fn name(params) -> type: INDENT body DEDENT"""
    self._expect(TokenType.FN, "Expected 'fn'")

    name_token = self._expect(TokenType.IDENT, "Expected function name")
    name = name_token.value

    self._expect(TokenType.LPAREN, "Expected '('")
    params = self._parse_parameters()
    self._expect(TokenType.RPAREN, "Expected ')'")

    self._expect(TokenType.ARROW, "Expected '->'")
    return_type = self._parse_type()

    self._expect(TokenType.COLON, "Expected ':'")
    self._expect(TokenType.NEWLINE, "Expected newline after ':'")
    body = self._parse_block()

    return Function(name, tuple(params), return_type, tuple(body))

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
    """Parse: name: type or just 'self' for methods."""
    # Handle 'self' parameter (no type annotation needed)
    if self._check(TokenType.SELF):
      self._advance()
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
      return self._parse_let()
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

  def _parse_let(self) -> LetStmt:
    """Parse: let name: type = expr"""
    self._advance()  # consume 'let'
    name_token = self._expect(TokenType.IDENT, "Expected variable name")
    self._expect(TokenType.COLON, "Expected ':'")
    type_ann = self._parse_type()
    self._expect(TokenType.ASSIGN, "Expected '='")
    value = self._parse_expression()
    self._expect(TokenType.NEWLINE, "Expected newline after let statement")
    return LetStmt(name_token.value, type_ann, value)

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
    """Parse binary expression with precedence climbing."""
    left = self._parse_unary()

    while True:
      token = self._current()
      prec = PRECEDENCE.get(token.type)

      if prec is None or prec < min_prec:
        break

      op = OP_STRINGS[token.type]
      self._advance()
      right = self._parse_binary(prec + 1)
      left = BinaryExpr(left, op, right)

    return left

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

      # Check if it's an enum literal: EnumName::Variant or EnumName::Variant(payload)
      if self._check(TokenType.COLONCOLON):
        self._advance()  # consume '::'
        variant_name = self._expect(TokenType.IDENT, "Expected variant name")
        payload: Expr | None = None
        if self._check(TokenType.LPAREN):
          self._advance()
          payload = self._parse_expression()
          self._expect(TokenType.RPAREN, "Expected ')'")
        return self._parse_postfix(EnumLiteral(name, variant_name.value, payload))
      # Check for Ok(value) and Err(value) - Result constructors
      elif name == "Ok" and self._check(TokenType.LPAREN):
        self._advance()
        value = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')' after Ok value")
        expr: Expr = OkExpr(value)
      elif name == "Err" and self._check(TokenType.LPAREN):
        self._advance()
        value = self._parse_expression()
        self._expect(TokenType.RPAREN, "Expected ')' after Err value")
        expr = ErrExpr(value)
      # Check if it's a function call
      elif self._check(TokenType.LPAREN):
        self._advance()
        args, kwargs = self._parse_arguments()
        self._expect(TokenType.RPAREN, "Expected ')'")
        expr = CallExpr(name, tuple(args), tuple(kwargs))
      # Check if it's a struct literal: Name { field: value, ... }
      elif self._check(TokenType.LBRACE):
        expr = self._parse_struct_literal(name)
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

    else:
      raise ParseError(f"Unexpected token '{token.value}'", token)

  def _parse_postfix(self, expr: Expr) -> Expr:
    """Parse postfix operations: indexing [i], field access .field, tuple index .0, method calls .method(), try ?."""
    while True:
      if self._check(TokenType.LBRACKET):
        # Index expression: expr[index]
        self._advance()
        index = self._parse_expression()
        self._expect(TokenType.RBRACKET, "Expected ']'")
        expr = IndexExpr(expr, index)
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

  def _parse_struct_literal(self, name: str) -> StructLiteral:
    """Parse struct literal: Name { field: value, ... }."""
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
    return StructLiteral(name, tuple(fields))

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

      # Parse arm pattern: EnumName::Variant(binding) or Ok(binding)/Err(binding) for Result
      first_ident = self._expect(TokenType.IDENT, "Expected enum name or variant")

      # Check if this is Ok/Err (Result shorthand) or EnumName::Variant
      if first_ident.value in ("Ok", "Err") and self._check(TokenType.LPAREN):
        # Result shorthand: Ok(binding) or Err(binding)
        enum_name = "Result"
        variant_name = first_ident.value
        self._advance()  # consume '('
        binding_token = self._expect(TokenType.IDENT, "Expected binding name")
        binding: str | None = binding_token.value
        self._expect(TokenType.RPAREN, "Expected ')'")
      else:
        # Standard enum: EnumName::Variant or EnumName::Variant(binding)
        enum_name = first_ident.value
        self._expect(TokenType.COLONCOLON, "Expected '::'")
        variant_token = self._expect(TokenType.IDENT, "Expected variant name")
        variant_name = variant_token.value

        binding = None
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
