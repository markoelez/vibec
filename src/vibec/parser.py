"""Recursive descent parser for the Vibec language."""

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
  Parameter,
  UnaryExpr,
  WhileStmt,
  AssignStmt,
  BinaryExpr,
  IntLiteral,
  ReturnStmt,
  BoolLiteral,
  StringLiteral,
  TypeAnnotation,
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
    functions: list[Function] = []
    self._skip_newlines()

    while not self._at_end():
      functions.append(self._parse_function())
      self._skip_newlines()

    return Program(tuple(functions))

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
    """Parse: name: type"""
    name_token = self._expect(TokenType.IDENT, "Expected parameter name")
    self._expect(TokenType.COLON, "Expected ':'")
    type_ann = self._parse_type()
    return Parameter(name_token.value, type_ann)

  def _parse_type(self) -> TypeAnnotation:
    """Parse a type annotation."""
    token = self._expect(TokenType.IDENT, "Expected type name")
    return TypeAnnotation(token.value)

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
    elif self._check(TokenType.IDENT) and self._peek().type == TokenType.ASSIGN:
      return self._parse_assign()
    else:
      return self._parse_expr_stmt()

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

  def _parse_expr_stmt(self) -> ExprStmt:
    """Parse an expression statement."""
    expr = self._parse_expression()
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

    elif token.type == TokenType.IDENT:
      name = token.value
      self._advance()

      # Check if it's a function call
      if self._check(TokenType.LPAREN):
        self._advance()
        args = self._parse_arguments()
        self._expect(TokenType.RPAREN, "Expected ')'")
        return CallExpr(name, tuple(args))

      return VarExpr(name)

    elif token.type == TokenType.LPAREN:
      self._advance()
      expr = self._parse_expression()
      self._expect(TokenType.RPAREN, "Expected ')'")
      return expr

    else:
      raise ParseError(f"Unexpected token '{token.value}'", token)

  def _parse_arguments(self) -> list[Expr]:
    """Parse comma-separated arguments."""
    args: list[Expr] = []

    if self._check(TokenType.RPAREN):
      return args

    args.append(self._parse_expression())

    while self._check(TokenType.COMMA):
      self._advance()
      args.append(self._parse_expression())

    return args


def parse(tokens: list[Token]) -> Program:
  """Convenience function to parse tokens into a Program."""
  return Parser(tokens).parse()
