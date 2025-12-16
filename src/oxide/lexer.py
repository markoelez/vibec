"""Lexer for the Vibec language with Python-style indentation handling."""

from .tokens import KEYWORDS, Token, TokenType

# Single-character tokens (excluding : which needs special handling for ::)
SIMPLE_TOKENS: dict[str, TokenType] = {
  "(": TokenType.LPAREN,
  ")": TokenType.RPAREN,
  "[": TokenType.LBRACKET,
  "]": TokenType.RBRACKET,
  "{": TokenType.LBRACE,
  "}": TokenType.RBRACE,
  ",": TokenType.COMMA,
  ";": TokenType.SEMICOLON,
  ".": TokenType.DOT,
  "+": TokenType.PLUS,
  "*": TokenType.STAR,
  "/": TokenType.SLASH,
  "%": TokenType.PERCENT,
  "&": TokenType.AMP,
  "|": TokenType.PIPE,
  "?": TokenType.QUESTION,
}


class LexerError(Exception):
  """Raised when the lexer encounters invalid input."""

  def __init__(self, message: str, line: int, column: int) -> None:
    super().__init__(f"{message} at line {line}, column {column}")
    self.line, self.column = line, column


class Lexer:
  """Tokenizes Vibec source code, handling indentation-based blocks."""

  def __init__(self, source: str) -> None:
    self.source = source
    self.pos = 0
    self.line = 1
    self.column = 1
    self.indent_stack: list[int] = [0]
    self.tokens: list[Token] = []
    self.at_line_start = True

  def _current(self) -> str:
    return self.source[self.pos] if self.pos < len(self.source) else ""

  def _advance(self) -> str:
    ch = self._current()
    self.pos += 1
    self.line, self.column = (self.line + 1, 1) if ch == "\n" else (self.line, self.column + 1)
    return ch

  def _emit(self, type: TokenType, value: str, line: int, col: int) -> None:
    self.tokens.append(Token(type, value, line, col))

  def _read_while(self, pred) -> str:
    start = self.pos
    while self._current() and pred(self._current()):
      self._advance()
    return self.source[start : self.pos]

  def _handle_indentation(self) -> None:
    """Process indentation at the start of a line."""
    line, col = self.line, self.column
    indent = len(self._read_while(lambda c: c == " "))

    # Skip blank lines and comments
    if self._current() in ("\n", "#", ""):
      return

    current = self.indent_stack[-1]
    if indent > current:
      self.indent_stack.append(indent)
      self._emit(TokenType.INDENT, "", line, col)
    elif indent < current:
      while self.indent_stack[-1] > indent:
        self.indent_stack.pop()
        self._emit(TokenType.DEDENT, "", line, col)
      if self.indent_stack[-1] != indent:
        raise LexerError("Inconsistent indentation", line, col)

  def _two_char_token(self, second: str, two_type: TokenType, one_type: TokenType, line: int, col: int) -> None:
    """Handle potential two-character tokens like ==, !=, <=, >=, ->."""
    self._advance()
    if self._current() == second:
      self._advance()
      self._emit(two_type, f"{self.source[self.pos - 2]}{second}", line, col)
    else:
      self._emit(one_type, self.source[self.pos - 1], line, col)

  def _read_string(self, line: int, col: int) -> None:
    """Read a string literal with escape sequences."""
    self._advance()  # consume opening quote
    chars: list[str] = []
    while self._current() and self._current() != '"':
      if self._current() == "\n":
        raise LexerError("Unterminated string literal", line, col)
      if self._current() == "\\":
        self._advance()
        match self._current():
          case "n":
            chars.append("\n")
          case "t":
            chars.append("\t")
          case "\\":
            chars.append("\\")
          case '"':
            chars.append('"')
          case c:
            raise LexerError(f"Invalid escape sequence '\\{c}'", self.line, self.column)
        self._advance()
      else:
        chars.append(self._advance())
    if not self._current():
      raise LexerError("Unterminated string literal", line, col)
    self._advance()  # consume closing quote
    self._emit(TokenType.STRING, "".join(chars), line, col)

  def tokenize(self) -> list[Token]:
    """Tokenize the entire source and return a list of tokens."""
    while self.pos < len(self.source):
      if self.at_line_start:
        self._handle_indentation()
        self.at_line_start = False

      ch = self._current()
      line, col = self.line, self.column

      match ch:
        case "":
          break
        case "\n":
          self._advance()
          if self.tokens and self.tokens[-1].type not in (TokenType.NEWLINE, TokenType.INDENT):
            self._emit(TokenType.NEWLINE, "\\n", line, col)
          self.at_line_start = True
        case " ":
          self._advance()
        case "#":
          self._read_while(lambda c: c != "\n")
        case c if c.isdigit():
          self._emit(TokenType.INT, self._read_while(str.isdigit), line, col)
        case '"':
          self._read_string(line, col)
        case c if c.isalpha() or c == "_":
          ident = self._read_while(lambda c: c.isalnum() or c == "_")
          self._emit(KEYWORDS.get(ident, TokenType.IDENT), ident, line, col)
        case "-":
          self._two_char_token(">", TokenType.ARROW, TokenType.MINUS, line, col)
        case "=":
          self._two_char_token("=", TokenType.EQ, TokenType.ASSIGN, line, col)
        case "!":
          self._advance()
          if self._current() == "=":
            self._advance()
            self._emit(TokenType.NE, "!=", line, col)
          else:
            raise LexerError("Unexpected character '!'", line, col)
        case "<":
          self._two_char_token("=", TokenType.LE, TokenType.LT, line, col)
        case ">":
          self._two_char_token("=", TokenType.GE, TokenType.GT, line, col)
        case ":":
          self._two_char_token(":", TokenType.COLONCOLON, TokenType.COLON, line, col)
        case c if c in SIMPLE_TOKENS:
          self._advance()
          self._emit(SIMPLE_TOKENS[c], c, line, col)
        case _:
          raise LexerError(f"Unexpected character '{ch}'", line, col)

    # Emit remaining DEDENTs at end of file
    while len(self.indent_stack) > 1:
      self.indent_stack.pop()
      self._emit(TokenType.DEDENT, "", self.line, self.column)

    self._emit(TokenType.EOF, "", self.line, self.column)
    return self.tokens


def tokenize(source: str) -> list[Token]:
  """Convenience function to tokenize source code."""
  return Lexer(source).tokenize()
