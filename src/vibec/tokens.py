"""Token definitions for the Vibec language."""

from enum import Enum, auto
from dataclasses import dataclass


class TokenType(Enum):
  # Keywords
  FN = auto()
  LET = auto()
  STRUCT = auto()
  IF = auto()
  ELSE = auto()
  WHILE = auto()
  FOR = auto()
  IN = auto()
  RANGE = auto()
  RETURN = auto()
  AND = auto()
  OR = auto()
  NOT = auto()
  TRUE = auto()
  FALSE = auto()

  # Identifiers and literals
  IDENT = auto()
  INT = auto()
  STRING = auto()

  # Punctuation
  COLON = auto()
  ARROW = auto()
  LPAREN = auto()
  RPAREN = auto()
  LBRACKET = auto()
  RBRACKET = auto()
  LBRACE = auto()
  RBRACE = auto()
  COMMA = auto()
  SEMICOLON = auto()
  DOT = auto()

  # Operators
  PLUS = auto()
  MINUS = auto()
  STAR = auto()
  SLASH = auto()
  PERCENT = auto()

  # Comparison
  EQ = auto()
  NE = auto()
  LT = auto()
  GT = auto()
  LE = auto()
  GE = auto()

  # Assignment
  ASSIGN = auto()

  # Indentation
  INDENT = auto()
  DEDENT = auto()
  NEWLINE = auto()

  # End of file
  EOF = auto()


KEYWORDS: dict[str, TokenType] = {
  "fn": TokenType.FN,
  "let": TokenType.LET,
  "struct": TokenType.STRUCT,
  "if": TokenType.IF,
  "else": TokenType.ELSE,
  "while": TokenType.WHILE,
  "for": TokenType.FOR,
  "in": TokenType.IN,
  "range": TokenType.RANGE,
  "return": TokenType.RETURN,
  "and": TokenType.AND,
  "or": TokenType.OR,
  "not": TokenType.NOT,
  "true": TokenType.TRUE,
  "false": TokenType.FALSE,
}


@dataclass(frozen=True, slots=True)
class Token:
  type: TokenType
  value: str
  line: int
  column: int

  def __repr__(self) -> str:
    return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"
