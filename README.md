# Vibec

[![CI](https://github.com/markoelez/vibec/actions/workflows/ci.yml/badge.svg)](https://github.com/markoelez/vibec/actions/workflows/ci.yml)

A toy compiled programming language with Python/Rust hybrid syntax, targeting ARM64 macOS.

## Language Features

Vibec combines Python's indentation-based blocks with Rust's explicit type annotations:

```
# Ownership and borrowing (like Rust)
struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

fn read_point(p: &Point) -> i64:    # Borrow immutably
    return p.x + p.y

fn modify_point(p: &mut Point) -> i64:  # Borrow mutably
    p.x = p.x + 10
    return p.x

fn main() -> i64:
    let p: Point = Point { x: 10, y: 20 }
    
    # Multiple shared borrows are OK
    let sum1: i64 = read_point(&p)
    let sum2: i64 = read_point(&p)
    
    # Mutable borrow is exclusive
    let result: i64 = modify_point(&mut p)
    
    # Can still use after borrow ends
    print(p.sum())
    
    return 0
```

**Ownership System:**
- Values have a single owner; ownership transfers on assignment/function calls
- Primitives (`i64`, `bool`) are `Copy` - they clone instead of move
- `&T` - shared/immutable borrow (multiple allowed)
- `&mut T` - exclusive/mutable borrow (only one at a time)
- Cannot use values after they've been moved
- Cannot move values inside loops

**Python-style Keyword Arguments:**
```
fn greet(name: i64, count: i64) -> i64:
    return name * count

fn main() -> i64:
    # Positional arguments
    let a: i64 = greet(10, 5)
    
    # Keyword arguments (any order)
    let b: i64 = greet(count=5, name=10)
    
    # Mixed: positional first, then keyword
    let c: i64 = greet(10, count=5)
    
    return 0
```

**Supported:** ownership & borrowing, enums with `match`, functions with keyword args, structs with `impl` methods, tuples, arrays, vectors, closures, `if`/`else`, `while`, `for i in range()`, references `&T`/`&mut T`.


## Architecture

The compiler is structured as follows:

```
Source Code → Lexer → Parser → Type Checker → Code Generator → ARM64 Assembly
```

### Modules

- `tokens.py` - Token definitions
- `lexer.py` - Tokenization with Python-style indentation handling
- `ast.py` - AST node dataclasses
- `parser.py` - Recursive descent parser with precedence climbing
- `checker.py` - Type checking with scoped symbol tables
- `codegen.py` - ARM64 assembly generation for macOS
- `compiler.py` - Pipeline orchestration
- `cli.py` - Command-line interface


## Installation

```bash
# Clone the repository
git clone https://github.com/markoelez/vibec.git
cd vibec

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
# Compile a source file to executable
vibec source.vb

# Specify output file
vibec source.vb -o myprogram

# Output assembly only
vibec source.vb --emit-asm

# Keep assembly file alongside binary
vibec source.vb --keep-asm
```

## Requirements

- Python 3.12+
- macOS with ARM64 (Apple Silicon)
- Xcode Command Line Tools (for `as` and `ld`)

## License

MIT

