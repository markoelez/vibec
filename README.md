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

**Rust-style Implicit Return:**
```
fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    n * factorial(n - 1)  # Last expression is implicitly returned

fn square(x: i64) -> i64:
    x * x  # No 'return' keyword needed

fn main() -> i64:
    print(factorial(5))  # 120
    0  # Implicit return
```

**Result Type and Error Propagation (like Rust):**
```
# Function that can fail returns Result[OkType, ErrType]
fn divide(a: i64, b: i64) -> Result[i64, i64]:
    if b == 0:
        return Err(1)  # Error code for division by zero
    Ok(a / b)

# The ? operator unwraps Ok or propagates Err
fn calculate(x: i64, y: i64, z: i64) -> Result[i64, i64]:
    let step1: i64 = divide(x, y)?  # Returns early with Err if y == 0
    let step2: i64 = divide(step1, z)?  # Returns early with Err if z == 0
    Ok(step1 + step2)

fn main() -> i64:
    let result: Result[i64, i64] = calculate(100, 5, 2)
    match result:
        Ok(value):
            print(value)  # 30
        Err(error):
            print(error)
    0
```

**Python/Rust-style Hashmaps (dict):**
```
fn main() -> i64:
    # Create and use dict
    let scores: dict[i64, i64] = {}
    scores[1] = 100
    scores[2] = 85
    print(scores[1])  # 100
    print(scores.len())  # 2
    
    # Dict literals and methods
    let prices: dict[i64, i64] = {100: 50, 200: 75}
    if prices.contains(100):
        print(prices.get(100))  # 50
    prices.insert(300, 100)
    prices.remove(100)
    
    # Dict comprehension
    let squares: dict[i64, i64] = {x: x * x for x in range(0, 5)}
    let evens: dict[i64, i64] = {x: x * 2 for x in range(0, 10) if x % 2 == 0}
    0
```

**Python-style List Comprehensions:**
```
fn main() -> i64:
    # Basic comprehension
    let squares: vec[i64] = [x * x for x in range(0, 10)]
    print(squares.sum())  # 285
    
    # With filter condition
    let evens: vec[i64] = [x for x in range(0, 20) if x % 2 == 0]
    print(evens.len())  # 10
    
    # Complex expressions
    let transformed: vec[i64] = [x * 2 + 1 for x in range(0, 5)]
    0
```

**Functional Iterator Methods:**
```
fn main() -> i64:
    let v: vec[i64] = []
    for i in range(0, 10):
        v.push(i)
    
    # Rust-style method chaining
    let result: vec[i64] = v.into_iter().skip(2).take(5).map(|x: i64| -> i64: x * 2).filter(|x: i64| -> bool: x > 5).collect()
    print(result.sum())  # 36
    
    # Fold for reductions
    let factorial: i64 = v.take(5).fold(1, |acc: i64, x: i64| -> i64: acc * x)
    0
```

**Supported:** hashmaps with dict comprehensions (`dict[K,V]`), list comprehensions, `Result[T, E]` type with `?` operator, functional iterators (`map`, `filter`, `fold`, `skip`, `take`, `sum`), implicit return, ownership & borrowing, enums with `match`, keyword args, structs with `impl`, tuples, arrays, vectors, closures.


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

## Todo

- Type aliases
- Const declarations
- Python-style Chained comparison
- Pattern guards
- Generics
- Rust-style traits
- Python-style slice syntax 

## License

MIT

