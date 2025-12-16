# Oxide

[![CI](https://github.com/markoelez/oxide/actions/workflows/ci.yml/badge.svg)](https://github.com/markoelez/oxide/actions/workflows/ci.yml)

A toy compiled programming language with Python/Rust hybrid syntax, targeting ARM64 macOS.

## Installation

```bash
# Clone the repository
git clone https://github.com/markoelez/oxide.git
cd oxide

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
# Compile a source file to executable
oxide source.ox

# Specify output file
oxide source.ox -o myprogram

# Output assembly only
oxide source.ox --emit-asm

# Keep assembly file alongside binary
oxide source.ox --keep-asm
```

## Requirements

- Python 3.12+
- macOS with ARM64 (Apple Silicon)
- Xcode Command Line Tools (for `as` and `ld`)

## Language Features

Oxide combines Python's indentation-based blocks with Rust's explicit type annotations:

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

**Variables and Constants:**
```
fn main() -> i64:
    # Mutable variable (can be reassigned)
    let x: i64 = 10
    x = 20  # OK
    
    # Constant (cannot be reassigned)
    const PI_APPROX: i64 = 3
    # PI_APPROX = 4  # Error: Cannot assign to const variable
    
    # Const with structs
    const origin: Point = Point { x: 0, y: 0 }
    # origin.x = 1  # Error: Cannot assign to const variable
    
    x + PI_APPROX
```

**Rust-style Type Aliases:**
```
# Simple type aliases
type Integer = i64
type IntVec = vec[i64]
type IntPair = (i64, i64)

# Type alias for struct
struct Point:
    x: i64
    y: i64
type Vec2D = Point

fn add(a: Integer, b: Integer) -> Integer:
    a + b

fn main() -> i64:
    let x: Integer = 10
    let nums: IntVec = []
    nums.push(1)
    let p: Vec2D = Point { x: 5, y: 10 }
    add(x, p.x)  # 15
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

**Option Type and ? Operator (like Rust):**
```
# Generic Option enum for nullable values
enum Option<T>:
    Some(T)
    None

fn get_positive(x: i64) -> Option<i64>:
    if x > 0:
        Option<i64>::Some(x)
    else:
        Option<i64>::None

# The ? operator unwraps Some or propagates None
fn add_positives(a: i64, b: i64) -> Option<i64>:
    let x: i64 = get_positive(a)?  # Returns early with None if a <= 0
    let y: i64 = get_positive(b)?  # Returns early with None if b <= 0
    Option<i64>::Some(x + y)

fn main() -> i64:
    let result: Option<i64> = add_positives(10, 20)
    match result:
        Option<i64>::Some(val):
            print(val)  # 30
        Option<i64>::None:
            print(0)
    0
```

**Rust-style Pattern Guards:**
```
enum Option:
    Some(i64)
    None

fn classify(opt: Option) -> i64:
    match opt:
        # Pattern guards with 'if' clause
        Option::Some(x) if x < 0:
            print(-1)  # Negative
            -1
        Option::Some(x) if x == 0:
            print(0)   # Zero
            0
        Option::Some(x) if x < 10:
            print(1)   # Small positive
            1
        Option::Some(x):
            print(2)   # Large positive (fallback)
            2
        Option::None:
            print(99)
            99

fn main() -> i64:
    let a: Option = Option::Some(5)
    classify(a)  # Prints 1 (small positive)
    
    # Guards can use chained comparisons
    let b: Option = Option::Some(50)
    match b:
        Option::Some(x) if 0 < x < 100:
            print(1)  # In range
        Option::Some(x):
            print(0)  # Out of range
        Option::None:
            print(-1)
    
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

**Python-style Slice Syntax:**
```
fn main() -> i64:
    let v: vec[i64] = [x * 10 for x in range(0, 10)]
    # v = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # Basic slice: [start:stop] - elements from index 2 to 5 (exclusive)
    let slice1: vec[i64] = v[2:5]  # [20, 30, 40]
    
    # From start: [:stop] - first 3 elements
    let first_three: vec[i64] = v[:3]  # [0, 10, 20]
    
    # To end: [start:] - elements from index 7 onwards  
    let last_three: vec[i64] = v[7:]  # [70, 80, 90]
    
    # Full copy: [:] - copy entire vector
    let copy: vec[i64] = v[:]
    
    # With step: [::step] - every 2nd element
    let evens: vec[i64] = v[::2]  # [0, 20, 40, 60, 80]
    
    # Full form: [start:stop:step]
    let middle_evens: vec[i64] = v[2:8:2]  # [20, 40, 60]
    
    # Negative indices (from end)
    let last_two: vec[i64] = v[-2:]  # [80, 90]
    let without_last_two: vec[i64] = v[:-2]  # first 8 elements
    
    # Array slicing (returns a vec)
    let arr: [i64; 5] = [1, 2, 3, 4, 5]
    let arr_slice: vec[i64] = arr[1:4]  # [2, 3, 4]
    
    # Chain with other operations
    let sum: i64 = v[0:5].sum()  # 0 + 10 + 20 + 30 + 40 = 100
    0
```

**Python-style Chained Comparisons:**
```
fn main() -> i64:
    let x: i64 = 5
    
    # Chained comparisons: a < b < c is equivalent to (a < b) and (b < c)
    if 0 < x < 10:
        print(1)  # x is in range (0, 10)
    
    # Works with any comparison operators
    if 0 <= x <= 100:
        print(2)  # x is in range [0, 100]
    
    # Multiple chains
    let a: i64 = 1
    let b: i64 = 2
    let c: i64 = 3
    if a < b < c:
        print(3)  # all comparisons true
    
    # Can mix different operators
    if 0 < x <= 10:
        print(4)
    
    # Works with expressions
    if 0 < x + 1 < 20:
        print(5)
    
    0
```

**Operator Overloading (like Rust/C++):**
```
struct Vec2:
    x: i64
    y: i64

impl Vec2:
    # Implement + operator
    fn __add__(self: Vec2, other: Vec2) -> Vec2:
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    
    # Implement - operator
    fn __sub__(self: Vec2, other: Vec2) -> Vec2:
        Vec2 { x: self.x - other.x, y: self.y - other.y }
    
    # Implement * operator (scalar multiplication)
    fn __mul__(self: Vec2, scalar: i64) -> Vec2:
        Vec2 { x: self.x * scalar, y: self.y * scalar }
    
    # Implement == operator
    fn __eq__(self: Vec2, other: Vec2) -> bool:
        self.x == other.x and self.y == other.y
    
    # Implement < operator (comparison by magnitude)
    fn __lt__(self: Vec2, other: Vec2) -> bool:
        self.x * self.x + self.y * self.y < other.x * other.x + other.y * other.y

fn main() -> i64:
    let a: Vec2 = Vec2 { x: 1, y: 2 }
    let b: Vec2 = Vec2 { x: 3, y: 4 }
    
    let c: Vec2 = a + b  # Uses __add__
    print(c.x)  # 4
    print(c.y)  # 6
    
    let d: Vec2 = b - a  # Uses __sub__
    print(d.x)  # 2
    
    let e: Vec2 = a * 3  # Uses __mul__
    print(e.x)  # 3
    
    # Chained operators work
    let f: Vec2 = a + b + c
    print(f.x)  # 8
    
    # Comparison operators
    if a == a:
        print(1)  # true
    
    if a < b:  # Compare by magnitude
        print(2)  # true
    
    0
```

Supported operator methods:
- `__add__` for `+`
- `__sub__` for `-`
- `__mul__` for `*`
- `__div__` for `/`
- `__mod__` for `%`
- `__eq__` for `==`
- `__ne__` for `!=`
- `__lt__` for `<`
- `__gt__` for `>`
- `__le__` for `<=`
- `__ge__` for `>=`

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

**Rust-style Generics with Monomorphization:**
```
# Generic struct with type parameter
struct Box<T>:
    value: T

# Generic impl block - methods on generic structs
impl Box<T>:
    fn get(self: Box<T>) -> T:
        self.value

# Generic function with type inference
fn identity<T>(x: T) -> T:
    x

fn make_box<T>(val: T) -> Box<T>:
    Box<T> { value: val }

# Generic enum (like Rust's Option)
enum Option<T>:
    Some(T)
    None

fn main() -> i64:
    # Generic struct with methods
    let b: Box<i64> = Box<i64> { value: 42 }
    print(b.get())  # 42
    
    # Generic functions with type inference (types inferred from arguments)
    print(identity(100))  # T inferred as i64
    let boxed: Box<i64> = make_box(77)  # T inferred as i64
    print(boxed.get())  # 77
    
    # Explicit type args still work
    let x: i64 = identity<i64>(42)
    
    # Generic enum with pattern matching
    let opt: Option<i64> = Option<i64>::Some(50)
    match opt:
        Option<i64>::Some(val):
            print(val)  # 50
        Option<i64>::None:
            print(0)
    
    return 0
```

**Supported:** operator overloading (`__add__`, `__eq__`, etc.), generics (structs, enums, functions, impl blocks) with type inference, type aliases, `const` declarations, hashmaps with dict comprehensions (`dict[K,V]`), list comprehensions, slice syntax (`v[1:3]`, `v[::2]`, negative indices), chained comparisons (`0 < x < 10`), pattern guards (`Some(x) if x > 0:`), `Result[T, E]` type with `?` operator, functional iterators (`map`, `filter`, `fold`, `skip`, `take`, `sum`), implicit return, ownership & borrowing, enums with `match`, keyword args, structs with `impl`, tuples, arrays, vectors, closures.


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


## Todo

- Traits (rust)
- Default Parameter Values (Python)
- String Interpolation / F-strings (Python)
- Or-Patterns in Match (Rust)
- if let / while let (Rust)
- Tuple Unpacking / Destructuring (Python/Rust)
- Struct Update Syntax (Rust)
- Wildcard Patterns (Rust)
- Async/Await (Rust/Python)

## License

MIT
