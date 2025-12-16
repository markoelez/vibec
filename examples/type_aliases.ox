# Type Aliases Example
# Rust-style type aliases for creating type synonyms

# Simple type aliases
type Integer = i64
type Flag = bool

# Generic type aliases
type IntVec = vec[i64]
type Coordinates = (i64, i64, i64)
type IntPair = (i64, i64)

# Complex type aliases
type IntMap = dict[i64, i64]
type MaybeError = Result[i64, i64]

# Type alias for structs
struct Point:
    x: i64
    y: i64

type Vec2D = Point

# Chained aliases (alias of alias)
type MyInteger = Integer

fn add(a: Integer, b: Integer) -> Integer:
    a + b

fn main() -> i64:
    # Using simple type aliases
    let x: Integer = 10
    let y: Integer = 20
    let sum: Integer = add(x, y)
    print(sum)  # 30
    
    # Using collection type aliases
    let numbers: IntVec = []
    numbers.push(1)
    numbers.push(2)
    numbers.push(3)
    print(numbers.sum())  # 6
    
    # Using tuple type aliases
    let point: Coordinates = (10, 20, 30)
    print(point.0 + point.1 + point.2)  # 60
    
    let pair: IntPair = (5, 7)
    print(pair.0 + pair.1)  # 12
    
    # Using dict type aliases
    let scores: IntMap = {}
    scores[1] = 100
    scores[2] = 85
    print(scores[1])  # 100
    print(scores.len())  # 2
    
    # Using struct type alias
    let p: Vec2D = Point { x: 3, y: 4 }
    print(p.x + p.y)  # 7
    
    # Using chained alias
    let z: MyInteger = 42
    print(z)  # 42
    
    return 0

