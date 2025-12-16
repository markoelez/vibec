# Operator Overloading Example
# Demonstrates how to implement custom operators for user-defined types

# A 2D vector struct
struct Vec2:
    x: i64
    y: i64

impl Vec2:
    # Addition: vec1 + vec2
    fn __add__(self: Vec2, other: Vec2) -> Vec2:
        Vec2 { x: self.x + other.x, y: self.y + other.y }

    # Subtraction: vec1 - vec2
    fn __sub__(self: Vec2, other: Vec2) -> Vec2:
        Vec2 { x: self.x - other.x, y: self.y - other.y }

    # Scalar multiplication: vec * scalar
    fn __mul__(self: Vec2, scalar: i64) -> Vec2:
        Vec2 { x: self.x * scalar, y: self.y * scalar }

    # Equality comparison: vec1 == vec2
    fn __eq__(self: Vec2, other: Vec2) -> bool:
        self.x == other.x and self.y == other.y

    # Inequality comparison: vec1 != vec2
    fn __ne__(self: Vec2, other: Vec2) -> bool:
        self.x != other.x or self.y != other.y

    # Less than (compares magnitude squared): vec1 < vec2
    fn __lt__(self: Vec2, other: Vec2) -> bool:
        self.x * self.x + self.y * self.y < other.x * other.x + other.y * other.y

# A wrapper for integers with custom division
struct SafeInt:
    value: i64

impl SafeInt:
    # Division that returns 0 for division by zero
    fn __div__(self: SafeInt, other: SafeInt) -> SafeInt:
        if other.value == 0:
            SafeInt { value: 0 }
        else:
            SafeInt { value: self.value / other.value }

    # Modulo that returns 0 for mod by zero
    fn __mod__(self: SafeInt, other: SafeInt) -> SafeInt:
        if other.value == 0:
            SafeInt { value: 0 }
        else:
            SafeInt { value: self.value % other.value }

fn main() -> i64:
    # === Vec2 Addition ===
    let a: Vec2 = Vec2 { x: 1, y: 2 }
    let b: Vec2 = Vec2 { x: 3, y: 4 }
    let c: Vec2 = a + b  # Uses __add__
    print(c.x)  # Output: 4
    print(c.y)  # Output: 6

    # === Vec2 Subtraction ===
    let d: Vec2 = b - a  # Uses __sub__
    print(d.x)  # Output: 2
    print(d.y)  # Output: 2

    # === Scalar Multiplication ===
    let e: Vec2 = a * 3  # Uses __mul__
    print(e.x)  # Output: 3
    print(e.y)  # Output: 6

    # === Chained Operations ===
    let f: Vec2 = a + b + c  # Chained __add__
    print(f.x)  # Output: 8 (1 + 3 + 4)
    print(f.y)  # Output: 12 (2 + 4 + 6)

    # === Equality Comparison ===
    let g: Vec2 = Vec2 { x: 1, y: 2 }
    if a == g:  # Uses __eq__
        print(1)  # Output: 1 (true)
    else:
        print(0)

    if a != b:  # Uses __ne__
        print(1)  # Output: 1 (true)
    else:
        print(0)

    # === Less Than Comparison ===
    # Compares by magnitude squared: (1^2 + 2^2 = 5) < (3^2 + 4^2 = 25)
    if a < b:  # Uses __lt__
        print(1)  # Output: 1 (true)
    else:
        print(0)

    # === Safe Division ===
    let n1: SafeInt = SafeInt { value: 10 }
    let n2: SafeInt = SafeInt { value: 3 }
    let n3: SafeInt = SafeInt { value: 0 }

    let result1: SafeInt = n1 / n2  # Uses __div__
    print(result1.value)  # Output: 3

    let result2: SafeInt = n1 / n3  # Division by zero returns 0
    print(result2.value)  # Output: 0

    # === Safe Modulo ===
    let result3: SafeInt = n1 % n2  # Uses __mod__
    print(result3.value)  # Output: 1

    let result4: SafeInt = n1 % n3  # Mod by zero returns 0
    print(result4.value)  # Output: 0

    0

