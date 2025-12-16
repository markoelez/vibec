# Generics Example
# Demonstrates Rust-style generic types with monomorphization

# =============================================================================
# Generic Structs
# =============================================================================

# Generic struct with one type parameter
struct Box<T>:
    value: T

# Generic struct with two type parameters
struct Pair<T, U>:
    first: T
    second: U

# =============================================================================
# Generic Enums
# =============================================================================

# Generic enum (like Rust's Option)
enum Option<T>:
    Some(T)
    None

# =============================================================================
# Generic Impl Blocks (Methods on Generic Structs)
# =============================================================================

impl Box<T>:
    fn get(self: Box<T>) -> T:
        self.value

    fn set(self: Box<T>, new_val: T) -> T:
        # Note: this returns the new value since we can't mutate self
        new_val

impl Pair<T, U>:
    fn get_first(self: Pair<T, U>) -> T:
        self.first

    fn get_second(self: Pair<T, U>) -> U:
        self.second

    fn swap(self: Pair<T, U>) -> Pair<U, T>:
        Pair<U, T> { first: self.second, second: self.first }

# =============================================================================
# Generic Functions
# =============================================================================

# Identity function - returns what it receives
fn identity<T>(x: T) -> T:
    x

# Return the first of two values
fn first<T, U>(a: T, b: U) -> T:
    a

# Return the second of two values
fn second<T, U>(a: T, b: U) -> U:
    b

# Wrap a value in a Box
fn make_box<T>(val: T) -> Box<T>:
    Box<T> { value: val }

# =============================================================================
# Main - Demonstration
# =============================================================================

fn main() -> i64:
    print(111)  # Separator

    # === Generic Structs ===
    let int_box: Box<i64> = Box<i64> { value: 42 }
    print(int_box.value)  # Output: 42

    let bool_box: Box<bool> = Box<bool> { value: true }
    if bool_box.value:
        print(1)  # Output: 1

    let pair: Pair<i64, i64> = Pair<i64, i64> { first: 10, second: 20 }
    print(pair.first + pair.second)  # Output: 30

    print(222)  # Separator

    # === Generic Methods (Impl Blocks) ===
    let b: Box<i64> = Box<i64> { value: 100 }
    print(b.get())  # Output: 100

    let p: Pair<i64, i64> = Pair<i64, i64> { first: 5, second: 15 }
    print(p.get_first())  # Output: 5
    print(p.get_second())  # Output: 15

    print(333)  # Separator

    # === Generic Functions with Type Inference ===
    # Type arguments are automatically inferred from the actual arguments
    print(identity(77))  # T inferred as i64, Output: 77

    let x: i64 = first(99, false)  # T=i64, U=bool inferred
    print(x)  # Output: 99

    let y: bool = second(0, true)  # T=i64, U=bool inferred
    if y:
        print(1)  # Output: 1

    let boxed: Box<i64> = make_box(123)  # T inferred as i64
    print(boxed.get())  # Output: 123

    # Explicit type args still work if you prefer
    print(identity<i64>(42))  # Output: 42

    print(444)  # Separator

    # === Generic Enums with Pattern Matching ===
    let opt: Option<i64> = Option<i64>::Some(50)
    match opt:
        Option<i64>::Some(val):
            print(val)  # Output: 50
        Option<i64>::None:
            print(0)

    let none_opt: Option<i64> = Option<i64>::None
    match none_opt:
        Option<i64>::Some(val):
            print(val)
        Option<i64>::None:
            print(88)  # Output: 88

    print(555)  # Separator

    return 0
