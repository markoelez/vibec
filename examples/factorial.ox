# Factorial example - demonstrates implicit return (Rust-style)

fn factorial(n: i64) -> i64:
    if n <= 1:
        return 1
    n * factorial(n - 1)  # Implicit return - no 'return' keyword needed

fn main() -> i64:
    print(factorial(5))  # 120
    print(factorial(10)) # 3628800
    0  # Implicit return for main too
