# Demonstrate closures / lambdas

fn main() -> i64:
    # Basic closure with two parameters
    let add: Fn(i64, i64) -> i64 = |a: i64, b: i64| -> i64: a + b
    print(add(10, 20))  # 30
    
    # Closure with no parameters
    let get_answer: Fn() -> i64 = || -> i64: 42
    print(get_answer())  # 42
    
    # Closure with single parameter
    let double: Fn(i64) -> i64 = |x: i64| -> i64: x * 2
    print(double(21))  # 42
    
    # Closure with complex arithmetic
    let calc: Fn(i64, i64, i64) -> i64 = |a: i64, b: i64, c: i64| -> i64: a * b + c
    print(calc(5, 8, 2))  # 42
    
    # Multiple calls to same closure
    let add_ten: Fn(i64) -> i64 = |x: i64| -> i64: x + 10
    print(add_ten(5))   # 15
    print(add_ten(90))  # 100
    
    # Closure returning bool
    let is_even: Fn(i64) -> bool = |x: i64| -> bool: x % 2 == 0
    if is_even(4):
        print(1)  # 1 (true)
    
    if is_even(7):
        print(0)
    else:
        print(2)  # 2 (odd)
    
    return 0

