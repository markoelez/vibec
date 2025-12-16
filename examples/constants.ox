# Constants Example
# Demonstrating const vs let variable declarations

struct Config:
    max_retries: i64
    timeout: i64

fn main() -> i64:
    # Constants - cannot be reassigned
    const MAX_VALUE: i64 = 100
    const SCALE_FACTOR: i64 = 10
    const DEBUG: bool = true
    
    # Mutable variables - can be reassigned
    let counter: i64 = 0
    let result: i64 = 0
    
    # Use constants in expressions
    result = MAX_VALUE * SCALE_FACTOR
    print(result)  # 1000
    
    # Reassign mutable variables
    counter = 1
    counter = counter + 1
    print(counter)  # 2
    
    # Constants work with all types
    const DEFAULT_CONFIG: Config = Config { max_retries: 3, timeout: 30 }
    print(DEFAULT_CONFIG.max_retries)  # 3
    print(DEFAULT_CONFIG.timeout)  # 30
    
    # Use const in conditionals
    if DEBUG:
        print(1)  # 1
    
    # Use const in loops
    let sum: i64 = 0
    for i in range(0, MAX_VALUE):
        sum = sum + 1
    print(sum)  # 100
    
    # Attempting to reassign const would cause compile error:
    # MAX_VALUE = 200  # Error: Cannot assign to const variable 'MAX_VALUE'
    # DEFAULT_CONFIG.max_retries = 5  # Error: Cannot assign to const variable 'DEFAULT_CONFIG'
    
    return 0

