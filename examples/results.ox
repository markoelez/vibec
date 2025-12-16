# Result Type Example
# Demonstrates Rust-style error handling with Result, Ok, Err, and the ? operator

# A function that can fail - division by zero returns an error
fn divide(a: i64, b: i64) -> Result[i64, i64]:
    if b == 0:
        return Err(1)  # Error code 1 for division by zero
    Ok(a / b)

# A function that chains multiple operations using the ? operator
# If any step fails, the error propagates automatically
fn complex_calculation(x: i64, y: i64, z: i64) -> Result[i64, i64]:
    let step1: i64 = divide(x, y)?  # Will return early if y == 0
    let step2: i64 = divide(step1, z)?  # Will return early if z == 0
    Ok(step1 + step2)

# A function that validates input before processing
fn validate_and_double(n: i64) -> Result[i64, i64]:
    if n < 0:
        return Err(2)  # Error code 2 for negative input
    if n > 50:
        return Err(3)  # Error code 3 for input too large
    Ok(n * 2)

# Process a range of numbers, collecting successes
fn process_range(start: i64, end: i64) -> Result[i64, i64]:
    let sum: i64 = 0
    for i in range(start, end):
        let result: i64 = validate_and_double(i)?
        sum = sum + result
    Ok(sum)

fn main() -> i64:
    # Test successful division
    let res1: Result[i64, i64] = divide(10, 2)
    match res1:
        Ok(v):
            print(v)  # Prints: 5
        Err(e):
            print(e)

    # Test division by zero
    let res2: Result[i64, i64] = divide(10, 0)
    match res2:
        Ok(v):
            print(v)
        Err(e):
            print(e)  # Prints: 1 (error code)

    # Test the ? operator with successful chain
    let res3: Result[i64, i64] = complex_calculation(100, 5, 2)
    match res3:
        Ok(v):
            print(v)  # Prints: 30 (100/5=20, 20/2=10, 20+10=30)
        Err(e):
            print(e)

    # Test the ? operator with failure in middle
    let res4: Result[i64, i64] = complex_calculation(100, 0, 2)
    match res4:
        Ok(v):
            print(v)
        Err(e):
            print(e)  # Prints: 1 (division by zero error)

    # Test validation
    let res5: Result[i64, i64] = validate_and_double(25)
    match res5:
        Ok(v):
            print(v)  # Prints: 50
        Err(e):
            print(e)

    # Test validation failure (negative)
    let res6: Result[i64, i64] = validate_and_double(0 - 5)
    match res6:
        Ok(v):
            print(v)
        Err(e):
            print(e)  # Prints: 2 (negative input error)

    # Test processing a range
    let res7: Result[i64, i64] = process_range(0, 5)
    match res7:
        Ok(v):
            print(v)  # Prints: 20 (0*2 + 1*2 + 2*2 + 3*2 + 4*2 = 20)
        Err(e):
            print(e)

    return 0

