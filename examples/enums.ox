# Enums example - Rust-style enums with pattern matching

enum Option:
    Some(i64)
    None

enum Result:
    Ok(i64)
    Err(i64)

fn unwrap_or(opt: Option, default: i64) -> i64:
    match opt:
        Option::Some(val):
            return val
        Option::None:
            return default

fn is_ok(res: Result) -> bool:
    match res:
        Result::Ok(val):
            return true
        Result::Err(code):
            return false

fn main() -> i64:
    # Create enum values
    let x: Option = Option::Some(42)
    let y: Option = Option::None

    # Pattern matching
    match x:
        Option::Some(val):
            print(val)
        Option::None:
            print(0)

    # Use helper functions
    let a: i64 = unwrap_or(x, 0)
    let b: i64 = unwrap_or(y, 99)
    print(a)
    print(b)

    # Result enum
    let success: Result = Result::Ok(100)
    let failure: Result = Result::Err(404)

    if is_ok(success):
        print(1)

    if not is_ok(failure):
        print(2)

    return 0

