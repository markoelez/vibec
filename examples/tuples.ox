# Tuple example

fn main() -> i64:
    # Simple tuple
    let point: (i64, i64) = (10, 20)
    print(point.0)
    print(point.1)

    # Mixed type tuple
    let data: (i64, bool, i64) = (42, true, 100)
    print(data.0)
    print(data.2)

    # Use tuple elements in expressions
    let sum: i64 = point.0 + point.1
    print(sum)

    # Conditional based on tuple element
    if data.1:
        print(1)
    else:
        print(0)

    return 0

