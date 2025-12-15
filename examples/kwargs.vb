# Demonstrate Python-style keyword arguments

# Basic function with multiple parameters
fn greet(count: i64, value: i64) -> i64:
    return count * value

# Function showing argument order matters
fn subtract(a: i64, b: i64) -> i64:
    return a - b

# Function with many parameters
fn combine(first: i64, second: i64, third: i64, fourth: i64) -> i64:
    return first * 1000 + second * 100 + third * 10 + fourth

fn main() -> i64:
    # === Basic Keyword Arguments ===
    
    # Call with keyword arguments (same order as parameters)
    let result1: i64 = greet(count=3, value=7)
    print(result1)  # 21
    
    # Call with keyword arguments (different order)
    let result2: i64 = greet(value=7, count=3)
    print(result2)  # 21 - same result, order doesn't matter
    
    # === Keyword Args Show Intent ===
    
    # Without kwargs: which is a, which is b?
    let diff1: i64 = subtract(10, 3)
    print(diff1)  # 7
    
    # With kwargs: crystal clear!
    let diff2: i64 = subtract(a=10, b=3)
    print(diff2)  # 7
    
    # Reverse order - kwargs make this obvious
    let diff3: i64 = subtract(b=3, a=10)
    print(diff3)  # 7 (not -7!)
    
    # === Mixed Positional and Keyword ===
    
    # First arg positional, rest keyword (in any order)
    let mixed: i64 = combine(1, fourth=4, second=2, third=3)
    print(mixed)  # 1234
    
    # All keyword args, completely reordered
    let reordered: i64 = combine(fourth=4, second=2, first=1, third=3)
    print(reordered)  # 1234
    
    # === Keyword Args with Expressions ===
    
    let x: i64 = 5
    let computed: i64 = greet(count=x + 1, value=x * 2)
    print(computed)  # (5+1) * (5*2) = 6 * 10 = 60
    
    # === Builtin print with kwarg ===
    print(value=99)  # 99
    
    return 0

