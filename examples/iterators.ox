# Functional programming with Rust-style iterator methods

fn main() -> i64:
    # Create a vector of numbers 0-9
    let numbers: vec[i64] = []
    for i in range(0, 10):
        numbers.push(i)
    
    # Method chaining: skip first 2, take next 5, double each, keep > 5
    # Input:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # skip(2): [2, 3, 4, 5, 6, 7, 8, 9]
    # take(5): [2, 3, 4, 5, 6]
    # map(*2): [4, 6, 8, 10, 12]
    # filter(>5): [6, 8, 10, 12]
    let result: vec[i64] = numbers.into_iter().skip(2).take(5).map(|x: i64| -> i64: x * 2).filter(|x: i64| -> bool: x > 5).collect()
    
    print(result.sum())  # 6 + 8 + 10 + 12 = 36
    
    # Using fold to compute factorial of 5
    let nums: vec[i64] = []
    nums.push(1)
    nums.push(2)
    nums.push(3)
    nums.push(4)
    nums.push(5)
    let factorial: i64 = nums.fold(1, |acc: i64, x: i64| -> i64: acc * x)
    print(factorial)  # 120
    
    # Filter even numbers and sum them
    let all: vec[i64] = []
    for i in range(1, 11):
        all.push(i)
    let even_sum: i64 = all.filter(|x: i64| -> bool: x % 2 == 0).sum()
    print(even_sum)  # 2 + 4 + 6 + 8 + 10 = 30
    
    # Map to squares
    let small: vec[i64] = []
    small.push(1)
    small.push(2)
    small.push(3)
    small.push(4)
    let squares: vec[i64] = small.map(|x: i64| -> i64: x * x)
    print(squares[0])  # 1
    print(squares[1])  # 4
    print(squares[2])  # 9
    print(squares[3])  # 16
    
    0
