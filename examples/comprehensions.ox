# List Comprehension Example
# Python-style list comprehensions for concise vector creation

fn main() -> i64:
    # Basic list comprehension: squares of 0-9
    let squares: vec[i64] = [x * x for x in range(0, 10)]
    print(squares.len())  # 10
    print(squares.sum())  # 285 (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81)

    # With filter: only even numbers
    let evens: vec[i64] = [x for x in range(0, 20) if x % 2 == 0]
    print(evens.len())  # 10 (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)
    print(evens.sum())  # 90

    # Complex expression: doubled and incremented
    let transformed: vec[i64] = [x * 2 + 1 for x in range(0, 5)]
    print(transformed[0])  # 1
    print(transformed[1])  # 3
    print(transformed[2])  # 5
    print(transformed[3])  # 7
    print(transformed[4])  # 9

    # With condition: multiples of 3
    let multiples_of_3: vec[i64] = [x for x in range(0, 30) if x % 3 == 0]
    print(multiples_of_3.len())  # 10 (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)

    # Combining with method chaining
    let squares_to_5: vec[i64] = [x * x for x in range(1, 6)]
    print(squares_to_5.sum())  # 55 (1 + 4 + 9 + 16 + 25)

    # Filter squares greater than 10
    let big_squares: vec[i64] = [x * x for x in range(0, 10) if x * x > 10]
    print(big_squares.len())  # 6 (16, 25, 36, 49, 64, 81)
    print(big_squares[0])  # 16

    return 0

