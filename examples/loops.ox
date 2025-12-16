fn main() -> i64:
    # Sum 0..9 using for loop
    let sum: i64 = 0
    for i in range(10):
        sum = sum + i
    print(sum)

    # Range with start: 5..10
    let partial: i64 = 0
    for i in range(5, 10):
        partial = partial + i
    print(partial)

    # Nested loops: 3x3 grid
    for i in range(3):
        for j in range(3):
            print(i * 3 + j)

    return 0

