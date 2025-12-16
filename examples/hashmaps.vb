# HashMap/Dict Example
# Python/Rust-style hashmaps with O(1) average lookup

fn main() -> i64:
    # Create empty dict
    let scores: dict[i64, i64] = {}
    
    # Insert using indexing
    scores[1] = 100
    scores[2] = 85
    scores[3] = 92
    print(scores.len())  # 3
    
    # Lookup values
    print(scores[1])  # 100
    print(scores[2])  # 85
    
    # Create dict with literal syntax
    let prices: dict[i64, i64] = {100: 50, 200: 75, 300: 100}
    print(prices[200])  # 75
    
    # Check if key exists
    if prices.contains(100):
        print(1)  # prints 1
    
    if prices.contains(999):
        print(999)
    else:
        print(0)  # prints 0
    
    # Update existing key
    prices[100] = 60
    print(prices[100])  # 60
    
    # Insert using method
    prices.insert(400, 125)
    print(prices.len())  # 4
    
    # Get using method
    print(prices.get(400))  # 125
    
    # Remove key
    let removed: bool = prices.remove(100)
    if removed:
        print(prices.len())  # 3
    
    # Dict with many entries (tests hash collision handling and growth)
    let big: dict[i64, i64] = {}
    for i in range(0, 20):
        big[i] = i * 10
    
    print(big.len())  # 20
    print(big[15])  # 150
    
    # Dict comprehension (Python-style)
    let squares: dict[i64, i64] = {x: x * x for x in range(0, 5)}
    print(squares[4])  # 16
    print(squares.len())  # 5
    
    # Dict comprehension with filter
    let even_cubes: dict[i64, i64] = {x: x * x * x for x in range(0, 10) if x % 2 == 0}
    print(even_cubes.len())  # 5 (0, 2, 4, 6, 8)
    print(even_cubes[4])  # 64
    
    return 0

