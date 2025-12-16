# Slice Syntax Example
# Python-style slicing for arrays and vectors

fn main() -> i64:
    # Create a vector using list comprehension
    let numbers: vec[i64] = [x * 10 for x in range(0, 10)]
    # numbers = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    
    # Basic slice: [start:stop] - elements from index 2 to 5 (exclusive)
    let slice1: vec[i64] = numbers[2:5]
    print(slice1[0])  # 20
    print(slice1[1])  # 30
    print(slice1[2])  # 40
    
    # Slice from start: [:stop] - first 3 elements
    let first_three: vec[i64] = numbers[:3]
    print(first_three.len())  # 3
    print(first_three[0])     # 0
    
    # Slice to end: [start:] - elements from index 7 onwards
    let last_three: vec[i64] = numbers[7:]
    print(last_three.len())  # 3
    print(last_three[0])     # 70
    
    # Full copy: [:] - copy entire vector
    let copy: vec[i64] = numbers[:]
    print(copy.len())  # 10
    
    # Slice with step: [::step] - every 2nd element
    let evens: vec[i64] = numbers[::2]
    print(evens.len())  # 5
    print(evens[0])     # 0
    print(evens[1])     # 20
    print(evens[2])     # 40
    
    # Slice with start, stop, and step: [start:stop:step]
    let middle_evens: vec[i64] = numbers[2:8:2]
    print(middle_evens.len())  # 3
    print(middle_evens[0])     # 20
    print(middle_evens[1])     # 40
    print(middle_evens[2])     # 60
    
    # Negative start index: [-2:] - last 2 elements
    let last_two: vec[i64] = numbers[-2:]
    print(last_two[0])  # 80
    print(last_two[1])  # 90
    
    # Negative stop index: [:-2] - all except last 2
    let without_last_two: vec[i64] = numbers[:-2]
    print(without_last_two.len())  # 8
    
    # Array slicing (returns a vec)
    let arr: [i64; 5] = [100, 200, 300, 400, 500]
    let arr_slice: vec[i64] = arr[1:4]
    print(arr_slice[0])  # 200
    print(arr_slice[1])  # 300
    print(arr_slice[2])  # 400
    
    # Empty slice when start >= stop
    let empty: vec[i64] = numbers[5:3]
    print(empty.len())  # 0
    
    # Chaining with other operations
    let sum: i64 = numbers[0:5].sum()
    print(sum)  # 0 + 10 + 20 + 30 + 40 = 100
    
    return 0


