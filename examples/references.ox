# Demonstrate ownership, borrowing, and move mechanics

struct Point:
    x: i64
    y: i64

fn main() -> i64:
    # === REFERENCES ===
    
    # Shared reference (&T) - read-only borrow
    let x: i64 = 42
    let r: &i64 = &x
    print(*r)  # 42 - dereference to read
    
    # Multiple shared borrows allowed
    let r2: &i64 = &x
    print(*r2)  # 42 - can have many &T
    
    # Mutable reference (&mut T) - exclusive borrow
    let y: i64 = 10
    let mr: &mut i64 = &mut y
    *mr = 100   # Modify through reference
    print(y)    # 100
    
    # === COPY TYPES ===
    
    # i64 is Copy - values are cloned, not moved
    let a: i64 = 5
    let b: i64 = a   # a is COPIED to b
    print(a + b)     # 10 - both still valid!
    
    # Can use 'a' in a loop - Copy types don't move
    let sum: i64 = 0
    for i in range(0, 3):
        sum = sum + a  # 'a' is copied each iteration
    print(sum)  # 15
    
    # === MOVE SEMANTICS ===
    
    # Structs are NOT Copy - they MOVE on assignment
    let p1: Point = Point { x: 3, y: 4 }
    print(p1.x)  # 3
    
    let p2: Point = p1  # p1 is MOVED to p2
    print(p2.x)  # 3 - p2 now owns the data
    print(p2.y)  # 4
    
    # p1 is now invalid! Uncommenting would error:
    # print(p1.x)  # ERROR: Use of moved variable 'p1'
    
    # === BORROW CHECKER ERRORS (try uncommenting!) ===
    
    # 1. Double mutable borrow:
    # let z: i64 = 5
    # let m1: &mut i64 = &mut z
    # let m2: &mut i64 = &mut z  # ERROR: already borrowed as mutable
    
    # 2. Shared + mutable borrow conflict:
    # let w: i64 = 5
    # let sr: &i64 = &w
    # let mw: &mut i64 = &mut w  # ERROR: already borrowed as immutable
    
    # 3. Mutate while borrowed:
    # let v: i64 = 5
    # let rv: &i64 = &v
    # v = 10  # ERROR: cannot mutate 'v' while borrowed
    
    return 0
