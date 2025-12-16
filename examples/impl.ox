# Example: impl blocks (methods on structs)

struct Point:
    x: i64
    y: i64

impl Point:
    fn sum(self) -> i64:
        return self.x + self.y

    fn distance_squared(self) -> i64:
        return self.x * self.x + self.y * self.y

struct Counter:
    value: i64

impl Counter:
    fn get(self) -> i64:
        return self.value

    fn add(self, n: i64) -> i64:
        return self.value + n

fn main() -> i64:
    let p: Point = Point { x: 3, y: 4 }
    print(p.sum())              # 7
    print(p.distance_squared()) # 25

    let c: Counter = Counter { value: 100 }
    print(c.get())              # 100
    print(c.add(42))            # 142

    return 0

