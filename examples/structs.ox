# Struct example: 2D Point

struct Point:
    x: i64
    y: i64

struct Rectangle:
    x: i64
    y: i64
    width: i64
    height: i64

fn area(w: i64, h: i64) -> i64:
    return w * h

fn main() -> i64:
    # Create a point
    let p: Point = Point { x: 10, y: 20 }
    print(p.x)
    print(p.y)

    # Modify fields
    p.x = 100
    p.y = 200
    print(p.x + p.y)

    # Create a rectangle
    let rect: Rectangle = Rectangle { x: 0, y: 0, width: 50, height: 30 }
    let a: i64 = area(rect.width, rect.height)
    print(a)

    return 0

