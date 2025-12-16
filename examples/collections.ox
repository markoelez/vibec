fn main() -> i64:
    # Fixed-size array (stack allocated)
    let arr: [i64; 5] = [10, 20, 30, 40, 50]
    print(arr[2])
    arr[2] = 99
    print(arr[2])
    print(arr.len())

    # Sum array elements
    let sum: i64 = 0
    for i in range(5):
        sum = sum + arr[i]
    print(sum)

    # Dynamic vector (heap allocated)
    let nums: vec[i64] = []
    nums.push(1)
    nums.push(2)
    nums.push(3)
    print(nums.len())
    print(nums[0])

    let last: i64 = nums.pop()
    print(last)
    print(nums.len())

    return 0

