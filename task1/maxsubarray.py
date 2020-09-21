def findMaxSubArray(A):
    size = len(A)
    max_so_far = -1 * 10 ** 200
    max_sum, start, end, s = 0, 0, 0, 0

    for i in range(0, size):
        max_sum += A[i]
        if max_so_far < max_sum:
            max_so_far = max_sum
            start = s
            end = i

        if max_sum < 0:
            max_sum = 0
            s = i + 1

    return A[start:end + 1]


my_list = [int(el) for el in input().split()]
print(findMaxSubArray(my_list))
