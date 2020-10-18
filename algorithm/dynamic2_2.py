t = int(input())
for _ in range(t):
    k = int(input())
    case = [int(value) for value in input().split(' ')]

    temp = [[0] * k for i in range(0, k)]

    for i in range(0, k - 1):
        temp[i][i + 1] = case[i] + case[i + 1]
        for j in range(i + 2, k):
            temp[i][j] = temp[i][j - 1] + case[j]

    for i in range(2, k):
        for j in range(k - i):
            x = i + j
            minimum = [temp[j][y] + temp[y + 1][x] for y in range(j, x)]
            temp[j][x] += min(minimum)

    for value in temp:
        print(value)

    print(sum(case))
    print(sum(case) + temp[0][-1])