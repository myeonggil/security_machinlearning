def recursive(K, queue, visited):
    count = 0
    while queue:
        temp = queue.pop(0)
        e, v = temp[0], temp[1]
        count = v
        if not visited[e]:
            visited[e] = True
            if e == K:
                return count
            count += 1
            if e * 2 <= 100000:
                queue.append([e * 2, count])
            if e + 1 <= 100000:
                queue.append([e + 1, count])
            if e - 1 >= 0:
                queue.append([e - 1, count])
    return count

N, K = map(int, input().split(' '))
queue = [[N, 0]]
visited = [False for i in range(0, 100001)]
result = recursive(K, queue, visited)
print(result)