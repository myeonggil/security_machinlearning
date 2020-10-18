# 문제가 개편되었습니다. 이로 인해 함수 구성이나 테스트케이스가 변경되어, 과거의 코드는 동작하지 않을 수 있습니다.
# 새로운 함수 구성을 적용하려면 [코드 초기화] 버튼을 누르세요. 단, [코드 초기화] 버튼을 누르면 작성 중인 코드는 사라집니다.
count = 0
minCount = 9
results = []


def BFS(N, number):
    global count
    global minCount
    global results
    print(results)
    if count >= minCount: return
    if len(results) >= minCount: return
    if len(results) == 0:
        lastNumber = 0
    else:
        lastNumber = results[-1]

    if lastNumber == number:
        minCount = min(count, minCount)
        return minCount

    n = 0
    addCount = 0

    c = 1
    while c <= 10000000:
        addCount += 1
        if count + addCount >= minCount:
            c *= 10
            continue

        n += (N * c)

        count += addCount

        results.append(lastNumber + n)
        BFS(N, number)
        results.pop(-1)

        if lastNumber - n != 0:
            results.append(lastNumber - n)
            BFS(N, number)
            results.pop(-1)

        results.append(lastNumber * n)
        BFS(N, number)
        results.pop(-1)

        if int(lastNumber / n) != 0:
            results.append(int(lastNumber / n))
            BFS(N, number)
            results.pop(-1)

        count -= addCount

        c *= 10

    return


def solution(N, number):
    answer = -1

    BFS(N, number)
    if minCount <= 8:
        answer = minCount
    print(answer)
    return answer

solution(5, 12)