# 1



# 2
# def solution(ball, order):
#     answer = []
#     hold_state = []
#     while order or hold_state:
#         print(ball, order, hold_state)
#         if hold_state:
#             exist = False
#             for i in range(0, len(hold_state)):
#                 if hold_state[i] == ball[0]:
#                     hold_state.pop(i)
#                     answer.append(ball.pop(0))
#                     exist = True
#                     break
#                 elif hold_state[i] == ball[-1]:
#                     hold_state.pop(i)
#                     answer.append(ball.pop())
#                     exist = True
#                     break
#             if exist: continue
#         if order and ball:
#             if order[0] == ball[0]:
#                 order.pop(0)
#                 answer.append(ball.pop(0))
#             elif order[0] == ball[-1]:
#                 order.pop(0)
#                 answer.append(ball.pop())
#             else:
#                 hold_state.append(order.pop(0))
#     print(answer)
#     return answer
#
# solution([11, 2, 9, 13, 24], [9, 2, 13, 24, 11])


# 3
def solution(n):
    answer = []
    if n < 10:
        answer.append(0)
        answer.append(n)
    else:
        temp = str(n)
        count = 0
        while True:
            first = ''
            second = ''
            half = len(temp) // 2
            for i in range(0, len(temp) - 1):
                if temp[i] == '0' or temp[i + 1] == '0' or i < half:
                    first += temp[i]
                else:
                    second += temp[i:]
                    break
            if len(second) == 0: second += temp[len(first): ]
            definition = first + '+' + second
            temp = str(eval(definition))
            count += 1

            if len(temp) == 1: break

        answer.append(count)
        answer.append(int(temp))

    return answer

solution(10007)


# 4



# 5



# 6


