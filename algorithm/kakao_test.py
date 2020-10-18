solution = "100-200*300-500+20"
# solution = "50*6-3*2"
# solution = "100-200*300-500+20+100-200*300-500+20+100-200*300-500+20+100-200*300-500+20+100-200*300-500+20+100-200*300-500+20+100-200*300-500+20+100-200*300-500+20"

results = []

def recursive(temp, results):
    if temp.count(',') == 2:
        results.append(temp)
        return

    if '+' not in temp:
        value = temp
        value += ', +'
        recursive(value, results)
    if '-' not in temp:
        value = temp
        value += ', -'
        recursive(value, results)
    if '*' not in temp:
        value = temp
        value += ', *'
        recursive(value, results)


operations = ['+', '-', '*']
for operator in operations:
    recursive(operator, results)

response = []
for result in results:
    value = result.split(', ')
    response.append(value)

answer = []
for res in response:
    temp = solution
    for i in range(0, len(res)):
        first = ''
        second = ''
        count = 0
        j = 0
        first_switch = 1
        second_switch = 1
        replaces = []
        eval_results = []
        while j < len(temp):
            if j == 0 and len(first) == 0:
                if temp[j] == '-':
                    j += 1
                    first_switch = -1
                    continue
            if count == 0:
                if temp[j] not in operations:
                    first += temp[j]
                elif temp[j] == res[i]:
                    count += 1
                else:
                    first = ''
                    count = 0
                    first_switch = 1
            else:
                if temp[j - 1] in operations and len(second) == 0:
                    if temp[j] == '-':
                        j += 1
                        second_switch = -1
                        continue
                if temp[j] not in operations:
                    second += temp[j]

                if temp[j] in operations or j == len(temp) - 1:
                    if first == '' or second == '':
                        first = ''
                        second = ''
                        first_switch = 1
                        second_switch = 1
                        count = 0
                        j += 1
                        continue
                    eval_result = ''
                    if res[i] == '+':
                        eval_result = str(int(first) * first_switch + int(second) * second_switch)
                    elif res[i] == '-':
                        eval_result = str(int(first) * first_switch - int(second) * second_switch)
                    elif res[i] == '*':
                        eval_result = str((int(first) * first_switch) * (int(second) * second_switch))
                    else:
                        eval_result = str((int(first) * first_switch) / (int(second) * second_switch))
                    replace = str(int(first) * first_switch) + res[i] + str(int(second) * second_switch)
                    replaces.append(replace)
                    eval_results.append(eval_result)
                    # temp = temp.replace(replace, eval_result)
                    # 초기화
                    count = 0
                    second = ''
                    first = ''
                    first_switch = 1
                    second_switch = 1
            j += 1

        for j in range(0, len(replaces)):
            temp = temp.replace(replaces[j], eval_results[j])

    temp = abs(int(eval(temp)))
    answer.append(temp)

print(answer)
print(max(answer))