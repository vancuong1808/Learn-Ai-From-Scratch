def levenshtein_distance(source, target):
    token1 = len(source)
    token2 = len(target)

    distances = [[0] * (token2 + 1) for _ in range(token1 + 1)]

    # 
    for i in range(token2 + 1):
        distances[0][i] = i
    #
    for i in range(token1 + 1):
        distances[i][0] = i

    delcost = 0
    inscost = 0
    subcost = 0

    for i in range(1, token1 + 1):
        for j in range(1, token2 + 1):
            if (source[i - 1] == target[j - 1]):
                distances[i][j] = distances[i - 1][j - 1]
            else:
                delcost = distances[i][j - 1]
                inscost = distances[i - 1][j]
                subcost = distances[i - 1][j - 1]

                if (delcost <= inscost and delcost <= subcost):
                    distances[i][j] = delcost + 1
                elif (inscost <= subcost and inscost <= delcost):
                    distances[i][j] = inscost + 1
                else:
                    distances[i][j] = subcost + 1

    for i in range(token1+1):
        print(distances[i])
    return distances[token1][token2]


if __name__ == "__main__":
    print(levenshtein_distance("hola", "hello"))

