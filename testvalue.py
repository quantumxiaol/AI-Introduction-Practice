# 参数
REWARD = -1 # constant reward for non-terminal states
DISCOUNT = 1
MAX_ERROR = 10**(-3)


NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right
NUM_ROW = 3
NUM_COL = 3
U = [   [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0]]


def printEnvironment(arr, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            # if r == c == 1:
            #     val = "WALL"

            if r == 2 and c == 1:
                val = "+1"

            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# 获得通过从给定状态执行给定动作所达到的状态的效用
def getU(U, r, c, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc

    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL : # boundary or the wallor (newR == newC == 1)
        return U[r][c]
    else:
        return U[newR][newC]

# 计算给定动作状态的效用
def calculateU(U, r, c, action):
    u = REWARD
    u += 0.1 * DISCOUNT * getU(U, r, c, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, (action+1)%4)
    return u

def valueIteration(U):
    print("值迭代:\n")
    while True:
        nextU =[[0, 0, 0],
                [0, 0, 0],
                [0, 1, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                # 到达食物
                if (r == 2 and c == 1) :#or (r == c == 1)
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        printEnvironment(U)
        if error < MAX_ERROR :#* (1-DISCOUNT) / DISCOUNT:
            break
    return U

# 从U得到最优策略
def getOptimalPolicy(U):
    policy = [[-1, -1, -1, -1] for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if (r <= 1 and c == 3) or (r == c == 1):
                continue
            # 选择使效用最大化的行动
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy


print("初始值:\n")
printEnvironment(U)

# 值迭代
U = valueIteration(U)

# 从U中得到最优策略并打印出来
policy = getOptimalPolicy(U)
print("最优策略:\n")
printEnvironment(policy, True)