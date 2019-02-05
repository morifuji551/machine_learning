import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (5,5))
ax = plt.gca()

plt.plot([1,1], [0,1], color = 'red', linewidth = 2)
plt.plot([1,2], [2,2], color = 'red', linewidth = 2)
plt.plot([2,2], [2,1], color = 'red', linewidth = 2)
plt.plot([2,3], [1,1], color = 'red', linewidth = 2)

plt.xlim(0,3)
plt.ylim(0,3)
#plt.show()

theta_0 = np.array([
#↑, →, ↓, ←
[np.nan, 1, 1, np.nan],
[np.nan, 1, np.nan, 1],
[np.nan, np.nan, 1, 1],
[1, 1, 1, np.nan],
[np.nan, np.nan, 1, 1],
[1, np.nan, np.nan, np.nan],
[1, np.nan, np.nan, np.nan],
[1, 1, np.nan, np.nan]
])

def softmax_convert_into_pi_from_theta(theta):
    beta = 1.0
    [m,n] = theta.shape
    pi = np.zeros((m,n))
    exp_theta = np.exp(beta * theta)
    for i in range(m):
        pi[i] = exp_theta[i] / np.nansum(exp_theta[i])
    pi = np.nan_to_num(pi)
    return pi

pi_0 = softmax_convert_into_pi_from_theta(theta_0)

def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    next_direction = np.random.choice(direction, p = pi[s])

    if next_direction == "up":
        action = 0
        s_next = s - 3
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "down":
        action = 2
        s_next = s + 3
    else: #next_direction == "left"
        action = 3
        s_next = s - 1

    return action, s_next

def goal_maze_ret_s_a(pi):
    s = 0
    s_a_history = [[0, np.nan]]

    while(1):
        action, next_s = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        s_a_history.append([next_s, np.nan])
        if next_s == 8:
            break
        else:
            s = next_s

    return s_a_history

def update_theta(theta, pi, s_a_history):
    eta = 0.1
    T = len(s_a_history) - 1
    [m, n] = theta.shape
    delta_theta = theta.copy()

    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i,j])):
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                SA_ij = [SA for SA in s_a_history if SA ==[i, j]]

                N_i = len(SA_i)
                N_ij = len(SA_ij)
                delta_theta[i, j] = (N_ij - pi[i,j] * N_i) / T

    new_theta = theta + eta * delta_theta

    return new_theta

stop_epsilon = 1e-06

theta = theta_0
pi = pi_0

is_continue = True
count = 1

while is_continue:
    s_a_history = goal_maze_ret_s_a(pi)
    new_theta = update_theta(theta, pi, s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)

    print("迷路を解くのにかかった回数は", len(s_a_history)-1)

    if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
        is_continue = False
        print(pi)
    else:
        theta = new_theta
        pi = new_pi
