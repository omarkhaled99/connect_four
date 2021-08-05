import numpy as np
from Node import Node
import time
rows = int(6)
columns = int(7)

def get_positions(node):
    board_state = node.get_state()
    positions = []
    for row in range(rows):
        for column in range(columns):
            if board_state[row][column] == 0 and (row == 0 or board_state[row - 1][column] != 0):
                positions.append([row, column])
    return positions



def terminal_test(node):
    node_board = node.get_state()
    return True if np.count_nonzero(node_board) == 42 else False



def evaluate_state(node):
    window_size = 4
    AI_count = 0
    human_count = 0
    node_board = node.get_state()
    heuristics_matrix = np.array([[3, 4, 5, 7, 5, 4, 3],
                                   [4, 6, 8, 10, 8, 6, 4],
                                   [5, 8, 11, 13, 11, 8, 5],
                                   [5, 8, 11, 13, 11, 8, 5],
                                   [4, 6, 8, 10, 8, 6, 4],
                                   [3, 4, 5, 7, 5, 4, 3]])
    ###number of connected fours horizaontally
    for i in range(rows):
        for j in range(columns - 3):

            if np.sum(node_board[i][j:j + window_size]) == np.count_nonzero(node_board[i][j:j + window_size]):
                human_count += np.dot(heuristics_matrix[i][j:j + window_size], node_board[i][j:j + window_size].T)
                # break
            elif np.sum(node_board[i][j:j + window_size]) == -np.count_nonzero(node_board[i][j:j + window_size]):
                AI_count -= np.dot(heuristics_matrix[i][j:j + window_size], node_board[i][j:j + window_size].T)
    ### number of connected fours vertically
    for i in range(rows - 4):
        sum_array = np.sum(node_board[i:i + window_size], axis=0)
        count_array =  np.count_nonzero(node_board[i:i + window_size], axis=0)
        sum_heuristic = np.sum(heuristics_matrix[i:i + window_size], axis=0)
       # print(node_board[i:i + window_size].T[0])
        for j in range(sum_array.shape[0]):
            if sum_array[j] == count_array[j]:
                ## we used transpose to get the coloumn
                human_count += np.dot(node_board[i:i + window_size].T[j], heuristics_matrix[i:i + window_size].T[j].T)
                # break
            elif sum_array[j] == -count_array[j]:
                AI_count -= np.dot(node_board[i:i + window_size].T[j], heuristics_matrix[i:i + window_size].T[j].T)
                # break
    ##number of connected fours diagonally
    for r in range(rows - 3):
        for c in range(columns - 3):
            window = np.array([node_board[r + i][c + i] for i in range(window_size)])
            heuristic_window = np.array([heuristics_matrix[r + i][c + i] for i in range(window_size)])
            if np.sum(window) == np.count_nonzero(window):
                human_count += np.dot(window,heuristic_window.T)
                # break
            elif np.sum(window) == -np.count_nonzero(window):
                AI_count -= np.dot(window, heuristic_window.T)
                # break

    for r in range(rows - 3):
        for c in range(columns - 3):
            window = np.array([node_board[r + 3 - i][c + i] for i in range(window_size)])
            heuristic_window = np.array([heuristics_matrix[r + 3 - i][c + i] for i in range(window_size)])
            if np.sum(window) == np.count_nonzero(window):
                human_count += np.dot(window,heuristic_window.T)
                # break
            elif np.sum(window) == -np.count_nonzero(window):
                AI_count -= np.dot(window, heuristic_window.T)
                # break

    return AI_count - human_count
def window_score(window):
    score =0


    if np.count_nonzero(window) == np.sum(window):
        sum = np.sum(window)
        if sum==4:
            score += 100
        elif sum ==3:
            score +=20
        elif sum ==2:
            score +=10
        elif sum ==1:
            score +=5

    elif np.count_nonzero(window)== -np.sum(window):
        sum = np.sum(window)
        if sum == -4:
            score -= 100
        elif sum == -3:
            score -= 20
        elif sum == -2:
            score -= 10
        elif sum == -1:
            score -= 5

    if np.sum(window) == 2 and np.count_nonzero(window == 1) == 3:

        score += 20000
    return score
def evaluate_state_2(node):
    ###number of connected fours horizaontally
    window_size = 4
    score = 0
    ROW_COUNT = 6
    COLUMN_COUNT = 7
    node_board = node.get_state()
    for i in range(ROW_COUNT):
        for j in range(COLUMN_COUNT - 3):
            window = node_board[i][j:j + window_size]
            score+= window_score(window[:])
    ### number of connected fours vertically
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(node_board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + window_size]
            score+= window_score(window[:])
    # break
    ##number of connected fours diagonally
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = np.array([node_board[r + i][c + i] for i in range(window_size)])
            score+= window_score(window[:])
        # break
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = np.array([node_board[r + 3 - i][c + i] for i in range(window_size)])
            score += window_score(window[:])
        # break
    return score


def maximize(node,depth, alpha, beta):

    if terminal_test(node) or depth == 0:
        # print(evaluate_state(node))
        # print(node.get_state())
        return None, evaluate_state(node)
    max_child = None
    max_utility = float('-inf')


    for child in node.get_children(True):

        _, utility = minimize(child,depth - 1, alpha, beta)

        if utility > max_utility:
            max_child = child
            max_utility = utility
        if alpha is not None and beta is not None:
            if max_utility >= beta:
                break

            if max_utility > alpha:
                alpha = max_utility

    return max_child, max_utility


def minimize(node,depth, alpha, beta):

    if terminal_test(node) or depth == 0:
        # print(evaluate_state(node))
        # print(node.get_state())
        return None, evaluate_state(node)
    min_child = None
    min_utility = float('inf')

    for child in node.get_children(False):

        _, utility = maximize(child,depth-1, alpha, beta)

        if utility < min_utility:
            min_child = child
            min_utility = utility
        if alpha is not None and beta is not None:
            if min_utility <= alpha:
                break

            if min_utility < beta:
                beta = min_utility

    return min_child, min_utility


def decision(node,K,alpha_beta):
    start_time = time.time()
    depth =K
    if alpha_beta:
        print("using alpha beta")
        child, _ = maximize(node, depth,float('-inf'), float('inf'))
    else:
        child, _ = maximize(node, depth, None, None)
    print("--- %s seconds ---" % (time.time() - start_time))
    return child



visited = []
queue = []

def get_tree(child):
    node = child.parent
    visited.append(node)
    queue.append([node,0])
    depth = 0
    while queue:
        s , depth = queue.pop(0)
        if s.children is None:
            break
        print(depth, "\t")
        print(s.get_state())
        for child in s.children:
            if child not in visited:
                visited.append(child)
                queue.append([child,depth+1])
