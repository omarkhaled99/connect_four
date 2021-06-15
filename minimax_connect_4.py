import numpy as np

rows = int(6)
columns = int(7)
##human --> 1 AI ---> -1
#board = np.array([[1,  1,  1,  1, 1,  -1, -1],
                  # [1,  1,  1,  1, -1,  1, -1],
                  # [1, -1, -1, -1, -1,  -1, -1],
                  # [1,  1,  1, -1, -1, -1, -1],
                  # [1, -1, -1, -1, -1, -1, -1],
                  # [1,  1,  1,  1,  1, -1, 1]])
board = np.zeros((rows,columns))

board = np.flip(board, axis=0)
board[0][0] = 1
board[1][0] = -1
board[0][1] = 1
# print(board)
# unique, counts = np.unique(board, return_counts=True)
# dictionary = dict(zip(unique, counts))
# print(dictionary[1])


def get_positions(node):
    board_state = node.get_state()
    board_state = np.flip(board_state, axis=0)
    positions = []
    for row in range(board_state.shape[0]):
        for column in range(board_state.shape[1]):
            if board[row][column] == 0 and (row == 0 or board[row - 1][column] != 0):
                positions.append([row, column])
    return positions


class Node:
    def get_children(self,max):
        self.max = max
        children = []
        positions = get_positions(self)
        # print(positions)
        for position in positions:
            # print(position)
            new_board = np.copy(self.board)
            if self.max:
                new_board[position[0]][position[1]] = -1
            else:
                new_board[position[0]][position[1]]= 1
            child = Node(new_board,self,None,None)
            children.append(child)
        return children

    def __init__(self, board, parent, movement, score):
        # Contains the state of the node, [list of the state of the board at this node]
        self.f = None
        self.board = board
        self.max = None
        # Contains the node that generated this node
        self.parent = parent
        self.movement = movement
        self.score = score

        if self.board is not None:
            self.map = ''.join(str(e) for e in self.board)

    def update_score(self, score):
        self.score = score

    def update_movement(self, movement):
        self.movement = movement

    def get_state(self):
        return self.board

    def __lt__(self, other):
        return self.f < other.f


def terminal_test(node):
    node_board = node.get_state()
    return True if np.count_nonzero(node_board) == 42 else False


# def evaluate_heuristic(node):
#     heuristics_matrix = np.array([[3, 4, 5, 7, 5, 4, 3],
#                                   [4, 6, 8, 10, 8, 6, 4],
#                                   [5, 8, 11, 13, 11, 8, 5],
#                                   [5, 8, 11, 13, 11, 8, 5],
#                                   [4, 6, 8, 10, 8, 6, 4],
#                                   [3, 4, 5, 7, 5, 4, 3]])
#     # print(heuristics_matrix.shape)
#     positions = np.array(get_positions(node))
#     positions_trans = positions.T
#     # print(positions)
#     positions_values = heuristics_matrix[positions_trans[0], positions_trans[1]]
#     node.update_score(positions_values[np.argmax(positions_values)])
#     return positions_values[np.argmax(positions_values)]


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
            if np.sum(node_board[i][j:j + window_size]) == 4:

                human_count += np.sum(heuristics_matrix[i][j:j + window_size])
                # break
            elif np.sum(node_board[i][j:j + window_size]) == -4:
                AI_count +=  np.sum(heuristics_matrix[i][j:j + window_size])
                # break
    ### number of connected fours vertically
    for i in range(rows - 4):
        sum_array = np.sum(node_board[i:i + window_size], axis=0)
        sum_heuristic = np.sum(heuristics_matrix[i:i + window_size], axis=0)
        for j in range(sum_array.shape[0]):
            if sum_array[j] == 4:
                human_count += sum_heuristic[j]
                # break
            elif sum_array[j] == -4:
                AI_count += sum_heuristic[j]
                # break
    ##number of connected fours diagonally
    for r in range(rows - 3):
        for c in range(columns - 3):
            window = [node_board[r + i][c + i] for i in range(window_size)]
            if np.sum(window) == 4:
                human_count += 1
                # break
            elif np.sum(window) == -4:
                AI_count += 1
                # break

    for r in range(rows - 3):
        for c in range(columns - 3):
            window = [node_board[r + 3 - i][c + i] for i in range(window_size)]
            heuristic_matrix_window = [heuristics_matrix[r + 3 - i][c + i] for i in range(window_size)]
            if np.sum(window) == 4:
                human_count += np.sum(heuristic_matrix_window)
                # break
            elif np.sum(window) == -4:
                AI_count +=  np.sum(heuristic_matrix_window)
                # break


    return AI_count - human_count


# print(evaluate_state())


def maximize(node,depth, alpha, beta):
    if terminal_test(node) or depth == 0:
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
                beta = max_utility

    return max_child, max_utility


def minimize(node,depth, alpha, beta):
    if terminal_test(node) or depth == 0:
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

            if min_utility <= beta:
                beta = min_utility

    return min_child, min_utility


def decision(node,K,alpha_beta):
    depth =K
    if alpha_beta:
        child, _ = maximize(node, depth,float('-inf'), float('inf'))
    else:
        child, _ = maximize(node, depth, None, None)
    return child

node = Node(board, None, None, None)
K=7
alpha_beta = True
child = decision(node,K,alpha_beta)
print(child.get_state())
