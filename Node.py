
import numpy as np

rows = int(6)
columns = int(7)

def get_positions(node):
    board_state = node.get_state()
    board_state = np.flip(board_state, axis=0)
    positions = []
    for row in range(board_state.shape[0]):
        for column in range(board_state.shape[1]):
            if board_state[row][column] == 0 and (row == 0 or board_state[row - 1][column] != 0):
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
