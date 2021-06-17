import numpy as np
import random
import pygame
import sys
import math
import minimax_connect_4
import Node
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = -1

WINDOW_LENGTH = 4
game_over = False
def game_ended(board):
	return True if np.count_nonzero(board)==42 else False
def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0
def calculate_score(board):
	###number of connected fours horizaontally
	window_size = 4
	AI_count =0
	human_count = 0
	for i in range(ROW_COUNT):
		for j in range(COLUMN_COUNT - 3):
			if np.sum(board[i][j:j + window_size]) == 4:
				human_count += 1
			# break
			elif np.sum(board[i][j:j + window_size]) == -4:
				AI_count += 1
	### number of connected fours vertically
	for i in range(ROW_COUNT - 4):
		sum_array = np.sum(board[i:i + window_size], axis=0)

		for j in range(sum_array.shape[0]):
			if sum_array[j] == 4:
				## we used transpose to get the coloumn
				human_count += 1
			# break
			elif sum_array[j] == -4:
				AI_count += 1
			# break
	##number of connected fours diagonally
	for r in range(ROW_COUNT - 3):
		for c in range(COLUMN_COUNT - 3):
			window = np.array([board[r + i][c + i] for i in range(window_size)])

			if np.sum(window) == 4:
				human_count += 1
			# break
			elif np.sum(window) == -4:
				AI_count += 1
			# break

	for r in range(ROW_COUNT - 3):
		for c in range(COLUMN_COUNT - 3):
			window = np.array([board[r + 3 - i][c + i] for i in range(window_size)])

			if np.sum(window) == 4:
				human_count += 1
			# break
			elif np.sum(window) == -4:
				AI_count += 1
			# break
	print(AI_count)
	print(human_count)
	return AI_count - human_count


def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
            int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)
board = np.zeros((ROW_COUNT, COLUMN_COUNT))
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER, AI)

while not game_over:

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()

		if event.type == pygame.MOUSEMOTION:
			pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
			posx = event.pos[0]
			if turn == PLAYER:
				pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)

		pygame.display.update()

		if event.type == pygame.MOUSEBUTTONDOWN:
			pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))

			# Ask for Player 1 Input
			if turn == PLAYER:
				posx = event.pos[0]
				col = int(math.floor(posx/SQUARESIZE))

				if is_valid_location(board, col):
					row = get_next_open_row(board, col)
					drop_piece(board, row, col, PLAYER_PIECE)



					if game_ended(board):
							label = None
							if calculate_score(board) > 0:
								label = myfont.render(" AI wins!!", 1, YELLOW)
							else:
								label = myfont.render(" human wins!!", 1, YELLOW)
							screen.blit(label, (40,10))
							pygame.display.update()
							game_over = True
							print(board)
					turn += 1
					turn = turn % 2

					draw_board(board)


	# # Ask for Player 2 Input
	if turn == AI and not game_over:
			node = Node.Node( np.copy(board), None, None, None, None)
			K = 5
			alpha_beta = False
			print("AI working")
			child = minimax_connect_4.decision(node, K, alpha_beta)
			minimax_connect_4.get_tree(child)
			print("AI done")


			if is_valid_location(board, child.col):

				row = get_next_open_row(board, child.col)
				drop_piece(board, row, child.col, AI_PIECE)
				draw_board(board)
				if game_ended(board):
					label = None
					if calculate_score(board) > 0:
						label = myfont.render(" AI wins!!", 1, YELLOW)
					else:
						label = myfont.render(" human wins!!", 1, YELLOW)
					screen.blit(label, (40,10))

					pygame.display.update()
					game_over = True


				turn += 1
				turn = turn % 2

	if game_over:
		pygame.time.wait(10000)