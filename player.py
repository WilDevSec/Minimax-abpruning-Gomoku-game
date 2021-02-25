""" Method:
- First move: place in centre of board
- Calculate score from each board as heuristic
- Minimax function attempts to minimise/maximise score
- AB pruning implemented inside minimax function
to prevent searching routes that don't need searching
"""

import numpy as np

from misc import legalMove, winningTest
from gomokuAgent import GomokuAgent


# Returns the first position that has X in a line, used for searching for winning moves.
def nInARowMove(playerID, board, X_IN_A_LINE):
    # BOARD_SIZE = board.shape[0]
    # mask = np.ones(X_IN_A_LINE, dtype=int) * playerID
    for i in range(0, 11):
        count = 0
        for j in range(0, 11):
            if count >= X_IN_A_LINE - 1:
                if board[i, j] == 0:
                    return i, j
                elif board[i, j - 6] == 0:
                    return i, j - 6
            if board[i, j] != playerID:
                count = 0
            else:
                count += 1
    return None, None


def nInARowCount(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    count = 0
    for i in range(BOARD_SIZE):
        consecutiveCount = 0
        for j in range(BOARD_SIZE):
            if consecutiveCount >= X_IN_A_LINE:
                if board[i, j] == 0:
                    count += 1
                elif board[i, j - X_IN_A_LINE - 1] == 0:
                    count += 1
                consecutiveCount = 0
            if board[i, j] != playerID:
                consecutiveCount = 0
            else:
                consecutiveCount += 1
    return count


def diagCount(playerID, board, X_IN_A_LINE):
    BOARD_SIZE = board.shape[0]
    count = 0
    for r in range(BOARD_SIZE - X_IN_A_LINE + 1):
        for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
            flag = True
            for i in range(X_IN_A_LINE):
                if board[r + i, c + i] != playerID:
                    flag = False
                    break
            if flag:
                count += 1
    return count


# Evaluation of the board state for the heuristic to be used in min max algorithm.
# Looks for, firstly, places where 5 in a row is possible with spaces on either end,
# then 4, 3, 2; each one weighted, as having larger lines is of course more important.
def evaluateScore(playerID, board):
    if nInARowCount(playerID, board, 5) != 0 or diagCount(playerID, board, 5) != 0:
        return np.inf
    score = nInARowCount(playerID, board, 4) * 4
    score += nInARowCount(playerID, board, 3) * 3
    score += nInARowCount(playerID, board, 2)
    score += diagCount(playerID, board, 4) * 4
    score += diagCount(playerID, board, 3) * 3
    score += diagCount(playerID, board, 2)
    # Rotate the board then re-count and add horizontal and diagonal lines
    np.rot90(board)
    if nInARowCount(playerID, board, 5) != 0 or diagCount(playerID, board, 5) != 0:
        return np.inf
    score += nInARowCount(playerID, board, 4) * 4
    score += nInARowCount(playerID, board, 3) * 3
    score += nInARowCount(playerID, board, 2)
    score += diagCount(playerID, board, 4) * 4
    score += diagCount(playerID, board, 3) * 3
    score += diagCount(playerID, board, 2)
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    if maximizingPlayer == 1:
        minimizingPlayer = -1
    else:
        minimizingPlayer = 1
    moveLoc = nInARowMove(maximizingPlayer, board, 5)
    if depth == 0 or moveLoc != (None, None):
        return evaluateScore(maximizingPlayer, board), moveLoc

    if maximizingPlayer:
        maxEval = [1_000_000, None, None]
        for child in children(maximizingPlayer, board):
            evaluation = minimax(child[0], depth - 1, alpha, beta, minimizingPlayer)
            maxEval = maxEval if maxEval[0] > evaluation[0] else [evaluation[0], child[1], child[2]]
            alpha = max(alpha, evaluation[0])
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = [1_000_000, None, None]
        for child in children(minimizingPlayer, board):
            evaluation = minimax(child[0], depth - 1, alpha, beta, maximizingPlayer)
            minEval = minEval if minEval[0] < evaluation[0] else [evaluation[0], child[1], child[2]]
            beta = min(beta, evaluation[0])
            if beta <= alpha:
                break
        return minEval


# Return list of board states and their moves after all viable moves have taken place
def children(playerID, board):
    boardList = []
    for i in range(0, 10):
        for j in range(0, 10):
            if board[i][j] == 0:
                if board[i + 1][j] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i - 1][j] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i + 1][j + 1] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i - 1][j + 1] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i + 1][j - 1] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i - 1][j - 1] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i][j + 1] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
                elif board[i][j - 1] != 0:
                    tempBoard = board.copy()
                    tempBoard[i][j] = playerID
                    boardList.append([tempBoard, i, j])
    return boardList


class Player(GomokuAgent):
    def move(self, board):
        # First moves
        if legalMove(board, (5, 5)):
            return 5, 5
        if legalMove(board, (5, 6)):
            return 5, 6
        x, i, j = minimax(board, 3, -np.inf, np.inf, self.ID)
        return i, j
