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


class Player(GomokuAgent):

    # Returns the first position that will create X number of pieces in a line horizontally
    # def nInARowMove(playerID, board, X_IN_A_LINE):
    #     for i in range(0, 11):
    #         count = 0
    #         for j in range(0, 11):
    #             if count >= X_IN_A_LINE - 1:
    #                 if board[i, j] == 0:
    #                     return i, j
    #                 elif board[i, j - 6] == 0:
    #                     return i, j - 6
    #             if board[i, j] != playerID:
    #                 count = 0
    #             else:
    #                 count += 1
    #     return None, None

    # Returns same as above but diagonally
    # def nInDiagMove(playerID, board, X_IN_A_LINE):
    #     BOARD_SIZE = board.shape[0]
    #     for r in range(BOARD_SIZE - X_IN_A_LINE + 1):
    #         for c in range(BOARD_SIZE - X_IN_A_LINE + 1):
    #             count = 0
    #             for i in range(X_IN_A_LINE):
    #                 if count >= X_IN_A_LINE - 1:
    #                     if board[r + i + 1, c + i + 1] == 0:
    #                         return r + i + 1, c + i + 1
    #                     elif board[r - 1, c - 1] == 0:
    #                         return r - 1, c - 1
    #                 if board[r + i, c + i] != playerID:
    #                     count = 0
    #                 else:
    #                     count += 1
    #     return None, None

    # Returns the number given player's pieces in an unbroken row of a given length X
    def nInARowCount(self, playerID, board, X_IN_A_LINE):
        count = 0
        for i in range(self.BOARD_SIZE):
            consecutiveCount = 0
            for j in range(self.BOARD_SIZE):
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

    # Returns number of given player's pieces in unbroken diagonal line of length X
    def diagCount(self, playerID, board, X_IN_A_LINE):
        count = 0
        for r in range(self.BOARD_SIZE - X_IN_A_LINE + 1):
            for c in range(self.BOARD_SIZE - X_IN_A_LINE + 1):
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
    def evaluateScore(self, playerID, board):
        # If winnable move then return big number so will 100% be chosen as next move.
        if self.nInARowCount(playerID, board, 5) != 0 or self.diagCount(playerID, board, 5) != 0:
            return 1_000_000
        score = self.nInARowCount(playerID, board, 4) * 4
        score += self.nInARowCount(playerID, board, 3) * 3
        score += self.nInARowCount(playerID, board, 2)
        score += self.diagCount(playerID, board, 4) * 4
        score += self.diagCount(playerID, board, 3) * 3
        score += self.diagCount(playerID, board, 2)
        # Rotate the board then re-count and add horizontal and diagonal lines
        np.rot90(board)
        if self.nInARowCount(playerID, board, 5) != 0 or self.diagCount(playerID, board, 5) != 0:
            return 1_000_000
        score += self.nInARowCount(playerID, board, 4) * 4
        score += self.nInARowCount(playerID, board, 3) * 3
        score += self.nInARowCount(playerID, board, 2)
        score += self.diagCount(playerID, board, 4) * 4
        score += self.diagCount(playerID, board, 3) * 3
        score += self.diagCount(playerID, board, 2)
        return score

    # Minimax algorithm, returns an evaluation of the board if the max depth has been reached
    # Otherwise it calls itself recursively with the children of the node and depth -1
    # AB pruning checks that
    def minimax(self, playerID, board, depth, alpha, beta, maximizingPlayer):
        # Checking for a winning move
        # i, j = nInARowMove(maximizingPlayer, board, 5)
        # x, y = nInDiagMove(maximizingPlayer, board, 5)
        if depth == 0:
            return [self.evaluateScore(playerID, board), None, None]
        # if x is not None:
        #     return [evaluateScore(maximizingPlayer, board), x, y]
        if maximizingPlayer:
            maxEval = [-1_000_000, None, None]
            for child in self.children(playerID, board):
                evaluation = self.minimax(1 if playerID == -1 else -1, child[0], depth - 1, alpha, beta, False)
                maxEval = maxEval if maxEval[0] > evaluation[0] else [evaluation[0], child[1], child[2]]
                alpha = max(alpha, evaluation[0])
                if beta <= alpha:
                    break
            return maxEval
        else:
            minEval = [1_000_000, None, None]
            for child in self.children(playerID, board):
                evaluation = self.minimax(1 if playerID == -1 else -1, child[0], depth - 1, alpha, beta, True)
                minEval = minEval if minEval[0] < evaluation[0] else [evaluation[0], child[1], child[2]]
                beta = min(beta, evaluation[0])
                if beta <= alpha:
                    break
            return minEval

    # def minimax(board, depth, alpha, beta, maximizingPlayer):
    #     # Setting opposing player's ID
    #     if maximizingPlayer == 1:
    #         minimizingPlayer = -1
    #     else:
    #         minimizingPlayer = 1
    #     # Checking for a winning move
    #     # i, j = nInARowMove(maximizingPlayer, board, 5)
    #     # x, y = nInDiagMove(maximizingPlayer, board, 5)
    #     if depth == 0:
    #         return [evaluateScore(maximizingPlayer if maximizingPlayer else minimizingPlayer, board), None, None]
    #     # if x is not None:
    #     #     return [evaluateScore(maximizingPlayer, board), x, y]
    #     if maximizingPlayer:
    #         maxEval = [-1_000_000, None, None]
    #         for child in children(maximizingPlayer, board):
    #             evaluation = minimax(child[0], depth - 1, alpha, beta, False)
    #             maxEval = maxEval if maxEval[0] > evaluation[0] else [evaluation[0], child[1], child[2]]
    #             previousMoves.append(maxEval)
    #             alpha = max(alpha, evaluation[0])
    #             if beta <= alpha:
    #                 break
    #         return maxEval
    #     else:
    #         minEval = [1_000_000, None, None]
    #         for child in children(minimizingPlayer, board):
    #             evaluation = minimax(child[0], depth - 1, alpha, beta, True)
    #             minEval = minEval if minEval[0] < evaluation[0] else [evaluation[0], child[1], child[2]]
    #             previousMoves.append(minEval)
    #             beta = min(beta, evaluation[0])
    #             if beta <= alpha:
    #                 break
    #         return minEval

    # Return list of board states and their moves after all viable moves have taken place
    def children(self, playerID, board):
        BOARD_SIZE = board.shape[0]
        boardList = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] == 0:
                    if i + 1 < BOARD_SIZE and board[i + 1][j] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif i - 1 > 0 and board[i - 1][j] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif i + 1 < BOARD_SIZE and j + 1 < BOARD_SIZE and board[i + 1][j + 1] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif i - 1 > 0 and j + 1 < BOARD_SIZE and board[i - 1][j + 1] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif i + 1 < BOARD_SIZE and j - 1 > 0 and board[i + 1][j - 1] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif i - 1 > 0 and j - 1 > 0 and board[i - 1][j - 1] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif j + 1 < BOARD_SIZE and board[i][j + 1] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
                    elif j - 1 > 0 and board[i][j - 1] != 0:
                        tempBoard = board.copy()
                        tempBoard[i][j] = playerID
                        boardList.append([tempBoard, i, j])
        return boardList

    def move(self, board):
        # First moves
        if legalMove(board, (5, 5)):
            return 5, 5
        evaluations = []
        moves = []
        for child in self.children(self.ID, board):
            moves.append([child[1], child[2]])
            evaluation = self.minimax(self.ID, child[0], 2, -1_000_000, 1_000_000, True)
            evaluations.append(evaluation[0])
            bestValueIndex = np.argmax(np.array(evaluations))
        return moves[bestValueIndex][0], moves[bestValueIndex][1]
