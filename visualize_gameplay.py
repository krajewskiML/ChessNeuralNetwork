# this module is used to visualize the gameplay of the game in which the model is playing against itself
# we will use pygame to visualize the gameplay
# we will use the model to evaluate the board state
# also we will use chess module to emulate the game

import chess
from chessboard import display
import random
import keras
import numpy as np
import pygame as pg
from tensorflow import keras


# define a function that will convert the board state to a tensor
def board_to_tensor(board_instance):
    tensor = np.zeros((12, 8, 8))
    to_move = board_instance.turn # true -> white, false -> black
    for i in range(8): # rows
        for j in range(8): # columns
            piece = board_instance.piece_at(i * 8 + j)
            if piece is not None:
                if to_move: # white -> we do not invert anything
                    if piece.color:
                        tensor[piece.piece_type - 1][i][j] = 1
                    else:
                        tensor[piece.piece_type + 5][i][j] = 1
                else:  # black
                    if not piece.color: # upside down
                        tensor[piece.piece_type - 1][7 - i][j] = 1
                    else:
                        tensor[piece.piece_type + 5][7 - i][j] = 1
    return tensor


# define a function that will play a game of chess
def play_random_game():
    game_board = display.start()
    board = chess.Board()
    while not board.is_game_over():
        display.check_for_quit()
        fen = board.fen()
        display.update(fen, game_board)
        # get possible moves
        moves = list(board.legal_moves)
        # get the move
        move = random.choice(moves)
        # make the move
        board.push(move)
        # sleep for 1 second
        pg.time.delay(500)
    fen = board.fen()
    display.update(fen, game_board)
    display.terminate()


def get_best_move(board: chess.Board, model):
    moves = list(board.legal_moves)
    positions = []
    for move in moves:
        board.push(move)
        positions.append(board_to_tensor(board))
        board.pop()
    positions = np.array(positions)
    scores = model.predict(positions)
    return moves[np.argmin(scores)]


def play_game_against_random(model):
    game_board = display.start()
    # set fps to 1
    display.fps_clock.tick(1)
    board = chess.Board()
    while not board.is_game_over():
        display.check_for_quit()
        fen = board.fen()
        display.update(fen, game_board)
        # if it is white's turn get the best move else get a random move
        if board.turn:
            move = get_best_move(board, model)
        else:
            moves = list(board.legal_moves)
            move = random.choice(moves)
        # make the move
        board.push(move)
        # sleep for 1 second
        pg.time.delay(500)

    fen = board.fen()
    display.update(fen, game_board)
    pg.time.delay(3000)
    display.terminate()

def play_game_against_self(model):
    game_board = display.start()
    # set fps to 1
    display.fps_clock.tick(1)
    board = chess.Board()
    while not board.is_game_over():
        display.check_for_quit()
        fen = board.fen()
        display.update(fen, game_board)
        # if it is white's turn get the best move else get a random move
        move = get_best_move(board, model)
        # make the move
        board.push(move)
        # sleep for 1 second
        pg.time.delay(500)

    fen = board.fen()
    display.update(fen, game_board)
    pg.time.delay(3000)
    display.terminate()


def main():
    # load the model
    model = keras.models.load_model('biggest_negative.h5')
    # play the game
    play_game_against_self(model)


if __name__ == "__main__":
    main()
