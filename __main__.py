import graphics
import hnefatafl
import pygame
import time
from utilities import *

from testing import run_tests
from test_boards import test_boards as tb


if __name__ == "__main__":
    #b = hnefatafl.TaflBoard()
    #display = graphics.initialize()
    #graphics.refresh(b.board, display)
    #time.sleep(5)
    #while True:
    #    piece = input("Choose a piece: row, col: ")
    #    piece = piece.split(' ')
    #    row, col = int(piece[0]), int(piece[1])
    #    target = input("Choose a target: move int: ")
    #    target = int(target)
    #    make_move(b.board, (row, col), target)
    #    graphics.refresh(b.board, display)

    run_tests()
