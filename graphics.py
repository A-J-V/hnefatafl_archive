import numpy as np
import pygame


WIN_SIZE = 1024
TILE_SIZE = 110
BLK = (0, 0, 0)
BGCOLOR = BLK
board_image = pygame.image.load('./assets/hnefatafl_board.png')
viking_white = pygame.image.load('./assets/viking_white.png')
viking_white = pygame.transform.scale(viking_white, (TILE_SIZE, TILE_SIZE))
viking_black = pygame.image.load('./assets/viking_black.png')
viking_black = pygame.transform.scale(viking_black, (TILE_SIZE, TILE_SIZE))
viking_king = pygame.image.load('./assets/viking_king.png')
viking_king = pygame.transform.scale(viking_king, (140, 140))
plane_to_img = {0: viking_black,
                1: viking_white,
                2: viking_king,
                }


def initialize():
    # Pygame & Camera setup
    pygame.init()
    main_display = pygame.display.set_mode(size=(WIN_SIZE, WIN_SIZE))
    pygame.display.set_caption('Hnefatafl')
    main_display.fill(BGCOLOR)
    rect = main_display.get_rect()
    main_display.blit(board_image, rect)
    return main_display


def refresh(board: np.array, display: pygame.surface):
    display_rect = display.get_rect()
    display.blit(board_image, display_rect)
    for i, row in enumerate(range(board.shape[1])):
        for j, col in enumerate(range(board.shape[2])):
            for k, plane in enumerate(range(board.shape[0])):
                if not board[k, i, j].any():
                    continue
                else:
                    cc = 112 - 15 * (plane == 2)
                    rc = 90 - 30 * (plane == 2)
                    piece_sprite = plane_to_img[k]
                    piece_rect = pygame.Rect((col * 68 + cc, row * 65 + rc, TILE_SIZE, TILE_SIZE))
                    display.blit(piece_sprite, piece_rect)
    pygame.display.update()



