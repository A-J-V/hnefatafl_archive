import numpy as np
import pygame


WIN_SIZE = 1024
TILE_SIZE = 110
RED_HIGHLIGHT = (240, 50, 50, 100)
BLUE_HIGHLIGHT = (50, 50, 240, 100)
RED_TILE = None
BLUE_TILE = None

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
    """Initialize Pygame and set up the camera and related variables."""
    pygame.init()
    main_display = pygame.display.set_mode(size=(WIN_SIZE, WIN_SIZE))
    pygame.display.set_caption('Hnefatafl')
    main_display.fill(BGCOLOR)
    rect = main_display.get_rect()
    main_display.blit(board_image, rect)
    global RED_TILE
    global BLUE_TILE
    RED_TILE = (pygame.Surface((TILE_SIZE//1.7, TILE_SIZE//1.7)).convert_alpha())
    RED_TILE.fill(RED_HIGHLIGHT)
    BLUE_TILE = pygame.Surface((TILE_SIZE//1.7, TILE_SIZE//1.7)).convert_alpha()
    BLUE_TILE.fill(BLUE_HIGHLIGHT)
    return main_display


def refresh(board: np.array,
            display: pygame.surface,
            piece_flags: np.array,
            show_cache: bool = False,
            dirty_flags: set = None,
            show_dirty: bool = False,
            ):
    """
    Update the camera.

    This function updates the view of the board that the player sees after each move. It can also be used for debugging
    by highlighting cache info.

    :param board: The 3D NumPy "board" array on which the game is being played.
    :param display: The Pygame surface on which all graphics are drawn.
    :param piece_flags: A binary 2D NumPy array where 1 indicates that a piece is present and 0 indicates no piece.
    :param show_cache: If True, tiles that are occupied by a piece according to the cache will be highlighted.
    :param dirty_flags: A set of (row, col) tuples indicating which tile needs a cache refresh.
    :param show_dirty: If true, tiles that are due for a cache refresh are highlighted red, clean caches are blue.
    """
    display_rect = display.get_rect()
    display.blit(board_image, display_rect)
    for i, row in enumerate(range(board.shape[1])):
        for j, col in enumerate(range(board.shape[2])):
            if piece_flags[i, j] != 1:
                if show_cache:
                    cc = 134
                    rc = 135
                    piece_rect = pygame.Rect((col * 68 + cc, row * 65 + rc, TILE_SIZE, TILE_SIZE))
                    display.blit(BLUE_TILE, piece_rect)

            else:
                if board[0, i, j] == 1:
                    plane = 0
                elif board[1, i, j] == 1:
                    plane = 1
                else:
                    plane = 2
                cc = 112 - 15 * (plane == 2)
                rc = 90 - 30 * (plane == 2)
                piece_sprite = plane_to_img[plane]
                piece_rect = pygame.Rect((col * 68 + cc, row * 65 + rc, TILE_SIZE, TILE_SIZE))
                if show_cache:
                    display.blit(RED_TILE, pygame.Rect((col * 68 + 134, row * 65 + 135, TILE_SIZE, TILE_SIZE)))
                elif show_dirty and dirty_flags is not None:
                    if (i, j) in dirty_flags:
                        display.blit(RED_TILE, pygame.Rect((col * 68 + 134, row * 65 + 135, TILE_SIZE, TILE_SIZE)))
                    else:
                        display.blit(BLUE_TILE, pygame.Rect((col * 68 + 134, row * 65 + 135, TILE_SIZE, TILE_SIZE)))
                display.blit(piece_sprite, piece_rect)
    pygame.display.update()



