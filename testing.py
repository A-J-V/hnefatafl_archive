from test_boards import test_boards as tb
from hnefatafl import TaflBoard
from utilities import *


def run_tests():
    print("Running tests...")
    # King is not captured when surrounded by defenders.
    b = TaflBoard(tb['starting_board'])
    assert check_king(b.board_array) == 0, \
        "Expected no terminal state, as the King has neither escaped nor been captured."

    # King is surrounded! This should result in immediate capture.
    b = TaflBoard(tb['instant_loss_surround'])
    assert check_king(b.board_array) == -1, \
        "Expected instant loss upon checking the King, because the King is surrounded by 4 enemies."

    # King is surrounded! This should result in immediate capture.
    b = TaflBoard(tb['instant_loss_throne'])
    assert check_king(b.board_array) == -1, \
        "Expected instant loss upon checking the King, because the King is surrounded by 3 enemies + throne."

    # King has escaped to a corner! This should be an immediate defender win.
    b = TaflBoard(tb['instant_win_corner'])
    assert check_king(b.board_array) == 1, \
        "Expected instant win upon checking the King, because the King is in a corner."

    # A defender is flanked by the empty throne and an attacker. It should be captured upon checking.
    b = TaflBoard(tb['instant_capture_empty_throne_5_7'])
    check_capture(b.board_array, (5, 7))
    assert not (b.board_array[:, 5, 6].any()), \
        "Expected the piece at (5, 6) to be captured, because it was flanked by enemy and empty throne."

    # A group of three attackers is pinned in a shield wall on the right edge.
    # They should all be captured upon checking.
    b = TaflBoard(tb['instant_capture_shield_wall_2_10'])
    check_capture(b.board_array, (3, 9))
    assert not (b.board_array[:, 3, 10].any()), \
        "Expected the piece at (3, 10) to be captured, because it was caught in a shield wall."
    assert not (b.board_array[:, 4, 10].any()), \
        "Expected the piece at (4, 10) to be captured, because it was caught in a shield wall."
    assert not (b.board_array[:, 5, 10].any()), \
        "Expected the piece at (5, 10) to be captured, because it was caught in a shield wall."

    # A group of two defenders is pinned in a shield wall on the bottom edge.
    # They should all be captured upon checking.
    b = TaflBoard(tb['instant_capture_shield_wall_9_4'])
    check_capture(b.board_array, (9, 4))
    assert not (b.board_array[:, 10, 4].any()), \
        "Expected the piece at (10, 4) to be captured, because it was caught in a shield wall."
    assert not (b.board_array[:, 10, 5].any()), \
        "Expected the piece at (10, 5) to be captured, because it was caught in a shield wall."

    # A group of two defenders and the King are pinned in a shield wall on the bottom edge.
    # The defenders should be captured upon checking, but not the King.
    b = TaflBoard(tb['instant_capture_shield_wall_9_5'])
    check_capture(b.board_array, (9, 5))
    assert not (b.board_array[:, 10, 4].any()), \
        "Expected the piece at (10, 4) to be captured, because it was caught in a shield wall."
    assert not (b.board_array[:, 10, 5].any()), \
        "Expected the piece at (10, 5) to be captured, because it was caught in a shield wall."
    assert (b.board_array[:, 10, 6].any()), \
        "Expected the King to remain at (10, 6), because it can't be captured by a shield wall."

    print("All tests finished.")
