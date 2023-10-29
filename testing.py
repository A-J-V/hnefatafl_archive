from test_boards import test_boards as tb
from hnefatafl import TaflBoard
from utilities import *


def run_tests():
    print("Running tests...")
    # King is not captured when surrounded by defenders.
    b = TaflBoard(tb['starting_board'])
    assert check_king(b.board) == 0, \
        "Expected no terminal state, as the King has neither escaped nor been captured."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") is None, "Expected no terminal state, but found terminal."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found terminal."

    # King is surrounded! This should result in immediate capture.
    b = TaflBoard(tb['instant_loss_surround'])
    assert check_king(b.board) == -1, \
        "Expected instant loss upon checking the King, because the King is surrounded by 4 enemies."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "attackers", "Expected a terminal state, but found none."

    # King is surrounded! This should result in immediate capture.
    b = TaflBoard(tb['instant_loss_throne'])
    assert check_king(b.board) == -1, \
        "Expected instant loss upon checking the King, because the King is surrounded by 3 enemies + throne."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "attackers", "Expected a terminal state, but found none."

    # King has escaped to a corner! This should be an immediate defender win.
    b = TaflBoard(tb['instant_win_corner'])
    assert check_king(b.board) == 1, \
        "Expected instant win upon checking the King, because the King is in a corner."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "defenders", "Expected a terminal state, but found none."

    # A defender is flanked by the empty throne and an attacker. It should be captured upon checking.
    b = TaflBoard(tb['instant_capture_empty_throne_5_7'])
    check_capture(b.board, (5, 7))
    assert not (b.board[:, 5, 6].any()), \
        "Expected the piece at (5, 6) to be captured, because it was flanked by enemy and empty throne."

    # A group of three attackers is pinned in a shield wall on the right edge.
    # They should all be captured upon checking.
    b = TaflBoard(tb['instant_capture_shield_wall_2_10'])
    check_capture(b.board, (3, 9))
    assert not (b.board[:, 3, 10].any()), \
        "Expected the piece at (3, 10) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 4, 10].any()), \
        "Expected the piece at (4, 10) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 5, 10].any()), \
        "Expected the piece at (5, 10) to be captured, because it was caught in a shield wall."

    # A group of two defenders is pinned in a shield wall on the bottom edge.
    # They should all be captured upon checking.
    b = TaflBoard(tb['instant_capture_shield_wall_9_4'])
    check_capture(b.board, (9, 4))
    assert not (b.board[:, 10, 4].any()), \
        "Expected the piece at (10, 4) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 10, 5].any()), \
        "Expected the piece at (10, 5) to be captured, because it was caught in a shield wall."

    # A group of two defenders and the King are pinned in a shield wall on the bottom edge.
    # The defenders should be captured upon checking, but not the King.
    b = TaflBoard(tb['instant_capture_shield_wall_9_5'])
    check_capture(b.board, (9, 5))
    assert not (b.board[:, 10, 4].any()), \
        "Expected the piece at (10, 4) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 10, 5].any()), \
        "Expected the piece at (10, 5) to be captured, because it was caught in a shield wall."
    assert (b.board[:, 10, 6].any()), \
        "Expected the King to remain at (10, 6), because it can't be captured by a shield wall."

    # A group of four defenders is pinned in a shield wall on the left edge between attackers and a corner.
    # All four defenders should be captured upon checking.
    b = TaflBoard(tb['instant_capture_shield_wall_1_1'])
    check_capture(b.board, (1, 1))
    assert not (b.board[:, 1, 0].any()), \
        "Expected the piece at (1, 0) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 2, 0].any()), \
        "Expected the piece at (2, 0) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 3, 0].any()), \
        "Expected the piece at (3, 0) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 4, 0].any()), \
        "Expected the piece at (4, 0) to be captured, because it was caught in a shield wall."

    # A group of three defenders is ALMOST pinned in a shield wall on the left edge, but not yet enclosed.
    # All three defenders should NOT be captured upon checking.
    b = TaflBoard(tb['no_capture_shield_wall_2_1'])
    check_capture(b.board, (2, 1))
    assert (b.board[:, 2, 0].any()), \
        "Expected the piece at (2, 0) to NOT be captured, because the shield wall isn't closed."
    assert (b.board[:, 3, 0].any()), \
        "Expected the piece at (3, 0) to NOT be captured, because the shield wall isn't closed."
    assert (b.board[:, 4, 0].any()), \
        "Expected the piece at (4, 0) to NOT be captured, because the shield wall isn't closed."

    # A group of two attackers is pinned in a shield wall on the top edge between defenders.
    # Both attackers should be captured upon checking.
    b = TaflBoard(tb['instant_capture_shield_wall_0_3'])
    check_capture(b.board, (0, 3))
    assert not (b.board[:, 0, 4].any()), \
        "Expected the piece at (0, 4) to be captured, because it was caught in a shield wall."
    assert not (b.board[:, 0, 5].any()), \
        "Expected the piece at (0, 5) to be captured, because it was caught in a shield wall."
    assert (b.board[:, 0, 7].any()), \
        "Expected the piece at (0, 7) to not be captured, because it was outside the shield wall."

    b = TaflBoard(tb['instant_capture_1_6'])
    check_capture(b.board, (1, 6))
    assert not (b.board[:, 1, 7].any()), \
        "Expected the defender at (1, 7) to be captured, because it was flanked by two attackers."

    b = TaflBoard(tb['king_5_5'])
    assert is_king(b.board, 5, 5), "Expected to recognize the King at (5, 5), but did not."

    b = TaflBoard(tb['king_5_10'])
    assert is_king(b.board, 5, 10), "Expected to recognize the King at (5, 10), but did not."

    b = TaflBoard(tb['fort_not_exit_fort_5_10'])
    defender_tags = []
    interior_tags = []
    assert is_fort(b.board, (5, 10), defender_tags, interior_tags), \
        "Expected the King at (5, 10) to be considered in a (non-exit) fort, but it was not."

    b = TaflBoard(tb['fort_not_exit_fort_10_3'])
    defender_tags = []
    interior_tags = []
    assert is_fort(b.board, (10, 3), defender_tags, interior_tags), \
        "Expected the King at (10, 3) to be considered in a (non-exit) fort, but it was not."
    assert not is_impenetrable(b.board, defender_tags, interior_tags), \
        "Expected the fort around (10, 3) to be penetrable, but it was deemed impenetrable."

    b = TaflBoard(tb['no_fort_10_3'])
    defender_tags = []
    interior_tags = []
    assert not is_fort(b.board, (10, 3), defender_tags, interior_tags), \
        "Expected the King at (10, 3) to NOT be considered in a (non-exit) fort, but is was."

    b = TaflBoard(tb['exit_fort_10_5'])
    defender_tags = []
    interior_tags = []
    assert is_fort(b.board, (10, 5), defender_tags, interior_tags), \
        "Expected the King at (10, 5) to be considered in a fort, but it was not."
    assert is_impenetrable(b.board, defender_tags, interior_tags), \
        "Expected the fort around (10, 5) to be impenetrable, but it was deemed penetrable."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "defenders", "Expected a terminal state, but found none."

    b = TaflBoard(tb['exit_fort_0_4'])
    defender_tags = []
    interior_tags = []
    assert is_fort(b.board, (0, 4), defender_tags, interior_tags), \
        "Expected the King at (0, 4) to be considered in a fort, but it was not."
    assert is_impenetrable(b.board, defender_tags, interior_tags), \
        "Expected the fort around (0, 4) to be impenetrable, but it was deemed penetrable."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "defenders", "Expected a terminal state, but found none."

    b = TaflBoard(tb['exit_fort_6_0'])
    defender_tags = []
    interior_tags = []
    assert is_fort(b.board, (6, 0), defender_tags, interior_tags), \
        "Expected the King at (6, 0) to be considered in a fort, but it was not."
    assert is_impenetrable(b.board, defender_tags, interior_tags), \
        "Expected the fort around (6, 0) to be impenetrable, but it was deemed penetrable."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "defenders", "Expected a terminal state, but found none."

    b = TaflBoard(tb['exit_fort_2_0'])
    defender_tags = []
    interior_tags = []
    assert is_fort(b.board, (2, 0), defender_tags, interior_tags), \
        "Expected the King at (2, 0) to be considered in a fort, but it was not."
    assert is_impenetrable(b.board, defender_tags, interior_tags), \
        "Expected the fort around (2, 0) to be impenetrable, but it was deemed penetrable."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "defenders", "Expected a terminal state, but found none."

    b = TaflBoard(tb['encirclement_test_1'])
    assert check_encirclement(b.board), \
        "Expected encirclement to be True, but it was False."
    attacker_walls, visited = verify_encirclement(b.board)
    assert is_impenetrable(b.board, attacker_walls, visited, option='encirclement'), \
        "Expected encirclement to be impenetrable, but it was not."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "attackers", "Expected a terminal state, but found none."

    b = TaflBoard(tb['encirclement_test_2'])
    assert check_encirclement(b.board), \
        "Expected encirclement to be True, but it was False."
    attacker_walls, visited = verify_encirclement(b.board)
    assert is_impenetrable(b.board, attacker_walls, visited, option='encirclement'), \
        "Expected encirclement to be impenetrable, but it was not."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "attackers", "Expected a terminal state, but found none."

    b = TaflBoard(tb['encirclement_test_3'])
    assert check_encirclement(b.board), \
        "Expected encirclement to be True, but it was False."
    attacker_walls, visited = verify_encirclement(b.board)
    assert is_impenetrable(b.board, attacker_walls, visited, option='encirclement'), \
        "Expected encirclement to be impenetrable, but it was not."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "attackers", "Expected a terminal state, but found none."

    b = TaflBoard(tb['no_encirclement_test_1'])
    assert not check_encirclement(b.board), \
        "Expected encirclement to be False, but it was True."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['no_encirclement_test_2'])
    assert not check_encirclement(b.board), \
        "Expected encirclement to be False, but it was True."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['no_encirclement_test_3'])
    assert not check_encirclement(b.board), \
        "Expected encirclement to be False, but it was True."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['partial_encirclement_test_1'])
    assert check_encirclement(b.board), \
        "Expected encirclement to be True, but it was False."
    attacker_walls, visited = verify_encirclement(b.board)
    assert not is_impenetrable(b.board, attacker_walls, visited, option='encirclement'), \
        "Expected encirclement to NOT be impenetrable, but it was."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['partial_encirclement_test_2'])
    assert check_encirclement(b.board), \
        "Expected encirclement to be True, but it was False."
    attacker_walls, visited = verify_encirclement(b.board)
    assert not is_impenetrable(b.board, attacker_walls, visited, option='encirclement'), \
        "Expected encirclement to NOT be impenetrable, but it was."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['partial_encirclement_test_3'])
    assert check_encirclement(b.board), \
        "Expected encirclement to be True, but it was False."
    attacker_walls, visited = verify_encirclement(b.board)
    assert not is_impenetrable(b.board, attacker_walls, visited, option='encirclement'), \
        "Expected encirclement to NOT be impenetrable, but it was."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['legal_move_check_3_0'])
    legal_moves = get_moves(b.board, (3, 0), cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags)
    expected_moves = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(legal_moves, expected_moves), "Expected legal moves did not match the returned legal moves."

    b = TaflBoard(tb['legal_move_check_3_0_king'])
    legal_moves = get_moves(b.board, (3, 0), cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags)
    expected_moves = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(legal_moves, expected_moves), "Expected legal moves did not match the returned legal moves."

    b = TaflBoard(tb['legal_move_check_5_1'])
    legal_moves = get_moves(b.board, (5, 1), cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags)
    expected_moves = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(legal_moves, expected_moves), "Expected legal moves did not match the returned legal moves."

    b = TaflBoard(tb['legal_move_check_5_1_king'])
    legal_moves = get_moves(b.board, (5, 1), cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags)
    expected_moves = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(legal_moves, expected_moves), "Expected legal moves did not match the returned legal moves."

    b = TaflBoard(tb['defenders_no_moves'])
    assert not has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player='defenders'), \
        "Expected defenders to have no legal moves, but they did."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "attackers", "Expected a terminal state, but found none."

    b = TaflBoard(tb['defenders_no_moves_2'])
    assert not has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have no legal moves, but they did."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "attackers", "Expected a terminal state, but found none."

    b = TaflBoard(tb['defenders_no_moves_3'])
    assert not has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have no legal moves, but they did."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="defenders") == "attackers", "Expected a terminal state, but found none."

    b = TaflBoard(tb['defenders_has_moves'])
    assert has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have legal moves, but they did not."

    b = TaflBoard(tb['defenders_has_moves_2'])
    assert has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have legal moves, but they did not."

    b = TaflBoard(tb['defenders_has_moves_3'])
    assert has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have legal moves, but they did not."

    b = TaflBoard(tb['defenders_has_moves_4'])
    assert has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have legal moves, but they did not."

    b = TaflBoard(tb['attackers_no_moves_1'])
    assert not has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'attackers'), \
        "Expected defenders to have no legal moves, but they did."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "defenders", "Expected a terminal state, but found none."

    b = TaflBoard(tb['attackers_no_moves_2'])
    assert not has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'attackers'), \
        "Expected defenders to have no legal moves, but they did."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") == "defenders", "Expected a terminal state, but found none."

    b = TaflBoard(tb['defenders_has_moves_3'])
    assert has_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders'), \
        "Expected defenders to have legal moves, but they did not."
    assert is_terminal(board=b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player="attackers") is None, "Expected no terminal state, but found one."

    b = TaflBoard(tb['legal_move_test_4_3'])
    moves = all_legal_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player='defenders')
    assert np.sum(moves) == 4, "Expected exactly four legal moves for the defenders, but found something else."

    b = TaflBoard(tb['legal_move_test_10_5'])
    moves = all_legal_moves(b.board, cache=b.cache, dirty_map=b.dirty_map, dirty_flags=b.dirty_flags, player= 'defenders')
    assert np.sum(moves) == 17, "Expected exactly 17 legal moves for the defenders, but found something else."

    print("All tests finished.")
