"""Exhaustive checks on the game rules, with no simulator involved.

The rules decide which piece the arm is told to pick up, so an error here is
invisible in simulation -- the robot executes an illegal move perfectly. Running
thousands of random games offline costs milliseconds and catches what watching a
demo cannot.

.. code-block:: bash

    python lab/scripts/test_chess_rules.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

# Imported by path rather than through the package, whose ``__init__`` pulls in Isaac
# Lab. Neither module needs it, so the rules stay testable with a bare interpreter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "robochess"
                       / "tasks" / "manager_based" / "chess"))

from board import PLAYABLE_SCENARIOS, make_layout  # noqa: E402
from chess_rules import Position, opponent, scenario_variant  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def start(scenario: str) -> Position:
    return Position.from_layout(make_layout(scenario))


# --------------------------------------------------------------------- opening moves
def test_opening_positions() -> None:
    """The documented starting position and its opening moves, variant by variant."""
    p = start("1d")
    kinds = [(piece.color, piece.kind, piece.square[0]) for piece in p.pieces]
    check(kinds == [("white", "king", 0), ("white", "knight", 1), ("white", "rook", 2),
                    ("black", "rook", 5), ("black", "knight", 6), ("black", "king", 7)],
          f"1d start position wrong: {kinds}")

    moves = {str(m) for m in p.legal_moves()}
    # King on cell 0 can only step to 1, which its own knight occupies -> no king move.
    # Knight on 1 jumps to 3 (2 is its own rook, but the knight jumps *over*).
    # Rook on 2 slides to 3 and 4, and captures on 5.
    check("1,0-3,0" in moves, f"1d knight should jump from 1 to 3: {sorted(moves)}")
    check("2,0-3,0" in moves and "2,0-4,0" in moves, f"1d rook slide missing: {sorted(moves)}")
    check("2,0x5,0" in moves, f"1d rook should capture the black rook on 5: {sorted(moves)}")
    check("2,0-6,0" not in moves, "1d rook slid straight through an enemy piece")
    check(not any(m.origin == (0, 0) for m in p.legal_moves()), "1d king moved onto its own knight")

    p = start("3x3")
    check(len(p.legal_moves()) == 3, f"hexapawn should open with 3 pawn pushes, got {len(p.legal_moves())}")
    check(all(m.captured is None for m in p.legal_moves()), "hexapawn opened with a capture")

    p = start("minichess")
    white = [(pc.kind, pc.square) for pc in p.pieces if pc.color == "white"]
    black = [(pc.kind, pc.square) for pc in p.pieces if pc.color == "black"]
    check(("king", (0, 0)) in white and ("king", (3, 3)) in black,
          f"minichess kings should be mirrored: {white} / {black}")
    check(len(p.legal_moves()) > 0, "minichess has no opening move")


def test_knight_jumps_over_blockers() -> None:
    """The 1D knight is defined as jumping, so a piece in between must not block it."""
    p = start("1d")
    knight = next(i for i, pc in enumerate(p.pieces) if pc.color == "white" and pc.kind == "knight")
    targets = {m.target for m in p.legal_moves() if m.piece == knight}
    check((3, 0) in targets, f"knight failed to jump over the rook on cell 2: {targets}")


def test_pawn_cannot_capture_forwards() -> None:
    p = start("3x3")
    p = p.apply(next(m for m in p.legal_moves() if m.origin == (0, 0)))   # white a-pawn to (0,1)

    # Black's a-pawn on (0,2) now faces a white pawn head-on. It cannot advance, and a
    # pawn may not capture forwards, so that pawn has no move at all -- the other two
    # do. Any move generated for it would mean forward capture had slipped through.
    stuck = [m for m in p.legal_moves() if m.origin == (0, 2)]
    check(not stuck, f"blocked pawn should have no move, got {[str(m) for m in stuck]}")
    # Three in total: the a-pawn is stuck, the b-pawn can advance or take diagonally,
    # and the c-pawn can advance.
    check(len(p.legal_moves()) == 3, f"black should have exactly 3 pawn moves, got {len(p.legal_moves())}")

    forwards = [m for m in p.legal_moves() if m.captured is not None and m.origin[0] == m.target[0]]
    check(not forwards, f"a pawn captured straight ahead: {[str(m) for m in forwards]}")

    # ...but the diagonal capture onto that same white pawn is legal.
    diagonals = [m for m in p.legal_moves() if m.captured is not None]
    check(len(diagonals) == 1 and diagonals[0].target == (0, 1),
          f"expected one diagonal capture onto (0,1), got {[str(m) for m in diagonals]}")


def test_check_is_respected() -> None:
    """No legal move may leave one's own king attacked."""
    for scenario in ("1d", "minichess"):
        position = start(scenario)
        rng = random.Random(7)
        for _ in range(40):
            moves = position.legal_moves()
            if not moves or position.result():
                break
            for move in moves:
                check(not position.apply(move).is_check(position.side_to_move),
                      f"{scenario}: legal move {move} left own king in check")
            position = position.apply(rng.choice(moves))


def test_capture_bookkeeping() -> None:
    """A captured piece leaves the board, keeps its index, and frees its square."""
    for scenario in PLAYABLE_SCENARIOS:
        position = start(scenario)
        rng = random.Random(11)
        for _ in range(60):
            moves = position.legal_moves()
            if not moves or position.result():
                break
            captures = [m for m in moves if m.captured is not None]
            move = rng.choice(captures or moves)
            before = sum(p.alive for p in position.pieces)
            after_pos = position.apply(move)
            after = sum(p.alive for p in after_pos.pieces)
            expected = before - (1 if move.captured is not None else 0)
            check(after == expected, f"{scenario}: {before} pieces -> {after}, expected {expected}")
            if move.captured is not None:
                check(not after_pos.pieces[move.captured].alive,
                      f"{scenario}: captured piece {move.captured} still on the board")
            check(after_pos.at(move.target) == move.piece,
                  f"{scenario}: mover is not standing on its target square")
            check(after_pos.at(move.origin) is None,
                  f"{scenario}: origin square {move.origin} still occupied")
            position = after_pos


def test_no_two_pieces_share_a_square() -> None:
    for scenario in PLAYABLE_SCENARIOS:
        rng = random.Random(3)
        for _ in range(50):
            position = start(scenario)
            for _ in range(80):
                moves = position.legal_moves()
                if not moves or position.result():
                    break
                position = position.apply(rng.choice(moves))
                squares = [p.square for p in position.pieces if p.alive]
                check(len(squares) == len(set(squares)),
                      f"{scenario}: two pieces on one square: {squares}")
                check(all(position.on_board(s) for s in squares),
                      f"{scenario}: a piece left the board: {squares}")


def test_promotion() -> None:
    """A minichess pawn reaching the far rank becomes a queen."""
    rng = random.Random(5)
    seen = False
    for _ in range(400):
        position = start("minichess")
        for _ in range(60):
            moves = position.legal_moves()
            if not moves or position.result():
                break
            promotions = [m for m in moves if m.promotion]
            move = rng.choice(promotions or moves)
            if move.promotion:
                after = position.apply(move)
                check(after.pieces[move.piece].kind == "queen",
                      f"promoted piece is a {after.pieces[move.piece].kind}, not a queen")
                seen = True
                break
            position = position.apply(move)
        if seen:
            break
    check(seen, "no promotion occurred in 400 random minichess games")


def test_games_terminate() -> None:
    """Random games reach a defined result rather than running forever."""
    for scenario in PLAYABLE_SCENARIOS:
        results = Counter()
        rng = random.Random(19)
        for _ in range(300):
            position = start(scenario)
            for _ in range(200):
                if position.result() is not None:
                    break
                moves = position.legal_moves()
                if not moves:
                    break
                position = position.apply(rng.choice(moves))
            results[position.result()] += 1
        undecided = results[None]
        check(undecided < 300 * 0.5,
              f"{scenario}: {undecided}/300 random games did not finish within 200 plies")
        print(f"    {scenario:10s} outcomes over 300 random games: "
              + ", ".join(f"{k or 'unfinished'}={v}" for k, v in sorted(results.items(), key=lambda kv: str(kv[0]))))


def test_checkmate_is_really_mate() -> None:
    """When the result names a winner, the loser is in check with no way out."""
    rng = random.Random(23)
    checked = 0
    for scenario in ("1d", "minichess"):
        for _ in range(300):
            position = start(scenario)
            for _ in range(120):
                result = position.result()
                if result is not None:
                    if result in ("white", "black"):
                        loser = opponent(result)
                        check(position.side_to_move == loser,
                              f"{scenario}: {result} won but it is {position.side_to_move} to move")
                        check(position.is_check(loser), f"{scenario}: {loser} lost without being in check")
                        check(not position.legal_moves(), f"{scenario}: {loser} lost but still has moves")
                        checked += 1
                    break
                moves = position.legal_moves()
                if not moves:
                    break
                position = position.apply(rng.choice(moves))
    check(checked > 0, "no checkmate was reached in any random game")
    print(f"    verified {checked} checkmates")


def test_scenario_variant_mapping() -> None:
    for scenario in PLAYABLE_SCENARIOS:
        variant = scenario_variant(scenario)
        position = start(scenario)
        check(position.variant == variant, f"{scenario} built variant {position.variant}, expected {variant}")
    for scenario in ("pieces", "4x4", "8x8"):
        try:
            scenario_variant(scenario)
        except ValueError:
            continue
        FAILURES.append(f"'{scenario}' should not be playable as a game")


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"  {test.__name__}")
        test()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"\nAll {len(tests)} rule checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
