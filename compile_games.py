import chess
import chess.pgn
import io
import re
import subprocess
import sys
from typing import Optional, Tuple
import argparse

class UciInterface:
    def __init__(self):
        path = "../../Stockfish/src/stockfish"

        self.process = subprocess.Popen(
            path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Use text mode (not binary)
        )

        assert self.process is not None
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        self.process.stdin.write(f'isready')
        self.process.stdin.flush()

    def evaluate(self, fen: str, target_depth: int) -> Tuple[int, str]:
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        pattern = re.compile(
            r'(\b[a-zA-Z]+)\s+(\d+)'  # Matches '<characters> <numbers>'
            r'|'  # Logical OR
            r'(score cp)\s+(-?\d+)'  # Matches 'score cp <±numbers>'
            r'|'  # Logical OR
            r'(score mate)\s+(-?\d+)'  # Matches 'score mate <numbers>'
            r'|'  # Logical OR
            r'(pv)\s+((?:(?:[a-h][1-8][qrbn]?){2,}\s+)+)'  # Matches 'pv <chess moves>+'
            #r'(pv)\s+((?:[a-h][1-8]){2,}\s+)+'  # Matches 'pv <chess moves>+'
        )

        output_dict = {}
        self.process.stdin.write(f'position fen {fen}\n')
        self.process.stdin.write(f'go depth {target_depth}\n')
        self.process.stdin.flush()

        all_lines = []
        found_score, found_move = None, None
        while True:
            line = self.process.stdout.readline()
            if not line:
                assert False, "Engine died"
                break


            if line.startswith("info"):
                all_lines += [line]
                depth, score, mate, pv = None, None, False, None
                for match in pattern.finditer(line):
                    if match.group(1) == "depth":
                        depth = int(match.group(2))
                    elif match.group(3) == "score cp":
                        score = int(match.group(4))
                    elif match.group(5) == "score mate":
                        if match.group(6) == "0":
                            print(f"Warning got 'score mate 0': {fen}")
                            continue
                        mate = True
                        if int(match.group(6)) > 0:
                            score = 5000
                        else:
                            score = -5000
                    elif match.group(7) == "pv":
                        pv = match.group(8)
                        if pv[-1] == "\n":
                            pv = pv[:-1]

                if (depth is not None and depth >= target_depth) or mate:
                    assert score is not None
                    if mate:
                        found_score = score
                        found_move = "-"
                    assert pv is not None
                    found_score = score
                    found_move = pv
                    self.process.stdin.write(f'stop\n')
                    self.process.stdin.flush()

            elif line.startswith("bestmove"):
                break
        assert found_score is not None, "Engine did not return a score"
        assert found_move is not None, "Engine did not return a move"
        return found_score, found_move

if __name__ == "__main__":
    # Read command line arguments: ELO bounds, time control bounds (lower and
    # upper) to filter games
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--max-elo", type=int, default=2200)
    parser.add_argument("--min-time_control", type=int, default=300)
    parser.add_argument("--max-time_control", type=int, default=900)
    args = parser.parse_args()

    print(f"# Elo: [{args.min_elo}, {args.max_elo}], Time control: [{args.min_time_control}, {args.max_time_control}]")

    uci = UciInterface()

    num_games = 0
    # Read games from stdin
    while True:
        game = chess.pgn.read_game(sys.stdin)
        if game is None:
            break

        # Does it have all the required headers?
        if "headers" not in game.__dict__ or \
            not all(header in game.headers for header in ["WhiteElo", "BlackElo", "TimeControl", 
                                                          "Result"]):
            continue

        try:
            # Elo
            white_elo = int(game.headers["WhiteElo"])
            black_elo = int(game.headers["BlackElo"])

            # Time
            time_control = int(game.headers["TimeControl"].split("+")[0])
        except:
            continue
        
        # Filter games
        if min(white_elo, black_elo) < args.min_elo \
            or max(white_elo, black_elo) > args.max_elo \
            or time_control < args.min_time_control \
            or time_control > args.max_time_control:
            continue

        num_games += 1

        board = game.board()
        for human_move in game.mainline_moves():
            fen = board.fen()

            try:
                eval, engine_moves = uci.evaluate(fen, 8)
            except:
                continue
            print(eval, engine_moves, human_move, game.headers["Result"], fen, 
                  game.headers["TimeControl"], sep=",")
            
            board.push(human_move)
            # Check if game is already over
            if board.is_game_over():
                break