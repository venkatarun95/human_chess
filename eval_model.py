from calendar import c
import chess
import sys
from networkx import number_of_isolates
import torch

from data_loader import to_halfkp
from train import HumanNNUE

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 eval_model.py <model_file> <data_file>")
        sys.exit(1)

    device = "cpu"

    print(f"Loading model {sys.argv[1]}")
    model = torch.load(sys.argv[1])
    model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    with open(sys.argv[2], 'r') as file:
        top_1, top_3, engine, total = 0, 0, 0, 0
        for line in file:
            if line[0] == "#":
                continue
            try:
                eval, engine_moves, human_move, result, fen, time_control = line.strip().split(",")
            except:
                continue

            human_move = chess.Move.from_uci(human_move)

            board = chess.Board(fen)
            indices = to_halfkp(fen, chess.Move.from_uci(engine_moves.split(' ')[0]))
            num_features = 2 * 64 * 64 * 10 + 64 * 64
            indices = torch.sparse_coo_tensor([indices], # type: ignore
                                              [True] * len(indices),
                                              size=(num_features,), dtype=torch.float)
            indices = indices.to(device)
            indices = indices.to_dense()

            pred = model(indices)
            moves = []
            for move in board.legal_moves:
                move_idx = move.from_square * 64 + move.to_square
                # print(move, pred[0][move_idx].item())
                moves.append((move, pred[move_idx].item()))

            moves.sort(key=lambda x: x[1], reverse=True)

            if moves[0][0] == human_move:
                top_1 += 1
            if human_move in [move[0] for move in moves[:3]]:
                top_3 += 1
            if chess.Move.from_uci(engine_moves.split(' ')[0]) == moves[0][0]:
                engine += 1
            total += 1

            if total % 1000 == 0:
                print(f"Accuracy: {top_1 / total}, top-3: {top_3 / total} engine: {engine / total}")

        print(f"Accuracy: {top_1 / total}, top-3: {top_3 / total} engine: {engine / total}")

