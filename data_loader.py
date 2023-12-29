import chess
from networkx import number_of_isolates
from numpy import indices
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class ChessDataset(Dataset):
    def __init__(self, file_path: str, device: str):
        self.num_features = 2 * 64 * 64 * 10 + 64 * 64
        self.device = device
        self.data = self._read_fens(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        indices, human = self.data[idx]
        human = torch.tensor(human).to(self.device)

        return (indices.to_dense(), human.to(self.device))
    
    def _read_fens(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line[0] == "#":
                    continue
                try:
                    eval, engine_moves, human_move, result, fen, time_control = line.strip().split(",")
                except:
                    continue

                engine_move = chess.Move.from_uci(engine_moves.split(' ')[0])
                human_move = chess.Move.from_uci(human_move)
                human_move_idx = human_move.from_square * 64 + human_move.to_square
                
                if result == "1-0":
                    result_idx = 0
                elif result == "0-1":
                    result_idx = 1
                elif result == "1/2-1/2":
                    result_idx = 2
                else:
                    assert False
                
                indices = to_halfkp(fen, engine_move)
                indices = torch.sparse_coo_tensor([indices], # type: ignore
                                                  [True] * len(indices),
                                                  size=(self.num_features,), dtype=torch.float)
                indices = indices.to(self.device)

                data.append((indices, human_move_idx))

                if len(data) % 10000 == 0:
                    print(len(data))
                        
        return data
    
def to_halfkp(fen: str, engine_move: chess.Move):
        ''' Convert FEN to a HalfKP tensor '''
        board = chess.Board(fen)

        us = board.turn
        # Position of our king and opponent's king
        our_king = board.king(us)
        their_king = board.king(not us)
        assert our_king is not None
        assert their_king is not None

        # Indices in the sparse tensor we will create where pieces exist
        indices: List[int] = []
        # Iterate over all pieces in the board
        for square, piece in board.piece_map().items():
            for (king, koffset) in [(our_king, 0), (their_king, 64 * 64 * 10)]:
                piece_idx = koffset + king * 64 * 10 + square * 10 + (piece.piece_type - 1) * 2 + int(piece.color != us)
                indices.append(piece_idx)
        
        # Index for the move the engine recommends. A number between 0 and 64 * 64
        move_idx = engine_move.from_square * 64 + engine_move.to_square + 2 * 64 * 64 * 10
        indices.append(move_idx)

        # TODO: give engine eval as input

        return indices
    
if __name__ == "__main__":
    file_path = 'positions_2000-2200_300_900.games'
    dataset = ChessDataset(file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
