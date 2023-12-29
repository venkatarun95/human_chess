Database files: https://database.lichess.org/
compile_games.py reads the database file and outputs positions along with Stockfish evaluation. Example command `zstdcat <database file name.pgn.zst> | python3 compile_games.py --min-elo 2000 --max-elo 2200 >test2.games`
train.py trains the neural net
eval.py evaluates its accuracy
