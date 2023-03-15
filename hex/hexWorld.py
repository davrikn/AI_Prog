from gameWorld import GameWorld, Piece
from game import Game

class HexWorld(Game):
    def __init__(self, size: int):
        super(HexWorld, self).__init__(size=size)
