class Piece:
    def __init__(self, x: int, y:int, player: int):
        self.x = x
        self.y = y
        self.player = player


class GameWorld:
    def __init__(self, size: int):
        if size < 3 or size > 10:
            raise Exception("Gameworld size must be between 3 and 10")
        self.size = size
        self.world = [[Piece(x, y, -1) for y in range(size)] for x in range(size)]

    def check_valid_point(self, point: tuple[int, int]):
        return self.size > point[0] >= 0 and self.size > point[1] >= 0

    def neighbors_of(self, x: int, y: int):
        neighbors: list[tuple[int, int]] = [(x, y-1), (x-1, y+1), (x-1, y), (x+1, y), (x-1, y+1), (x, y+1)]
        return list(filter(self.check_valid_point, neighbors))

    def neighbors_of_piece(self, p: Piece):
        return self.neighbors_of(p.r, p.c)

    def place_piece(self, x: int, y: int, player: int):
        if not self.check_valid_point((x, y)):
            raise Exception("Cannot place piece outside board range")
        if self.world[x][y].player != -1:
            raise Exception("Cannot overwrite a populated location")
        piece = Piece(x, y, player)
        self.world[x][y] = piece

    def is_player_n(self, player: int):
        def check(p: Piece):
            return p.player == player
        return check