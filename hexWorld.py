from gameWorld import GameWorld, Piece



class HexWorld(GameWorld):
    def check_win(self, player: int):
        # Check if the player has at least one piece on every layer from start to finish
        for i in range(self.size):
            check_list = self.world[i] if player == 1 else [x[i] for x in self.world]
            if len(list(filter(self.is_player_n(player), check_list))) > 0:
                pass
            else:
                return False
        nodes = list(filter(self.is_player_n(player), self.world[i] if player == 1 else [x[i] for x in self.world]))
        frontier = list()

        def is_in_frontier(node: Piece):
            for front in frontier:
                if front[0] == node.x and front[1] == node.y:
                    return True
            return False

        while (len(nodes) > 0):
            node = nodes.pop()
            frontier.append((node.x, node.y))
            neighbors = list(filter(is_in_frontier, list(filter(self.is_player_n(player), self.neighbors_of_piece(node)))))

        # TODO complete search


