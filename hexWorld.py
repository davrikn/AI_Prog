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
        frontier = list(filter(self.is_player_n(player), self.world[0] if player == 1 else [x[0] for x in self.world]))
        visited = list()

        def is_in_visited(node: Piece):
            for visit in visited:
                if visit[0] == node.x and visit[1] == node.y:
                    return True
            return False

        def is_not_in_visited(node: Piece):
            return not is_in_visited(node)

        def is_in_frontier(node: Piece):
            for front in frontier:
                if front[0] == node.x and front[1] == node.y:
                    return True
            return False

        def is_not_in_frontier(node: Piece):
            return not is_in_frontier(node)

        while len(frontier) > 0:
            node = frontier.pop()
            if player == 1:
                if node.y == self.size - 1:
                    return True
            else:
                if node.x == self.size - 1:
                    return True

            visited.append((node.x, node.y))
            neighbors = list(filter(is_not_in_frontier, list(filter(is_not_in_visited, list(filter(self.is_player_n(player), self.neighbors_of_piece(node)))))))
            frontier.extend(neighbors)

        return False


