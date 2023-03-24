import itertools
import random

import configs
from hex.hexAgent import HexAgent
from hex.hexWorld import HexWorld


class Tournament:

    def __init__(self, agents: list[HexAgent], UI: bool, G=25):
        self.agents = agents
        self.G = G
        self.M = len(agents)
        self.UI = UI

    def run_tournament(self):
        for agent in self.agents:
            for param in agent.model.parameters():
                print(param.data)
            print('---------------------------------------------------------------')

        matches = self.create_match_ups()
        random.shuffle(matches)
        for match_up in matches:
            self.play_out_series(match_up[0], match_up[1])
        self.print_final_scoreboard()


    def play_out_series(self, agent1: HexAgent, agent2: HexAgent):
        agent1_curr_wins = 0
        agent2_curr_wins = 0
        starting_player = 1
        for i in range(self.G):
            winner = self.play_game(agent1, agent2, starting_player)
            if winner == 1:
                agent1_curr_wins += 1
                agent1.wins += 1
                agent2.losses += 1
            else:
                agent2_curr_wins += 1
                agent1.losses += 1
                agent2.wins += 1

            if winner != starting_player:
                print("debug")
            starting_player = 2 if starting_player == 1 else 1
        if agent1_curr_wins > agent2_curr_wins:
            print(f'{agent1.name} won vs {agent2.name} with {agent1_curr_wins}/{self.G} wins')
        else:
            print(f'{agent2.name} won vs {agent1.name} with {agent2_curr_wins}/{self.G} wins')

    def play_game(self, agent1: HexAgent, agent2: HexAgent, starting_player: int) -> int:
        game = HexWorld(size=configs.size)
        finished = False
        agents_turn = starting_player
        winning_agent = 0
        while not finished:
            if agents_turn == 1:
                game = agent1.perform_move(game)
                finished = game.is_final_state()
                if finished:
                    winning_agent = agents_turn
            else:
                game = agent2.perform_move(game)
                finished = game.is_final_state()
                if finished:
                    winning_agent = agents_turn
            if self.UI:
                pass
        return winning_agent

    def create_match_ups(self):
        matches = self.unique_combinations()
        return matches

    def unique_combinations(self) -> list[tuple[HexAgent, HexAgent]]:
        return list(itertools.combinations(self.agents, 2))

    def print_final_scoreboard(self):
        self.agents = sorted(self.agents, reverse=True, key=lambda agent: agent.wins)
        print('\nAgent name \t\t\t\t Wins \t\t Losses \t\t Win rate')
        for agent in self.agents:
            print(f'{agent.name} \t\t {agent.wins} \t\t {agent.losses} \t\t {agent.wins/(agent.wins +agent.losses)}%')

