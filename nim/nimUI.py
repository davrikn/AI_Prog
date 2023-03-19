import pygame
import pygame.gfxdraw

import configs
from monteCarlo import MonteCarlo
from nim.NimWorld import NimSimWorld
from nim.nimModel import NimModel
from math import *


class NimUI(NimSimWorld):
    def __init__(self, size: int = configs.size, model: NimModel = NimModel(configs.size, '../model_dicts')):
        super().__init__(size)
        self.model = model
        self.screen_width = 300
        self.screen_height = 400

        self.running = True
        self.game_over = False

        pygame.init()

        # Game window
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])

        # Fill the background with white
        self.screen.fill((255, 255, 255))

    def start_game(self):
        self.draw_board()

        while self.running:
            for event in pygame.event.get():
                self.draw_board()
                if self.player == 1:
                    self.pick_from_pile()
                else:
                    self.computer_move()

                pygame.display.flip()

    # Initializes an empty game board
    def draw_board(self):
        white = (255, 255, 255)

        self.screen.fill(white)
        start_x = self.screen_width * 0.2
        start_y = self.screen_height * 0.2
        for i in range(self.size):
            for j in range(self.board[i]):
                rect = pygame.Rect(start_x + j * 20, start_y + i * 40, 5, 30)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 3)

    def pick_from_pile(self):
        pile = int(input("What pile?"))
        sticks = int(input("How many sticks?"))
        action = "0"*self.size
        action = action[:pile]+str(sticks)+action[pile+1:]
        print(action)
        self.apply(action)

    def computer_move(self):
        actions = self.model.classify(self.state())
        for action in actions:
            try:
                self.apply(action)
                break
            except:
                pass
        self.draw_board()

    def do_move(self):
        if self.player == -1:
            print("computer move..")
            print("curr state: ", self.board)
            self.computer_move()
            print("next state: ", self.board)
        else:
            self.pick_from_pile()


if __name__ == "__main__":
    model = NimModel(configs.size, '../model_dicts')
    ui = NimUI()
    ui.start_game()


    pygame.quit()
    exit(0)

# Done! Time to quit.
pygame.quit()
