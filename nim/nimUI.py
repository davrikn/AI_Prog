import pygame
import pygame.gfxdraw

from monteCarlo import MonteCarlo
from nim.NimSimWorld import NimSimWorld
from math import *


class NimUI(NimSimWorld):
    def __init__(self, sim_world: NimSimWorld):
        super().__init__(sim_world.size)
        self.screen_width = 300
        self.screen_height = 400

        self.sim_world = sim_world
        self.running = True
        self.game_over = False

        self.curr_player = 1
        self.computer = -1
        self.player = 1

        pygame.init()

        # Game window
        self.screen = pygame.display.set_mode([self.screen_width, self.screen_height])

        # Fill the background with white
        self.screen.fill((255, 255, 255))

    def start_game(self):
        self.draw_board()

        while self.running:

            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.do_move()

                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    # print(event)
                    pass
                # Update display
                self.draw_board()
                if self.sim_world.is_final_state():
                    if self.curr_player == self.computer:
                        print("You lost")
                    else:
                        print("You won")

                pygame.display.flip()

    # Initializes an empty game board
    def draw_board(self):
        white = (255, 255, 255)

        self.screen.fill(white)
        start_x = self.screen_width * 0.2
        start_y = self.screen_height * 0.2
        for i in range(self.size):
            for j in range(self.sim_world.board[i]):
                rect = pygame.Rect(start_x + j * 20, start_y + i * 40, 5, 30)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 3)

    def pick_from_pile(self):
        pile = int(input("What pile?"))
        sticks = int(input("How many sticks?"))

        self.sim_world.pick_piece_from_pile(pile - 1, sticks)

    def computer_move(self):
        next_game_state = MonteCarlo(root=self.sim_world, player=self.computer).run()
        self.sim_world.board = next_game_state.state.enumerate_state2()
        self.draw_board()

    def do_move(self):
        if self.curr_player == self.computer:
            print("computer move..")
            print("curr state: ", self.sim_world.board)
            self.computer_move()
            print("next state: ", self.sim_world.board)
            self.curr_player = self.player
        else:
            self.pick_from_pile()
            self.curr_player = self.computer


# Done! Time to quit.
pygame.quit()
