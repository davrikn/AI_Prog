import pygame
import pygame.gfxdraw
from nim.NimSimWorld import NimSimWorld
from math import *


class NimUI(NimSimWorld):
    def __init__(self, sim_world: NimSimWorld):
        super().__init__(sim_world.size)
        self.screen_width = 300
        self.screen_height = 300

        self.sim_world = sim_world
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
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_0:
                        self.pick_piece_from_pile(0)
                    if event.key == pygame.K_1:
                        self.pick_piece_from_pile(1)
                    if event.key == pygame.K_2:
                        self.pick_piece_from_pile(2)
                    if event.key == pygame.K_3:
                        self.pick_piece_from_pile(3)
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    # print(event)
                    pass
                # Update display
                self.draw_board()
                if self.is_final_state():
                    print("WInner winner chicken dinner")

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

    def get_input_from_user(self):
        row_input = int(input("enter row number"))
        if len(self.board) >= row_input >= 0:
            self.pick_piece_from_pile(row_input)
# Done! Time to quit.
pygame.quit()
