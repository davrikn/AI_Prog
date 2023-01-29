import pygame
import pygame.gfxdraw
from hexWorld import HexWorld
from math import *


class HexGame(HexWorld):
    def __init__(self, size: int):
        super().__init__(size)
        screen_width = 750
        screen_height = 500

        self.running = True
        self.player_won = False

        pygame.init()

        # Game window
        self.screen = pygame.display.set_mode([screen_width, screen_height])

        # Fill the background with white
        self.screen.fill((255, 255, 255))

    def start_game(self):
        self.draw_board()

        while self.running:
            # Did the user click the window close button?
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # Update display
                pygame.display.flip()

    # Initializes an empty game board
    def draw_board(self):
        start_x = 100
        start_y = 100
        for i in range(self.size):
            extra_indent = i * 25
            for j in range(self.size):
                piece = self.world[i][j]
                if piece.player == -1:
                    self.draw_hexagon(position=(start_x + extra_indent + (j * 50), start_y + (i * 37)), color="white")
                if piece.player == 0:
                    self.draw_hexagon(position=(start_x + extra_indent + (j * 50), start_y + (i * 37)), color="blue")
                if piece.player == 1:
                    self.draw_hexagon(position=(start_x + extra_indent + (j * 50), start_y + (i * 37)), color="red")

    # Draws a single Hexagon on the board
    def draw_hexagon(self, position: tuple[int, int], color: str):
        white_hex = pygame.image.load('ui/hex-white.png')
        blue_hex = pygame.image.load('ui/hex_blue_50x50.png')
        red_hex = pygame.image.load('ui/hex_red_50x50.png')

        if color == "white":
            self.screen.blit(white_hex, position)
        if color == "blue":
            self.screen.blit(blue_hex, position)
        if color == "red":
            self.screen.blit(red_hex, position)


# Done! Time to quit.
pygame.quit()
