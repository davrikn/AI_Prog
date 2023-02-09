import pygame
import pygame.gfxdraw
from hexWorld import HexWorld
from math import *


class HexGame(HexWorld):
    def __init__(self, hex_world: HexWorld):
        super().__init__(hex_world.size)
        screen_width = 750
        screen_height = 500

        self.hex_world = hex_world
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
                else:
                    # print(event)
                    pass
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
        # white_hex = HexSprite(color="red")
        blue_hex = pygame.image.load('ui/hex_blue_50x50.png')
        # blue_hex = HexSprite(color="blue")
        red_hex = pygame.image.load('ui/hex_red_50x50.png')
        # red_hex = HexSprite(color="red")

        if color == "white":
            self.screen.blit(white_hex, position)
        if color == "blue":
            self.screen.blit(blue_hex, position)
        if color == "red":
            self.screen.blit(red_hex, position)

    def place_piece(self, x: int, y: int, player: int):
        self.hex_world.world[x][y] = player


# Done! Time to quit.
pygame.quit()


class HexSprite(pygame.sprite.Sprite):
    def __init__(self, color):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)

        # Create an image of the block, and fill it with a color.
        # This could also be an image loaded from the disk.
        if color == "red":
            self.image = pygame.image.load('ui/hex_red_50x50.png')
        if color == "blue":
            self.image = pygame.image.load('ui/hex_blue_50x50.png')
        if color == "white":
            self.image = pygame.image.load('ui/hex-white.png')

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y
        self.rect = self.image.get_rect()


def draw_ngon(Surface, color, n, radius, position):
    pi2 = 2 * 3.14

    for i in range(0, n):
        pygame.draw.line(Surface, color, position,
                         (cos(i / n * pi2) * radius + position[0], sin(i / n * pi2) * radius + position[1]))

    return pygame.draw.lines(Surface,
                             color,
                             True,
                             [(cos(i / n * pi2) * radius + position[0], sin(i / n * pi2) * radius + position[1]) for i
                              in range(0, n)])
