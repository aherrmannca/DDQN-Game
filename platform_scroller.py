"""
Sample Python/Pygame Programs
Simpson College Computer Science
http://programarcadegames.com/
http://simpson.edu/computer-science/

From:
http://programarcadegames.com/python_examples/f.php?file=platform_jumper.py

Explanation video: http://youtu.be/BCxWJgN4Nnc

Part of a series:
http://programarcadegames.com/python_examples/f.php?file=move_with_walls_example.py
http://programarcadegames.com/python_examples/f.php?file=maze_runner.py
http://programarcadegames.com/python_examples/f.php?file=platform_jumper.py
http://programarcadegames.com/python_examples/f.php?file=platform_scroller.py
http://programarcadegames.com/python_examples/f.php?file=platform_moving.py
http://programarcadegames.com/python_examples/sprite_sheets/
"""

import pygame
from spritesheet_functions import SpriteSheet
import numpy as np
import pandas as pd
import random
import os, sys
import time
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt

# Run on google cloud without the need of opening a window
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# Global constants

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800


class Player(pygame.sprite.Sprite):
    """ This class represents the bar at the bottom that the player
        controls. """

    # -- Methods
    def __init__(self):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # This holds all the images for the animated walk left/right
        # of our player
        self.walking_frames_l = []
        self.walking_frames_r = []

        # What direction is the player facing?
        self.direction = "S"

        sprite_sheet = SpriteSheet("platformer_sprites.png")
        # Load all the right facing images into a list
        image = sprite_sheet.get_image(64*4+7, 0, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(64*5+5, 0, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(64*6+5, 0, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(64*7+5, 0, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(0+5, 64, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(64+5, 64, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(64*2+5, 64, 45, 64)
        self.walking_frames_r.append(image)
        image = sprite_sheet.get_image(64*3+5, 64, 45, 64)


        # Load all the right facing images, then flip them
        # to face left.
        image = sprite_sheet.get_image(64*4+5, 0, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(64*5+5, 0, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(64*6+5, 0, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(64*7+5, 0, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(0+5, 64, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(64+5, 64, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(64*2+5, 64, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)
        image = sprite_sheet.get_image(64*3+5, 64, 45, 64)
        image = pygame.transform.flip(image, True, False)
        self.walking_frames_l.append(image)

        self.stillR_frame = sprite_sheet.get_image(0+5, 64*8, 45, 64)
        self.stillL_frame = pygame.transform.flip(sprite_sheet.get_image(0+5, 64*8, 45, 64), True, False)

        # Set the image the player starts with
        self.image = sprite_sheet.get_image(0 + 18, 64*8, 28, 64)

        # Set a referance to the image rect.
        self.rect = self.image.get_rect()

        # Set speed vector of player
        self.change_x = 0
        self.change_y = 0

        # List of sprites we can bump against
        self.level = None

    def update(self):
        """ Move the player. """
        # Gravity
        self.calc_grav()

        # Move left/right
        self.rect.x += self.change_x

        pos = self.rect.x
        if self.direction == "R":
            frame = (pos // 30) % len(self.walking_frames_r)
            self.image = self.walking_frames_r[frame]
        elif self.direction == "L":
            frame = (pos // 30) % len(self.walking_frames_l)
            self.image = self.walking_frames_l[frame]
        elif self.direction == "RS":
            self.image = self.stillR_frame
        elif self.direction == "LS":
            self.image = self.stillL_frame

        # See if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right

        # See if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.created_blocks, False)
        for block in block_hit_list:
            # If we are moving right,
            # set our right side to the left side of the item we hit
            if self.change_x > 0:
                self.rect.right = block.rect.left
            elif self.change_x < 0:
                # Otherwise if we are moving left, do the opposite.
                self.rect.left = block.rect.right

        # Move up/down
        self.rect.y += self.change_y

        # Check and see if we hit anything
        block_hit_list = pygame.sprite.spritecollide(self, self.level.created_blocks, False)
        for block in block_hit_list:

            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # Stop our vertical movement
            self.change_y = 0

        block_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        for block in block_hit_list:

            # Reset our position based on the top/bottom of the object.
            if self.change_y > 0:
                self.rect.bottom = block.rect.top
            elif self.change_y < 0:
                self.rect.top = block.rect.bottom

            # Stop our vertical movement
            self.change_y = 0



    def calc_grav(self):
        """ Calculate effect of gravity. """
        if self.change_y == 0:
            self.change_y = 1
        else:
            self.change_y += .95

        # See if we are on the ground.
        if self.rect.y >= SCREEN_HEIGHT - self.rect.height and self.change_y >= 0:
            self.change_y = 0
            self.rect.y = SCREEN_HEIGHT - self.rect.height

    def jump(self):
        """ Called when user hits 'jump' button. """

        # move down a bit and see if there is a platform below us.
        # Move down 2 pixels because it doesn't work well if we only move down
        # 1 when working with a platform moving down.
        self.rect.y += 2
        platform_hit_list = pygame.sprite.spritecollide(self, self.level.platform_list, False)
        self.rect.y -= 2

        # If it is ok to jump, set our speed upwards
        if len(platform_hit_list) > 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.change_y = -17

        self.rect.y += 2
        block_hit_list = pygame.sprite.spritecollide(self, self.level.created_blocks, False)
        self.rect.y -= 2

        if len(block_hit_list) > 0:
            self.change_y = -17

    # Player-controlled movement:
    def go_left(self):
        """ Called when the user hits the left arrow. """
        self.change_x = -10
        self.direction = "L"

    def go_right(self):
        """ Called when the user hits the right arrow. """
        self.change_x = 10
        self.direction = "R"

    def stop(self):
        """ Called when the user lets off the keyboard. """
        self.change_x = 0
        if self.direction == "R":
            self.direction = "RS"
        elif self.direction == "L":
            self.direction = "LS"


class Platform(pygame.sprite.Sprite):
    """ Platform the user can jump on """

    def __init__(self, width, height, rgb=GREEN):
        """ Platform constructor. Assumes constructed with user passing in
            an array of 5 numbers like what's defined at the top of this
            code. """
        super().__init__()

        self.image = pygame.Surface([width, height])
        self.image.fill(rgb)

        self.rect = self.image.get_rect()


class Level(object):
    """ This is a generic super-class used to define a level.
        Create a child class for each level with level-specific
        info. """

    def __init__(self, player):
        """ Constructor. Pass in a handle to player. Needed for when moving platforms
            collide with the player. """
        self.platform_list = pygame.sprite.Group()
        self.end_list = pygame.sprite.Group()
        self.enemy_list = pygame.sprite.Group()
        self.created_blocks = pygame.sprite.Group()
        self.player = player

        # Background image
        self.background = None

    # Update everythign on this level
    def update(self):
        """ Update everything in this level."""
        self.platform_list.update()
        self.enemy_list.update()
        self.end_list.update()
        self.created_blocks.update()

    def draw(self, screen):
        """ Draw everything on this level. """

        # Draw the background
        screen.fill((66,140,244))

        # Draw all the sprite lists that we have
        self.end_list.draw(screen)
        self.platform_list.draw(screen)
        self.enemy_list.draw(screen)
        self.created_blocks.draw(screen)


# Create platforms for the level
class Level_01(Level):
    """ Definition for level 1. """

    def __init__(self, player):
        """ Create level 1. """

        # Call the parent constructor
        Level.__init__(self, player)

        # Outer walls of level
        barriers = [[20, SCREEN_HEIGHT, 0, 0],
                    [SCREEN_WIDTH, 20, 0, 0],
                    [184, SCREEN_HEIGHT, SCREEN_WIDTH-184, 0],
                    [200, 30, 20, SCREEN_HEIGHT-30]]

        # Finishing line
        end = [[210-184, SCREEN_HEIGHT, SCREEN_WIDTH-210, 0]]

        for platform in barriers:
            block = Platform(platform[0], platform[1], rgb=(77,83,91))
            block.rect.x = platform[2]
            block.rect.y = platform[3]
            block.play = self.player
            self.platform_list.add(block)

        for platform in end:
            block = Platform(platform[0], platform[1], rgb=(58,196,51))
            block.rect.x = platform[2]
            block.rect.y = platform[3]
            block.play = self.player
            self.end_list.add(block)

    def add_block(self, block_list, rgb=(127,119,43)):
        block = Platform(block_list[0], block_list[1], rgb=rgb)
        block.rect.x = block_list[2]
        block.rect.y = block_list[3]
        block.play = self.player
        self.created_blocks.add(block)

class button():
    def __init__(self, color, x, y, width, height, text=''):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self,win,outline=BLACK):
        #Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x-2,self.y-2,self.width+4,self.height+4),0)

        pygame.draw.rect(win, self.color, (self.x,self.y,self.width,self.height),0)

        if self.text != '':
            font = pygame.font.SysFont('comicsans', 60)
            text = font.render(self.text, 1, (0,0,0))
            win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

    def isOver(self, pos):
        #Pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True

        return False

# Returns the saved levels
def get_saved_levels():
    stored_levels = pd.read_csv('saved_levels.csv', sep='|', header=None,names=['A']) # Get already stored levels
    stored_levels_transformed = []
    for level in stored_levels.A.str.split(',').values:
        level_transform = []
        level = list(map(int, level))
        for i in range(int(len(level) / 4)):
            platform_transform = []
            platform_transform.append(level[i*4])
            platform_transform.append(level[i*4+1])
            platform_transform.append(level[i*4+2])
            platform_transform.append(level[i*4+3])
            level_transform.append(platform_transform)
        stored_levels_transformed.append(level_transform)

    return stored_levels_transformed

def main():
    """ Main Program """
    pygame.init()

    # Set the height and width of the screen
    size = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("Platformer Jumper")

    # Create the player
    player = Player()

    # Create all the levels
    level_list = []
    level_list.append( Level_01(player) )

    # Set the current level
    current_level_no = 0
    current_level = level_list[current_level_no]

    active_sprite_list = pygame.sprite.Group()
    player.level = current_level

    player.rect.x = 60
    player.rect.y = SCREEN_HEIGHT - player.rect.height - 30
    active_sprite_list.add(player)

    # Draw the buttons
    start_button = button((201,136,72), SCREEN_WIDTH-182, SCREEN_HEIGHT-202, 180, 100, text='Start')
    reset_button = button((170, 135, 46), SCREEN_WIDTH-182, SCREEN_HEIGHT-102, 180, 100, text='Reset')
    save_button = button((211, 180, 101), SCREEN_WIDTH-182, SCREEN_HEIGHT-302, 180, 100, text='Save')
    rand_lvl_button = button((145, 107, 10), SCREEN_WIDTH-182, SCREEN_HEIGHT-402, 180, 100, text='Random')

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # Whether or not we can draw, dependent on if the model is running
    allow_drawing = True
    mouse_dragging = False

    # Start and end of drag
    start_drag = []
    end_drag = []

    stored_levels_transformed = get_saved_levels()
    curr_stored_level =[]
    chooc = 100

    # -------- Main Program Loop -----------
    while not done:

        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()

            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    player.go_left()
                if event.key == pygame.K_RIGHT:
                    player.go_right()
                if event.key == pygame.K_UP:
                    player.jump()

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and player.change_x < 0:
                    player.stop()
                if event.key == pygame.K_RIGHT and player.change_x > 0:
                    player.stop()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.isOver(pos):
                    print('Clicked the start button')
                    allow_drawing = False
                elif rand_lvl_button.isOver(pos):
                    print('Creating random level')
                    # Reset the screen first
                    current_level.created_blocks.empty()
                    player.rect.x = 60
                    player.rect.y = SCREEN_HEIGHT - player.rect.height - 30
                    curr_stored_level = []

                    # Choose and create random level
                    level_num = random.randint(0, len(stored_levels_transformed)-1)
                    for platform in stored_levels_transformed[level_num]:
                        current_level.add_block(platform, rgb=(104,101,93))
                        curr_stored_level.extend(tuple(platform))
                elif save_button.isOver(pos):
                    print('Saving this level...')
                    saver = pd.DataFrame([curr_stored_level])
                    with open('saved_levels.csv', 'a') as f:
                        saver.to_csv(f, header=None, index=False)
                        print('Saved!')

                    # Get saved levels including the newly saved one
                    stored_levels_transformed = get_saved_levels()
                elif reset_button.isOver(pos):
                    print('You reset the screen')
                    current_level.created_blocks.empty()
                    player.rect.x = 60
                    player.rect.y = SCREEN_HEIGHT - player.rect.height - 30
                    curr_stored_level = []
                elif pos[0] > SCREEN_WIDTH - 204:
                    print('Can\'t draw in finish zone')
                elif pos[0] > 20 and pos[1] > 20 and allow_drawing == True:
                    mouse_dragging = True
                    start_drag = pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if mouse_dragging:
                    end_drag = pos
                    mouse_dragging = False
                    width = np.abs(start_drag[0] - end_drag[0])
                    height = np.abs(start_drag[1] - end_drag[1])
                    x = np.min([start_drag[0], end_drag[0]])
                    y = np.min([start_drag[1], end_drag[1]])
                    current_level.add_block([width, height, x, y], rgb=(104,101,93))
                    curr_stored_level.extend((width, height, x, y))


        # Update the player.
        active_sprite_list.update()

        # Update items in the level
        current_level.update()

        # If the player gets near the right side, shift the world left (-x)
        if player.rect.right > SCREEN_WIDTH-210:
            player.rect.x = 60
            player.rect.y = SCREEN_HEIGHT - player.rect.height - 30
            allow_drawing = True
            print("YOU WON!")

        if player.rect.bottom >= SCREEN_HEIGHT:
            player.rect.x = 60
            player.rect.y = SCREEN_HEIGHT - player.rect.height - 30
            allow_drawing = True
            print("YOU LOST!")

        # If the player gets near the left side, shift the world right (+x)
        if player.rect.left < 0:
            player.rect.left = 0


        # ALL CODE TO DRAW SHOULD GO BELOW THIS COMMENT
        current_level.draw(screen)
        active_sprite_list.draw(screen)
        start_button.draw(screen)
        reset_button.draw(screen)
        save_button.draw(screen)
        rand_lvl_button.draw(screen)

        # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT


        # Limit to 60 frames per second
        clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()

if __name__ == "__main__":
    main()
