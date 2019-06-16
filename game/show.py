# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import sys
import pygame

def main():
    pygame.init()
    size = (320, 240)
    screen = pygame.display.set_mode(size)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

    pygame.quit()

if __name__ == '__main__':
    main()
