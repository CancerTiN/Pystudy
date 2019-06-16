# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import sys
import pygame
import random

def main():
    pygame.init()
    width = 640
    height = 480
    size = (width, height)
    screen = pygame.display.set_mode(size)
    color = (0, 0, 0)

    ball = pygame.image.load('cat.png')
    ballrect = ball.get_rect()

    speed = [random.randint(1, 9), random.randint(1, 9)]
    clock = pygame.time.Clock()
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        ballrect = ballrect.move(speed)
        if ballrect.left < 0:
            speed[0] = random.randint(1, 9)
        if ballrect.right > width:
            speed[0] = - random.randint(1, 9)
        if ballrect.top < 0 or ballrect.bottom > height:
            speed[1] = random.randint(1, 9)
        if ballrect.bottom > height:
            speed[1] = - random.randint(1, 9)
        screen.fill(color)
        screen.blit(ball, ballrect)
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
