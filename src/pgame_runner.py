import pandas as pd
import pygame
import os
import sys
from configparser import ConfigParser
import argparse

config = ConfigParser()
config.read('config.cfg')


class Runner(object):
    def __init__(self, x_max, x_min, y_max, y_min, width, height, scaled_base_path):
        self.x_max = x_max
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.screen = None
        self.point_num = 0
        self.width = width
        self.height = height
        self.scaled_base_path = scaled_base_path
        self.next_path = False

    def pgame_wait(self):
        """ Wait until user press a key. """
        pygame.display.update()
        print("Press button to continue ..")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    return event.key

    def read_coordinates(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((255, 255, 255))

        count = 0
        for k in range(12, 18):
            csv_file = '%s_%d.csv' % (self.scaled_base_path, k)
            count += 1
            df = pd.read_csv(csv_file, index_col=0)
            points = [(row['X(m)'], row['Y(m)']) for idx, row in df.iterrows()]
            points = [(point[0] * 500, -point[1] * 500) for point in points]
            self.draw_lines(points)

    def update_state(self):
        self.point_num += 1
        if self.point_num > 150:
            self.point_num = 0
            self.pgame_wait()
            self.screen.fill((255, 255, 255))
        if self.next_path:
            # self.pgame_wait()
            self.screen.fill((255, 255, 255))
            self.next_path = False

    def draw_lines(self, points, black=(0, 0, 0)):
        count = 0
        start_point = (-points[0][0] + self.x_max, points[0][1] + self.y_max)

        for (x, y) in points:
            if_draw = True
            end_point = (-x + self.x_max, y + self.y_max)
            pygame.draw.line(self.screen, black, start_point, end_point, 4)
            pygame.display.update()
            start_point = end_point
            if if_draw:
                # start point
                if count == 0:
                    self.update_state()
                self.update_state()
                # last point
                if count == len(points) - 1:
                    self.update_state()
            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='sliver-maestro')
    parser.add_argument('-rp', '--rootpath')
    args = parser.parse_args()
    root_path = args.rootpath
    if not root_path:
        root_path = os.getcwd()

    x_max = int(config['pgame']['x_max'])
    x_min = int(config['pgame']['x_min'])

    y_max = int(config['pgame']['y_max'])
    y_min = int(config['pgame']['y_min'])

    width = (x_max - x_min) * 3
    height = y_max - y_min

    scaled_base_path = os.path.join(root_path, config['generate_motion']['scaled_base_path'])
    runner = Runner(x_max=x_max, x_min=x_min, y_max=y_max, y_min=y_min, width=width, height=height,
                    scaled_base_path=scaled_base_path)

    runner.read_coordinates()
    pygame.display.quit()
    pygame.quit()
