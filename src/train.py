import os
import sys

base_path = os.getcwd().split('sliver-maestro')[0]
base_path = os.path.join(base_path, "sliver-maestro")
sys.path.insert(1, base_path)
from src.draw_model import *

config = ConfigParser()
config.read('config.cfg')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='sliver-maestro')
    parser.add_argument('-phase', '--phase')  # 'train', 'test'
    parser.add_argument('-category', '--category')

    args = parser.parse_args()
    phase = args.phase
    category = args.category


    model = DRAW(category)
    print("training...")
    model.start(phase='train')
