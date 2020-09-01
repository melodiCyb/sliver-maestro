import os


def create_folders():
    if not os.path.exists('data'):
        os.mkdir('data')
        if os.path.exists('data'):
            print("data folder is created...")

    if not os.path.exists('data/input'):
        os.mkdir('data/input')
        if os.path.exists('data/input'):
            print("data/input folder is created...")        

    if not os.path.exists('data/output'):
        os.mkdir('data/output')
        if os.path.exists('data/output'):
            print("data/output folder is created...")

    if not os.path.exists('image'):
        os.mkdir('image')
        if os.path.exists('image'):
            print("image folder is created...")

    if not os.path.exists('weights'):
        os.mkdir('weights')
        if os.path.exists('weights'):
            print("weights folder is created...")

    if not os.path.exists('model'):
        os.mkdir('model')
        if os.path.exists('model'):
            print("model folder is created...")
