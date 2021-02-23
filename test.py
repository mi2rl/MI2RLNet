import argparse
import sys
import os


##### test.py guide lines

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--input_path', '-i', type=str, default=None)
    parser.add_argument('--weights', '-w', type=str, default=None)
    parser.add_argument('--save_path', '-o', type=str, default=None)

    return parser.parse_args()


def main(args):
    input_path = os.path.abspath(args.input_path)
    save_path = os.path.abspath(args.save_path)

    if args.mode == 'blackblood_segmentation':
        from medimodule.Brain.module import BlackbloodSegmentation
        ### scratch
        bbs = BlackbloodSegmentation()
        ### fine-tune
        #bbs = BlackbolldSegmentation('/weights_path')
        bbs.predict(input_path, save_path)
        
        


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
