import numpy as np
import cv2
import argparse
import os
import sys


def parse_arguments(argv):
    """
        argument explaining

        Args: 
            (mode) : 
            (string) img : input file name. please pass the absolute path.
                           main function will check your data input type and it will raise error if there is an issue.
            (string) weights :  model weight file name. please pass the absolute path(dcm, png, nii, hdr, ...) 
            (string) save_path : output mask will be saved in this folder. 
                                Mask file name will be same as input file name. 

        
            
        """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    return parser.parse_args()


def main(args):
    sys.path.append('../')
    

    img_path = os.path.abspath(args.img)
    
    #Checker.check_input_type(img_path, 'jpg')

    if args.mode == 'polyp_segmentation':
        from Endoscopy import PolypSegmentation
        polyp_seg = PolypSegmentation()
        polyp_seg.init(args.weights)
        out = polyp_seg.predict(img_path)
        print(args.save_path + args.img.split('/')[-1])
        out = out.squeeze().squeeze()
        cv2.imwrite(args.save_path + args.img.split('/')[-1], out)
        
if __name__ == '__main__':
   argv = parse_arguments(sys.argv[1:])
   main(argv)
    
