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
    parser.add_argument('--gpus', type=str, default='0')

    return parser.parse_args()


def main(args):
    input_path = os.path.abspath(args.input_path)
    save_path = os.path.abspath(args.save_path)
    
    ##### BRAIN EXAMPLES
    if args.mode == 'blackblood_segmentation':
        from medimodule.Brain import BlackbloodSegmentation
        ### scratch
        bbs = BlackbloodSegmentation()
        ### fine-tune
        #bbs = BlackbolldSegmentation('/weights_path')
        bbs.predict(input_path, save_path)
        

    ##### CHEST EXAMPLES
    ### Viewpoint Classifier Example (PA / Lateral / Others)
    elif args.mode == 'viewpoint_classification':
        from medimodule.Chest import ViewpointClassifier
        view_classifier = ViewpointClassifier()
        view_classifier.init(args.weights)
        out = view_classifier.predict(dcm_path)
        print(out)

    ### Enhance Classifier Example (Non-Enhanced / Enhanced)
    elif args.mode == 'enhance_classification':
        from medimodule.Chest import EnhanceCTClassifier
        enhanceCT_classifier = EnhanceCTClassifier()
        enhanceCT_classifier.init(args.weights)
        out = enhanceCT_classifier.predict(dcm_path)
        print(out)

    ### LR Detection (L / R)
    elif args.mode == 'lr_detection':
        from medimodule.Chest import ChestLRDetection
        detection = ChestLRDetection()
        detection.init(args.weights)
        predict = detection.predict(args.img)
    
    ##### ENDOSCOPY EXAMPLES
    ### Polyp Segmentation
    elif args.mode == 'polyp_segmentation':
        from medimodule.Endoscopy import PolypSegmentation
        polyp_seg = PolypSegmentation()
        polyp_seg.init(args.weights)
        out = polyp_seg.predict(img_path)
        print(args.save_path + args.img.split('/')[-1])
        out = out.squeeze().squeeze()

    ##### Liver EXAMPLES
    ### Liver Segmentation
    elif args.mode == 'liver_segmentation':
        from medimodule.Liver import LiverSegmentation
        Checker.set_gpu(args.gpus, 'tf2')
        liver_segmentation = LiverSegmentation()
        liver_segmentation.init(os.path.abspath(args.weights))
        result = liver_segmentation.predict(img_path)
        if args.save_path is not None:
            temp2 = np.swapaxes(result, 1, 2)
            temp2 = np.swapaxes(temp2, 0, 1)
            temp2 = np.swapaxes(temp2, 1, 2)
            mask_pair = nib.Nifti1Pair(
                temp2, np.diag([-liver_segmentation.space[0], 
                                -liver_segmentation.space[1], 
                                5., 1]))
            nib.save(mask_pair, args.save_path)
        print(result.shape, type(result))
    
    ### Lung Segmentation
    elif args.mode == 'lung_segmentation':
        from medimodule.Lung import LungSegmentation
        Lung_segmentation = LungSegmentation()
        Lung_segmentation.init(args.weights)
        out = Lung_segmentation.predict(dcm_path)
        if args.save_path is not None:
            if (not os.path.isdir(args.save_path)):
                os.makedirs(args.save_path)
            mask = out[0, :, :, 0]
            mask = mask * 255

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
