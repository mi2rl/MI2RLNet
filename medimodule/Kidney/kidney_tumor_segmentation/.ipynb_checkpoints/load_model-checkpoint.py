from .models.ACE_CNet import ACE_CNet
from .utils.run_eval_cascaded import TransAxis, resample_img_asdim, normalize_vol, CCL_check_1ststg, CCL_1ststg_post, CCL
from .models.model_2_5 import MyModel
from .utils.load_data import Preprocessing

def get_config(mode):
    config = {
        "1": { # 1st cascade
            'checkpoint': '../checkpoint/model0.h5',
            'depth': 3,
            'wlower': -300,
            'wupper': 600,
            'input_dim': (200, 200, 200),
            'num_labels_1ststg': 1
            }, 
        "2_1": {
            'checkpoint': '../checkpoint/model1.h5',
            'depth': 3,
            'wlower': -300,
            'wupper': 600,
            'input_dim': (200, 200, 200)
            },
        "2_2": {
            'checkpoint': '../checkpoint/model2.h5',
            'lossfn': 'dice',
            'depth': 4,
            'standard': 'normal',
            'task': 'tumor',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_3": {
            'checkpoint': '../checkpoint/model3.h5',
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_4": {
            'checkpoint': '../checkpoint/model4.h5',
            'lossfn': 'focaldice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_5": {
            'checkpoint': '../checkpoint/model5.h5',
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'normal',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            }}

    return config[mode]

def kidney_tumor_segmentation(mode) :

    if mode == '1':
        print("mode 1 is called")
        ''' coreline '''
#         from models.ACE_CNet import ACE_CNet
#         from utils.run_eval_cascaded import TransAxis, resample_img_asdim, normalize_vol, CCL_check_1ststg, CCL_1ststg_post

        config = get_config(mode)

        model = ACE_CNet(
            input_shape=(None, None, None, 1), 
            num_labels=1, 
            base_filter=32,
            depth_size=config['depth'], 
            se_res_block=True, 
            se_ratio=16, 
            last_relu=True
            )
        model.load_weights(config['checkpoint'])
        return model

    else:
        if mode== '2_1':
            ''' coreline '''
#             from models.ACE_CNet import ACE_CNet
#             from utils.run_eval_cascaded import TransAxis, resample_img_asdim, normalize_vol, CCL
            
            config = get_config(mode)

            model = ACE_CNet(
                input_shape=(None, None, None, 1), 
                num_labels=3, 
                base_filter=32,
                depth_size=config['depth'], 
                se_res_block=True, 
                se_ratio=16, 
                last_relu=False
                )

            model.load_weights(config['checkpoint'])
            print("mode 2_1 is called")
            return model

        else:
            ''' mi2rl '''
#             from models.model_2_5 import MyModel
#             from utils.load_data import Preprocessing
            
            config = get_config(mode)

            model = MyModel(
                model=mode,
                input_shape=(None, None, None, 1),
                lossfn=config['lossfn'],
                classes=3,
                depth=config['depth']
                )

            model.mymodel.load_weights(config['checkpoint'])
            print("mode 2_5 is called")
            return model

        
if __name__ == "__main__":
    model = kidney_tumor_segmentation('2_1')
    print("model is called")