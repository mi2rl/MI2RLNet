from .models.ACE_CNet import ACE_CNet
from .utils.run_eval_cascaded import TransAxis, resample_img_asdim, normalize_vol, CCL_check_1ststg, CCL_1ststg_post, CCL
from .models.model_2_5 import MyModel
from .utils.load_data import Preprocessing

def get_config(mode):
    config = {
        "1": { # 1st cascade
            'depth': 3,
            'wlower': -300,
            'wupper': 600,
            'input_dim': (200, 200, 200),
            'num_labels_1ststg': 1
            }, 
        "2_1": {
            'depth': 3,
            'wlower': -300,
            'wupper': 600,
            'input_dim': (200, 200, 200)
            },
        "2_2": {
            'lossfn': 'dice',
            'depth': 4,
            'standard': 'normal',
            'task': 'tumor',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_3": {
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_4": {
            'lossfn': 'focaldice',
            'depth': 3,
            'standard': 'minmax',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            },
        "2_5": {
            'lossfn': 'dice',
            'depth': 3,
            'standard': 'normal',
            'task': 'tumor1',
            'wlevel': 100,
            'wwidth': 400
            }}

    return config[mode]

def kidney_tumor_segmentation(mode, weight_path) :

    if mode == '1':
        ''' coreline '''

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
        model.load_weights(weight_path)
        return model

    else:
        if mode== '2_1':
            ''' coreline '''

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

            model.load_weights(weight_path)
            return model

        else:
            ''' mi2rl '''
            
            config = get_config(mode)

            model = MyModel(
                model=mode,
                input_shape=(None, None, None, 1),
                lossfn=config['lossfn'],
                classes=3,
                depth=config['depth']
                )

            model.mymodel.load_weights(weight_path)
            return model

        