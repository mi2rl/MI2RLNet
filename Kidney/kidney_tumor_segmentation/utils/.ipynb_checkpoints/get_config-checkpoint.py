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