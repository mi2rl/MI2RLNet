from efficientdet import efficientdet

def lr_detection(weight_path:str ):
    _, prediction_model = efficientdet(phi=0,
                                           weighted_bifpn=False,
                                           num_classes=2,
                                           score_threshold=0.85)
    prediction_model.load_weights(weight_path, by_name=True)
    
    return prediction_model
