from models.swin_upernet import Swin_UperNet
from models.swinres_upernet import SwinRes_UperNet
from models.dinknet import DinkNet34_multiclass, DinkNet34, DinkNet34_three_stages

def build_model(model_name, num_classes, sync_bn=False):
    if model_name == 'Swin_UperNet':
        return Swin_UperNet(num_classes, sync_bn)
    elif model_name == 'SwinRes_UperNet':
        return SwinRes_UperNet(num_classes, sync_bn)
    elif model_name == 'DinkNet34':
        return DinkNet34(num_classes=num_classes)
    elif model_name == 'DinkNet34_multiclass':
        return DinkNet34_multiclass(num_classes=num_classes)
    elif model_name == 'DinkNet34_three_stages':
        return DinkNet34_three_stages(num_classes=num_classes)
    else:
        raise NotImplementedError

