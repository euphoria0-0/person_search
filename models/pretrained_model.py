import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign
from models.resnet_simclr import ResNetSimCLR
import torch


def load_pretrained_faster_rcnn():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    backbone = models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                            aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = models.detection.FasterRCNN(backbone, num_classes=2,
                                        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
    return model


def load_trained_simclr_model(device, model_name = 'checkpoint_1000.pth.tar'):
    #simclr_model = models.resnet18(pretrained=False, num_classes=6978).to(device)
    simclr_model = ResNetSimCLR(base_model='resnet18', out_dim=128).to(device)

    simclr_checkpoint = torch.load('checkpoints/simclr/'+model_name, map_location=device)
    state_dict = simclr_checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    log = simclr_model.load_state_dict(state_dict, strict=False)
    #assert log.missing_keys == ['fc.weight', 'fc.bias']

    # freeze all layers but the last fc
    for name, param in simclr_model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    parameters = list(filter(lambda p: p.requires_grad, simclr_model.parameters()))
    #assert len(parameters) == 2  # fc.weight, fc.bias
    return simclr_model