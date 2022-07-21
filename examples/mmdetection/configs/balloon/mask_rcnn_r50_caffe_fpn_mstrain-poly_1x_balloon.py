# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('balloon',)
data_dir = '/workspace/data/balloon'
data = dict(
    train=dict(
        img_prefix=f'{data_dir}/train/',
        classes=classes,
        ann_file=f'{data_dir}/train/annotation_coco.json'
    ),
    val=dict(
        img_prefix=f'{data_dir}/val/',
        classes=classes,
        ann_file=f'{data_dir}/val/annotation_coco.json'
    ),
    test=dict(
        img_prefix=f'{data_dir}/val/',
        classes=classes,
        ann_file=f'{data_dir}/val/annotation_coco.json'
    )
)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = f'{data_dir}/ckpt/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
