from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    HookBase,
    default_argument_parser,
)
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils import comm

import os
import argparse
import weakref
import periflow_sdk as pf


def main(args: argparse.Namespace):

    # Register custom dataset
    register_coco_instances(
        name="fruits_nuts",
        metadata={},
        json_file=os.path.join(args.data_dir, "trainval.json"),
        image_root=os.path.join(args.data_dir, "images"),
    )

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Specify dataset to use
    cfg.DATASETS.TRAIN = ("fruits_nuts",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # Create trainer
    trainer = PeriflowTrainer(cfg, args.save_interval)
    trainer.resume_or_load(resume=args.resume)

    # Start training
    pf.init(None)
    trainer.train()


class PeriflowTrainer(DefaultTrainer):

    def __init__(self, cfg, save_interval):
        super().__init__(cfg)
        
        # Register hook to use Periflow
        checkpointer = PeriflowCheckpointer(
            model=self._trainer.model,
            save_dir=self.cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        hook = PeriflowHook(checkpointer, save_interval)
        self.register_hooks([hook])


class PeriflowCheckpointer(DetectionCheckpointer):

    def save(self, name, **kwargs) -> None:
        super().save(name, **kwargs)
        pf.upload_checkpoint()


class PeriflowHook(HookBase):

    def __init__(
        self,
        checkpointer: PeriflowCheckpointer,
        save_interval: int
    ):
        super().__init__()
        self.checkpointer = checkpointer
        self.save_interval = save_interval

    def before_step(self):
        pf.start_step()
        
    def after_step(self):
        pf.end_step()
        
        # Save checkpoint
        step = self.trainer.iter
        if step % self.save_interval == 0:
            self.checkpointer.save(f"step{step:05}.pth")


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument("--output-dir", type=str, default='')
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()
    print("Program arguments:", args)

    main(args)
