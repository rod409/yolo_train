import os
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.loss import v8SegmentationLoss, BboxLoss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


class CustomBboxLoss(BboxLoss):
    """Subclasses BboxLoss to zero out bounding box loss for stuff classes on GPU."""
    def __init__(self, reg_max, stuff_classes=None):
        super().__init__(reg_max)
        stuff_tensor = (
            torch.tensor(stuff_classes, dtype=torch.long)
            if stuff_classes
            else torch.empty(0, dtype=torch.long)
        )
        self.register_buffer("stuff_tensor", stuff_tensor, persistent=False)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        *args,
        **kwargs,
    ):
        if self.stuff_tensor.numel() > 0 and fg_mask.any():
            # Extract foreground class IDs directly from target_scores tensor
            fg_classes = target_scores[fg_mask].argmax(dim=-1)

            # Ensure stuff_tensor matches active GPU device
            if self.stuff_tensor.device != fg_classes.device:
                self.stuff_tensor = self.stuff_tensor.to(fg_classes.device)

            is_stuff = torch.isin(fg_classes, self.stuff_tensor)

            # Suppress target scores for stuff class foreground anchors
            target_scores = target_scores.clone()
            stuff_mask = fg_mask.clone()
            stuff_mask[fg_mask] = is_stuff
            target_scores[stuff_mask] = 0.0

        return super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            *args,
            **kwargs,
        )


class CustomSegmentationLoss(v8SegmentationLoss):
    """Subclasses v8SegmentationLoss to swap in CustomBboxLoss."""
    def __init__(self, model, stuff_classes=None):
        super().__init__(model)
        m = model.model[-1] if hasattr(model, "model") else getattr(model, "head", None)
        # Fix: Pass m.reg_max directly (16) instead of m.reg_max - 1 (15)
        self.bbox_loss = CustomBboxLoss(
            m.reg_max,
            stuff_classes=stuff_classes if stuff_classes is not None else [],
        ).to(self.device)

    def __call__(self, preds, batch):
        return super().__call__(preds, batch)


class CustomSegmentationTrainer(SegmentationTrainer):
    """Custom Segmentation Trainer that initializes custom loss lazily on GPU."""
    stuff_classes = [2, 4, 5, 7, 9, 14, 16, 17, 18, 21]

    def setup_model(self):
        ckpt = super().setup_model()

        # 1. Attach trainer hyperparameters to model before loss instantiation
        self.model.args = self.args

        # 2. Define lazy constructor so loss is built after model moves to GPU
        stuff_classes = self.stuff_classes

        def custom_init_criterion():
            return CustomSegmentationLoss(self.model, stuff_classes=stuff_classes)

        self.model.init_criterion = custom_init_criterion
        # Set criterion to None so model.loss() triggers custom_init_criterion on GPU
        self.model.criterion = None

        return ckpt


def main():
    model = YOLO("yolo11m-seg.pt")

    model.train(
        data="/data/common/rod/cognata_yolo_1280/data.yaml",
        epochs=30,
        batch=8,
        nbs=32,
        imgsz=1280,
        device=1,
        workers=8,
        cache="ram",
        amp=False,
        trainer=CustomSegmentationTrainer,
        project="custom_yolo11_seg",
        name="stuff_class_nobox_experiment",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()