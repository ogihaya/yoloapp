from math import ceil
from pathlib import Path
from typing import List, Union

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        # optimizerに .next_epoch() が存在するかチェックする
        optimizer = self.trainer.optimizers[0]
        if hasattr(optimizer, "next_epoch"):
            optimizer.next_epoch(
                ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
            )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):

        # optimizerに .next_batch() が存在するかチェックする
        optimizer = self.trainer.optimizers[0]
        if hasattr(optimizer, "next_batch"):
            lr_dict = optimizer.next_batch()
            self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)

        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        # 1. モデルからピクセル単位・xyxy形式で予測結果を取得
        predicts_list = self.post_process(self(images), rev_tensor=rev_tensor)
        
        # 検出結果がある場合のみ処理
        # predicts_listはList[Tensor]で、各テンソルは(N, 6)の形状
        # [class_id, x1, y1, x2, y2, confidence]の形式（すでにxyxy形式）
        if predicts_list is not None and len(predicts_list) > 0:
            # バッチの最初の画像の検出結果を取得
            predictions_tensor = predicts_list[0]  # (N, 6)のテンソル
            
            # テンソルをリストのリストに変換（draw_bboxesが期待する形式）
            if len(predictions_tensor) > 0:
                # テンソルをCPUに移動してnumpy配列に変換し、リストに変換
                predictions_list = predictions_tensor.cpu().tolist()
            else:
                predictions_list = []

            # 3. 変換後のデータを描画関数に渡す
            # draw_bboxesはList[List[...]]を期待し、最初の要素を使用する
            img = draw_bboxes(origin_frame, [predictions_list], idx2label=self.cfg.dataset.class_list)
        else:
            # 何も検出されなかった場合は元の画像を使用
            img = origin_frame

        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
            
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
        

    # format_predictions_for_drawingメソッドは不要になったので削除できます
    # （または、将来の互換性のために残しておくこともできます）
