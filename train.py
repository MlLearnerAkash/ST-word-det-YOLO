# import typer
import yaml
from ultralytics import YOLO
from ultralytics import settings
import typer
import mlflow
import re
import os
import dagshub


try:
    os.environ['MLFLOW_TRACKING_URI'] = 'http://10.10.16.13:5000'
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "ST-word-det-YOLO"
    dagshub.init(repo_owner='manna.phys', repo_name='ST-word-det-YOLO', mlflow=True)
except ImportError:
    print("mlflow not initlaized")


settings.update({
    'mlflow': True,
    'wandb' : False})



def on_fit_epoch_end(trainer):
    print('in the on_fit_epoch_end')
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def main(
    base_model: str,
    datasets: str = "/home/akash/ws/ST-YOLO-word-det/ultralytics/cfg/datasets/mlt2k19.yaml",
    epochs: int = 150,
    imgsz: int = 1024,
    batch: int = 6,
    dropout: float = 0.05,
    resume: bool = False,
    device = "0",
    name: str= "ST-word-det-YOLO_mlt2k19_221124_",
    project = "/home/akash/ws/artifacts/ST/ST-word-det-YOLO_mlt2k19",
    freeze: int = 0,
    single_cls :bool = True,
    auto_augment: str = "randaugment",
    patience: int = 30
):
    
    with mlflow.start_run():
        model = YOLO(base_model)
        model.add_callback("on_fit_epoch_end",on_fit_epoch_end)
        results = model.train(
            data=datasets,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            # dropout=dropout,
            resume=resume,
            device = device,
            name= name,
            project = project,
            freeze= freeze,
            single_cls = single_cls,
            auto_augment = auto_augment,
            patience=patience

        )
        mlflow.end_run()


if __name__ == "__main__":
    typer.run(main)