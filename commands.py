import fire
from hydra import compose, initialize
from mlops_project.infer import infer as inference
from mlops_project.train import train as training


def main_training(cfg="default_cfg"):
    initialize(version_base="1.3", config_path="configs")
    cfg = compose(config_name="training/" + cfg)
    training(**cfg["training"])


def main_inference(cfg="cfg") -> None:
    initialize(version_base="1.3", config_path="configs")
    cfg = compose(config_name="postprocess/" + cfg)
    inference(**cfg["postprocess"])


if __name__ == "__main__":
    fire.Fire({"train": main_training, "infer": main_inference})
