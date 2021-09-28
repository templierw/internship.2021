from gluonts.model.predictor import Predictor
from pathlib import Path
import os

MODEL_DIR = f"{os.environ['ROOT_DIR']}/notebooks/saved_models/pbfcast"

def save_model(
    predictor: Predictor,
    name: str
) -> None:

    path = Path(f'{MODEL_DIR}/{name}')

    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)

    predictor.serialize(path)

def load_model(
    name: str
) -> Predictor:
    return Predictor.deserialize(Path(f'{MODEL_DIR}/{name}'))