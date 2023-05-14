import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

def infer(image: Image, weights_path: Path, output_path: Optional[Path] = None, visualization_mode: str = "simple", show: bool = False):
    """Run inference."""
    config_path = "/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib-main/tools/inference/config.yaml"
    config = get_configurable_parameters(config_path)
    config.trainer.resume_from_checkpoint = str(weights_path)
    config.visualization.show_images = show
    config.visualization.mode = visualization_mode
    if output_path:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = str(output_path)
    else:
        config.visualization.save_images = False

    # Create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # Get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )

    # Convert PIL image to numpy array
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    img_arr = np.frombuffer(img_byte_arr, dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create the dataset
    dataset = InferenceDataset(img, image_size=tuple(config.dataset.image_size), transform=transform)
    dataloader = DataLoader(dataset)

    # Generate predictions
    trainer.predict(model=model, dataloaders=[dataloader])
    
    
    
    
    
    
    
from PIL import Image

image_path = "/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib-main/tools/inference/000.png"
config_path = "/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib-main/tools/inference/config.yaml"
weights_path = "/home/grajebhosle/Documents/IPML/Projects/Anamoly/anomalib-main/tools/inference/model.ckpt"

# Load the image
image = Image.open(image_path)

# Call the infer function with the image
infer(image,weights_path)    

