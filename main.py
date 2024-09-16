import os

from lightning.pytorch.cli import ArgsType, LightningCLI

# Allows classes to be referenced using only the class name
import torchgeo.trainers  # noqa: F401
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask


def main(args: ArgsType = None) -> None:
    """Command-line interface to TorchGeo."""
    rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(rasterio_best_practices)
    # Initialize LightningCLI without resume_from_checkpoint in trainer_defaults
    cli = LightningCLI(
        model_class=BaseTask,
        datamodule_class=BaseDataModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        args=args,
    )



if __name__ == "__main__":
    main()
