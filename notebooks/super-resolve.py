import argparse
from pathlib import Path

import mlstac
import rioxarray as rio
import torch
from rasterio.enums import Resampling
from tqdm import tqdm


class FTWDataset(torch.utils.data.Dataset):
    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        self.images = list(self.root.glob("**/s2_images/**/*.tif"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = rio.open_rasterio(path)

        # Resample down from 5m to 10m resolution and normalize by 10k
        image = image.rio.reproject(
            image.rio.crs, shape=(128, 128), resample=Resampling.bilinear
        )
        image = torch.from_numpy(image.to_numpy()).to(torch.float) / 10_000
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return dict(image=image, path=str(path))


def main(args):
    # Download model
    if not Path("model/SEN2SR_RGBN").exists():
        mlstac.download(
            file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
            output_dir="model/SEN2SR_RGBN",
        )

    # Load model
    device = torch.device(args.device)
    model = mlstac.load("model/SEN2SR_RGBN").compiled_model(device=device)
    model.eval()
    model = model.to(device)

    dataset = FTWDataset(args.root)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    for batch in tqdm(dataloader, total=len(dataloader)):
        # Predict and convert to uint16 format (need to multiply by 10k to original scale)
        with torch.inference_mode():
            sr_images = model(batch["image"].to(device)).cpu()
            sr_images = sr_images.mul(10_000).clip(0, 65_535).to(torch.uint16).numpy()

        for path, sr in zip(batch["path"], sr_images):
            image = rio.open_rasterio(path)

            # Resample to 2.5m resolution and override pixels with super-resolved image
            image = image.rio.reproject(
                image.rio.crs, shape=(512, 512), resample=Resampling.bilinear
            )
            image.data = sr

            output_path = path.replace("ftw", args.out)
            output_path.mkdir(parents=True, exist_ok=True)
            image.rio.to_raster(output_path, driver="COG", compress="zstd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/data/data/ftw/")
    parser.add_argument("--out", type=str, default="ftw-sr")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
