import click
from ftw_cli.download import download
from ftw_cli.unpack import unpack
from ftw_cli.model import model
from ftw_cli.inference import inference

@click.group()
def ftw():
    """Fields of The World (FTW) - Command Line Interface"""
    pass

@ftw.group()
def data():
    """Downloading, unpacking, and preparing the FTW dataset."""
    pass

data.add_command(download, "download")
data.add_command(unpack, "unpack")

ftw.add_command(model, "model")
ftw.add_command(inference, "inference")

if __name__ == "__main__":
    ftw()
