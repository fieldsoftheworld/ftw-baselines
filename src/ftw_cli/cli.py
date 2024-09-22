import click
from ftw_cli.download import download
from ftw_cli.unpack import unpack
from ftw_cli.model import model

@click.group()
def ftw():
    """Fields of The World (FTW) - Command Line Interface"""
    pass

ftw.add_command(download, "download")
ftw.add_command(unpack, "unpack")
ftw.add_command(model, "model")

if __name__ == "__main__":
    ftw()
