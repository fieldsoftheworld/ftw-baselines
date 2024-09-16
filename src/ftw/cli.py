import click
from ftw.download_dataset import download
from ftw.unpack_dataset import unpack

@click.group()
def ftw():
    """Fields of The World (FTW) - Command Line Interface"""
    pass

ftw.add_command(download, "download")
ftw.add_command(unpack, "unpack")

if __name__ == "__main__":
    ftw()
