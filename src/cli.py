import click
import logging

@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('-o', '--save-overlay', is_flag=True, help="Save segmented overlay to disk")
@click.option('-v', '--verbose', count=True, help="Increase verbosity level")
@click.option('-p', '--postsize', required=True, type=click.Choice(['2.5', '5', '10']), help="Size of posts")
def segment(directory, save_overlay, verbose, postsize):
    '''Segment an image or directory of images and saves extracted cells to disk'''
    if postsize == "10":
        from .data.segmentation.segment_cells_10 import segment_cells_to_file10

        # Setup logging
        if verbose == 1:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        elif verbose >= 2:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
        

        logging.info("Extracting cells from: %s", directory)

        segment_cells_to_file10(directory, save_overlay=save_overlay)

    elif postsize == "5":
        from .data.segmentation.segment_cells_5 import segment_cells_to_file5

        # Setup logging
        if verbose == 1:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        elif verbose >= 2:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
        

        logging.info("Extracting cells from: %s", directory)

        segment_cells_to_file5(directory, save_overlay=save_overlay)

    elif postsize == "2.5":
        from .data.segmentation.segment_cells_2p5 import segment_cells_to_file2p5

        # Setup logging
        if verbose == 1:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        elif verbose >= 2:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
        

        logging.info("Extracting cells from: %s", directory)

        segment_cells_to_file2p5(directory, save_overlay=save_overlay)
    