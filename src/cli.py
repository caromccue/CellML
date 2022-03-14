import click
import logging

@click.group()
@click.version_option()
def cli():
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('-c', '--check-segmentation', is_flag=True, help="Displays the segmented image to check accuracy")
@click.option('-o', '--save-overlay', is_flag=True, help="Save images with cell detection overlay")
@click.option('-p', '--save-plot', is_flag=True, help="Generate and save plots of number of posts per cell")
@click.option('-v', '--verbose', count=True, help="Increase verbosity level")
def process(directory, check_segmentation, save_overlay, save_plot, verbose):
    '''Process a directory containing a z stack of images'''

    from .cell_processing.process_image_folder import process_image_folder

    # Setup logging
    if verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    elif verbose >= 2:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    logging.info("Processing directory: %s", directory)

    process_image_folder(directory, show_plot=save_plot, save_overlay=save_overlay, show_segmentation=check_segmentation)


@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('-o', '--save-overlay', is_flag=True, help="Save segmented overlay to disk")
@click.option('-v', '--verbose', count=True, help="Increase verbosity level")
@click.option('-p', '--postsize', required=True, type=click.Choice(['2.5', '5', '10']), help="Size of posts")
def segment(directory, save_overlay, verbose, postsize):
    '''Segment an image or directory of images and saves extracted cells to disk'''
    if postsize == "2.5":
        from .data.segmentation.segment_cells_2p5 import segment_cells_to_file

        # Setup logging
        if verbose == 1:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        elif verbose >= 2:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
        

        logging.info("Extracting cells from: %s", directory)

        segment_cells_to_file(directory, save_overlay=save_overlay)

    elif postsize == "5":
        from .data.segmentation.segment_cells_5 import segment_cells_to_file

        # Setup logging
        if verbose == 1:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        elif verbose >= 2:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
        

        logging.info("Extracting cells from: %s", directory)

        segment_cells_to_file(directory, save_overlay=save_overlay)

    elif postsize == "10":
        from .data.segmentation.segment_cells_10 import segment_cells_to_file

        # Setup logging
        if verbose == 1:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        elif verbose >= 2:
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
        

        logging.info("Extracting cells from: %s", directory)

        segment_cells_to_file(directory, save_overlay=save_overlay)
    

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('-m', '--model', required=True, type=click.Choice(['cnn', 'cnn-transfer']), help="Type of model to train")
@click.option('-tb', '--tensorboard', is_flag=True, help="Save logs for tensorboard visualization")
@click.option('-v', '--verbose', count=True, help="Increase verbosity level")
@click.option('-l', '--layer', default=1, help="For transfer learning, how many layers to skim from the top.")
@click.option('-f', '--frozen', default=0, help="For transfer learning, how many layers to unfreeze at the top.")
def train(directory, model, verbose, tensorboard, layer, frozen):
    '''Train a model from a directory of labeled images'''

    training_directory = directory

    # Setup logging
    if verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    elif verbose >= 2:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    if model == "cnn":
        from .models.train.cnn_simple import train_cnn_simple_from_directory
        train_cnn_simple_from_directory(training_directory, tensorboard)

    elif model == "cnn-transfer":
        from .models.train.cnn_transfer import train_cnn_transfer_from_directory
        train_cnn_transfer_from_directory(training_directory, tensorboard, -1 * layer, frozen)
