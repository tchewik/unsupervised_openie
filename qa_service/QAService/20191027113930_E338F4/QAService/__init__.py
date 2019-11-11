import os
import sys
import logging

from bentoml import archive
from bentoml.cli import create_bento_service_cli
from bentoml.utils.log import configure_logging

# By default, ignore warnings when loading BentoService installed as PyPI distribution
# CLI will change back to default log level in config(info), and by adding --quiet or
# --verbose CLI option, user can change the CLI output behavior
configure_logging(logging.ERROR)

__VERSION__ = "20191027113930_E338F4"

__module_path = os.path.abspath(os.path.dirname(__file__))

QAService = archive.load_bento_service_class(__module_path)

cli=create_bento_service_cli(__module_path)


def load():
    return archive.load(__module_path)


__all__ = ['__version__', 'QAService', 'load']
