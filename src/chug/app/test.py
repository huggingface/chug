import logging
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pprint import pprint
from typing import Dict, Optional, Union

import simple_parsing

from chug.common import ImageInputCfg, ImageAugCfg, DataArg
from chug.wds import create_loader_wds

@dataclass
class TestArgs:
    data: DataArg
    # FIXME need TaskArg form to define subset of task cfg options from command line
    input: ImageInputCfg
    aug: ImageAugCfg


def main():
    args = simple_parsing.parse(
        TestArgs,
        add_option_string_dash_variants=simple_parsing.DashVariant.DASH,
        argument_generation_mode=simple_parsing.ArgumentGenerationMode.BOTH,
        add_config_path_arg=True,
    )

    pprint(args)

    loader = create_loader_wds(...)

    # FIXME WIP app to demo iteration / analysis for supported datasets


if __name__ == '__main__':
    main()
