#!/usr/bin/env python3

import os
from gcp_common import GCPHelper
from argparse import ArgumentParser

DTS_SAMPLE = 'hello_world'

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-v', '--version', required=True, type=str)
    ap.add_argument('-o', '--output', required=True, type=str)
    args, _ = ap.parse_known_args()

    version = args.version
    output = args.output

    gcp = GCPHelper(version)
    builder_platforms = gcp.get_builder_results()

    # Apply filters from the dashboard
    # XXX: this is a temporary solution. This will be dropped
    #      after migration to a new CI testing solution
    #      (decoupling from the dashboard)
    omit_arch = ('arc', 'posix')
    omit_board = ('acrn', 'qemu', 'native', 'nsim', 'xenvm', 'xt-sim')
    filtered_boards = {
        k: v for k, v in builder_platforms.items()
        if v['arch'] not in omit_arch
        and k not in omit_board
        and not k.startswith('fvp_')
        and not k.startswith('qemu_')
    }

    built_platforms = [key for key, value in filtered_boards.items() if value['samples'][DTS_SAMPLE]['status'] == 'BUILT']

    os.makedirs(output, exist_ok=True)
    for platform in built_platforms:
        gcp.download_file(platform, DTS_SAMPLE, 'dts', output)
        # GCP helper doesn't support renaming downloaded files
        # This will be propably fixed in the dashboard's upstream
        os.rename(f'{output}/hello_world.dts', f'{output}/{platform}.dts')
