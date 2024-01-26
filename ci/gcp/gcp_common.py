#!/usr/bin/env python3

# !!! this is a trimmed down version of a file !!!
# !!! copied from the zephyr dashboard tree    !!!

import json
import requests
from typing import Optional
from colorama import Fore, Style
from argparse import ArgumentParser, BooleanOptionalAction


def bold(text):
    return Style.BRIGHT + (text or '') + Style.RESET_ALL


def red(text):
    return Fore.RED + (text or '') + Style.RESET_ALL


def green(text):
    return Fore.GREEN + (text or '') + Style.RESET_ALL


class GCPHelper:
    def __init__(self, project_version: str):
        # Load artifact names from the configuration file
        self.artifacts = {'dts': "{sample_name}.dts"}
        self.gcp_base = "https://new-zephyr-dashboard.renode.io/zephyr"
        if project_version is None:
            raise Exception('Zephyr version not set')
        elif project_version == 'latest':
            # Get the SHA of latest version of the project
            with requests.Session() as s:
                r = s.get(f'{self.gcp_base}/latest')
                r.raise_for_status()
                self.project_version = r.text.strip()
        else:
            self.project_version = project_version

        self.session = requests.Session()

    # Methods
    def _resolve_filename(self, sample: str, file: str) -> str:
        return self.artifacts[file].replace('{sample_name}', sample)

    def get_file_path(self, board: str, sample: str, file: str) -> str:
        return f'{self.gcp_base}/{self.project_version}/{board}/{sample}/{self._resolve_filename(sample, file)}'

    def _download_file_url(self, url: str, suppress_output: bool = False) -> Optional[bytes]:
        # Helper function to suppress output
        def dprint(*a, **k):
            if not suppress_output:
                print(*a, **k)

        dprint(f'Downloading {bold(url)}', end='')
        r = self.session.get(url)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            dprint(f'\n{red("Failed to download")}: {url}! Error: {e}')
            return

        dprint(f' {green("✓")}')
        return r.content

    def download_file(self, board: str, sample: str, file: str, dest: str) -> None:
        if not (data := self._download_file_url(self.get_file_path(board, sample, file))):
            return

        with open(f'{dest}/{self._resolve_filename(sample, file)}', 'wb') as d:
            d.write(data)

    def get_builder_results(self) -> dict:
        url = f'{self.gcp_base}/{self.project_version}/result.json'
        return json.loads(self._download_file_url(url, suppress_output=True).decode())
