#!/usr/bin/env python3
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from dts2repl import dts2repl


def main(args):
    zephyr_dts_filename = args.board_dir / f"{args.board_name}.dts"
    dep_chain = dts2repl.get_cpu_dep_chain(args.arch, zephyr_dts_filename, args.zephyr_path, [])
    dts2repl_args = Namespace(
        filename=args.flat_dts, overlays=",".join(dep_chain + [args.board_name])
    )
    repl = dts2repl.generate(dts2repl_args)
    sys.stdout.write(repl)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "-z", "--zephyr-path", required=True, type=Path, help="path to the Zephyr source"
    )
    ap.add_argument(
        "-f", "--flat-dts", required=True, type=Path, help="path to the flattened device tree"
    )
    ap.add_argument("arch", help="the architecture of the board")
    ap.add_argument("board_name", help="the name of the board as used in Zephyr")
    ap.add_argument("board_dir", type=Path, help="the board directory in the Zephyr source tree")
    main(ap.parse_args())
