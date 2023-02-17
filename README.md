# Renode dts2repl

Copyright (c) 2022 [Antmicro](https://www.antmicro.com)

[![View on Antmicro Open Source Portal](https://img.shields.io/badge/View%20on-Antmicro%20Open%20Source%20Portal-332d37?style=flat-square)](https://opensource.antmicro.com/projects/dts2repl) 

The `dts2repl` project is a tool for converting device tree sources into [Renode's](https://renode.io) `.repl` files.

## Installation

Use pip to install this package:

```
pip3 install git+https://github.com/antmicro/dts2repl.git
```

## Usage

### Generating the device tree

For this tool to work, you need to have access to the flattened device tree
file. You can obtain it by manually preprocessing your device tree sources with
GCC, for example:

```
gcc -H -E -P -x assembler-with-cpp -I include/ -I dts/riscv -I dts/common boards/riscv/hifive1/hifive1.dts 1>flat.dts 2>includes.txt
```

Some boards need additional overlays located under ``dts2repl/overlay`` to be
used to generate a proper repl file. Check the ``includes.txt`` output file
from the command mentioned above to see if any files overlap listed there
correspond to overlay files provided by this tool. For the previous command,
the `includes.txt` file has the following content:

```
. dts/riscv/riscv32-fe310.dtsi
.. include/dt-bindings/gpio/gpio.h
Multiple include guards may be useful for:
dts/riscv/riscv32-fe310.dtsi
```

This suggests that you should use the `riscv32-fe310` overlay by passing it through the `--overlay` option.

### Generating the repl file

You can use this tool either directly from the commandline, i.e.:

```
dts2repl --overlays riscv32-fe310 path/to/flattened_devicetree.dts
```

or by importing and using it in your Python script:

```
from dts2repl import dts2repl
from argparse import Namespace

print(dts2repl.generate(Namespace(overlays="riscv32-fe310",
                         filename="path/to/flattened_devicetree.dts")))
```

## Development

You can use the CI scripts locally to quickly verify your dts2repl changes. First, run [setup-env.sh](./ci/setup_env.sh)
after sourcing [config.sh](./ci/config.sh)to prepare your environment:

```sh
source ./ci/config.sh
./ci/setup_env.sh
```

Then, you can use:

```sh
(cd ci-output && ../ci/get_dashboard_replkit.sh)
```

to download all repls from the dashboard for comparison, and

```sh
./ci/entrypoint.sh
```

to build new repls for all platforms for which the dashboard has device trees and compare them against the downloaded
ones. `entrypoint.sh` will download source code for the dashboard projects as necessary. The only step you need to
re-run after making a new change to dts2repl is `entrypoint.sh`.
