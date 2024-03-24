# Renode dts2repl

Copyright (c) 2022-2024 [Antmicro](https://www.antmicro.com)

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
used to generate a proper repl file. These overlays are matched automatically
by using the platform-level compatible string from the ``/`` root node, and
from the ``/soc`` node.

### Generating the repl file

You can use this tool either directly from the commandline, i.e.:

```
dts2repl path/to/flattened_devicetree.dts
```

or by importing and using it in your Python script:

```
from dts2repl import dts2repl

print(dts2repl.generate("path/to/flattened_devicetree.dts"))
```
