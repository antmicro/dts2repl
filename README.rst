===============
Renode dts2repl
===============

Copyright (c) 2022 `Antmicro <https://www.antmicro.com>`_

``dts2repl`` is a tool for converting device tree sources from
`Zephyr RTOS <https://www.zephyrproject.org>`_ into
`Renode's <https://renode.io>`_ ``.repl`` files.

Installation
------------

Use pip to install this package::

   pip install git+https://github.com/antmicro/dts2repl.git

Required setup and data
-----------------------

Make sure you have set up the Zephyr development environment. If you're not
sure how to prepare it, follow the `Getting Started Guide
<https://docs.zephyrproject.org/latest/getting_started/index.html>`_ from
the Zephyr documentation.

For this tool to work, you need to have access to the following files:

* flattened device tree, provided by Zephyr build tools when building a sample in ``zephyrproject/zephyr/build/zephyr/zephyr.dts``,
* top-level device tree source from the ``board`` directory, e.g. ``zephyrproject/zephyr/boards/arm/96b_aerocore2/96b_aerocore2.dts``.

Note that providing the top-level device tree source file is optional if you
pass the comma-separated CPU dependency chain instead. It consists of
device tree includes, starting from the top-level file from the ``board``
directory. For example, the CPU dependency chain for ``96b_aerocore2`` would
be::

   st/f4/stm32f427vi,st/f4/stm32f427vX,st/f4/stm32f427,st/f4/stm32f407,st/f4/stm32f405,st/f4/stm32f401,st/f4/stm32f4,arm/armv7-m

Usage
-----

You can use this tool either directly from the commandline, i.e.::

   # using the top-level dts
   dts2repl --arch riscv --board litex_vexriscv --base-dts board/riscv/litex_vexriscv/litex_vexriscv.dts path/to/flattened_devicetree.dts 

   # providing CPU dependency chain manually
   dts2repl --arch riscv --board litex_vexriscv --cpu-dep-chain riscv32-litex-vexriscv path/to/flattened_devicetree.dts

or by importing and using it in your Python script::

   from dts2repl import dts2repl
   from argparse import Namespace

   print(dts2repl.generate(Namespace(arch="riscv", board="litex_vexriscv",
                                     cpu_dep_chain="riscv32-litex-vexriscv",
                                     filename="path/to/flattened_devicetree.dts")))
