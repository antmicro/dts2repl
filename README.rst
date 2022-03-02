===============
Renode dts2repl
===============

Copyright (c) 2022 `Antmicro <https://www.antmicro.com>`_

``dts2repl`` is a tool for converting device tree sources into
`Renode's <https://renode.io>`_ ``.repl`` files.

Installation
------------

This tool uses the ``devicetree`` library from Zephyr RTOS. You can either
install it directly with ``pip``::

    pip install devicetree

or by setting up the Zephyr development environment by following the
`Getting Started Guide <https://docs.zephyrproject.org/latest/getting_started/index.html>`_
from the Zephyr documentation and then installing it manually::

    pushd ~/zephyrproject/zephyr/scripts/dts/python-devicetree/
    python3 setup.py install
    popd

Note that if you plan to use Zephyr RTOS build system, you need to use the
``devicetree`` library version provided by its repository.

Next, use pip to install this package::

   pip install git+https://github.com/antmicro/dts2repl.git

Usage
-----

You can use this tool either directly from the commandline, i.e.::

   dts2repl --overlays riscv32-fe310 path/to/flattened_devicetree.dts

or by importing and using it in your Python script::

   from dts2repl import dts2repl
   from argparse import Namespace

   print(dts2repl.generate(Namespace(overlays="riscv32-fe310",
                                     filename="path/to/flattened_devicetree.dts")))

Required data
-------------

For this tool to work, you need to have access to the flattened device tree
file. You can obtain it by manually preprocessing your device tree sources with
GCC, for example::

   gcc -H -E -P -x assembler-with-cpp -I include/ -I dts/riscv -I dts/common boards/riscv/litex_vexriscv/litex_vexriscv.dts 1>flat.dts 2>includes.txt

Some boards need additional overlays located under ``dts2repl/overlay`` to be
used to generate a proper repl file. Check the ``includes.txt`` output file
from the command mentioned above to see if any files overlap listed there
correspond to overlay files provided by this tool. For the previous command,
the ``include.txt`` file has the following content::

   . dts/riscv/riscv32-fe310.dtsi
   .. include/dt-bindings/gpio/gpio.h
   Multiple include guards may be useful for:
   dts/riscv/riscv32-fe310.dtsi

This suggests that you should use the ``riscv32-fe310`` overlay by passing it through the ``--overlay`` option.
