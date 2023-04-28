#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import pathlib
from pathlib import Path
import subprocess
import sys
import json
import tempfile
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Set
import itertools
from dts2repl import dtlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--loglevel',
                        default='warning',
                        help='Provide logging level. Example --loglevel debug, default=warning',
                        choices=['info', 'warning', 'debug', 'error'])
    parser.add_argument('--overlays',
                        default='',
                        help='Comma-separated CPU dependency chain. Can be omitted if top-level dts from board directory is provided')
    parser.add_argument('--output',
                        default='output.repl',
                        help='Output filename')
    parser.add_argument('--include',
                        default='',
                        help='Comma-separated dtsi include directories')
    parser.add_argument('--preprocessor-args',
                        default='',
                        help='Extra arguments to pass to the C preprocessor')
    parser.add_argument('--automatch',
                        action='store_true',
                        help='Match overlays automatically. Only available when dtsi include dirs are provided')

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel.upper())
    return args


def dump(obj):
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))


def get_cpu_dep_chain(arch, dts_filename, zephyr_path, chain):
    next_include = ''
    if os.path.exists(dts_filename):
        with open(dts_filename) as f:
            dts_file = f.readlines()

        for l in dts_file:
            if next_include == '' and l.startswith('#include'):
                _, next_include = l.split()
                local = not (next_include.startswith('<') and next_include.endswith('>'))
                next_include = next_include.strip(' "<>')
                name, extension = os.path.splitext(next_include)

                # omit header files
                if extension.strip('.') == 'h':
                    next_include = ''
                    continue

                if local:
                    dtsi_filename = f'{os.path.dirname(dts_filename)}/{next_include}'
                    name = '!' + name
                else:
                    dtsi_filename = f'{zephyr_path}/dts/{arch}/{next_include}'

                return get_cpu_dep_chain(arch, dtsi_filename, zephyr_path, chain+[name])
    return chain


def get_uart(dts_filename):
    uart = ''
    if os.path.exists(dts_filename):
        with open(dts_filename, "r") as dts_file:
            for l in dts_file.readlines():
                if 'zephyr,shell-uart' in l:
                    uart = l[l.index('&')+1:l.index(';')].strip()
                    return uart


def get_user_led0(dts_filename):
    try:
        dt = dtlib.DT(dts_filename)
    except Exception:
        logging.exception('failed to load or parse dts')
        return None

    # we might come up with a different name for the LED here if we had duplicate
    # labels while generating the repl, but the same is true for get_uart and it
    # hasn't been an issue so far
    name_mapper = NameMapper()

    # find led0
    try:
        led0 = dt.get_node('led0')
    except dtlib.DTError:
        logging.exception('led0 not found')
        return None

    led_name = name_mapper.get_name(led0)

    try:
        gpio, num, gpio_flags = next(get_node_prop(led0, 'gpios'))
        gpio_name = name_mapper.get_name(gpio)
    except Exception:
        logging.exception('failed to get led0 gpio name')
        return None

    return {'name': gpio_name, 'led_name': led_name}


def get_dt(filename):
    with open(filename) as f:
        dts_file = f.readlines()
        dts_file = filter(lambda x: 'pinctrl-0;' not in x, dts_file)
        dts_file = ''.join(dts_file)

    # Workaround for NamedTemporaryFile not being able to be opened for reading while
    # it's open here. This can be simplified on Python 3.12 with `delete_on_close`,
    # see https//github.com/python/cpython/pull/97015
    try:
        f = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False)
        f.write(dts_file)
        f.close()
        return dtlib.DT(f.name)
    finally:
        os.remove(f.name)


def get_node_prop(node, prop, default=None, inherit=False):
    # look up inherited properties
    if inherit:
        while node.parent is not None:
            if prop in node.props:
                break
            node = node.parent

    if prop not in node.props:
        return default

    val = node.props[prop]
    if prop in ('compatible', 'device_type'):
        val = val.to_strings()
    elif prop in ('interrupts', 'reg', 'ranges'):
        val = val.to_nums()
    elif prop in ('#address-cells', '#size-cells', 'cc-num', 'clock-frequency', 'riscv,ndev'):
        val = val.to_num()
    elif prop in ('interrupt-parent',):
        val = val.to_node()
    elif prop in ('interrupts-extended',):
        val = get_prop_value(val, 'pn')
    elif prop in ('gpios',):
        val = get_prop_value(val, 'pnn')
    else:
        val = val.to_string()

    return val

def renode_model_overlay(compat, mcu, models, overlays):
    model = models[compat]

    # this hack is needed for stm32f072b_disco, as needs UART.STM32F7_USART
    # model to work properly while using the same compat strings as boards
    # which require UART.STM32_UART model
    if compat == "st,stm32-usart" and mcu in ("arm,cortex-m0", "arm,cortex-m7", "arm,cortex-m33"):
        compat = "st,stm32-lpuart"
        model = models[compat]

    # this hack is required for stm32f3, stm32g0 and stm32l0 based boards uarts
    # to work properly
    if compat == 'st,stm32-usart' and any(map(lambda x: x in overlays, ('stm32f3', 'stm32g0', 'stm32l0'))):
        model = 'UART.STM32F7_USART'

    # compat-based mapping of peripheral models for the following SoCs is not enough
    # as there are ifdefs in the driver; adding a manual map for now as a workaround
    if any(x in overlays for x in ('stm32g4', 'stm32l4', 'stm32wl', 'stm32l0')):
        if compat == "st,stm32-usart":
            compat = "st,stm32-lpuart"
            model = models[compat]

        if compat == "st,stm32-rcc":
            if any(map(lambda x: x in overlays, ('stm32l4', 'stm32g4'))):
                model = 'Python.PythonPeripheral'
            elif 'stm32l0' in overlays:
                model = 'Miscellaneous.STM32L0_RCC'
            else:
                model = 'Miscellaneous.STM32F4_RCC'

    if compat == 'st,stm32-gpio' and 'stm32f1' in overlays:
        model = 'GPIOPort.STM32F1GPIOPort'

    if compat == "atmel,sam0-uart" and 'samd20' in overlays:
        model = 'UART.SAMD20_UART'

    # LiteX on Fomu is built in the 8-bit CSR data width configuration
    if compat == "litex,timer0" and "fomu" in overlays:
        model = 'Timers.LiteX_Timer'

    # EFR32xG22 USART uses different offsets, but the compatible is identical to EFR32xG12 USART
    # The HAL uses compile-time defines to choose the right register layout
    if compat == 'silabs,gecko-usart' and 'efr32bg22' in overlays:
        model = 'UART.EFR32xG22_USART'

    # remap some core types to the closest supported equivalent
    if compat in ('arm,armv8', 'arm,cortex-a57'):
        compat = 'arm,cortex-a53'
    elif compat == 'arm,cortex-a72':
        compat = 'arm,cortex-a75'

    return model, compat


def get_cells(cells, n):
    current, rest = cells[:n], cells[n:]
    value = 0
    for cell in current:
        value <<= 32
        value |= cell
    return value, rest


def get_ranges(node):
    if not node.props['ranges'].value:  # ranges;
        return []

    ranges = get_node_prop(node, 'ranges')
    if not ranges:  # ranges = < >;
        return []
    # #address-cells from this node only applies to its address space (child addresses in ranges)
    address_cells = get_node_prop(node, '#address-cells')
    size_cells = get_node_prop(node, '#size-cells')
    parent_address_cells = 1
    if node.parent and '#address-cells' in node.parent.props:
        parent_address_cells = get_node_prop(node.parent, '#address-cells')

    while ranges:
        child_addr, ranges = get_cells(ranges, address_cells)
        parent_addr, ranges = get_cells(ranges, parent_address_cells)
        size, ranges = get_cells(ranges, size_cells)
        yield child_addr, parent_addr, size


def get_reg(node):
    if node.parent:
        address_cells = get_node_prop(node.parent, '#address-cells', inherit=True)
        size_cells = get_node_prop(node.parent, '#size-cells', inherit=True)
    else:
        address_cells = 1
        size_cells = 1

    reg = get_node_prop(node, 'reg')
    while reg:
        address, reg = get_cells(reg, address_cells)
        size, reg = get_cells(reg, size_cells)
        yield (address, size)


class NameMapper:
    def __init__(self):
        self._counter = Counter()
        self._mapping = {}

    def get_name(self, node):
        if node.path in self._mapping:
            return self._mapping[node.path]

        # Allow getting the name of a node whose name doesn't contain a unit address
        name, _, _ = node.name.partition('@')
        if len(node.labels) > 0:
            name = node.labels[0].lower().replace("_", "")

        # make name a valid repl GeneralIdentifier
        name = re.sub('[^A-Za-z0-9_]', '_', name)

        if name.startswith('cpu'):
            # Rename all cpus in order so we always have cpu0
            name = f"cpu{self._counter['cpu']}"
            self._counter['cpu'] += 1
        else:
            # "timer" becomes "timer1", "timer2", etc
            # if we have "timer" -> "timer1" but there was already a peripheral named "timer1",
            # we'll end up with "timer" -> "timer1" -> "timer11"
            while name in self._counter:
                self._counter[name] += 1
                name += str(self._counter[name] - 1)
            self._counter[name] += 1

        self._mapping[node.path] = name
        return name


@dataclass
class ReplBlock:
    depends: Set[str]
    provides: Set[str]
    content: List[str]

    def __str__(self):
        return '\n'.join(self.content)


def filter_available_blocks(blocks):
    available_blocks = []
    visited_blocks = set()

    def dfs(block):
        visited_blocks.add(id(block))
        for dep in block.depends:
            for other_block in blocks:
                if dep in other_block.provides:
                    if id(other_block) not in visited_blocks:
                        dfs(other_block)
                    break
            else:
                return False
        available_blocks.append(block)
        return True

    for block in blocks:
        if id(block) not in visited_blocks:
            dfs(block)

    return available_blocks


# Allowed characters in format string:
#   p: phandle
#   n: number
def get_prop_value(prop: dtlib.Property, fmt: str):
    node = prop.node
    byte_count = len(fmt) * 4

    for i in range(0, len(prop.value), byte_count):
        values = []
        for j, value_type in enumerate(fmt):
            offset = j * 4
            cell_value = int.from_bytes(prop.value[i + offset:i + offset + 4], 'big')
            if value_type == 'p':
                phandle_node = node.dt.phandle2node[cell_value]
                values.append(phandle_node)
            elif value_type == 'n':
                values.append(cell_value)
            else:
                raise ValueError(f'Invalid character {value_type} in format string')
        yield tuple(values)


def parse_overlay(path):
    with open(path) as f:
        lines = [line.rstrip() for line in f.readlines()]

    blocks = []
    parts = [list(g) for k, g in itertools.groupby(lines, lambda x: x == "") if not k]
    for part in parts:
        depends = set()
        provides = set()

        registration_point = re.search(r":\s*[\w.]+\s*@\s*(\w+)", part[0])
        if registration_point:
            depends.add(registration_point.group(1))

        # IRQ destinations are not treated as dependencies, see the comment starting with
        # the same prefix in `generate` for reasoning
        # properties (such as `timeProvider: clint`) could be used to derive additional
        # dependency information here

        node_name = part[0].split(':')[0].strip()
        provides.add(node_name)
        blocks.append(ReplBlock(depends, provides, part))

    return blocks


def can_be_memory(node):
    possible_names = ('ram', 'flash', 'partition', 'memory')
    return len(node.props) == 1 and 'reg' in node.props \
        and any(x in node.name.lower() for x in possible_names) \
        and 'reserved-memory' not in node.path.split('/')


def generate(args):
    def get_model(node):
        node_compatible = next(filter(lambda x: x in models, get_node_prop(node, 'compatible')), None)
        if node_compatible:
            node_model, _ = renode_model_overlay(node_compatible, mcu, models, args.overlays)
            return node_model
        return None

    name_mapper = NameMapper()
    dt = get_dt(args.filename)
    if dt is None:
        return ''

    models_path = f'{pathlib.Path(__file__).parent.resolve()}/models.json'
    with open(models_path) as f:
        models = json.load(f)

    # `sysbus` and `none` are registration points provided by Renode itself
    blocks = [ReplBlock(set(), {'sysbus', 'none'}, ['// autogenerated'])]
    nodes = sorted(dt.node_iter(), key=lambda x: get_node_prop(x, 'compatible')[0] if 'compatible' in x.props else '')

    # get mcu compat name
    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    mcu_compat = None
    if mcu is not None:
        mcu_compat = get_node_prop(mcu, 'compatible')[0]

    # get platform compat names
    platform = get_node_prop(dt.get_node('/'), 'compatible', [])

    for node in nodes:
        # those memory peripherals sometimes require changing the sysbus address of this peripheral
        is_heuristic_memory = False
        # filter out nodes without compat strings
        compatible = get_node_prop(node, 'compatible')
        if compatible and 'gpio-leds' in compatible:
            logging.debug(f'Skipping LED parent node {node.name}')
            continue
        else:
            parent_compat = get_node_prop(node.parent, 'compatible', []) if node.parent else []
            # if the paren't compat string is the one for LEDs, move it down to each individual LED
            if 'gpio-leds' in parent_compat:
                compatible = parent_compat

        if compatible is None:
            logging.debug(f'Node {node.name} has no compat string. Trying device_type...')
            compatible = get_node_prop(node, 'device_type')
            if compatible is None:
                # hack to generate entries for memory peripherals without a compat string on some platforms
                # we only want to treat it as such when the node meets all of the requirements described in the
                # 'can_be_memory' function
                if can_be_memory(node):
                    compatible = ['memory']
                    is_heuristic_memory = True
                    logging.debug(f'Node {node.name} will be treated as memory')
                else:
                    logging.debug(f'Node {node.name} has no compat string or device_type and cannot be treated as memory. Skipping...')
                    continue

        # look at all compat strings and take the first one that has a Renode model
        # if none of them do, skip the node
        compat = next(filter(lambda x: x in models, compatible), None)
        if compat is None:
            logging.info(f'Node {node.name} does not have a matching Renode model. Skipping...')
            continue

        # not sure why this is needed. We need to investigate the RCC->RTC dependency.
        if get_node_prop(node, 'status') == 'disabled' and not node.name.startswith('rtc'):
            logging.info(f'Node {node.name} disabled. Skipping...')
            continue

        # get model name and addr
        _, _, addr = node.name.partition('@')
        name = name_mapper.get_name(node)

        # decide which Renode model to use
        model, compat = renode_model_overlay(compat, mcu_compat, models, args.overlays)

        registration_point = 'sysbus'
        dependencies = set()
        provides = {name}
        address = ''
        if addr and not name.startswith('cpu'):
            parent_node = node.parent
            addr = int(addr, 16)
            addr_offset = 0
            while parent_node is not None and 'ranges' in parent_node.props:
                for child_addr, parent_addr, size in get_ranges(parent_node):
                    if child_addr <= addr + addr_offset < child_addr + size:
                        addr_offset += parent_addr - child_addr
                        break
                parent_node = parent_node.parent

            addr += addr_offset
            if addr % 4 != 0:
                logging.info(f'Node {node.name} has misaligned address {addr}. Skipping...')
                continue

            address = f'0x{addr:X}'
            if name == 'nvic':
                # weird mismatch, need to investigate, manually patching for now
                address = address[0:-3] + '0' + address[-2:]

            if (
                any(map(lambda x: x in compat,
                    ['stm32-gpio', 'stm32-timers', 'silabs,gecko', 'gaisler,irqmp',
                     'gaisler,gptimer', 'gaisler,apbuart', 'arm,cortex-a9-twd-timer',
                     'xlnx,xuartps']))
                or any(map(lambda x: x in model, 
                    ['UART.STM32_UART', 'UART.TrivialUart']))
            ):
                _, size = next(get_reg(node))
                address = f'<{address}, +{size:#x}>'
            
        # check the registration point of guessed memory peripherals
        if is_heuristic_memory:
            node_reg = next(get_reg(node), None)
            if node_reg:
                address = hex(node_reg[0])

        indent = []

        # additional parameters for peripherals
        if compat == "nordic,nrf-uarte":
            indent.append('easyDMA: true')
        if compat == "st,stm32-timers":
            indent.append('frequency: 10000000')
            indent.append('initialLimit: 0xFFFFFFFF')
        if compat == "st,stm32-lpuart":
            indent.append('frequency: 200000000')
        if compat.startswith('litex,timer'):
            indent.append('frequency: 100000000')
        if compat == 'ns16550':
            indent.append('wideRegisters: true')
        if compat == 'st,stm32-watchdog':
            indent.append('frequency: 32000')
        if compat == 'microsemi,coreuart':
            indent.append('clockFrequency: 66000000')
        if model == 'Timers.OMAP_Timer':
            indent.append('frequency: 4000000')
        if model == 'Timers.IMX_GPTimer':
            indent.append('frequency: 32000')
        if model == 'Timers.Marvell_Armada_Timer':
            indent.append('frequency: 100000000')
        if model == 'IRQControllers.PlatformLevelInterruptController':
            # the default of 1023 matches the highest one seen in Zephyr's dts
            ndev = get_node_prop(node, 'riscv,ndev', 1023)
            indent.append(f'numberOfSources: {ndev}')
            indent.append('numberOfContexts: 9')
            indent.append('prioritiesEnabled: true')

        # additional parameters for python peripherals
        if compat.startswith("st,stm32") and compat.endswith("rcc") and model == "Python.PythonPeripheral":
            indent.append('size: 0x400')
            indent.append('initable: true')
            if any(map(lambda x: x in args.overlays, ('stm32l4', 'stm32g4'))):
                indent.append('filename: "scripts/pydev/flipflop.py"')
            else:
                indent.append('filename: "scripts/pydev/rolling-bit.py"')
        elif compat == 'nordic,nrf91-flash-controller':
            indent.append('initable: true')
            indent.append('filename: "scripts/pydev/rolling-bit.py"')
            indent.append('size: 0x1000')
        elif compat == 'xlnx,zynq-slcr':
            indent.append('size: 0x200')
            indent.append('initable: false')
            indent.append('script: "request.value = {0x100: 0x0001A008, 0x120: 0x1F000400, 0x124: 0x18400003}.get(request.offset, 0)"')
        elif compat.startswith('fsl,imx6') and compat.endswith('-anatop'):
            indent.append('size: 0x1000')
            indent.append('initable: false')
            indent.append('// 0x10: usb1_pll_480_ctrl')
            indent.append('// 0xe0: pll_enet')
            indent.append('// 0x100: pfd_528')
            indent.append('// 0x150: ana_misc0')
            indent.append('// 0x180: tempsense0')
            indent.append('// 0x260: digprog, report mx6sl (0x60)')
            indent.append('// 0x280: digprog_sololite, report mx6sl (0x60)')
            indent.append('script: "request.value = {0x10: 0x80000000, 0xe0: 0x80000000, 0x100: 0xffffffff, 0x150: 0x80, 0x180: 0x4, 0x260: 0x600000, 0x280: 0x600000}.get(request.offset, 0)"')
        elif compat == 'fsl,imx6q-mmdc':
            indent.append('size: 0x4000')
            indent.append('initable: false')
            indent.append('// 0x0: ctl')
            indent.append('// 0x18: misc')
            indent.append('// these settings mean 256 MB of DRAM')
            indent.append('script: "request.value = {0x0: 0x4000000, 0x18: 0x0}.get(request.offset, 0)"')
        elif compat.startswith('fsl,imx') and compat.endswith('-fec'):
            indent.append('size: 0x4000')
            indent.append('initable: false')
            indent.append('// 0x4: ievent')
            indent.append('script: "request.value = {0x4: 0x800000}.get(request.offset, 0)"')
        elif compat.startswith('fsl,imx') and compat.endswith('-ccm'):
            indent.append('size: 0x4000')
            indent.append('initable: false')
            indent.append('// 0x14: cbcdr')
            indent.append('// 0x18: cbcmr')
            indent.append('// 0x1c: cscmr1')
            indent.append('script: "request.value = {0x14: 3<<8 | 7<<10, 0x18: 2<<12, 0x1c: 0x3f}.get(request.offset, 0)"')
        elif compat == 'ti,am4-prcm':
            indent.append('size: 0x11000')
            indent.append('initable: true')
            indent.append('filename: "scripts/pydev/rolling-bit.py"')
        elif compat in ('ti,am4372-i2c', 'ti,omap4-i2c'):
            indent.append('size: 0x1000')
            indent.append('initable: false')
            indent.append('script: "request.value = 1"')
        elif compat == 'marvell,mbus-controller':
            indent.append('size: 0x200')
            indent.append('initable: false')
            indent.append('// 0x180: win_bar')
            indent.append('// 0x184: win_sz')
            indent.append('script: "request.value = {0x180: 0x0, 0x184: 0xf000001}.get(request.offset, 0)"')
        elif compat.startswith('marvell,armada') and compat.endswith('-nand-controller'):
            indent.append('size: 0x100')
            indent.append('initable: false')
            indent.append('script: "request.value = 0xffffffe1"')
        elif compat.startswith('marvell,mv') and compat.endswith('-i2c'):
            indent.append('size: 0x100')
            indent.append('initable: false')
            indent.append('script: "request.value = 0xf8"')
        elif compat == 'nvidia,tegra210-mc':
            indent.append('size: 0x1000')
            indent.append('initable: false')
            indent.append('script: "request.value = 0x400"')  # report 1024 MiB, 512 gets reserved as carveout
        elif compat == 'nvidia,tegra210-i2c':
            indent.append('size: 0x100')
            indent.append('initable: false')
            indent.append('script: "request.value = 8<<4 | 1"')
        elif compat == 'xlnx,zynqmp-ipi-mailbox':
            # the address of the Xilinx ZynqMP IPI mailbox is defined in its child node
            for child in node.nodes.values():
                address = f'0x{child.unit_addr}'
                break
            else:
                logging.info('ZynqMP mailbox has no children: {node}')
            indent.append('size: 0x1000')
            indent.append('initable: false')
            indent.append('script: "request.value = {0x1e4: 0x10000}.get(request.offset, 0)"')
        elif model == 'Python.PythonPeripheral':
            indent.append('size: 0x1000')
            indent.append('initable: true')
            indent.append('filename: "scripts/pydev/flipflop.py"')

        # additional parameters for CPUs
        if compat.startswith('arm,cortex-a') and compat.count('-') == 1:
            cpu = compat.split(',')[1]
            indent.append(f'cpuType: "{cpu}"')
        if compat == 'marvell,sheeva-v7':
            indent.append('cpuType: "cortex-a9"')
        if compat.startswith('arm,cortex-m'):
            cpu = compat.split(',')[1]
            if cpu == 'cortex-m33f':
                cpu = cpu[:-1]
            indent.append(f'cpuType: "{cpu}"')
            indent.append('nvic: nvic')
            dependencies.add('nvic')
        if model == 'CPU.RiscV32':  # We use CPU.RiscV32 as a generic model for all RV CPUs and fix it up here
            isa = get_node_prop(node, 'riscv,isa', 'rv32imac')
            indent.append(f'cpuType: "{isa}"')
            if '64' in isa:
                model = 'CPU.RiscV64'

            # Use CPU.VexRiscv for LiteX and Fomu
            if set(platform) & {'litex,vexriscv', 'kosagi,fomu'}:
                model = 'CPU.VexRiscv'
            else:
                indent.append('timeProvider: clint')
                dependencies.add('clint')

            indent.append(f'hartId: {node.name.split("@")[1]}')

            if any(c.startswith('riscv,sifive') or
                   c.startswith('starfive,rocket') or
                   c == 'sifive,e31'
                   for c in compatible):
                indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_10')
            else:
                indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_09')
        if model == "CPU.ARMv8A":
            # Each ARMv8A core has its own generic timer and they are not individually
            # listed in device trees, so we add them here
            timer_lines = [f'{name}_timer: Timers.ARM_GenericTimer @ {name}', '    frequency: 62500000']
            generic_timer = ReplBlock({name}, {f'{name}_timer'}, timer_lines)
            blocks.append(generic_timer)

        if compat == 'gaisler,leon3':
            indent.append('cpuType: "leon3"')

        if model == 'UART.STM32F7_USART' and compat != 'st,stm32-lpuart':
            indent.append('frequency: 200000000')

        # additional parameters for STM32F4_RCC
        if model == 'Miscellaneous.STM32F4_RCC':
            indent.append('rtcPeripheral: rtc')
            dependencies.add('rtc')

        if model == 'Miscellaneous.LED':
            gpios = list(get_node_prop(node, 'gpios'))
            if not gpios:
                logging.info(f'LED {node.name} has no gpios property, skipping...')
                continue

            gpio, num, gpio_flags = gpios[0]
            active_low = (gpio_flags & 1) == 1
            if active_low:
                indent.append('invert: true')
            gpio_name = name_mapper.get_name(gpio)
            registration_point = gpio_name
            address = str(num)

            gpio_connection = ReplBlock({gpio_name, name}, set(),
                                        [f'{gpio_name}:\n    {num} -> {name}@0'])
            blocks.append(gpio_connection)

        if model.startswith('Timers'):
            if 'cc-num' in node.props:
                indent.append(f'numberOfEvents: {str(get_node_prop(node, "cc-num"))}')
        if model.startswith('Memory'):
            if 'reg' in node.props:
                _, size = next(get_reg(node))
                # increase OCRAM size for imx6 platforms
                # the device trees in U-Boot all have 0x20000, but some platforms
                # actually have 0x40000 and the config headers reflect this, which
                # would make the stack end up outside of memory if the size from
                # the device tree was used
                if any('imx6' in p for p in platform):
                    if node.labels and 'ocram' in node.labels[0]:
                        size = 0x40000
                if size != 0:
                    indent.append(f'size: {hex(size)}')
                else:
                    # do not generate memory regions of size 0
                    continue

        irq_dest_nodes = []
        irq_numbers = []
        # decide which IRQ destinations to use in Renode model
        # these IRQ ctrls get special treatment
        if compat.endswith('nvic'):
            indent.append('-> cpu0@0')
            dependencies.add('cpu0')
        elif compat == 'gaisler,irqmp':
            indent.append('0 -> cpu0@0 | cpu0@1 | cpu0@2')
            dependencies.add('cpu0')
        elif 'interrupts' in node.props:
            interrupt_parent = get_node_prop(node, 'interrupt-parent', inherit=True)
            if interrupt_parent is not None:
                # Note: this only works for #interrupt-cells = 2
                irq_numbers = get_node_prop(node, 'interrupts')[::2]
                irq_dest_nodes = [interrupt_parent] * len(irq_numbers)
        elif 'interrupts-extended' in node.props:
            irq_dest_nodes, irq_numbers = zip(*get_node_prop(node, 'interrupts-extended'))
            irq_dest_nodes = list(irq_dest_nodes)

        for i, irq_dest_node in enumerate(irq_dest_nodes):
            irq_dest_compatible = get_node_prop(irq_dest_node, 'compatible', [])
            # treat the RISC-V CPU interrupt controller as the CPU itself
            if 'riscv,cpu-intc' in irq_dest_compatible:
                irq_dest_nodes[i] = irq_dest_node.parent
            # treat the VexRiscv intc as the CPU itself
            if 'vexriscv-intc0' in irq_dest_compatible:
                irq_dest_nodes[i] = mcu

        # assign IRQ signals (but not when using TrivialUart)
        if irq_dest_nodes and model != 'UART.TrivialUart':
            # decide which IRQ names to use in Renode model
            if compat == 'st,stm32-rtc':
                irq_names = ['AlarmIRQ']
            elif compat in ['nxp,kinetis-lpuart', 'nxp,kinetis-uart', 'silabs,gecko-leuart', 'sifive,uart0', 'st,stm32-adc']:
                irq_names = ['IRQ']
            elif compat in ['silabs,gecko-uart', 'silabs,gecko-usart']:
                irq_names = ['ReceiveIRQ', 'TransmitIRQ']
            elif compat in ['gaisler,gptimer']:
                irq_names = ['0']
            # the Renode model for these UARTs only has 1 IRQ
            elif compat in ['arm,pl011', 'atmel,sam0-uart']:
                irq_names = ['']
            else:
                irq_names = [str(n) for n in range(len(irq_dest_nodes))]

                # use non-numbered GPIO output if we only have 1 GPIO
                # and this node is not an interrupt controller
                if irq_names == ['0'] and 'interrupt-controller' not in node.props:
                    irq_names = ['']

            for irq_name, irq_dest, irq in zip(irq_names, irq_dest_nodes, irq_numbers):
                # assume very large IRQ numbers which have all bits set (i.e. 2^n - 1) are invalid
                if irq >= 0xfff and (irq & (irq + 1)) == 0:
                    continue
                irq_dest_name = name_mapper.get_name(irq_dest)
                indent.append(f'{irq_name}->{irq_dest_name}@{irq}')
                # IRQ destinations are not treated as dependencies, we filter
                # out IRQ connections to missing peripherals at the end because
                # it is better to have a peripheral missing an interrupt connection
                # than no peripheral at all

        # devices other than CPUs require an address to register on the sysbus
        if registration_point == 'sysbus' and not address and not model.startswith('CPU.'):
            logging.info(f'Node {node} has sysbus registration without an address. Skipping...')
            continue

        # the registration point itself is also a dependency
        dependencies.add(registration_point)

        block_content = [f'{name}: {model} @ {registration_point} {address}']
        block_content.extend(map(lambda x: f'    {x}', indent))

        block = ReplBlock(dependencies, provides, block_content)
        blocks.append(block)

    # soc and board overlay
    overlay_path = f'{pathlib.Path(__file__).parent.resolve()}/overlay'
    overlay_blocks = []
    for cpu in map(lambda x: x.split("/")[-1], args.overlays.split(",")[::-1]):
        overlay = f'{overlay_path}/{cpu}.repl'
        if os.path.exists(overlay):
            overlay_blocks.append(ReplBlock(set(), set(), [f'// {cpu} overlay']))
            overlay_blocks.extend(parse_overlay(overlay))

    # build the repl out of the dts + overlay blocks filtering out unavailable blocks
    available_blocks = filter_available_blocks(blocks + overlay_blocks)
    repl_devices = set.union(*[b.provides for b in available_blocks])
    repl = [str(b) + '\n' for b in available_blocks]

    # convert it to a list of lines for the line-based interrupt removal step
    repl = '\n'.join(repl).splitlines()

    # remove interrupts connected to nonexistent peripherals
    filtered_repl = []
    for line in repl:
        irq_dest = re.search(r"->\s*(\w+)\s*@", line)
        if irq_dest is None or irq_dest.group(1) in repl_devices:
            filtered_repl.append(line)

    return '\n'.join(filtered_repl) + '\n'

def get_mcu_compat(filename):
    dt = get_dt(filename)
    if dt is None:
        return ''

    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    if mcu is not None:
        mcu = get_node_prop(mcu, 'compatible')[0]
    return mcu

def generate_cpu_freq(filename):
    result = {}
    par = ''
    irq_nums = []
    reg = None

    dt = get_dt(filename)
    if dt is None:
        return ''

    print(f"Checking for CPU Freq in {str(Path(filename).stem)}...")

    for node in dt.node_iter():
        if node.name == 'cpus':
            par = node
            break

    for n in par.node_iter():
        if n.parent == par:
            if "clock-frequency" in n.props:
                freq = get_node_prop(n, 'clock-frequency')
                if freq < 1000:
                    freq = freq * 1000000
                print(f" * Found clock-frequency - {freq} Hz")
                return freq
    print(f" * Not found")
    return None

def generate_peripherals(filename, overlays, type):
    result = {}
    par = ''
    irq_nums = []
    reg = None

    dt = get_dt(filename)
    if dt is None:
        return ''

    models_path = f'{pathlib.Path(__file__).parent.resolve()}/models.json'
    with open(models_path) as f:
        models = json.load(f)

    mcu = get_mcu_compat(filename)

    print(f"Generating {type} peripherals for {str(Path(filename).stem)}")
    for node in dt.node_iter():
        if node.name == 'soc':
            par = node
            break

    for node in par.node_iter():
        compats = get_node_prop(node, 'compatible')

        if compats is None:
            logging.info(f"No compats (type) for node {node}. Skipping...")
            continue

        if type == "board":
            status = get_node_prop(node, 'status')
            if status == 'disabled':
                continue

        compat = get_node_prop(node, 'compatible')[0]

        if compat in models:
            model, compat = renode_model_overlay(compat, mcu, models, overlays)
        else:
            model = ''

        reg = list(get_reg(node))
        if reg:
            unit_addr = hex(reg[0][0])
            size = sum(r[1] for r in reg)
            if size == 0:
                logging.info(f"Regs for node {node} have total size 0. Skipping...")
                continue
        else:
            logging.info(f"No regs for node {node}. Skipping...")
            continue

        if node.labels:
            label = node.labels[0]
        else:
            label = ''

        if 'interrupts' in node.props:
            irq_nums = [irq for irq in get_node_prop(node, 'interrupts')[::2]]

        result[node.name] = {"unit_addr":unit_addr, "label":label, "model":model, "compats":compats.copy()}

        if reg:
            result[node.name]["size"] = hex(size)
        if irq_nums != []:
            result[node.name]["irq_nums"] = irq_nums.copy()

    return result


def remove_duplicates(xs):
    """Remove duplicates while preserving element order"""
    return list(dict.fromkeys(xs))


def get_includes(dts_filename, dirs):
    """Get the paths of all dts(i) files /include/d by the specified dts file"""
    include_directive = "/include/"
    # we use this dict as an ordered set of include names
    includes = {}
    with open(dts_filename) as dts_file:
        for line in dts_file:
            line = line.strip()
            if line.startswith(include_directive):
                quoted_name = line[len(include_directive):].strip()
                name = quoted_name.strip('"')
                # find the the include in one of the dirs from the provided list
                for path in (Path(dir) / name for dir in dirs):
                    if not path.is_file():
                        continue
                    # found, save its path and recurse into it
                    includes[str(path)] = None
                    for inc in get_includes(path, dirs):
                        includes[inc] = None
                    break
    return list(includes)


def main():
    args = parse_args()

    dirs = []
    for top in args.include.split(','):
        for root, _, _ in os.walk(top):
            dirs.append(root)

    incl_dirs = ' '.join(f'-I {dir}' for dir in dirs)

    if args.automatch:
        board_name = os.path.splitext(os.path.basename(args.filename))[0]
        # get list of #includes (C preprocessor)
        cmd = f'gcc -H -E -P -x assembler-with-cpp {incl_dirs} {args.preprocessor_args} {args.filename}'.split()
        ret = subprocess.run(cmd, capture_output=True)

        # save partially flattened device tree
        base = os.path.splitext(args.output)[0]
        flat_dts = f'{base}.flat.dts'
        with open(flat_dts, 'w') as f:
            f.write(ret.stdout.decode('utf-8'))

        # get list of /include/s (device tree mechanism)
        dts_includes = get_includes(flat_dts, dirs)

        # save fully flattened device tree (also /include/s)
        dts = dtlib.DT(flat_dts, dirs)
        with open(flat_dts, 'w') as f:
            f.write(str(dts))
        args.filename = flat_dts

        # try to automatch overlays
        includes = ret.stderr.decode('utf-8').split('\n')
        includes = filter(lambda x: '.dtsi' in x, includes)
        includes = itertools.chain(includes, dts_includes)
        includes = map(lambda x: x.lstrip('. '), includes)
        includes = remove_duplicates(includes)
        includes_file = f'{base}.includes'
        with open(includes_file, 'w') as f:
            f.writelines(f'{x}\n' for x in includes)
        includes = map(lambda x: os.path.basename(x), includes)
        includes = map(lambda x: os.path.splitext(x)[0], includes)
        includes = set(includes)
        includes.add(board_name)
        args.overlays = ','.join(includes)

    with open(args.output, 'w') as f:
        f.write(generate(args))

if __name__ == "__main__":
    main()
