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
from typing import List, Set, Optional
import itertools
from dts2repl import dtlib, name


with open(f'{pathlib.Path(__file__).parent.resolve()}/models.json') as f:
    MODELS = json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument('filename')
    parser.add_argument('--loglevel',
                        default='warning',
                        help='Provide logging level. Example --loglevel debug, default=warning',
                        choices=['info', 'warning', 'debug', 'error'])
    parser.add_argument('--output',
                        default='output.repl',
                        help='Output filename')
    parser.add_argument('--include',
                        default='',
                        help='Comma-separated dtsi include directories')
    parser.add_argument('--preprocessor-args',
                        default='',
                        help='Extra arguments to pass to the C preprocessor')
    parser.add_argument('--flatten',
                        action='store_true',
                        help='Flatten dtsi files to one dts automatically. Only available when dtsi include dirs are provided')

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
    try:
        dt = dtlib.DT(dts_filename)
    except FileNotFoundError:
        logging.error(f'File not found: "{dts_filename}"')
        return None
    except Exception:
        logging.exception('Error while parsing DT')
        return None

    # The same caveat as in get_user_led0 applies: if we end up with a duplicate uart
    # name while generating the repl, all bets are off
    name_mapper = NameMapper()

    # First, try to use the chosen zephyr,shell-uart
    try:
        chosen = dt.get_node('/chosen')
        return name_mapper.get_name(chosen.props['zephyr,shell-uart'].to_path())
    except Exception:
        pass

    # Then, chosen stdout-path = &uart;
    try:
        chosen = dt.get_node('/chosen')
        return name_mapper.get_name(chosen.props['stdout-path'].to_path())
    except Exception:
        pass

    # Then, chosen stdout-path = "uart:115200n8"; (Linux style)
    try:
        chosen = dt.get_node('/chosen')
        alias = chosen.props['stdout-path'].to_string().split(':')[0]
        return name_mapper.get_name(dt.get_node(alias))
    except Exception:
        pass

    # Finally, just return any non-disabled node that looks vaguely like a uart
    for node in dt.node_iter():
        if any(x in node.name.lower() for x in ('uart', 'usart', 'serial')):
            if get_node_prop(node, 'status', default='okay') != 'disabled' and 'reg' in node.props:
                return name_mapper.get_name(node)

    # No uart found
    return None



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
    dts_file = []

    # This cast is needed, as the `filename` might be of `Path` or `PosixPath` type
    filename = str(filename)

    if filename == '-':
       filename = '/dev/stdin'
    if filename.startswith("https://") or filename.startswith("http://"):
       try:
          import requests
       except:
          print("error: requests module is required to support remote files")
          sys.exit(1)
       dts_file = requests.get(filename).text
    else:
       dts_file = open(filename).readlines()
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
    elif prop in ('#address-cells', '#size-cells', '#interrupt-cells', 'cc-num', 'clock-frequency',
                  'riscv,ndev'):
        val = val.to_num()
    elif prop in ('interrupt-parent',):
        val = val.to_node()
    elif prop in ('interrupts-extended',):
        val = get_interrupts_extended(val)
    elif prop in ('gpios',):
        val = get_prop_value(val, 'pnn')
    else:
        val = val.to_string()

    return val

def renode_model_overlay(compat, mcu, overlays):
    model = MODELS[compat]

    # this hack is needed for stm32f072b_disco, as needs UART.STM32F7_USART
    # model to work properly while using the same compat strings as boards
    # which require UART.STM32_UART model
    if compat == "st,stm32-usart" and mcu in ("arm,cortex-m0", "arm,cortex-m7", "arm,cortex-m33"):
        compat = "st,stm32-lpuart"
        model = MODELS[compat]

    # compat-based mapping of peripheral models for the following SoCs is not enough
    # as there are ifdefs in the driver; adding a manual map for now as a workaround
    if any(x in overlays for x in ('st,stm32f3', 'st,stm32g0', 'st,stm32g4', 'st,stm32l4', 'st,stm32wl', 'st,stm32l0')):
        if compat in ("st,stm32-uart", "st,stm32-usart"):
            compat = "st,stm32-lpuart"
            model = MODELS[compat]

        if compat == "st,stm32-rcc":
            if any(map(lambda x: x in overlays, ('st,stm32g0', 'st,stm32g4', 'st,stm32l4'))):
                model = 'Python.PythonPeripheral'
            elif 'st,stm32l0' in overlays:
                model = 'Miscellaneous.STM32L0_RCC'
            else:
                model = 'Miscellaneous.STM32F4_RCC'

    if compat == 'st,stm32-gpio' and 'st,stm32f1' in overlays:
        model = 'GPIOPort.STM32F1GPIOPort'

    if compat == "atmel,sam0-uart" and 'atmel,samd20' in overlays:
        model = 'UART.SAMD20_UART'

    # LiteX on Fomu is built in the 8-bit CSR data width configuration
    if compat == "litex,timer0" and "fomu" in overlays:
        model = 'Timers.LiteX_Timer'

    # EFR32xG22 USART uses different offsets, but the compatible is identical to EFR32xG12 USART
    # The HAL uses compile-time defines to choose the right register layout
    if compat == 'silabs,gecko-usart' and any(overlay in overlays for overlay in ['silabs,efr32bg22', 'silabs,efr32mg24']):
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


# Returns the address in the root address space that corresponds to `addr`
# in the address space of `node`
def translate_address(addr, node):
    addr_offset = 0
    parent_node = node.parent
    while parent_node is not None and 'ranges' in parent_node.props:
        for child_addr, parent_addr, size in get_ranges(parent_node):
            if child_addr <= addr + addr_offset < child_addr + size:
                addr_offset += parent_addr - child_addr
                break
        parent_node = parent_node.parent
    return addr + addr_offset


# The returned addresses are in the root address space, that is they have all
# `ranges` translations applied.
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
        address = translate_address(address, node)
        yield (address, size)


# interrupts-extended entries are of the form < &phandle nn nn >, < &phandle2 nn > etc
# The number of numbers after each phandle is given by #interrupt-cells of the node
# pointed to by that phandle
def get_interrupts_extended(val):
    node = val.node
    phandle2node = node.dt.phandle2node
    cells = dtlib.to_nums(val.value)
    while cells:
        dest = phandle2node[cells[0]]
        interrupt_cells = get_node_prop(dest, '#interrupt-cells')
        if not interrupt_cells:
            logging.warn(f'Failed to parse interrupts_extended for {node.path}: {dest.path} has no #interrupt-cells')
            return
        params = cells[1:1 + interrupt_cells]
        cells = cells[1 + interrupt_cells:]
        yield (dest, params)


class NameMapper:
    OVERRIDES = {
        "IRQControllers.ARM_GenericInterruptController": "gic",
        "IRQControllers.CoreLevelInterruptor": "clint",
        "IRQControllers.GaislerMIC": "irqmp",
        "IRQControllers.NVIC": "nvic",
        "IRQControllers.PlatformLevelInterruptController": "plic",
    }

    def __init__(self):
        self._counter = Counter()
        self._mapping = {}
        # Start with cpu0, not cpu
        self._counter['cpu'] = 0

    def get_name(self, node):
        if node.path in self._mapping:
            return self._mapping[node.path]

        # Allow getting the name of a node whose name doesn't contain a unit address
        name, _, _ = node.name.partition('@')
        if len(node.labels) > 0:
            name = node.labels[0].lower().replace("_", "")

        # make name a valid repl GeneralIdentifier
        name = re.sub('[^A-Za-z0-9_]', '_', name)

        model = get_model(node) or ''
        if model.startswith('CPU.'):
            # Rename all cpus in order so we always have cpu0
            name = 'cpu'
        elif model in NameMapper.OVERRIDES:
            # Some models should always get the same name no matter what their dts node is called.
            # There can only be one of each of these models, so this should only be used for
            # peripherals like interrupt controllers
            name = NameMapper.OVERRIDES[model]
            if name in self._counter:
                logging.warn(f'Node {node.path} had duplicate overridden name {name}')

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
class RegistrationRegion:
    address: Optional[int] = None
    size: Optional[int] = None
    region_name: Optional[str] = None
    registration_point: str = "sysbus"

    @property
    def has_address_and_size(self) -> bool:
        return self.address is not None and self.size is not None

    @property
    def end(self) -> Optional[int]:
        if not self.has_address_and_size:
            return None
        return self.address + self.size

    @end.setter
    def end(self, value: int) -> None:
        self.size = value - self.address

    @staticmethod
    def to_repl(regions):
        if len(regions) == 0:
            return ''
        if len(regions) == 1:
            region = regions[0]
            if region.size is not None:
                return f'{region.registration_point} <{region.address:#x}, +{region.size:#x}>'
            registration = region.registration_point
            if region.address is not None:
                registration += f' {region.address:#x}'
            return registration

        # Multi-region registration
        # We assume that each region will have an address, size and name in this case
        parts = []
        for r in regions:
            parts.append(f'sysbus new Bus.BusMultiRegistration {{ address: {r.address:#x}; size: {r.size:#x}; region: "{r.region_name}" }}')
        return "{\n" + ";\n".join(f'{" "*8}{p}' for p in parts) + "\n    }"


@dataclass
class ReplBlock:
    name: str
    model: Optional[str]
    depends: Set[str]
    provides: Set[str]
    content: List[str]
    region: Optional[RegistrationRegion] = None

    def __str__(self):
        return '\n'.join(self.content)


def filter_available_blocks(blocks) -> List[ReplBlock]:
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


OVERLAY_NODE = re.compile(
    r"""
    ^                                   # Start of a line
    (?P<name>\w+):                      # Capture the name in `name:`
    (                                   # Optional part: model name and registration point. This part won't
     [^\S\n]*                           # be present on nodes that just override properties on existing ones
     (?P<model>[\w.]+)                  # Capture the model name in `model @ registration_point`
     \s*@\s*                            # Match the @ and whitespace
     (?P<registration_point>\w+)        # Capture the registration point in `model @ registration_point`
     (                                  # Optional: address
      \s+                               # Skip whitespace
      (?P<address>(0[xX])?[0-9a-fA-F]+) # Capture the address
     )?                                 # End address optional part
    )?                                  # Outer optional (model/registration point) end
    """,
    re.VERBOSE,
)


def parse_overlay(path):
    with open(path) as f:
        lines = [line.rstrip() for line in f.readlines()]

    blocks = []
    parts = [list(g) for k, g in itertools.groupby(lines, lambda x: x == "") if not k]
    for part in parts:
        depends = set()
        provides = set()

        non_comment_lines = [l for l in part if not l.lstrip().startswith('//')]
        if not non_comment_lines:
            continue

        node = OVERLAY_NODE.search(non_comment_lines[0])
        if not node:
            continue

        # IRQ destinations are not treated as dependencies, see the comment starting with
        # the same prefix in `generate` for reasoning
        # properties (such as `timeProvider: clint`) could be used to derive additional
        # dependency information here
        provides.add(node.group('name'))
        registration_point = node.group('registration_point')
        region = None
        if registration_point:
            depends.add(registration_point)
            if node.group('address'):
                address = node.group('address')
                address = int(address, 16 if address.lower().startswith('0x') else 10)
                region = RegistrationRegion(address=address, registration_point=registration_point)
                for line in part:
                    if 'size:' in line:
                        region.size = int(line.split()[1], 16)
                        break
        blocks.append(
            ReplBlock(node.group('name'), node.group('model'), depends, provides, part, region)
        )

    return blocks


def can_be_memory(node):
    possible_names = ('ram', 'flash', 'partition', 'memory')
    return len(node.props) == 1 and 'reg' in node.props \
        and any(x in node.name.lower() for x in possible_names) \
        and 'reserved-memory' not in node.path.split('/') \
        and (node.parent is None or not (get_model(node.parent) or '').startswith('Memory'))


def get_model(node, mcu=None, overlays=tuple()):
    node_compatible = next(filter(lambda x: x in MODELS, get_node_prop(node, 'compatible', [])), None)
    if node_compatible:
        node_model, _ = renode_model_overlay(node_compatible, mcu, overlays)
        return node_model
    return None


def generate(args):
    name_mapper = NameMapper()
    dt = get_dt(args.filename)
    if dt is None:
        return ''

    # `sysbus` and `none` are registration points provided by Renode itself
    blocks = [ReplBlock("sysbus", None, set(), {'sysbus', 'none'}, ['// autogenerated'])]
    nodes = sorted(dt.node_iter(), key=lambda x: get_node_prop(x, 'compatible')[0] if 'compatible' in x.props else '')

    # get mcu compat name
    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    mcu_compat = None
    if mcu is not None:
        mcu_compat = get_node_prop(mcu, 'compatible')[0]

    # get platform compat names
    platform = get_node_prop(dt.get_node('/'), 'compatible', [])

    # get soc compat names
    if dt.has_node('/soc'):
        soc = get_node_prop(dt.get_node('/soc'), 'compatible', [])
    else:
        soc = []

    overlays = set(soc + platform)

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
        compat = next(filter(lambda x: x in MODELS, compatible), None)
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
        model, compat = renode_model_overlay(compat, mcu_compat, overlays)

        dependencies = set()
        provides = {name}
        regions = []
        if addr and (not name.startswith('cpu') or name.endswith('tcm')):
            addr = int(addr.split(',')[0], 16)
            addr = translate_address(addr, node)
            if addr % 4 != 0:
                logging.info(f'Node {node.name} has misaligned address {addr}. Skipping...')
                continue

            if name == 'nvic':
                # weird mismatch, need to investigate, manually patching for now
                addr &= ~0x100

            if model == 'Timers.TegraUsecTimer':
                # the microsecond timer is at offset 0x10 from the base of the timer block
                addr += 0x10

            if (
                any(map(lambda x: x in compat,
                    ['stm32-gpio', 'stm32-timers', 'silabs,gecko', 'gaisler,irqmp',
                     'gaisler,gptimer', 'gaisler,apbuart', 'xlnx,xuartps']))
                or any(map(lambda x: x in model, 
                    ['UART.STM32_UART', 'UART.TrivialUart']))
            ):
                # sized sysbus registration for peripherals that require an explicit size
                _, size = next(get_reg(node))
                regions = [RegistrationRegion(addr, size)]
            else:
                # unsized sysbus registration
                regions = [RegistrationRegion(addr)]
        else:
            # assume sysbus registration without address like for CPUs
            regions = [RegistrationRegion()]
            
        # special multi-registration address for GIC
        if model == 'IRQControllers.ARM_GenericInterruptController':
            region_names = ('distributor', 'cpuInterface',)
            if compat in ('arm,gic-v3', 'arm,gic-v4',):
                region_names = ('distributor', 'redistributor',)

            regions = [
                RegistrationRegion(region_addr, region_size, region_name)
                for (region_addr, region_size), region_name
                in zip(get_reg(node), region_names)
            ]

        # check the registration point of guessed memory peripherals
        if is_heuristic_memory:
            node_reg = next(get_reg(node), None)
            if node_reg and regions:
                regions[0].address = node_reg[0]

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
        if compat == 'microchip,coreuart':
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
        if model == 'Timers.ARM_GenericTimer':
            if 'renesas,rzt2m_starter_kit' in overlays:
                indent.append('frequency: 20000000')
            else:
                indent.append('frequency: 62500000')
        if model == 'Timers.ARM_PrivateTimer':
            indent.append('frequency: 667000000')
        if model == 'Miscellaneous.STM32L0_RCC':
            indent.append('systick: nvic')
            dependencies.add('nvic')

        # additional parameters for python peripherals
        if compat.startswith("st,stm32") and compat.endswith("rcc") and model == "Python.PythonPeripheral":
            indent.append('size: 0x400')
            indent.append('initable: true')
            if any(map(lambda x: x in overlays, ('st,stm32l4', 'st,stm32g4'))):
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
                regions = [RegistrationRegion(int(child.unit_addr, 16))]
                break
            else:
                logging.info(f'ZynqMP mailbox has no children: {node}')
            indent.append('size: 0x1000')
            indent.append('initable: false')
            indent.append('script: "request.value = {0x24: 0x10000, 0x1e4: 0x10000}.get(request.offset, 0)"')
        elif compat == 'brcm,iproc-pcie-ep':
            indent.append('size: 0x2100')
            indent.append('initable: false')
            indent.append('script: "request.value = 0x1"')
        elif model == 'Python.PythonPeripheral':
            indent.append('size: 0x1000')
            indent.append('initable: true')
            indent.append('filename: "scripts/pydev/flipflop.py"')

        # additional parameters for CPUs
        if any(compat.startswith(x) for x in ('arm,cortex-a', 'arm,cortex-r')) and compat.count('-') == 1:
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
            overlays.add('cortex-m')
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
                indent.append('allowUnalignedAccesses: true')

            indent.append(f'hartId: {node.name.split("@")[1]}')

            if any(c.startswith('riscv,sifive') or
                   c.startswith('starfive,rocket') or
                   c == 'sifive,e31'
                   for c in compatible):
                indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_10')
            else:
                indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_09')
        if model in ("CPU.ARMv8A", "CPU.ARMv8R", "CPU.ARMv7A", "CPU.ARMv7R") and name != "cpu0":
            # We generate the cpu0 timer along with correct interrupt connections
            # while processing the timer node in the dts, and we generate 'fake'
            # timers for other cores here for now
            timer_name = f'{name}_timer'
            timer_lines = [f'{timer_name}: Timers.ARM_GenericTimer @ {name}', '    frequency: 62500000']
            generic_timer = ReplBlock(timer_name, 'Timers.ARM_GenericTimer', {name}, {timer_name}, timer_lines)
            blocks.append(generic_timer)
        if model in ("CPU.ARMv8A", "CPU.ARMv8R", "CPU.ARMv7A", "CPU.ARMv7R"):
            # We use our CPU number as the CPU ID instead of the reg address
            # This relies on the fact that the name will have been changed to "cpu{n}"
            indent.append(f'cpuId: {name.replace("cpu", "")}')
            indent.append('genericInterruptController: gic')
            dependencies.add('gic')

        if compat == 'gaisler,leon3':
            indent.append('cpuType: "leon3"')

        if model == 'UART.STM32F7_USART' and compat != 'st,stm32-lpuart':
            indent.append('frequency: 200000000')

        # additional parameters for STM32F4_RCC
        if model == 'Miscellaneous.STM32F4_RCC':
            indent.append('rtcPeripheral: rtc')
            dependencies.add('rtc')

        if model == 'Sensors.TMP103' or model == 'Sensors.TMP108':
            if len(node.parent.labels) == 0:
                logging.warning(f"Node {node} has no labels! Dropping {model}")
                continue

            i2c_name = name_mapper.get_name(node.parent)
            i2c_addr = int(node.unit_addr, 16)
            regions = [RegistrationRegion(address=i2c_addr, registration_point=i2c_name)]

        if model == 'Sensors.SI7210':
            if len(node.parent.labels) == 0:
                logging.warning(f"Node {node} has no labels! Dropping {model}")
                continue

            i2c_name = name_mapper.get_name(node.parent)
            i2c_addr = int(node.unit_addr, 16)
            regions = [RegistrationRegion(address=i2c_addr, registration_point=i2c_name)]

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
            regions = [RegistrationRegion(num, registration_point=gpio_name)]

            gpio_connection = ReplBlock(gpio_name, None, {gpio_name, name}, set(),
                                        [f'{gpio_name}:', f'    {num} -> {name}@0'])
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
        irq_local_indices = {}
        # decide which IRQ destinations to use in Renode model
        # these IRQ ctrls get special treatment
        if compat.endswith('nvic'):
            indent.append('-> cpu0@0')
            dependencies.add('cpu0')
        elif model == 'IRQControllers.ARM_GenericInterruptController':
            # Select correct GIC version
            gic_ver = compat.split(',')[-1]
            gic_renode_ver = {
                'gic-v1': 'GICv1',
                'gic-v2': 'GICv2',
                'gic-v3': 'GICv3',
                'gic-v4': 'GICv4',
            }.get(gic_ver)
            if gic_renode_ver:
                indent.append(f'architectureVersion: IRQControllers.ARM_GenericInterruptControllerVersion.{gic_renode_ver}')
            else:
                logging.warning(f'Unknown GIC version for compatible "{compat}"')


            # We route both the GIC's IRQ and FIQ outputs to the CPU's IRQ input
            # because FIQ support is not yet fully implemented in Renode
            combiner_model = 'Miscellaneous.CombinedInput'
            combiner_name = 'gicIrqCombiner'
            combiner_lines = [f'{combiner_name}: {combiner_model} @ none', '    numberOfInputs: 2', '    -> cpu0@0']
            combiner = ReplBlock(combiner_name, combiner_model, {'cpu0'}, {combiner_name}, combiner_lines)
            blocks.append(combiner)
            indent.append(f'[0, 1] -> {combiner_name}@[0, 1]')
            dependencies.add('cpu0')
            if 'renesas,rzt2m-soc' in overlays:
                indent.append('supportsTwoSecurityStates: false')
        elif compat == 'gaisler,irqmp':
            indent.append('0 -> cpu0@0 | cpu0@1 | cpu0@2')
            dependencies.add('cpu0')
        elif 'interrupts' in node.props:
            interrupt_parent = get_node_prop(node, 'interrupt-parent', inherit=True)
            if interrupt_parent is not None:
                interrupt_cells = get_node_prop(interrupt_parent, '#interrupt-cells', default=2)
                irq_numbers = get_node_prop(node, 'interrupts')[::interrupt_cells]
                irq_dest_nodes = [interrupt_parent] * len(irq_numbers)

                # Handle GIC IRQ number remapping
                parent_model = get_model(interrupt_parent, mcu_compat, overlays)
                if parent_model == 'IRQControllers.ARM_GenericInterruptController':
                    irq_types = get_node_prop(node, 'interrupts')[0::interrupt_cells]
                    irq_numbers = get_node_prop(node, 'interrupts')[1::interrupt_cells]
                    # Add 16 and route to local receiver (currently always #0) for GIC_PPI
                    for i, t in enumerate(irq_types):
                        if t == 1:
                            irq_numbers[i] += 16
                            irq_local_indices[i] = 0
        elif 'interrupts-extended' in node.props:
            # For now we assume that there is only one parameter: the IRQ number, otherwise
            # we skip the interrupt
            irq_dests = [(d, ps) for d, ps in get_node_prop(node, 'interrupts-extended') if len(ps) == 1]
            irq_dest_nodes = [d for d, _ in irq_dests]
            irq_numbers = [ps[0] for _, ps in irq_dests]
        for i, irq_dest_node in enumerate(irq_dest_nodes):
            irq_dest_compatible = get_node_prop(irq_dest_node, 'compatible', [])
            # treat the RISC-V CPU interrupt controller as the CPU itself
            if 'riscv,cpu-intc' in irq_dest_compatible:
                irq_dest_nodes[i] = irq_dest_node.parent
            # treat the VexRiscv intc as the CPU itself
            if 'vexriscv-intc0' in irq_dest_compatible:
                irq_dest_nodes[i] = mcu

        # assign IRQ signals (but not when using TrivialUart or PythonPeripheral)
        if irq_dest_nodes and model not in ('UART.TrivialUart', 'Python.PythonPeripheral'):
            # decide which IRQ names to use in Renode model
            if compat == 'st,stm32-rtc':
                irq_names = ['AlarmIRQ']
            elif (compat in ['nxp,kinetis-lpuart', 'nxp,kinetis-uart', 'silabs,gecko-leuart', 'sifive,uart0', 'st,stm32-adc']
                or model in ['UART.STM32F7_USART', 'SPI.STM32SPI']):
                irq_names = ['IRQ']
            elif compat in ['silabs,gecko-uart', 'silabs,gecko-usart']:
                irq_names = ['ReceiveIRQ', 'TransmitIRQ']
            elif compat in ['gaisler,gptimer', 'ambiq,counter']:
                irq_names = ['0']
            elif model == 'Timers.ARM_GenericTimer':
                irq_names = ['EL3PhysicalTimerIRQ', 'EL1PhysicalTimerIRQ', 'EL1VirtualTimerIRQ', 'NonSecureEL2PhysicalTimerIRQ']
            # the Renode model for these UARTs only has 1 IRQ
            elif (compat in ['arm,pl011', 'atmel,sam0-uart']
                  or model in ['UART.Cadence_UART', 'UART.NS16550']):
                irq_names = ['']
            elif compat in ['st,stm32-i2c-v1', 'st,stm32-i2c-v2']:
                irq_names = ['EventInterrupt', 'ErrorInterrupt']
            elif compat in ['arm,cmsdk-uart']:
                irq_names = ['TxInterrupt', 'RxInterrupt']
            elif compat in ['ambiq,stimer']:
                irq_names = ['IRQA', 'IRQB', 'IRQC', 'IRQD', 'IRQE', 'IRQF', 'IRQG', 'IRQH', 'IRQI']
            elif compat in ['renesas,rzt2m-uart']:
                irq_names = ['// RxErrIRQ', 'RxIRQ', 'TxIRQ', 'TxEndIRQ']
            else:
                irq_names = [str(n) for n in range(len(irq_dest_nodes))]

                # use non-numbered GPIO output if we only have 1 GPIO
                # and this node is not an interrupt controller
                if irq_names == ['0'] and 'interrupt-controller' not in node.props:
                    irq_names = ['']

            for i, (irq_name, irq_dest, irq) in enumerate(zip(irq_names, irq_dest_nodes, irq_numbers)):
                # assume very large IRQ numbers which have all bits set (i.e. 2^n - 1) are invalid
                if irq >= 0xfff and (irq & (irq + 1)) == 0:
                    continue
                irq_dest_name = name_mapper.get_name(irq_dest)
                if i in irq_local_indices:
                    irq_dest_name += f'#{irq_local_indices[i]}'
                indent.append(f'{irq_name}->{irq_dest_name}@{irq}')
                # IRQ destinations are not treated as dependencies, we filter
                # out IRQ connections to missing peripherals at the end because
                # it is better to have a peripheral missing an interrupt connection
                # than no peripheral at all

        # the ARM generic timer is registered at cpu0 without an address
        # for proper multi-CPU support we will need to generate one for
        # each CPU with the proper interrupt connections
        if model == "Timers.ARM_GenericTimer":
            regions = [RegistrationRegion(address=None, registration_point="cpu0")]

        # devices other than CPUs require an address to register on the sysbus
        if any(r.registration_point == 'sysbus' and r.address is None for r in regions) and not model.startswith('CPU.'):
            logging.info(f'Node {node} has sysbus registration without an address. Skipping...')
            continue

        # the registration points themselves are also dependencies
        for r in regions:
            dependencies.add(r.registration_point)

        block_content = [f'{name}: {model} @ {RegistrationRegion.to_repl(regions)}']
        block_content.extend(map(lambda x: f'    {x}', indent))

        region = regions[0] if len(regions) == 1 else None # for now we store at most one for merging
        if model.startswith('Memory'):
            region.size = size
        block = ReplBlock(name, model, dependencies, provides, block_content, region)
        blocks.append(block)

    # soc and board overlay
    overlay_path = f'{pathlib.Path(__file__).parent.resolve()}/overlay'
    overlay_blocks = []
    for compat in overlays:
        overlay = f'{overlay_path}/{compat}.repl'
        if os.path.exists(overlay):
            overlay_blocks.append(ReplBlock('', None, set(), set(), [f'// {compat} overlay']))
            overlay_blocks.extend(parse_overlay(overlay))

    # filter out unavailable blocks (with unsatisfied depends)
    available_blocks = filter_available_blocks(blocks + overlay_blocks)

    # split into blocks of known and unknown size
    sized = []
    unsized = []
    for block in available_blocks:
        (unsized, sized)[block.region is not None and block.region.has_address_and_size].append(block)
    # merge overlapping sized blocks
    # NOTE: currently, only memory blocks are merged, other overlapping blocks are removed
    sized_merged = []
    for block in sorted(sized, key=lambda b: (b.region.address, b.region.end)):
        if (
            sized_merged
            and sized_merged[-1].model == block.model
            and sized_merged[-1].region.end >= block.region.address + 1
        ):
            target = sized_merged[-1]
            target.region.end = max(target.region.end, block.region.end)
            # merge memory blocks by updating their name and size
            if target.model.startswith('Memory'):
                target.name += f'_{block.name}'
                # for now we manually update the string representation
                header = target.content[0].split(':')
                target.content[0] = target.name + ':' + header[1]
                target.content[1] = f'    size: {hex(target.region.size)}'

        else:
            sized_merged.append(block)

    # build the repl out of the filtered and merged blocks
    available_blocks = sized_merged + unsized
    repl_devices = set.union(*[b.provides for b in available_blocks])
    repl = [str(b) + '\n' for b in available_blocks]

    # convert it to a list of lines for the line-based interrupt removal step
    repl = '\n'.join(repl).splitlines()

    # remove interrupts connected to nonexistent peripherals
    filtered_repl = []
    for line in repl:
        irq_dest = re.search(r"->\s*(\w+)(#\d+)?\s*@", line)
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
    par = None

    dt = get_dt(filename)
    if dt is None:
        return None

    print(f"Checking for CPU Freq in {str(Path(filename).stem)}...")

    par = next((n for n in dt.node_iter() if n.name == 'cpus'), None)
    if not par:
        return None

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

def generate_peripherals(filename, overlays, type, get_snippets=False):
    result = {}
    par = ''
    irq_nums = []
    reg = None

    dt = get_dt(filename)
    if dt is None:
        return None

    mcu = get_mcu_compat(filename)

    print(f"Generating {type} peripherals for {str(Path(filename).stem)}")
    par = next((n for n in dt.node_iter() if n.name == 'soc'), None)
    if not par:
        return None

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

        if compat in MODELS:
            model, compat = renode_model_overlay(compat, mcu, overlays)
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

        if get_snippets:
            result[node.name]['snippet'] = str(node)

    return result


def get_buses(dt: dtlib.DT) -> list:
    '''
        Find buses:
        Iterate every node. Check for the following conditions:
        - The node name matches our collection of interesting nodes.
        - Check if the node is located somewhere in address space.

        Retruns: list of string paths for buses
    '''
    buses_paths = []
    for node in dt.node_iter():
        node_name = node.name
        if ('spi' in node_name or 'i2c' in node_name) and '@' in node_name:
            buses_paths.append(node.path)

    logging.debug(f'buses: {buses_paths}')
    return buses_paths


def generate_bus_sensors(filename, overlays, type):
    result = {}

    try:
        dt = get_dt(filename)
    except Exception:
        logging.exception(f'Failed on {filename}')
        return

    for bus_path in get_buses(dt):
        bus = dt.get_node(bus_path)    # bus node
        nodes = bus.nodes              # bus sub-nodes

        for node in nodes.values():
            # Get node compats
            compats = get_node_prop(node, 'compatible')

            compat = compats[0]
            if compat in MODELS:
                mcu = get_mcu_compat(filename)
                model, compat = renode_model_overlay(compat, mcu, overlays)
            else:
                model = ''

            result[node.name] = {
                "unit_addr": node.unit_addr,
                "label": node.labels[0] if node.labels else '',
                "model": model,
                "compats": compats.copy(),
                "bus": bus.name,
                "snippet": str(node)
            }

    return result


# dtlib does not support parsing props of type PHANDLES_AND_NUMS
# XXX: this will not raise any errors upon failure
def parse_phandles_and_nums(dt: dtlib.DT, node: dtlib.Node, prop: str, signed=False):
    value = node.props[prop].value
    return [int.from_bytes(value[i:i+4], "big", signed=signed) for i in range(0, len(value), 4)]


def generate_gpio(filename, overlays, type):
    result = {}

    KEYS_NODE = '/gpio_keys'
    LEDS_NODE = '/leds'
    LEDS_PWM_NODE = '/pwmleds'
    try:
        dt = get_dt(filename)
    except Exception:
        logging.exception(f'Failed on {filename}')
        return

    # Handle LEDS and Keys with the same flow, as they contain the same data
    for type in [KEYS_NODE, LEDS_NODE]:
        if dt.has_node(type):
            parent = dt.get_node(type)
            nodes = parent.nodes
            for node in nodes.values():
                gpio_info = parse_phandles_and_nums(dt, node, "gpios")
                # Assumes that gpios property is formatted as follows:
                # gpios = < &gpio_phandle 0xPin 0xMode >
                # If len(gpios) == 2 assumes that:
                # gpios = < &gpio_phandle 0xPin >
                result[node.name] = {
                    "label": node.labels[0] if node.labels else '',
                    "gpio": dt.phandle2node[gpio_info[0]].name,
                    "pin": gpio_info[1],
                    "mode": gpio_info[2] if len(gpio_info) == 3 else '',
                    "compats": get_node_prop(parent, 'compatible'),
                    "model": "",
                    "snippet": str(node)
                }

    if dt.has_node(LEDS_PWM_NODE):
        leds_pwm = dt.get_node(LEDS_PWM_NODE)
        nodes = leds_pwm.nodes
        for node in nodes.values():
            gpio_info = parse_phandles_and_nums(dt, node, "pwms")
            # Some PWM LEDs are driven by non-standard PWM sources
            # As a result of that, we implement alternative flows
            # XXX: this assumes that phandle to PWM source is always
            #      the first element in the properties list
            pwm_source = dt.phandle2node[gpio_info[0]].name

            result[node.name] = {
                "label": node.labels[0] if node.labels else '',
                "pwm": dt.phandle2node[gpio_info[0]].name,
                "channel": gpio_info[1],
                "period": gpio_info[2],
                "compats": get_node_prop(leds_pwm, 'compatible'),
                "model": "",
                "snippet": str(node)
            }

            if 'tcc' in pwm_source:
                # Atmel SAM0 TCC in PWM mode
                # pwms = < &pwm_phandle 0xChannel 0xPeriod >
                result[node.name]["flags"] = None
            elif 'pwm' in pwm_source:
                # "Standard" Zephyr PWM source
                # pwms = < &pwm_phandle 0xChannel 0xPeriod 0xFlags >
                # If len(gpio_info) == 3 assumes that:
                # pwms = < &pwm_phandle 0xChannel 0xPeriod >
                result[node.name]["flags"] = gpio_info[3] if len(gpio_info) == 4 else ''

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

    if args.flatten:
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

        # save info about dtsi includes
        includes = ret.stderr.decode('utf-8').split('\n')
        includes = filter(lambda x: '.dtsi' in x, includes)
        includes = itertools.chain(includes, dts_includes)
        includes = map(lambda x: x.lstrip('. '), includes)
        includes = remove_duplicates(includes)
        includes_file = f'{base}.includes'
        with open(includes_file, 'w') as f:
            f.writelines(f'{x}\n' for x in includes)

    if args.output == "-":
       args.output = "/dev/stdout"
    with open(args.output, 'w') as f:
        f.write(generate(args))

if __name__ == "__main__":
    main()
