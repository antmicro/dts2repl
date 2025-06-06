#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import pathlib
from pathlib import Path
import subprocess
import sys
import json
import tempfile
import re
import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Set, Optional
import itertools
import functools
from dts2repl import dtlib
from dts2repl.improved_json_decoder import ImprovedJsonDecoder

from dts2repl import name, version

try:
   from dts2repl import version_commit
   version = version + "-" + version_commit
except:
   pass

try:
   from dts2repl import version_additional
   version = version + "-" + version_additional
except:
   pass

def get_models_path() -> Path:
    return pathlib.Path(__file__).parent.resolve() / "models.json"

with open(get_models_path()) as f:
    MODELS = json.load(f, cls=ImprovedJsonDecoder)

def parse_args():
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument('-v', '--version', action="version", help="Display dts2repl version", version=name + " version " + version)
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
    parser.add_argument('--nonrecursive',
                        action='store_true',
                        help='Should flattening skip including directories recursively. Only used together with --flatten')
    parser.add_argument('--force-valid-aliases',
                    action='store_true',
                    help='Enables fatal errors when encountering invalid aliases')
    parser.add_argument('--override-system-clock-frequency',
                        action='store',
                        default=None,
                        help='Override default system clock frequency.')

    args = parser.parse_args()

    if args.nonrecursive and not args.flatten:
        parser.error("--nonrecursive requires --flatten")

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

def get_usb_cdc_acm_uart(dts_filename):
    # This function will check if CDC ACM UART is supported
    # and will return its USB device name if it is supported
    # and return None if it's not
    try:
        dt = dtlib.DT(dts_filename)
    except FileNotFoundError:
        logging.error(f'File not found: "{dts_filename}"')
        return None
    except Exception:
        logging.exception('Error while parsing DT')
        return None

    name_mapper = NameMapper()
    # Check if cdc-acm-uart is in dts file
    # Then get chosen shell-uart, then by that get parent of CDC ACM UART.
    # The CDC ACM UART is supposed to be a child of USB device.
    try:
        chosen = dt.get_node('/chosen')
        cdc_acm_uart_node = chosen.props['zephyr,shell-uart'].to_path()
        if cdc_acm_uart_node.props['compatible'].to_string() != 'zephyr,cdc-acm-uart':
            logging.info(f'cdc acm uart is not supported: "{dts_filename}"')
            return None
        return name_mapper.get_name(cdc_acm_uart_node.parent)
    except Exception:
        return None

def get_uart(dts_filename, only_compatible = False):
    try:
        dt = dtlib.DT(dts_filename)
        repl = generate(dts_filename)
    except FileNotFoundError:
        logging.error(f'File not found: "{dts_filename}"')
        return None
    except Exception:
        logging.exception('Error while parsing DT or generating REPL file')
        return None

    def verify_and_return_node(node):
        name = name_mapper.get_name(node)
        if only_compatible:
            # Try to match the node name to a peripheral in repl
            # it should mean that the UART is supported and can be used
            UART_REGEX = re.compile(fr"\s*{name}:\s*.+")
            if UART_REGEX.search(repl) is None:
                raise Exception(f'get_uart strict mode exception: node {name} not found in generated REPL')
        return name

    # The same caveat as in get_user_led0 applies: if we end up with a duplicate uart
    # name while generating the repl, all bets are off
    name_mapper = NameMapper()

    # First, try to use the chosen zephyr,shell-uart
    try:
        chosen = dt.get_node('/chosen')
        node = chosen.props['zephyr,shell-uart'].to_path()
        return verify_and_return_node(node)
    except Exception:
        pass

    # Then, chosen stdout-path = &uart;
    try:
        chosen = dt.get_node('/chosen')
        node = chosen.props['stdout-path'].to_path()
        return verify_and_return_node(node)
    except Exception:
        pass

    # Then, chosen stdout-path = "uart:115200n8"; (Linux style)
    try:
        chosen = dt.get_node('/chosen')
        alias = chosen.props['stdout-path'].to_string().split(':')[0]
        node = dt.get_node(alias)
        return verify_and_return_node(node)
    except Exception:
        pass

    # Then, let's check if it's a supported Xtensa platform. We only support
    # semihosting for those, so return a hardcoded string here
    try:
        node = dt.get_node('/cpus/cpu@0')
        compats = get_node_prop(node, 'compatible')
        if any('xtensa' in compat for compat in compats):
            return 'cpu0.uart'
    except Exception:
        pass

    # u-boot HACK: for "phytec,rk3288-pcm-947" hardcoded uart2 is used
    try:
        compats = get_node_prop(dt.get_node('/'), 'compatible', [])
        if any(compat == "phytec,rk3288-pcm-947" for compat in compats): 
            return "uart2"
    except Exception:
        pass
    
    # Then, chosen bootargs = "console=ttymxc1,115200";
    # ttymxc<i> = serial<i+1>
    try:
        node = dt.get_node('/chosen')
        bootargs = get_node_prop(node, 'bootargs')
        TTYMXC_REGEX = re.compile(r"ttymxc(?P<num>[\d]+)")
        node = TTYMXC_REGEX.search(bootargs)
        cons_index = int(node.group('num'))
        for node in dt.node_iter():
            node_lower = node.name.lower()
            if not (node_lower.startswith('serial') or
                   (node_lower.startswith('uart') and not node_lower.endswith('grp')) 
            ):
            # HACK: in u-boot some pinctrl nodes are called "uartXgrp" and they shouldn't be taken into account
                continue
            if not is_disabled(node) and 'reg' in node.props and cons_index == 0:
                return verify_and_return_node(node)
            cons_index -= 1
    except Exception:
       pass

    # Try to return any non-disabled node that looks vaguely like a uart
    try:
        for node in dt.node_iter():
            compats = get_node_prop(node, 'compatible')
            if any(x in node.name.lower() for x in ('uart', 'usart', 'serial')): 
                if any(keyword in node.name.lower() for keyword in ('clkuart', 'serial-flash')):
                    continue   
                if not is_disabled(node) and 'reg' in node.props:
                    return verify_and_return_node(node)
    except Exception:
       pass

    # Last check, search the generated repl itself for something UART look-alike
    UART_BACKUP_REGEX=re.compile(r"(?P<name>[0-9A-Za-z]+):\s+UART.")   
    node = UART_BACKUP_REGEX.search(repl)
    if node is not None:
        return node.group('name')

    # No uart found
    return None


def is_disabled(node):
    return get_node_prop(node, 'status', default='okay') == 'disabled'


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


# Returns the same as @functools.cache but is available before Python 3.9
# https://docs.python.org/3/library/functools.html#functools.cache
@functools.lru_cache(maxsize=None)
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
    if prop in ('compatible', 'device_type', 'model'):
        return val.to_strings()
    elif prop in ('interrupts', 'reg', 'ranges', 'alloc-ranges', 'dma-ranges', 'phandle'):
        return val.to_nums()
    elif prop in ('#address-cells', '#size-cells', '#interrupt-cells', 'cc-num', 'clock-frequency',
                  'riscv,ndev'):
        return val.to_num()
    elif prop in ('interrupt-parent',):
        return val.to_node()
    elif prop in ('interrupts-extended',):
        return get_interrupts_extended(val)
    elif prop in ('gpios',):
        fmt = 'pnn'
        while len(fmt) * 4 < len(val.value): 
            fmt += 'n'
        return get_prop_value(val, fmt)
    else:
        try:
            return val.to_string()
        except:
            pass
        try:
            return val.to_path()
        except:
            pass
    return None

def renode_model_overlay(compat, mcu, overlays):
    def _try_decode(e):
        if isinstance(e, str):
            # value is just a string - it's a model type in Renode
            return e, {}, None
        elif "type" in e.keys():
            # value is a dictionary containg the special 'type' entry containig a model type in Renode
            im = None
            m = e["type"]

            # treat all other entries in the dictionary as parameters for the model's construtor/properties
            a = copy.deepcopy(e)
            del a["type"]

            # check for yet another 'special' argument in the dictionary
            # here we assume 'irq_mappings' are only present if 'type' is provided as well
            if "irq_mappings" in e.keys():
                im = e["irq_mappings"]
                del a["irq_mappings"]

            return m, a, im
        else:
            # some other case, to be handled separately
            return None, {}, None

    models_entry = MODELS[compat]
    model, attribs, irq_mappings = _try_decode(models_entry)

    if model is None:
        # let's look for a special match notation containing multiple entries separated with '|' or a '_' else entry
        for entry in models_entry:
            for subentry in entry.split("|"):
                if subentry == "_" or subentry in overlays:
                    model, attribs, irq_mappings = _try_decode(models_entry[entry])
                    return model, compat, attribs, irq_mappings

    return model, compat, attribs, irq_mappings


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
def get_reg(node, variant='reg'):
    if node.parent:
        address_cells = get_node_prop(node.parent, '#address-cells', inherit=True)
        size_cells = get_node_prop(node.parent, '#size-cells', inherit=True)
    else:
        address_cells = 1
        size_cells = 1

    reg = get_node_prop(node, variant)
    while reg:
        address, reg = get_cells(reg, address_cells)
        if size_cells > 0:
            size, reg = get_cells(reg, size_cells)
        else:
            size = None
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
        "IRQControllers.NVIC": "nvic0",
        "IRQControllers.PlatformLevelInterruptController": "plic",
        "Miscellaneous.SiLabs.EFR32xG2_HFXO_2": "hfxo",
        "Miscellaneous.SiLabs.EFR32xG2_HFXO_3": "hfxo",
        "Miscellaneous.MAX32650_GCR": "gcr",
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
class RedistributorRegistrationRegion:
    address: int
    cpu: str

    @property
    def region_name(self) -> str:
        return 'redistributor'

    @property
    def registration_point(self) -> str:
        return 'sysbus'

    def get_constructor(self) -> str:
        return f'IRQControllers.ArmGicRedistributorRegistration {{ attachedCPU: {self.cpu}; address: {self.address:#x} }}'

@dataclass
class RegistrationRegion:
    addresses: list[int] = field(default_factory=list)
    size: Optional[int] = None
    region_name: Optional[str] = None
    registration_point: str = "sysbus"
    cpu: [Optional[str]] = None

    def __post_init__(self):
        if isinstance(self.addresses, list):
            return
        elif isinstance(self.addresses, int):
            self.addresses = [self.addresses]
        else:
            raise ValueError()

    @property
    def has_address_and_size(self) -> bool:
        return self.address is not None and self.size is not None

    @property
    def end(self) -> Optional[int]:
        if not self.has_address_and_size:
            return None
        return self.address + self.size

    @property
    def address(self) -> Optional[int]:
        if len(self.addresses) == 0:
            return None
        return self.addresses[0]

    @address.setter
    def address(self, value: Optional[int]) -> None:
        if value is None:
            self.addresses = []
            return
        self.addresses = [value]

    @end.setter
    def end(self, value: int) -> None:
        self.size = value - self.address

    def get_constructor(self) -> str:
        # We assume that each region will have an address, size and name in this case
        return f'Bus.BusMultiRegistration {{ address: {self.address:#x}; size: {self.size:#x}; region: "{self.region_name}" }}'

    @staticmethod
    def to_repl(regions):
        def _get_registration_str_simple(region):
            if region.cpu is None:
                if region.size is not None:
                    return f'{region.registration_point} <{region.address:#x}, +{region.size:#x}>'
                elif region.address is not None:
                    return f'{region.registration_point} {region.address:#x}'
                return region.registration_point
            else:
                NL = '\n'
                I4 = 4 * ' '
                # Per-core registration
                if region.address is None:
                    logging.error('Cannot perform per-core registration without an address!')
                    return None
                if region.size is not None:
                    bus_range_registartion = f'{I4}sysbus new Bus.BusPointRegistration {{ address: <{region.address:#x} +{region.size:#x}>; cpu: {region.cpu} }}'
                    return f'{{{NL}{bus_range_registartion}{NL}}}'
                bus_point_registartion = f'{I4}sysbus new Bus.BusPointRegistration {{ address: {region.address:#x}; cpu: {region.cpu} }}'
                return f'{{{NL}{bus_point_registartion}{NL}}}'

        if len(regions) == 0:
            return ''
        if len(regions) == 1:
            return _get_registration_str_simple(regions[0])

        # Multi-region registration
        parts = []
        for r in regions:
            if not r.region_name:
                # It's still possible to have a region without name here - use the same syntax as in single registration
                parts.append(_get_registration_str_simple(r))
            else:
                parts.append(f'sysbus new {r.get_constructor()}')
        return "{\n" + ";\n".join(f'{" "*8}{p}' for p in parts) + "\n    }"


@dataclass
class ReplFile:
    blocks: List[ReplBlock] = field(default_factory=list)
    tags: List[TagBlock] = field(default_factory=list)


    def get_block_by_name(self, name: str) -> Optional[ReplBlock]:
        return next((b for b in self.blocks if b.name == name), None)


    def get_blocks_by_type(self, type: str) -> iterable[ReplBlock] :
        if type == 'cpu':
            return filter(lambda x: 'cpu' in x.name and not 'timer' in x.name, self.blocks)

        raise Exception(f'Unsupported block type {type}')


    def add_block(self, block: ReplBlock, skip_duplicates: bool = False) -> None:
        block.owner = self

        # Check if the block already exists. If it does, skip it and display a warning.
        # XXX: This should never happen during the parsing stage of the `dts` file!
        # This mechanism is only intended to handle duplicates from overlay files.
        # There are two exceptions to this rule:
        # - `sysbus` named blocks, which are used for additional tags and SVDs,
        # - blocks where the `model` field is None, which are used for overwriting values from DTS-generated blocks.
        if skip_duplicates and self.get_block_by_name(block.name) and block.name != "sysbus" and block.model is not None:
            logging.warning(f'Skipping duplicated block {block.name}!')
            return

        self.blocks.append(block)


    def try_generate_tag(self, dts_node: dtlib.Node) -> bool:
        generated_any = False
        for node_reg in get_reg(dts_node):
            offset = node_reg[0]
            size = node_reg[1]
            if size is None or get_node_prop(dts_node.parent, 'device_type', [None], True)[0] == 'pci':
                # this should filter out CPUs, SPI, I2C, PCI devices, etc.
                continue
            name = dts_node.labels[0] if any(dts_node.labels) else dts_node.name
            tag = TagBlock(name, offset, size)
            if size == 0:
                continue
            self.tags.append(tag)
            generated_any = True
        return generated_any


    def filter_available_blocks(self) -> None:
        available_blocks = []
        visited_blocks = set()

        def dfs(block: ReplBlock) -> None:
            visited_blocks.add(id(block))
            for dep in block.depends:
                for other_block in self.blocks:
                    if dep in other_block.provides:
                        if id(other_block) not in visited_blocks:
                            dfs(other_block)
                        break
                else:
                    return
            available_blocks.append(block)

        for block in self.blocks:
            if id(block) not in visited_blocks:
                dfs(block)

        self.blocks = available_blocks


    def filter_64bit_reg_peripherals(self) -> None:
        self.blocks = [
            block for block in self.blocks
            if block.region is None or 
            all((block.region.end or address) <= 0x100000000
                for address in block.region.addresses)
        ]

    def merge_registration_points_duplicates(self, sized: List[ReplBlock]):
        # merge overlapping memory with multiple registration points
        # NOTE: this is a simplified solution 
        # 1. will work for this case:
        # memory: Memory.MappedMemory @ { sysbus 0x0; sysbus 0x80000000 }
        #   size: 0x40000000
        # memory: Memory.MappedMemory @ { sysbus 0x80000000 }
        #   size: 0x40000000
        # it'll output
        # memory: Memory.MappedMemory @ { sysbus 0x0; sysbus 0x80000000 }
        #   size: 0x40000000
        #
        # 2. will drop one of the addresses in the following case:
        # memory: Memory.MappedMemory @ { sysbus 0x0; sysbus 0x80000000 }
        #   size: 0x40000000
        # memory: Memory.MappedMemory @ { sysbus 0xb16b00b5; sysbus 0x80000000 }
        #   size: 0x40000000 
        # 0xb16b00b5 will be dropped and it'll output
        # memory: Memory.MappedMemory @ { sysbus 0x0; sysbus 0x80000000 }
        #   size: 0x40000000

        sized_not_memory = []
        sized_memory = []
        sized_memory_filtered = []

        for block in sized:
            (sized_not_memory, sized_memory)[block.model is not None and block.model.startswith('Memory')].append(block)
        for block in sized_memory:
            any_overlaps = False
            for target_block in sized_memory:
                target_region = target_block.region
                if (
                    block.region.size == target_region.size 
                    and set(block.region.addresses) & set(target_region.addresses)
                    and len(block.region.addresses) < len(target_region.addresses)
                ):
                    any_overlaps = True
                    break
            if not any_overlaps:
                sized_memory_filtered.append(block)
        return sized_not_memory + sized_memory_filtered


    def merge_overlapping_blocks(self) -> None:
        # split into blocks of known and unknown size
        sized = []
        unsized = []
        for block in self.blocks:
            (unsized, sized)[block.region is not None and block.region.has_address_and_size].append(block)

        def are_compatible(model_a, model_b):
            return model_a == model_b or (model_a.startswith('Memory') and model_b.startswith('Memory'))

        # merge overlapping sized blocks
        # NOTE: currently, only memory blocks are merged, other overlapping blocks are removed
        sized_merged = []
        for block in sorted(sized, key=lambda b: (b.region.address, b.region.end)):
            if (
                sized_merged
                and are_compatible(sized_merged[-1].model, block.model)
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
        self.blocks = self.merge_registration_points_duplicates(sized_merged) + unsized


    def remove_dangling_irqs(self) -> None:
        existing_devices = set.union(*[b.provides for b in self.blocks])
        for block in self.blocks:
            filtered_content = []
            for line in block.content:
                irq_dest = re.search(r"->\s*(\w+)(#\d+)?\s*@", line)
                if irq_dest is None or irq_dest.group(1) in existing_devices:
                    filtered_content.append(line)
            block.content = filtered_content


    def _get_grouped_tags(self) -> iterable[TagBlock]:
        if len(self.tags) < 2:
            return self.tags

        # aggregate tags that occupy the same area, Renode doesn't like that
        s = sorted(self.tags, key=lambda t: t.offset)
        r = []
        accumulator = copy.copy(s[0])
        for t in s[1:]:
            if t.offset < (accumulator.offset + accumulator.size):
                # the next tag partially overlaps the previous one - join them
                if (t.offset + t.size) > (accumulator.offset + accumulator.size):
                    accumulator.size = (t.offset + t.size) - accumulator.offset

                if accumulator.name != t.name:
                    accumulator.name += ' / ' + t.name
            else:
                r.append(accumulator)
                accumulator = copy.copy(t)
        r.append(accumulator)
        return r


    def __str__(self) -> str:
        # generate tags first
        # overlays can add more precise tagging with return values
        body = ''
        if any(self.tags):
            # Generate tags in a size-based descending order
            grouped_tags = self._get_grouped_tags()
            grouped_tags.sort(reverse=True)
            body = 'sysbus:\n    init:\n'
            for tag in grouped_tags:
                body += 8 * ' ' + str(tag) + '\n'
            body += '\n'
        body += '\n'.join([str(b) + '\n' for b in self.blocks])
        return body


@dataclass
class ReplBlock:
    name: str
    model: Optional[str]
    depends: Set[str]
    provides: Set[str]
    content: List[str]
    region: Optional[RegistrationRegion] = None
    owner: ReplFile = None # this field will be filled when adding this block to a ReplFile


    def get_depending_blocks(self) -> list[ReplBlock]:
        return [b for b in self.owner.blocks if self.name in b.depends]


    def __str__(self) -> str:
        return '\n'.join(self.content)


@dataclass
class TagBlock:
    name: str
    offset: int
    size: int

    def __str__(self) -> str:
        return  f'Tag <{hex(self.offset)} {hex(self.size)}> "{self.name}"'

    def __lt__(self, other):
        return self.size < other.size


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
     (?P<registration_point>[{\w]+)     # Capture the registration point (or '{') in `model @ registration_point`
     (                                  # Optional: address
      \s+                               # Skip whitespace
      (?P<address>(0[xX])?[0-9a-fA-F]+) # Capture the address
     )?                                 # End address optional part
    )?                                  # Outer optional (model/registration point) end
    """,
    re.VERBOSE,
)

# Warning: this will match multiple registration points where registration point matches [A-Za-z]+ (so no numbers)
OVERLAY_MULTIPLE_REG = re.compile(r"(?P<registration_point>[A-Za-z]+)(\s+(?P<address>(0[xX])?[0-9a-fA-F]+))")

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
        registration_point = node.group('registration_point')
    
        # We treat multiple registration points as sysbus
        addresses = []
        if registration_point == '{':
            points = OVERLAY_MULTIPLE_REG.findall(non_comment_lines[0])
            registration_point = 'sysbus'
            addresses = sorted([int(m[2], 0) for m in points]) # m[2] = address
            
        region = None
        if registration_point:
            # Only creating entries actually provide the name
            provides.add(node.group('name'))
            depends.add(registration_point)
            if node.group('address'):
                if len(addresses) == 0:
                    addresses = [0]
                addresses[0] = int(node.group('address'), 0)
            
            if len(addresses) != 0:
                region = RegistrationRegion(addresses=addresses, registration_point=registration_point)
                for line in part:
                    if 'size:' in line:
                        region.size = int(line.split()[1], 16)
                        break
        else:
            # Non-creating entries depend on a creating entry
            depends.add(node.group('name'))
        blocks.append(
            ReplBlock(node.group('name'), node.group('model'), depends, provides, part, region)
        )

    return blocks


def can_be_memory(node):
    chosen_node = None
    try:
        chosen_node = node.dt.get_node("/chosen")
    except dtlib.DTError:
        pass  # No "/chosen" node

    if chosen_node:
        sram = get_node_prop(chosen_node, "zephyr,sram")
        flash = get_node_prop(chosen_node, "zephyr,flash")
        if sram and node == sram:
            return True
        if flash and node == flash:
            return True
    possible_names = ('ram', 'flash', 'partition', 'memory')
    return len(node.props) == 1 and 'reg' in node.props \
        and any(x in node.name.lower() for x in possible_names) \
        and 'reserved-memory' not in node.path.split('/') \
        and (node.parent is None or not (get_model(node.parent) or '').startswith('Memory'))


def get_model(node, mcu=None, overlays=tuple()):
    node_compatible = next(filter(lambda x: x in MODELS, get_node_prop(node, 'compatible', [])), None)
    if node_compatible:
        node_model, _, _, _ = renode_model_overlay(node_compatible, mcu, overlays)
        return node_model
    return None


def get_overlays(dt):
    if not isinstance(dt, dtlib.DT):
        dt = get_dt(dt)
    if not dt:
        return set()

    # get platform compat names
    platform = get_node_prop(dt.get_node('/'), 'compatible', [])
    if not platform:
        # if there is no compatible for the platform, let's try model
        platform = get_node_prop(dt.get_node('/'), 'model', [])

    # get soc compat names
    if dt.has_node('/soc'):
        soc = get_node_prop(dt.get_node('/soc'), 'compatible', [])
    else:
        soc = []

    return set(soc + platform)


def generate(filename, override_system_clock_frequency=None):
    name_mapper = NameMapper()
    dt = get_dt(filename)
    if dt is None:
        return ''

    # `sysbus` and `none` are registration points provided by Renode itself
    repl_file = ReplFile()
    repl_file.add_block(ReplBlock("sysbus", None, set(), {'sysbus', 'none'}, ['// autogenerated']))
    nodes = sorted(dt.node_iter(), key=lambda x: get_node_prop(x, 'compatible')[0] if 'compatible' in x.props else '')

    # get mcu compat name
    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    mcu_compat = None
    if mcu is not None:
        mcu_compat = get_node_prop(mcu, 'compatible')[0]
    logging.debug(f'mcu_compat: {mcu_compat}')

    # get overlays
    overlays = get_overlays(dt)
    address_space_32bit = False

    main_compatible = None
    for node in nodes:
        compatible = get_node_prop(node, 'compatible')
        if compatible:
            if node.name == "/":
                main_compatible = compatible[0]
                logging.debug(f'main compat string is = {main_compatible}')
                break

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
                    logging.debug(f'Node {node.name} will be treated as memory')
                else:
                    logging.debug(f'Node {node.name} has no compat string or device_type and cannot be treated as memory. Skipping...')
                    repl_file.try_generate_tag(node)
                    continue

        if 'memory' in compatible:
            is_heuristic_memory = True

        # look at all compat strings and take the first one that has a Renode model
        # if none of them do, skip the node
        compat = next(filter(lambda x: x in MODELS, compatible), None)
        if compat is None:
            logging.info(f'Node {node.name} does not have a matching Renode model. Skipping...')
            repl_file.try_generate_tag(node)
            continue

        # not sure why this is needed. We need to investigate the RCC->RTC dependency.
        if is_disabled(node) and not node.name.startswith('rtc') and not "renesas,smartbond-timer" in compat:
            logging.info(f'Node {node.name} disabled. Skipping...')
            repl_file.try_generate_tag(node)
            continue


        # get model name and addr
        _, _, addr = node.name.partition('@')

        # Attempt to retrieve the first element from the "reg" property 
        # and verify whether the `unit-address` matches it. If not,
        # display a warning and prefer the addr from reg.
        try:
            first_address_in_reg = list(get_reg(node))[0][0]
            addr_int = translate_address(int(addr, 16), node)

            if first_address_in_reg != addr_int:
                logging.warning(
                    f'node: {node.name} unit-address ({hex(addr_int)}) does not match node.reg[0] ({hex(first_address_in_reg)}), '
                    'using address from node.reg[0]'
                )
                addr = format(first_address_in_reg, 'X')
        except:
            pass

        name = name_mapper.get_name(node)

        logging.info(f'Node {node.name} mapped to {name}...')

        # decide which Renode model to use
        model, compat, attribs, irq_mappings = renode_model_overlay(compat, mcu_compat, overlays)
        if model is None:
            # There is no model for the given "specialized" SoC compat string, but they might exist for other SoC variants
            logging.info(f'Node {node.name}, compat {compat} has no matching specific model - does the JSON have "_" clause? Skipping...')
            repl_file.try_generate_tag(node)
            continue
        model = str(model)
        logging.info(f'Node {node.name} mapped to {name} was assigned model {model}')

        dependencies = set()
        provides = {name}
        regions = []
        indent = []

        if model in ("CPU.ARMv7A",):
            address_space_32bit = True 
        
        # special handling of 'st,stm32-ethernet' compat. Sometimes the 'mac' node is a child of the 'ethernet' node,
        # and has no address specified. If possible, extract the addr from the parent node.
        if compat == 'st,stm32-ethernet' and not addr:
            if not (parent := node.parent):
                logging.warn(f'Unable to find address for {compat}. Dropping {model}')
                continue
            _, _, addr = parent.name.partition('@')

        # special handling of flash devices as those should be treated as memory
        flash_nodes = [
            'issi,is25lp128',
            'nxp,imx-flexspi-hyperflash',
            'nxp,imx-flexspi-mx25um51345g',
            'nxp,imx-flexspi-nor',
            'nxp,s32-qspi-nor',
            'nxp,xspi-mx25um51345g',
        ]
        if compat in flash_nodes:
            # try to take the 2nd reg entry from the parent controller
            if not node.parent:
                logging.info(f'Parent node for {name} of type {compat} not found. Skipping...')
                continue
            parent_regs = list(get_reg(node.parent))
            if len(parent_regs) != 2:
                logging.info(f'Expected 2 reg entries for {name} parent. Skipping...')
                continue

            base, size = parent_regs[1]
            regions = [RegistrationRegion(base)]
            indent.append(f'size: {hex(size)}')
        elif addr and not model.startswith('CPU.'):
            addr = int(addr.split(',')[0], 16)
            addr = translate_address(addr, node)
            if addr % 4 != 0:
                logging.info(f'Node {node.name} has misaligned address {addr}. Skipping...')
                repl_file.try_generate_tag(node)
                continue

            # hack for x86/ioport
            if model == 'UART.NS16550' and addr in [0x2f8, 0x3f8]:
                addr += 0xE0000000

            if model == 'Timers.TegraUsecTimer':
                # the microsecond timer is at offset 0x10 from the base of the timer block
                addr += 0x10

            # In nRF52 GPIO, registers are starting at offset 0x500.
            # Other nRF SoCs use the same peripheral, but in their case there is no offset.
            if dt.has_node('/soc'):
                soc_compat = get_node_prop(dt.get_node('/soc'), 'compatible', [])
            else:
                soc_compat = []
            if model == 'GPIOPort.NRF52840_GPIO' and 'nordic,nrf52' in soc_compat:
                addr += 0x500

            if (
                any(map(lambda x: x in compat,
                    ['stm32-gpio', 'stm32-timers', 'silabs,gecko', 'silabs,usart-uart', 'gaisler,irqmp',
                     'gaisler,gptimer', 'gaisler,apbuart', 'xlnx,xuartps']))
                or any(map(lambda x: x in model,
                    ['UART.STM32_UART', 'I2C.TegraI2CController', 'IRQControllers.PlatformLevelInterruptController']))
            ):
                # sized sysbus registration for peripherals that require an explicit size
                _, size = next(get_reg(node))
                regions = [RegistrationRegion(addr, size)]
            # All NVICs need to be per-core registered
            # Only handle the first one here, all others will be "faked" later
            elif name == 'nvic0':
                # weird mismatch, need to investigate, manually patching for now
                addr &= ~0x100
                regions = [RegistrationRegion(addr, cpu='cpu0')]
            else:
                # unsized sysbus registration
                regions = [RegistrationRegion(addr)]
        else:
            # assume sysbus registration without address like for CPUs
            regions = [RegistrationRegion()]

        if compat == 'renesas,ra-sci':
            # get the name from the first enabled child
            child = next((n for n in node.nodes.values() if not is_disabled(n)), None)
            if child:
                name = name_mapper.get_name(child)

        if 'nordic,nrf-vevif-task' in compat:
            ipc = dt.get_node('/ipc')
            if ipc == None:
                continue
            for ipc_node in (i for i in ipc.nodes.values() if not is_disabled(i)):
                mboxes = parse_phandles_and_nums(dt, ipc_node, "mboxes")
                if get_node_prop(node, 'phandle')[0] in mboxes:
                    bellboard = next((n for n in nodes \
                                        if (get_node_prop(n, 'phandle') != None \
                                        and get_node_prop(n, 'phandle')[0] == mboxes[0])), None)
                    if bellboard != None and get_node_prop(bellboard, 'phandle')[0] in mboxes:
                        target_name = bellboard.labels[0].replace('_','')
                        # indexes come from the parent "zephyr,ipc-icmsg" node, from the `mboxes` prop:
                        # mboxes = < &cpusec_bellboard 0xc >, < &cpuapp_bellboard 0x0 >;
                        indent.append(f'{mboxes[3]}->{target_name}@{mboxes[1]}')
                    break

        # special multi-registration address for GIC
        if model == 'IRQControllers.ARM_GenericInterruptController':
            if overlays & {'fsl,lx2160a', 'fsl,ls2080a', 'samsung,exynos5250'}:
                # fsl,lx2160a/ls2080a don't have CPU in uboot's device tree
                # we add gic together with cpu in the appropriate overlays
                continue

            def arm_gic_get_region(addr, size, name):
                if name == 'redistributor':
                    cpus = repl_file.get_blocks_by_type('cpu')
                    return [RedistributorRegistrationRegion(addr + (i * 0x20000), cpu.name) for i, cpu in enumerate(cpus)]
                else:
                    return [RegistrationRegion(addr, size, name)]

            region_names = ('distributor', 'cpuInterface',)
            if compat in ('arm,gic-v3', 'arm,gic-v4',):
                region_names = ('distributor', 'redistributor',)

            regions = [
                region
                for (region_addr, region_size), region_name
                in zip(get_reg(node), region_names)
                for region
                in arm_gic_get_region(region_addr, region_size, region_name)
            ]

        # check the registration point of guessed memory peripherals
        if is_heuristic_memory:
            node_reg = next(get_reg(node), None)
            if node_reg and regions:
                regions[0].address = node_reg[0]

        for attr in attribs:
            if (isinstance(attribs[attr], str)):
                indent.append("%s: \"%s\"" % (attr, attribs[attr]))
            elif (isinstance(attribs[attr], bool)):
                if attribs[attr]:
                    indent.append("%s: true" % (attr))
                else:
                    indent.append("%s: false" % (attr))
            else:
                indent.append("%s: %s" % (attr, str(attribs[attr])))

        # additional parameters for peripherals
        if model == 'IRQControllers.PlatformLevelInterruptController':
            # 1023 is currently maximum value supported by the PLIC model in Renode
            ndev = min(1023, get_node_prop(node, 'riscv,ndev', 1023))
            indent.append(f'numberOfSources: {ndev}')
        if model == "IRQControllers.CoreLevelInterruptor":
            frequency = override_system_clock_frequency or 1000000
            indent.append(f'frequency: {frequency}')
        if model == 'Miscellaneous.STM32L0_RCC':
            indent.append('systick: nvic0')
            dependencies.add('nvic0')
        if model == 'MTD.STM32WBA_FlashController':
            children = node.nodes.values()
            child = next(iter(children), None)
            if child == None:
                logging.warn(f'{model} should have exactly one flash child node, but got none. Dropping {model}')
                repl_file.try_generate_tag(node)
                continue
            if len(children) > 1:
                logging.warn(f'{model} should have only one flash assigned, but got: {[c.name for c in children]}. Selecting {child.name}.')
            child_name = name_mapper.get_name(child)
            indent.append(f'flash: {child_name}')
            dependencies.add(child_name)
        if model == 'Network.SynopsysDWCEthernetQualityOfService':
            regions += [RegistrationRegion(addr + 0xC00, 0x200, 'mtl'), RegistrationRegion(addr + 0x1000, 0x200, 'dma')]
        if model == 'IRQControllers.RenesasRA_ICU':
            indent.append('nvic: nvic0')
            dependencies.add('nvic0')
        if model == 'Timers.RenesasDA_Watchdog':
            indent.append('nvic: nvic0')
            dependencies.add('nvic0')
        if model == 'Miscellaneous.SiLabs.EFR32xG2_CMU_3':
            indent.append('hfxo: hfxo')
            dependencies.add('hfxo')
        if model == 'Miscellaneous.MAX32650_GCR':
            indent.append('nvic: nvic0')
            dependencies.add('nvic0')
        if model == 'UART.MAX32650_UART':
            indent.append('gcr: gcr')
            dependencies.add('gcr')

        # additional parameters for python peripherals
        if compat.startswith('fsl,imx6') and compat.endswith('-anatop'):
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
        elif compat.startswith('fsl,imx') and compat.endswith('-ccm'):
            indent.append('size: 0x4000')
            indent.append('initable: false')
            indent.append('// 0x14: cbcdr')
            indent.append('// 0x18: cbcmr')
            indent.append('// 0x1c: cscmr1')
            indent.append('script: "request.value = {0x14: 3<<8 | 7<<10, 0x18: 2<<12, 0x1c: 0x3f}.get(request.offset, 0)"')
        elif compat == 'marvell,mbus-controller':
            indent.append('size: 0x200')
            indent.append('initable: false')
            indent.append('// 0x180: win_bar')
            indent.append('// 0x184: win_sz')
            indent.append('script: "request.value = {0x180: 0x0, 0x184: 0xf000001}.get(request.offset, 0)"')
        elif compat == 'xlnx,zynqmp-ipi-mailbox':
            # the address of the Xilinx ZynqMP IPI mailbox is defined in its child node
            for child in node.nodes.values():
                regions = [RegistrationRegion(int(child.unit_addr, 16))]
                break
            else:
                logging.info(f'ZynqMP mailbox has no children: {node}')
        elif compat == 'nxp,s32-siul2-eirq':
            name = name_mapper.get_name(node.parent)
            provides = {name}
            regions = [RegistrationRegion(int(node.parent.unit_addr, 16))]

        if compat.startswith('arm,cortex-m'):
            cpu_number = name[-1]
            indent.append(f'nvic: nvic{cpu_number}')
            dependencies.add(f'nvic{cpu_number}')
            overlays.add('cortex-m')

        if model == 'CPU.RiscV32' or model == 'CPU.Ri5cy':  # We use CPU.RiscV32 as a generic model for all RV CPUs and fix it up here
            isa = get_node_prop(node, 'riscv,isa', 'rv32imac')
            # The g extension already contains Zicsr and Zifencei
            if 'rv64g' not in isa and 'rv32g' not in isa:
                if '_zicsr' not in isa:
                    isa += '_zicsr'
                if '_zifencei' not in isa:
                    isa += '_zifencei'
            indent.append(f'cpuType: "{isa}"')

            if '64' in isa:
                model = 'CPU.RiscV64'

            # Use CPU.VexRiscv for LiteX and Fomu
            if overlays & {'litex,vexriscv', 'kosagi,fomu'}:
                model = 'CPU.VexRiscv'
            elif "openisa,rv32m1" in overlays or "telink,tlsr9518adk80d" in overlays or "_xandes" in isa or compat in ["intel,niosv", "nuclei,bumblebee", "neorv32-cpu", "espressif,riscv", "ite,riscv-ite", "litex,vexriscv-standard", "sensry,sy1xx", "nordic,vpr"]:
                indent.append('timeProvider: empty')
            else:
                indent.append('timeProvider: clint')
                dependencies.add('clint')
                indent.append('allowUnalignedAccesses: true')
            # If no hartId is defined (from models.json), attempt to generate
            # it from CPU node index in DTS file.
            if not any('hartId: ' in s for s in indent):
                hartid = 0
                if len(node.name.split("@")) >= 2:
                    hartid = int(node.name.split("@")[1], 16)
                indent.append(f'hartId: {hartid}')
            if any(c.startswith('riscv,sifive') or
                   c.startswith('starfive,rocket') or
                   c in [
                        'sifive,e31', 'sifive,u54', 'sifive,u54-mc', 'sifive,u74-mc',
                        'espressif,riscv', 'qemu,riscv-virt', 'openhwgroup,cva6',
                        'thead,c910', 'thead,c906', 'amd,mbv64'
                   ] for c in compatible) \
            or "_xandes" in isa:
                indent.append('privilegedArchitecture: PrivilegedArchitecture.Priv1_10')
            else:
                indent.append('privilegedArchitecture: PrivilegedArchitecture.Priv1_09')
        if model in ("CPU.ARMv8A", "CPU.ARMv8R", "CPU.ARMv7A", "CPU.ARMv7R") and name != "cpu0":
            # We generate the cpu0 timer along with correct interrupt connections
            # while processing the timer node in the dts, and we generate 'fake'
            # timers for other cores here for now
            timer_name = f'{name}_timer'
            timer_lines = [f'{timer_name}: Timers.ARM_GenericTimer @ {name}', '    frequency: 62500000']
            generic_timer = ReplBlock(timer_name, 'Timers.ARM_GenericTimer', {name}, {timer_name}, timer_lines)
            repl_file.add_block(generic_timer)
        if model in ("CPU.ARMv8A", "CPU.ARMv8R", "CPU.ARMv7A", "CPU.ARMv7R"):
            # We use our CPU number as the CPU ID instead of the reg address
            # This relies on the fact that the name will have been changed to "cpu{n}"
            cpuIdx = int(name.replace("cpu", ""))
            # For Cortex-A55 core number is stored in affinity level 1.
            # For older cpus it's stored in affinity level 0.
            cpuId = f'0x{(cpuIdx << 8):x}' if attribs.get('cpuType', '') == 'cortex-a55' else cpuIdx
            indent.append(f'cpuId: {cpuId}')
            if not any(compat.startswith(x) for x in ["arm,arm11", "arm,arm9", "arm,cortex-a8"])  \
            and not main_compatible in ["beagle,beaglebone-ai64", "beagle,beagley-ai", "grinn,am335x-chiliboard"]  \
            and not compat == "arm,cortex-a5":
                indent.append('genericInterruptController: gic')
                dependencies.add('gic')
        if model in ("CPU.X86", "CPU.X86_64"):
            cpu_number = name[-1]
            lapic_name = f'intcloapic{cpu_number}'
            indent.append(f'lapic: {lapic_name}')
            dependencies.add(lapic_name)

        if model == "CPU.Sparc":
            sysbus_endianness = ReplBlock('sysbus', None, {'sysbus'}, set(), ['sysbus:', '    Endianess: Endianess.BigEndian'])
            repl_file.add_block(sysbus_endianness)

        # Create 'model_name' only visible to 'cpu_name' at 'offset'
        def create_per_cpu_model(cpu_name, model_name, csharp_model_name, offset, arguments):
            cpu_number = cpu_name[-1]
            model_name = f'{model_name}{cpu_number}'
            region = RegistrationRegion.to_repl([RegistrationRegion(offset, cpu=f'cpu{cpu_number}')])
            block = f'{model_name}: {csharp_model_name} @ {region}'
            arguments = [f'    {key}: {val}' for key, val in arguments.items()]

            repl_block = ReplBlock(model_name, csharp_model_name, {cpu_name}, {model_name}, [block, *arguments])
            repl_file.add_block(repl_block)

        # For CortexM and x86_64 multi core systems, we generate the IRQ controllers
        # along with correct interrupt connections while processing the IRQ controller node in the dts,
        # and we generate 'fake' IRQ controllers for other cores.

        # fake NVICs for multi-core ARM systems
        if model in ("CPU.CortexM") and name != "cpu0":
            create_per_cpu_model(name, "nvic", "IRQControllers.NVIC", 0xe000e000, {})

        # fake LAPICs for multi-core x86_64 systems
        if model in ("CPU.X86_64") and name != "cpu0":
            create_per_cpu_model(name, "intcloapic", "IRQControllers.LAPIC", 0xfee00000, {"id": name[-1]})

        # additional parameters for STM32F4_RCC
        if model == 'Miscellaneous.STM32F4_RCC':
            indent.append('rtcPeripheral: rtc')
            dependencies.add('rtc')

        # Temporary solution until Zephyr merges model name in dts
        if model == "Timers.NEORV32_MachineSystemTimer":
            indent.append('-> cpu0@23')


        if compat == "openisa,rv32m1-intmux":
            interruptNumbers = [
                get_node_prop(entry, "interrupts", 0)[0]
                for entry in node.nodes.values()
                if 'openisa,rv32m1-intmux-ch' in get_node_prop(entry, "compatible", False)
            ]
            indent.append(f"{list(range(len(interruptNumbers)))} -> cpu0@{interruptNumbers}")


        if compat in ["openisa,rv32m1-lptmr", "openisa,rv32m1-lpuart"]:
            interrupt_parent = get_node_prop(node, "interrupt-parent")
            interrupt_number = get_node_prop(node, "interrupts", 0)[0]

            if "openisa,rv32m1-intmux" in get_node_prop(interrupt_parent.parent, "compatible", ""):
                channel_match = re.match(r"intmux(\d)_ch(\d)", interrupt_parent.labels[0])
                if channel_match:
                    indent.append(f"IRQ -> intmux{channel_match.group(1)}#{channel_match.group(2)}@{interrupt_number}")
            else:
                indent.append(f"IRQ -> cpu0@{interrupt_number}")


        i2c_sensors = [
            'Sensors.TMP103',
            'Sensors.TMP108',
            'Sensors.SI7210',
            'I2C.BME280',
            'I2C.SHT45'
        ]

        if model in i2c_sensors:
            if len(node.parent.labels) == 0:
                logging.warning(f"Node {node} has no labels! Dropping {model}")
                continue

            i2c_name = name_mapper.get_name(node.parent)
            if not i2c_name.startswith("i2c"):
                logging.warning(f"Parent of {node} is not an I2C controller! Dropping {model}")
                continue
            i2c_addr = int(node.unit_addr, 16)
            regions = [RegistrationRegion(addresses=[i2c_addr], registration_point=i2c_name)]

        if model == 'Miscellaneous.LED':
            gpios = list(get_node_prop(node, 'gpios'))
            if not gpios:
                logging.info(f'LED {node.name} has no gpios property, skipping...')
                continue
            gpio, num, gpio_flags = gpios[0][:3]
            gpio_compat = get_node_prop(gpio, 'compatible', [])
            if 'nxp,s32-gpio' in gpio_compat:
                # We have to translate gpio pin to pad
                gpio_addr = int(gpio.unit_addr, 16)
                gpio_base_addr = gpio_addr & ~0xFFFF
                port_index = int((gpio_addr - gpio_base_addr - 0x1700) / 0x4)
                num = port_index * 32 + num
                gpio = gpio.parent
            
            gpio_model = get_model(gpio)
            if not gpio_model or not gpio_model.startswith("GPIO"):
                # don't add invalid GPIO connections
                continue

            active_low = (gpio_flags & 1) == 1
            if active_low:
                indent.append('invert: true')
            gpio_name = name_mapper.get_name(gpio)
            regions = [RegistrationRegion(num, registration_point=gpio_name)]

            gpio_connection = ReplBlock(gpio_name, None, {gpio_name, name}, set(),
                                        [f'{gpio_name}:', f'    {num} -> {name}@0'])
            repl_file.add_block(gpio_connection)

        if model.startswith('Timers'):
            if 'cc-num' in node.props:
                count = str(get_node_prop(node, "cc-num"))
                # hack for NRF52840
                if count == "8" and model == "Timers.NRF52840_Timer":
                    count = "6"
                indent.append(f'numberOfEvents: {count}')
        if model.startswith('Memory') and not any(i for i in indent if i.startswith('size:')):
            if 'dma-ranges' in node.props:
                # Device Tree Spec 2.3.9 (child-bus-address, parent-bus-address, length)
                # we assume just a single triplet (although it could be a list of triplets)
                dma_ranges = get_node_prop(node, 'dma-ranges')
                regions = [RegistrationRegion([dma_ranges[1]])]
                size = dma_ranges[2]
                indent.append(f'size: {hex(size)}')
            elif 'reg' in node.props:
                _, size = next(get_reg(node))
                # increase OCRAM size for imx6 platforms
                # the device trees in U-Boot all have 0x20000, but some platforms
                # actually have 0x40000 and the config headers reflect this, which
                # would make the stack end up outside of memory if the size from
                # the device tree was used
                if any('imx6' in p for p in overlays):
                    if node.labels and 'ocram' in node.labels[0]:
                        size = 0x40000
                if size:
                    indent.append(f'size: {hex(size)}')
                else:
                    # do not generate memory regions of size 0
                    continue
            elif 'alloc-ranges' in node.props:
                addr, size = next(get_reg(node, 'alloc-ranges'))
                regions = [RegistrationRegion(addr)]
                indent.append(f'size: {hex(size)}')
            else:
                logging.warning('Memory node {node.name} has no reg, skipping')
                continue

        irq_dest_nodes = []
        irq_numbers = []
        irq_local_indices = {}
        # decide which IRQ destinations to use in Renode model
        # these IRQ ctrls get special treatment
        if compat.endswith('nvic'):
            indent.append('-> cpu0@0')
            dependencies.add('cpu0')
            if override_system_clock_frequency:
                indent.append(f'systickFrequency: {override_system_clock_frequency}')
        elif model == 'IRQControllers.ARM_GenericInterruptController':
            # Select correct GIC version
            gic_ver = compat.split(',')[-1]
            gic_renode_ver = {
                'gic-v1': 'GICv1',
                'cortex-a9-gic': 'GICv1',
                'gic-v2': 'GICv2',
                'gic-400': 'GICv2',
                'cortex-a7-gic': 'GICv2',
                'cortex-a15-gic': 'GICv2',
                'gic-v3': 'GICv3',
                'gic-v4': 'GICv4',
            }.get(gic_ver)
            if gic_renode_ver:
                indent.append(f'architectureVersion: IRQControllers.ARM_GenericInterruptControllerVersion.{gic_renode_ver}')
            else:
                logging.warning(f'Unknown GIC version for compatible "{compat}"')
        elif compat == 'gaisler,irqmp':
            indent.append('0 -> cpu0@0 | cpu0@1 | cpu0@2')
            dependencies.add('cpu0')
        elif 'interrupts' in node.props and model != "Memory.MappedMemory":
            interrupt_parent = get_node_prop(node, 'interrupt-parent', inherit=True)
            if interrupt_parent is not None:
                interrupt_cells = get_node_prop(interrupt_parent, '#interrupt-cells', default=2)
                irq_numbers = get_node_prop(node, 'interrupts')[::interrupt_cells]

                if compat == 'renesas,ra-sci' and 'uart' in node.nodes and node.nodes['uart'].props['compatible'].to_string() == 'renesas,ra-uart-sci':
                    # there is a difference between IRQ description for 'renesas,ra-uart-sci' and 'renesas,ra-sci-uart' drivers!
                    # we need to use the last cell and additionally use upper two bytes of the value there
                    irq_numbers = get_node_prop(node, 'interrupts')[(interrupt_cells - 1)::interrupt_cells]
                    irq_numbers = [i >> 8 for i in irq_numbers]

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
        elif model == 'IRQControllers.LAPIC':
            # Each CPU has it's own LAPIC
            name += '0'
            provides = {name}
            indent.append('IRQ -> cpu0@0')
            indent.append('id: 0')
            dependencies.add('cpu0')

        for i, irq_dest_node in enumerate(irq_dest_nodes):
            irq_dest_compatible = get_node_prop(irq_dest_node, 'compatible', [])
            # treat the RISC-V CPU interrupt controller as the CPU itself
            if 'riscv,cpu-intc' in irq_dest_compatible:
                irq_dest_nodes[i] = irq_dest_node.parent
            # treat the VexRiscv intc as the CPU itself
            if 'vexriscv-intc0' in irq_dest_compatible:
                irq_dest_nodes[i] = mcu

        # assign IRQ signals (but not when using TrivialUart or PythonPeripheral)
        if irq_dest_nodes and model != 'Python.PythonPeripheral':
            if irq_mappings:
                irq_names = irq_mappings
            # decide which IRQ names to use in Renode model
            # to be moved to models.json
            elif compat == 'st,stm32-rtc':
                irq_names = ['AlarmIRQ']
            elif (compat in ['nxp,kinetis-lpuart', 'nxp,lpuart', 'nxp,kinetis-uart', 'silabs,gecko-leuart', 'sifive,uart0', 'st,stm32-adc']
                or model in ['UART.STM32F7_USART', 'SPI.STM32SPI']):
                irq_names = ['IRQ']
            elif (compat in ['nuvoton,npcx-uart']):
                irq_names = ['IRQ', 'DMAReceive']
            elif compat in ['silabs,gecko-uart', 'silabs,gecko-usart', 'silabs,usart-uart']:
                irq_names = ['ReceiveIRQ', 'TransmitIRQ']
            elif compat in ['gaisler,gptimer', 'ambiq,counter']:
                irq_names = ['0']
            elif model == "Timers.IMX_GPTimer":
                irq_names = ['IRQ']
            elif model == 'Timers.IMXRT_PWM':
                irq_names = ['0', '1', '2']
            elif model == 'Timers.ARM_GenericTimer':
                irq_names = ['EL3PhysicalTimerIRQ', 'EL1PhysicalTimerIRQ', 'EL1VirtualTimerIRQ', 'NonSecureEL2PhysicalTimerIRQ', 'NonSecureEL2VirtualTimerIRQ']
            # the Renode model for these only has 1 IRQ
            elif (compat in ['arm,pl011', 'atmel,sam0-uart']
                  or model in ['UART.Cadence_UART', 'UART.NS16550', 'Timers.ARM_SP804_Timer']):
                irq_names = ['']
            elif compat in ['st,stm32-i2c-v1', 'st,stm32-i2c-v2']:
                irq_names = ['EventInterrupt', 'ErrorInterrupt']
            elif compat in ['arm,cmsdk-uart']:
                irq_names = ['TxInterrupt', 'RxInterrupt']
            elif compat in ['infineon,xmc4xxx-uart']:
                irq_names = ['TxInterrupt', 'RxInterrupt']
            elif compat in ['ambiq,stimer']:
                irq_names = ['IRQA', 'IRQB', 'IRQC', 'IRQD', 'IRQE', 'IRQF', 'IRQG', 'IRQH', 'IRQI']
            elif compat in ['nordic,nrf-bellboard-rx']:
                irq_names = ['0']
            elif compat in ['renesas,rzt2m-uart']:
                irq_names = ['// RxErrIRQ', 'RxIRQ', 'TxIRQ', 'TxEndIRQ']
            else:
                irq_names = [str(n) for n in range(len(irq_dest_nodes))]

                # use non-numbered GPIO output if we only have 1 GPIO
                # and this node is not an interrupt controller
                if irq_names == ['0'] and 'interrupt-controller' not in node.props:
                    irq_names = ['']

            visited_irqs = set()
            for i, (irq_name, irq_dest, irq) in enumerate(zip(irq_names, irq_dest_nodes, irq_numbers)):
                if irq_name is None:
                    continue
                if irq in visited_irqs:
                    # some u-boot peripherals contain duplicated irqs in device trees e.g. 
                    # https://github.com/u-boot/u-boot/blob/69bd83568c57813cd23bc2d100c066a17e7e349d/dts/upstream/src/riscv/microchip/mpfs-icicle-kit.dts#L86
                    continue
                # assume very large IRQ numbers which have all bits set (i.e. 2^n - 1) are invalid
                if irq >= 0xfff and (irq & (irq + 1)) == 0:
                    continue
                if is_disabled(irq_dest):
                    continue
                irq_dest_name = name_mapper.get_name(irq_dest)
                if i in irq_local_indices:
                    irq_dest_name += f'#{irq_local_indices[i]}'
                indent.append(f'{irq_name}->{irq_dest_name}@{irq}')
                visited_irqs.add(irq)
                # IRQ destinations are not treated as dependencies, we filter
                # out IRQ connections to missing peripherals at the end because
                # it is better to have a peripheral missing an interrupt connection
                # than no peripheral at all

        # the ARM generic timer is registered at cpu0 without an address
        # for proper multi-CPU support we will need to generate one for
        # each CPU with the proper interrupt connections
        if model == "Timers.ARM_GenericTimer":
            regions = [RegistrationRegion(addresses=[], registration_point="cpu0")]

        if model == 'IRQControllers.LAPIC':
            regions = [RegistrationRegion(addr, cpu='cpu0')]

        # devices other than CPUs require an address to register on the sysbus
        if any(r.registration_point == 'sysbus' and r.address is None for r in regions) and not model.startswith('CPU.'):
            logging.info(f'Node {node} has sysbus registration without an address. Skipping...')
            continue

        # the registration points themselves are also dependencies
        for r in regions:
            dependencies.add(r.registration_point)

        if model.startswith('Memory'):
            if model == 'Memory.MappedMemory' and size % 0x400 != 0:
                # there is a requirement for mapped memory regions to be of certain size in tlib;
                # let's fallback to array memory for smaller regions
                model = 'Memory.ArrayMemory'

        block_content = [f'{name}: {model} @ {RegistrationRegion.to_repl(regions)}']
        block_content.extend(map(lambda x: f'    {x}', indent))

        region = regions[0] if len(regions) == 1 else None # for now we store at most one for merging
        if model.startswith('Memory'):
            region.size = size
        block = ReplBlock(name, model, dependencies, provides, block_content, region)
        repl_file.add_block(block)

    # now all blocks originating from the device tree are processed, let's add overlays next
    # soc and board overlay
    overlay_path = f'{pathlib.Path(__file__).parent.resolve()}/overlay'
    overlay_files = os.listdir(overlay_path)
    for compat in sorted(overlays):
        overlay = f'{compat}.repl'
        for file in overlay_files:
            if overlay.lower() == file.lower():
                logging.info(f'Applying {overlay} overlay file')
                repl_file.add_block(ReplBlock('', None, set(), set(), [f'// {compat} overlay']))
                for overlay_block in parse_overlay(f'{overlay_path}/{file}'):
                    repl_file.add_block(overlay_block, skip_duplicates=True)
                break

    # filter out unavailable blocks (with unsatisfied depends)
    repl_file.filter_available_blocks()
    if address_space_32bit:
        repl_file.filter_64bit_reg_peripherals()

    # set number of targets for CLINT if necessary
    # note that IRQ destinations don't create a dependency; this should pick up just CPUs because they get a dependency on the CLINT when we add timeProvider
    clint_block = repl_file.get_block_by_name('clint')
    if clint_block and (clint_targets := len(clint_block.get_depending_blocks())):
        clint_block.content += [f'    numberOfTargets: {clint_targets}']

    repl_file.merge_overlapping_blocks()
    repl_file.remove_dangling_irqs()

    return str(repl_file)

def get_mcu_compat(filename):
    dt = get_dt(filename)
    if dt is None:
        return ''
    logging.debug(f'trying to find cpu for {filename}')
    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    logging.debug(f'Found: {mcu}')
    if mcu is not None:
        mcu = get_node_prop(mcu, 'compatible')[0]
    logging.debug(f'Result: {mcu}')
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
    print(" * Not found")
    return None


def get_compats(node):
    SKIP_COMPATS = ['arm,armv8m-mpu']
    compats = get_node_prop(node, 'compatible')

    if compats is None:
        logging.info(f"No compats (type) for node {node}. Skipping...")
        return None

    if any(item in SKIP_COMPATS for item in compats):
        return None
    return compats


def process_node(node, node_type, mcu, overlays, get_snippets, skip_disabled):
    result = {}
    label = None
    model = None

    if skip_disabled and is_disabled(node):
        return None

    if (compats := get_compats(node)) is None:
        return None

    compat = compats[0]

    if node.labels:
        label = node.labels[0]

    if compat in MODELS:
        model, compat, _, _ = renode_model_overlay(compat, mcu, overlays)

    if node_type == "cpu":
        if id := node.unit_addr:
            unit_addr = int(id, 16)
        else:
            unit_addr = 0x0

        size = 0x0
        compats = [compats[0]]
    else:
        reg = list(get_reg(node))
        if reg:
            # transform the second item of each tuple (representing the size):
            # 'set it to 0 if it's None'
            reg = [(r[0], r[1] if r[1] else 0) for r in reg]
            unit_addr = reg[0][0]
            size = sum(r[1] for r in reg)
            if size == 0:
                logging.info(f"Regs for node {node} have total size 0. Skipping...")
                return None
        else:
            logging.info(f"No regs for node {node}. Skipping...")
            return None

    result = {
        "unit_addr": hex(unit_addr),
        "label": label or "",
        "model": model or "",
        "compats": compats.copy(),
        "parent_node": node.parent.name,
    }

    result["size"] = hex(size)

    if 'interrupts' in node.props:
        irq_nums = [irq for irq in get_node_prop(node, 'interrupts')[::2]]
        if irq_nums:
            result["irq_nums"] = irq_nums.copy()

    if get_snippets:
        result['snippet'] = str(node)

    return result


def generate_peripherals(filename, overlays, generate_type, get_snippets=False):
    CPU_NODE_REGEX = re.compile(r"^cpu(@[A-Fa-f0-9]+)?$")

    skip_disabled = generate_type == "board"
    result = {}

    dt = get_dt(filename)
    if dt is None:
        return None

    mcu = get_mcu_compat(filename)

    print(f"Generating {generate_type} peripherals for {str(Path(filename).stem)}")

    # Go through /cpus node
    try:
        cpus = dt.get_node("/cpus")
        for node in cpus.nodes.values():
            logging.debug(f'Checking cpu node {node}')
            if not CPU_NODE_REGEX.match(node.name):
                continue

            if res := process_node(node, "cpu", mcu, overlays, get_snippets, skip_disabled):
                result[node.name] = res

    except dtlib.DTError:
        pass

    try:
        par = dt.get_node("/soc")
    except dtlib.DTError:
        par = dt.root

    # Go through /soc node
    for node in par.node_iter():
        # There is a rare case, when `/soc` node isn't present in DT, when we
        # may end up processing `/cpus` node as a peripheral because we proces
        # all children of the root node.
        path = node.path
        if path.startswith("/cpus/") or path == "/cpus":
            continue

        if res := process_node(node, "peripheral", mcu, overlays, get_snippets, skip_disabled):
            result[node.name] = res

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


def generate_bus_sensors(filename, overlays):
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
            if not compats:
                continue

            compat = compats[0]
            if compat in MODELS:
                mcu = get_mcu_compat(filename)
                model, compat, _, _ = renode_model_overlay(compat, mcu, overlays)
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


def generate_gpio(filename, overlays):
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
    for typ in [KEYS_NODE, LEDS_NODE]:
        if dt.has_node(typ):
            parent = dt.get_node(typ)
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

    dirs = args.include.split(',')

    if not args.nonrecursive:
        dirs = []
        for top in args.include.split(','):
            for root, _, _ in os.walk(top):
                dirs.append(root)

    incl_dirs = ' '.join(f'-I {dir}' for dir in dirs)

    if args.flatten:
        # get list of #includes (C preprocessor)
        cmd = f'gcc -H -E -P -x assembler-with-cpp {incl_dirs} {args.preprocessor_args} {args.filename}'.split()
        ret = subprocess.run(cmd, capture_output=True)

        if ret.returncode != 0:
            raise ValueError(f'Preprocessing failed. Compiler output:\n{ret.stderr.decode("utf-8")}')

        # save partially flattened device tree
        base = os.path.splitext(args.output)[0]
        flat_dts = f'{base}.flat.dts'
        with open(flat_dts, 'w') as f:
            f.write(ret.stdout.decode('utf-8'))

        # get list of /include/s (device tree mechanism)
        dts_includes = get_includes(flat_dts, dirs)

        # save fully flattened device tree (also /include/s)
        dts = dtlib.DT(flat_dts, dirs, args.force_valid_aliases)
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
        f.write(generate(args.filename, args.override_system_clock_frequency))

if __name__ == "__main__":
    main()
