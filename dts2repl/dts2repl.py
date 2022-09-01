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


def get_dt(filename):
    with open(filename) as f:
        dts_file = f.readlines()
        dts_file = filter(lambda x: 'pinctrl-0;' not in x, dts_file)
        dts_file = ''.join(dts_file)

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8') as f:
        f.write(dts_file)
        f.flush()
        return dtlib.DT(f.name)


def get_node_prop(node, prop):
    if prop not in node.props:
        return None

    val = node.props[prop]
    if prop in ('compatible', 'device_type'):
        val = val.to_strings()
    elif prop in ('interrupts', 'reg', 'ranges'):
        val = val.to_nums()
    elif prop == 'cc-num':
        val = val.to_num()
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
            if 'stm32l4' in overlays:
                model = 'Python.PythonPeripheral'
            elif 'stm32l0' in overlays:
                model = 'Miscellaneous.STM32L0_RCC'
            else:
                model = 'Miscellaneous.STM32F4_RCC'

    if compat == "atmel,sam0-uart" and 'samd20' in overlays:
        model = 'UART.SAMD20_UART'

    # LiteX on Fomu is built in the 8-bit CSR data width configuration
    if compat == "litex,timer0" and "fomu" in overlays:
        model = 'Timers.LiteX_Timer'

    # compat-based mapping for MiV and PolarFire SoC is not enough, as one is 32-bit
    # and the other 64-bit
    if compat == "microsemi,miv" and 'mpfs_icicle' in overlays:
        model = 'CPU.RiscV64'

    return model, compat


def get_ranges(ranges):
    while ranges:
        child_addr, parent_addr, size, *ranges = ranges
        yield child_addr, parent_addr, size


def generate(args):
    name_counter = Counter()
    dt = get_dt(args.filename)
    if dt is None:
        return ''

    models_path = f'{pathlib.Path(__file__).parent.resolve()}/models.json'
    with open(models_path) as f:
        models = json.load(f)

    repl = [f'// autogenerated']
    nodes = sorted(dt.node_iter(), key=lambda x: get_node_prop(x, 'compatible')[0] if 'compatible' in x.props else '')

    # get mcu compat name
    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    if mcu is not None:
        mcu = get_node_prop(mcu, 'compatible')[0]

    for node in nodes:
        # filter out nodes without compat strings
        compatible = get_node_prop(node, 'compatible')
        if compatible is None:
            logging.debug(f'Node {node.name} has no compat string. Trying device_type...')
            compatible = get_node_prop(node, 'device_type')
            if compatible is None:
                logging.debug(f'Node {node.name} has no compat string or device_type. Skipping...')
                continue

        # filter out nodes without a sysbus address
        if len(node.name.split('@')) < 2:
            logging.info(f'Node {node.name} has no sysbus address. Skipping...')
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
        name, addr = node.name.split('@')
        if len(node.labels) > 0:
            name = node.labels[0].lower().replace("_", "")

        # make name a valid repl GeneralIdentifier
        name = re.sub('[^A-Za-z0-9_]', '_', name)

        if name == 'cpu':
            name += addr

        # decide which Renode model to use
        model, compat = renode_model_overlay(compat, mcu, models, args.overlays)

        address = ''
        if not name.startswith('cpu'):
            parent_node = node.parent
            addr_offset = '0'
            ranges = []
            if parent_node is not None and '@' in parent_node.name and 'ranges' in parent_node.props:
                _, addr_offset = parent_node.name.split('@')
                if parent_node.props['ranges'].value:
                    ranges = get_ranges(get_node_prop(parent_node, 'ranges'))

            addr = int(addr, 16)
            addr_offset = int(addr_offset, 16)

            for child_addr, parent_addr, size in ranges:
                if child_addr <= addr < child_addr + size:
                    addr_offset = parent_addr - child_addr
                    break

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
                or model == 'UART.STM32_UART'
            ):
                start, size = list(map(lambda x: hex(x), get_node_prop(node, 'reg')))
                address = f'<{start}, +{size}>'

        # "timer" becomes "timer1", "timer2", etc
        # if we have "timer" -> "timer1" but there was already a peripheral named "timer1",
        # we'll end up with "timer" -> "timer1" -> "timer11"
        while name in name_counter:
            name_counter[name] += 1
            name += str(name_counter[name] - 1)
        name_counter[name] += 1

        repl.append(f'{name}: {model} @ sysbus {address}')
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
            indent.append('frequency: 240000000')
        if model == 'Timers.Marvell_Armada_Timer':
            indent.append('frequency: 100000000')

        # additional parameters for python peripherals
        if compat.startswith("st,stm32") and compat.endswith("rcc") and model == "Python.PythonPeripheral":
            indent.append('size: 0x400')
            indent.append('initable: true')
            if 'stm32l4' in args.overlays:
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
            indent.append('// 0x260: digprog, report mx6ul')
            indent.append('// 0x280: digprog_sololite, report mx6ul')
            indent.append('script: "request.value = {0x10: 0x80000000, 0xe0: 0x80000000, 0x100: 0xffffffff, 0x150: 0x80, 0x180: 0x4, 0x260: 0x640000, 0x280: 0x640000}.get(request.offset, 0)"')
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
        if compat.startswith('riscv,sifive') or compat == 'sifive,e31':
            indent.append('cpuType: "rv32imac"')
            indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_10')
            indent.append('timeProvider: clint')
        if compat.startswith('microsemi,miv'):
            isa = get_node_prop(node, 'riscv,isa')
            indent.append(f'cpuType: "{isa}"')
            indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_09')
            indent.append('timeProvider: clint')
        if compat == 'gaisler,leon3':
            indent.append('cpuType: "leon3"')
        if compat.startswith('starfive,rocket'):
            indent.append('cpuType: "rv64gc"')
            indent.append(f'hartId: {node.name.split("@")[1]}')
            indent.append('privilegeArchitecture: PrivilegeArchitecture.Priv1_10')
            indent.append('timeProvider: clint')

        if model == 'UART.STM32F7_USART' and compat != 'st,stm32-lpuart':
            indent.append('frequency: 200000000')

        # additional parameters for STM32F4_RCC
        if model == 'Miscellaneous.STM32F4_RCC':
            indent.append('rtcPeripheral: rtc')

        # additional parameters for IRQ ctrls
        if compat.endswith('nvic'):
            indent.append('-> cpu0@0')
        if compat == 'gaisler,irqmp':
            indent.append('0 -> cpu0@0 | cpu0@1 | cpu0@2')

        # for some reason the only compat string that VexRiscv has is "riscv"
        # check the board compat string and if doesn't match, remove last entry
        if compat == 'riscv':
            if get_node_prop(node.parent.parent, 'compatible')[0] == 'litex,vexriscv':
                indent.append('cpuType: "rv32imac"')
            elif get_node_prop(node.parent.parent, 'compatible')[0] == 'kosagi,fomu':
                indent.append('cpuType: "rv32im"')
            else:
                repl.pop()

        if model.startswith('Timers'):
            if 'cc-num' in node.props:
                indent.append(f'numberOfEvents: {str(get_node_prop(node, "cc-num"))}')
        if model.startswith('Memory'):
            if 'reg' in node.props:
                size = get_node_prop(node, "reg")[-1]
                if size != 0:
                    indent.append(f'size: {hex(size)}')
                else:
                    # do not generate memory regions of size 0
                    repl.pop()

        if 'interrupts' in node.props and mcu is not None:
            # decide which IRQ destination to use in Renode model
            if any(map(lambda x: mcu.startswith(x), ['microsemi,miv', 'riscv,sifive', 'starfive', 'sifive,e'])):
                irq_dest = 'plic'
            elif mcu.startswith('riscv'):  # this is for LiteX!
                irq_dest = 'cpu0'
            elif mcu.startswith('gaisler'):
                irq_dest = 'irqmp'
            elif mcu.startswith('arm,cortex-m'):
                irq_dest = 'nvic'
            else:
                irq_dest = None
                logging.warning(f'Unknown IRQ destination for {node.name}')

            # decide which IRQ names to use in Renode model
            if compat == 'st,stm32-rtc':
                irq_names = ['AlarmIRQ']
            elif compat in ['nxp,kinetis-lpuart', 'nxp,kinetis-uart', 'silabs,gecko-leuart', 'sifive,uart0', 'st,stm32-adc']:
                irq_names = ['IRQ']
            elif compat in ['silabs,gecko-uart', 'silabs,gecko-usart']:
                irq_names = ['ReceiveIRQ', 'TransmitIRQ']
            elif compat in ['gaisler,gptimer']:
                irq_names = ['0']
            else:
                irq_names = ['']

            # assign IRQ signals
            if irq_dest is not None:
                for name, irq in zip(irq_names, get_node_prop(node, 'interrupts')[::2]):
                    indent.append(f'{name}->{irq_dest}@{irq}')

        repl.extend(map(lambda x: f'    {x}', indent))
        repl.append('')

    # soc and board overlay
    overlay_path = f'{pathlib.Path(__file__).parent.resolve()}/overlay'
    for cpu in map(lambda x: x.split("/")[-1], args.overlays.split(",")[::-1]):
        overlay = f'{overlay_path}/{cpu}.repl'
        if os.path.exists(overlay):
            repl.append('')
            with open(overlay) as f:
                repl.extend(map(lambda x: x.rstrip(), f.readlines()))

    return '\n'.join(repl)

def get_mcu_compat(filename):
    dt = get_dt(filename)
    if dt is None:
        return ''

    mcu = next(filter(lambda x: 'cpu' in x.name and get_node_prop(x, 'compatible'), dt.node_iter()), None)
    if mcu is not None:
        mcu = get_node_prop(mcu, 'compatible')[0]
    return mcu

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

        if 'reg' in node.props:
            reg = get_node_prop(node, 'reg')
            if len(reg) == 1:
                reg = None
                size = None
                continue
            else:
                unit_addr = hex(reg[0]) if len(reg) > 0 else None
                if len(reg) > 1:
                    size = sum(reg[1::2])
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

def main():
    args = parse_args()

    dirs = []
    for top in args.include.split(','):
        for root, _, _ in os.walk(top):
            dirs.append(root)

    incl_dirs = ' '.join(f'-I {dir}' for dir in dirs)

    if args.automatch:
        board_name = os.path.splitext(os.path.basename(args.filename))[0]
        cmd = f'gcc -H -E -P -x assembler-with-cpp {incl_dirs} {args.filename}'.split()
        ret = subprocess.run(cmd, capture_output=True)

        # save partially flattened device tree
        flat_dts = f'{os.path.splitext(args.output)[0]}.flat.dts'
        with open(flat_dts, 'w') as f:
            f.write(ret.stdout.decode('utf-8'))

        # save fully flattened device tree (also /include/s)
        dts = dtlib.DT(flat_dts, dirs)
        with open(flat_dts, 'w') as f:
            f.write(str(dts))
        args.filename = flat_dts

        # try to automatch overlays
        includes = ret.stderr.decode('utf-8').split('\n')
        includes = filter(lambda x: '.dtsi' in x, includes)
        includes = map(lambda x: x.lstrip('. '), includes)
        includes = map(lambda x: os.path.basename(x), includes)
        includes = map(lambda x: os.path.splitext(x)[0], includes)
        includes = set(includes)
        includes.add(board_name)
        args.overlays = ','.join(includes)

    with open(args.output, 'w') as f:
        f.write(generate(args))

if __name__ == "__main__":
    main()
