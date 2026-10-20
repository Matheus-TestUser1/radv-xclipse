#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Record descriptors first; --submit explicitly enables isolated BC sampling.
Build this diagnostic branch with ./build.sh before running. No root or global property writes.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE = '/data/local/tmp/x940-bc45-native'
FORMATS = {'bc3': 137, 'bc4': 139, 'bc5': 141}
NATIVE_CODES = {'bc3': 113, 'bc4': 115, 'bc5': 117}
FIELDS = ('GHG_S', 'GPF_S', 'GTO_S', 'GCF_S', 'GRH_S', 'GRF_S')
TOLERANCE = 2.1 / 255
SOURCE_FILES = ['src/amd/vulkan/radv_physical_device.h', 'src/amd/vulkan/radv_physical_device.c',
                'src/amd/vulkan/radv_formats.c', 'src/amd/vulkan/radv_image_view.c',
                'src/amd/vulkan/meta/radv_meta_bc_decode.c', 'src/amd/common/ac_descriptors.c',
                'src/amd/registers/gfx11-rsrc.json', 'diagnostics/bc45_native/probe.c',
                'diagnostics/bc45_native/sample.comp', 'diagnostics/bc45_native/run.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counters(text):
    result = {key: int(value) for key, value in re.findall(r'"(' + '|'.join(FIELDS) + r')":"(\d+)"', text)}
    if set(result) != set(FIELDS):
        raise ValueError('Contadores gpu_info incompletos: ' + text)
    return result


def expected_blocks(bc):
    ends = [(240, 16), (20, 210), (127, 127), (0, 255)]
    def alpha(block, channel):
        a, b = ends[(block + channel) & 3]
        bits = sum(((p + block + channel * 3) & 7) << (3 * p) for p in range(16))
        return bytes([a, b]) + bits.to_bytes(6, 'little')
    blocks = []
    for block in range(4):
        value = alpha(block, 0)
        if bc == 'bc5':
            value += alpha(block, 1)
        if bc == 'bc3':
            value += struct.pack('<HHI', 0xf800, 0x07e0,
                                 sum(((p + block) & 3) << (2 * p) for p in range(16)))
        blocks.append(value)
    return b''.join(blocks)


def reference(bc, data):
    """Decode actual little-endian fields independently from the C pixel oracle."""
    stride = 8 if bc == 'bc4' else 16
    def channel(block):
        a, b = block[:2]
        values = [a / 255, b / 255]
        if a > b:
            values += [((6 - n) * a + (n + 1) * b) / (7 * 255) for n in range(6)]
        else:
            values += [((4 - n) * a + (n + 1) * b) / (5 * 255) for n in range(4)] + [0., 1.]
        bits = int.from_bytes(block[2:8], 'little')
        return [values[(bits >> (3 * p)) & 7] for p in range(16)]
    result = [None] * 64
    for block in range(4):
        raw = data[block * stride:(block + 1) * stride]
        red = channel(raw)
        green = channel(raw[8:16]) if bc == 'bc5' else [0.] * 16
        rgb = None
        if bc == 'bc3':
            c0, c1, bits = struct.unpack('<HHI', raw[8:16])
            def rgb565(c):
                return [(c >> 11) / 31, ((c >> 5) & 63) / 63, (c & 31) / 31]
            e0, e1 = rgb565(c0), rgb565(c1)
            palette = [e0, e1, [(2*a+b)/3 for a,b in zip(e0,e1)], [(a+2*b)/3 for a,b in zip(e0,e1)]]
            rgb = [palette[(bits >> (2 * p)) & 3] for p in range(16)]
        for p in range(16):
            x, y = (block % 2) * 4 + p % 4, (block // 2) * 4 + p // 4
            result[y * 8 + x] = rgb[p] + [red[p]] if rgb else [red[p], green[p], 0., 1.]
    return result


def analyze(log, bc, mode, submit=False):
    device = re.findall(r'x940-bc45-device: build=bc45-v1 eligible=(\d+) mode=(\d+) requested=(\w+)', log)
    expected_mode = 1 if mode == 'trace' else 2
    fmt = FORMATS[bc]
    pattern = (r'x940-bc45-view: requested=(\d+) effective=(\d+) planes=(\d+) plane=(\d+) '
               r'emulated=(\d+) img_format=(\d+) desc=([0-9a-f,]+)')
    views = [item for item in re.findall(pattern, log) if int(item[0]) == fmt]
    inputs = re.findall(r'BC45_INPUT: format=(\d+) size=(\d+) hex=([0-9a-f]+)', log)
    checks = {
        'diagnostic_940_gfx11_guard': device == [('1', str(expected_mode), mode)],
        'one_requested_view': len(views) == 1,
        'input_blocks_unchanged': inputs == [(str(fmt), str(len(expected_blocks(bc))), expected_blocks(bc).hex())],
        'recording_completed': bool(re.search(r'event=command_buffer_recorded value=' + bc[-1] + r'\b', log)),
        'no_watchdog': 'event=watchdog_timeout' not in log,
    }
    details = {}
    native = bc == 'bc3' or mode == 'native'
    if len(views) == 1:
        requested, effective, planes, plane, emulated, img_format = map(int, views[0][:6])
        words = [int(x, 16) for x in views[0][6].split(',')]
        checks['eight_descriptor_words'] = len(words) == 8
        checks['descriptor_format_matches_trace'] = len(words) == 8 and (words[1] >> 20) & 0x1ff == img_format
        checks['valid_texture_address'] = len(words) == 8 and bool(words[0] or words[1] & 255)
        if native:
            checks['native_original_format_single_plane'] = effective == requested == fmt and planes == 1 and plane == 0 and emulated == 0
            checks['existing_gfx11_native_code'] = img_format == NATIVE_CODES[bc]
        else:
            checks['baseline_emulation_preserved'] = emulated == 1 and plane == 1 and planes == 2 and effective == (153 if bc == 'bc4' else 155)
            checks['baseline_eac_code'] = img_format == (127 if bc == 'bc4' else 129)
        details['view'] = dict(requested=requested, effective=effective, planes=planes, plane=plane,
                               emulated=emulated, img_format=img_format, descriptor_words=words)
    decodes = len(re.findall(r'x940-bc45-meta-decode: format=' + str(fmt) + r'\b', log))
    checks['meta_decode_path'] = decodes == (0 if native else 1)
    begins = len(re.findall(r'event=submit_begin\b', log))
    if not submit:
        checks['probe_submits_zero'] = begins == 0 and 'event=submit_return' not in log and 'BC45_RECORD_ONLY_PASS: probe vkQueueSubmit calls=0' in log
    else:
        checks['one_submit_and_signaled_fence'] = (
            begins == 1 and len(re.findall(r'event=submit_return value=0\b', log)) == 1
            and len(re.findall(r'event=fence_return value=0\b', log)) == 1)
        found = re.findall(r'BC45_PIXEL: (\d+) (\S+) (\S+) (\S+) (\S+)', log)
        checks['all_64_pixels_once'] = len(found) == 64 and sorted(int(p[0]) for p in found) == list(range(64))
        if checks['all_64_pixels_once']:
            expected = reference(bc, expected_blocks(bc))
            errors = [abs(float(v) - expected[int(row[0])][c]) for row in found for c,v in enumerate(row[1:])]
            details['cpu_max_abs_error'] = max(errors) if all(math.isfinite(x) for x in errors) else None
            checks['pixels_match_independent_cpu_decode'] = all(math.isfinite(x) and x <= TOLERANCE for x in errors)
        checks['probe_sample_pass'] = 'BC45_SAMPLE_PASS' in log and 'BC45_SAMPLE_MISMATCH' not in log
    checks['teardown_completed'] = bool(re.search(r'event=teardown_complete value=0\b', log))
    return checks, details


def self_test():
    for bc in FORMATS:
        pixels = reference(bc, expected_blocks(bc))
        assert len(pixels) == 64 and all(len(p) == 4 for p in pixels)
        assert max(p[0] for p in pixels) - min(p[0] for p in pixels) > .8
        for mode in ('trace', 'native'):
            native = bc == 'bc3' or mode == 'native'
            effective = FORMATS[bc] if native else 153 if bc == 'bc4' else 155
            code = NATIVE_CODES[bc] if native else 127 if bc == 'bc4' else 129
            words = [0x100, code << 20, 0, 0, 0, 0, 0, 0]
            log = (f'x940-bc45-device: build=bc45-v1 eligible=1 mode={1 if mode=="trace" else 2} requested={mode}\n'
                   f'x940-bc45-view: requested={FORMATS[bc]} effective={effective} planes={1 if native else 2} '
                   f'plane={0 if native else 1} emulated={0 if native else 1} img_format={code} desc=' +
                   ','.join(f'{x:08x}' for x in words) + '\n' +
                   f'BC45_INPUT: format={FORMATS[bc]} size={len(expected_blocks(bc))} hex={expected_blocks(bc).hex()}\n'
                   f'bc45-flow: pid=1 mono_ns=1 event=command_buffer_recorded value={bc[-1]}\n'
                   'BC45_RECORD_ONLY_PASS: probe vkQueueSubmit calls=0; sampling not executed\n'
                   'bc45-flow: pid=1 mono_ns=1 event=teardown_complete value=0\n')
            if not native:
                log += f'x940-bc45-meta-decode: format={FORMATS[bc]}\n'
            assert all(analyze(log, bc, mode)[0].values())
            mutations = [log.replace('eligible=1', 'eligible=0'), log.replace('desc=', 'bad_desc='),
                         log.replace('hex=', 'bad_hex='), log + 'event=submit_begin value=1\n',
                         log.replace('command_buffer_recorded', 'not_recorded')]
            if native:
                mutations += [log.replace('emulated=0', 'emulated=1'),
                              log + f'x940-bc45-meta-decode: format={FORMATS[bc]}\n']
            for bad in mutations:
                assert not all(analyze(bad, bc, mode)[0].values())
            if native:
                sample = log.replace('BC45_RECORD_ONLY_PASS: probe vkQueueSubmit calls=0; sampling not executed',
                                     'event=submit_begin value=1\nevent=submit_return value=0\nevent=fence_return value=0\nBC45_SAMPLE_PASS')
                sample += ''.join(f'BC45_PIXEL: {i} ' + ' '.join(map(str,p)) + '\n' for i,p in enumerate(pixels))
                assert all(analyze(sample, bc, mode, True)[0].values())
                zeros = re.sub(r'BC45_PIXEL: (\d+) .*', r'BC45_PIXEL: \1 0 0 0 0', sample)
                assert not all(analyze(zeros, bc, mode, True)[0].values())
    assert counters(''.join(f'"{k}":"0",' for k in FIELDS)) == dict.fromkeys(FIELDS, 0)
    print('PASS: independent CPU decoder, six descriptor profiles, corrupted-record rejection, false-zero readback rejection.')


REMOTE_SCRIPT = '''
set -eu
cd /data/local/tmp/x940-bc45-native
for var in $(env | cut -d= -f1); do
 case "$var" in RADV_*|MESA_*|X940_*) unset "$var" ;; esac
done
export HOME=/data/local/tmp/x940-bc45-native
export XDG_CACHE_HOME=/data/local/tmp/x940-bc45-native/cache
export LD_LIBRARY_PATH=/data/local/tmp/x940-bc45-native
export MESA_SHADER_CACHE_DISABLE=1
export RADV_DEBUG=startup,info
export RADV_XCLIPSE_LOG=0
export RADV_XCLIPSE_BC5_ALIAS=0 RADV_XCLIPSE_BC7_ALIAS=0
export RADV_XCLIPSE_DIAG_BC45="$2"
exec ./bc45_probe ./vulkan.radeon.so ./sample.spv "$1" "$3"
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=ROOT / 'build-android/vulkan.radeon.so')
    parser.add_argument('--ndk', type=Path, default=Path(os.environ.get('ANDROID_NDK_HOME', str(Path.home() / 'android-ndk-r29'))))
    parser.add_argument('--serial')
    parser.add_argument('--submit', action='store_true', help='Execute BC3 control, then native BC4/BC5; stop on first failure')
    parser.add_argument('--record', type=Path, help='Approved record-only directory required for --submit')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    library = args.library.expanduser().resolve()
    if not library.is_file():
        raise ValueError('Biblioteca de diagnóstico ausente; compile esta branch com ./build.sh primeiro.')
    source_hashes = {p: sha(ROOT / p) for p in SOURCE_FILES}
    library_sha = sha(library)
    if args.submit:
        if not args.record:
            raise ValueError('--submit exige --record com a pasta de gravação aprovada.')
        previous = args.record.expanduser().resolve()
        saved = json.loads((previous / 'summary.json').read_text())
        if saved.get('status') != 0 or saved.get('gpu_execution_tested') or saved.get('library_sha256') != library_sha or saved.get('source_hashes') != source_hashes:
            raise ValueError('Gravação anterior não aprovada ou fontes/biblioteca mudaram.')
        for bc in FORMATS:
            for mode in ('trace', 'native'):
                case = f'{bc}_{mode}'
                info = saved['cases'][case]
                old_log = (previous / (case + '.txt')).read_text()
                if info['rc'] or info['counters_before'] != info['counters_after'] or not all(analyze(old_log, bc, mode)[0].values()):
                    raise ValueError('Gravação anterior inválida: ' + case)
    compiler = args.ndk.expanduser() / 'toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang'
    glslang = shutil.which('glslangValidator')
    adb_program = shutil.which('adb')
    if not compiler.is_file() or not glslang or not adb_program:
        raise ValueError('São necessários NDK r29/API34, glslangValidator e adb no Ubuntu.')
    root = Path.home() / 'xclipse-diag'
    root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='jim_bc45_submit_' if args.submit else 'jim_bc45_record_', dir=root))
    print('Logs: ' + str(out), flush=True)
    binary, spirv = out / 'bc45_probe', out / 'sample.spv'
    subprocess.run([str(compiler), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-I' + str(ROOT / 'include'),
                    str(HERE / 'probe.c'), '-o', str(binary), '-ldl', '-lm', '-llog'], check=True)
    elf = binary.read_bytes()[:20]
    if len(elf) < 20 or elf[:4] != b'\x7fELF' or elf[18:20] != b'\xb7\x00':
        raise ValueError('Probe não é ELF AArch64.')
    subprocess.run([glslang, '-V', '--target-env', 'vulkan1.1', '-S', 'comp', str(HERE / 'sample.comp'), '-o', str(spirv)], check=True)
    if shutil.which('spirv-val'):
        subprocess.run(['spirv-val', '--target-env', 'vulkan1.1', str(spirv)], check=True)
    if args.submit and (sha(binary) != saved['probe_sha256'] or sha(spirv) != saved['shader_sha256']):
        raise ValueError('Probe/SPIR-V diferem dos usados na gravação aprovada.')
    adb_base = [adb_program] + (['-s', args.serial] if args.serial else [])
    def adb(*a, **kw):
        return subprocess.run(adb_base + list(a), **kw)
    def gpu():
        result = adb('shell', 'cat /sys/kernel/gpu/gpu_info', capture_output=True, text=True, timeout=15, check=True)
        return counters(result.stdout)
    prop = adb('shell', 'getprop debug.radv_xclipse_no_bc_emu', capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    if prop and prop != '0':
        raise ValueError('debug.radv_xclipse_no_bc_emu está ativo globalmente; o experimento exige a emulação disponível como controle.')
    adb('shell', 'mkdir', '-p', REMOTE, check=True, timeout=15)
    for local, name in [(binary, 'bc45_probe'), (spirv, 'sample.spv'), (library, 'vulkan.radeon.so')]:
        adb('push', str(local), REMOTE + '/' + name, check=True, timeout=45)
    adb('shell', 'chmod', '755', REMOTE + '/bc45_probe', check=True, timeout=15)
    adb('shell', REMOTE + '/bc45_probe', '--self-test', check=True, timeout=15)
    remote_sha = adb('shell', 'sha256sum', REMOTE + '/vulkan.radeon.so', capture_output=True, text=True, timeout=15, check=True).stdout.split()[0]
    if remote_sha != library_sha:
        raise ValueError('Hash da biblioteca enviada difere do arquivo local.')
    metadata = adb('shell', 'date; uname -a; getprop ro.build.fingerprint', capture_output=True, text=True, timeout=15, check=True)
    (out / 'device_metadata.txt').write_text(metadata.stdout + metadata.stderr)
    (out / 'remote_script.txt').write_text(REMOTE_SCRIPT)
    for name in ('probe.c', 'sample.comp', 'run.py'):
        shutil.copyfile(HERE / name, out / name)
    summary = {'version': 'bc45-v1', 'library_sha256': library_sha, 'probe_sha256': sha(binary),
               'shader_sha256': sha(spirv), 'source_hashes': source_hashes,
               'gpu_execution_tested': bool(args.submit), 'cases': {}, 'status': 1}
    jobs = [(bc, mode) for bc in FORMATS for mode in ('trace', 'native')] if not args.submit else [('bc3', 'trace'), ('bc4', 'native'), ('bc5', 'native')]
    failed = False
    try:
        for bc, mode in jobs:
            case = f'{bc}_{mode}'
            print(f'BC45: {case}; ' + ('EXECUÇÃO REAL' if args.submit else 'gravação, sem vkQueueSubmit do probe'), flush=True)
            before = gpu()
            try:
                result = adb('shell', 'sh', '-s', '--', bc, mode, 'submit' if args.submit else 'record',
                             input=REMOTE_SCRIPT, capture_output=True, text=True, timeout=45)
            except subprocess.TimeoutExpired as error:
                def as_text(value):
                    return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                result = subprocess.CompletedProcess(adb_base, 124, as_text(error.stdout),
                                                     as_text(error.stderr) + '\nADB TIMEOUT\n')
            log = (result.stdout + result.stderr).replace('\r', '')
            (out / (case + '.txt')).write_text(log)
            checks, details = analyze(log, bc, mode, args.submit)
            snapshots = [gpu()]
            if args.submit:
                for _ in range(3):
                    time.sleep(2)
                    snapshots.append(gpu())
            after = snapshots[-1]
            checks['probe_rc_zero'] = result.returncode == 0
            checks['external_counters_unchanged'] = all(x == before for x in snapshots)
            summary['cases'][case] = {'rc': result.returncode, 'checks': checks, 'details': details,
                                      'counters_before': before, 'counters_after': after, 'counter_snapshots': snapshots}
            for name, good in checks.items():
                print('  ' + name + ': ' + ('OK' if good else 'FALHOU'), flush=True)
            if not all(checks.values()):
                failed = True
                if args.submit and (result.returncode in (70, 124) or not checks['external_counters_unchanged']):
                    print('Capturando bugreport após timeout/reset.', flush=True)
                    try:
                        summary['bugreport_rc'] = adb('bugreport', str(out / (case + '_bugreport.zip')), timeout=180).returncode
                    except subprocess.SubprocessError as error:
                        summary['bugreport_error'] = str(error)
                break
        summary['status'] = int(failed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        summary['error'] = str(error)
        failed = True
        print('Interrompido: ' + str(error), flush=True)
    finally:
        try:
            logcat = adb('logcat', '-d', '-v', 'threadtime', '-s', 'X940-BC45:I', 'RADV_HANG:I', '*:S', capture_output=True, text=True, timeout=20)
            (out / 'logcat.txt').write_text(logcat.stdout + logcat.stderr)
        except subprocess.SubprocessError as error:
            summary['logcat_error'] = str(error)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        archive = out.with_name(out.name + '_logs.tar.gz')
        with tarfile.open(archive, 'w:gz') as tar:
            for path in sorted(out.iterdir()):
                if path.is_file() and path.name != 'bc45_probe':
                    tar.add(path, arcname=path.name)
        print('Envie: ' + str(archive), flush=True)
    return int(failed)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print('BC45 interrompido: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
