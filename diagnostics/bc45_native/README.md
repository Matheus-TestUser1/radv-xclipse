# Xclipse 940: BC4/BC5 native sampling experiment

This branch is a diagnostic, based on JimVulkan/radv-xclipse commit
`269238e8538c971cbe5195ed3e0013dea59f3bf5`. It does not change the normal driver's defaults.
It does not establish that BC4/BC5 hardware decoding exists on this chip.

The process-local `RADV_XCLIPSE_DIAG_BC45` switch accepts:

- `trace`: original format paths with descriptor/decode diagnostics.
- `native`: BC4_UNORM and BC5_UNORM bypass the emulation decision, hidden plane and
  decode/transcode dispatch. Existing GFX11 format-table entries are used unchanged.

Both modes are accepted only for `AC_XCLIPSE_940`, with its GFX11 shader core and descriptor
table. There is no Android property for the switch. Other BC formats keep their existing paths.
BC3_UNORM is the native control. No format-number sweep or arbitrary register writes are made.

## First stage: record on the phone

On Ubuntu, with the build requirements from the root README, NDK r29 and adb:

```sh
./build.sh "$HOME/android-ndk-r29"
python3 -B diagnostics/bc45_native/run.py --self-test
python3 -B diagnostics/bc45_native/run.py
```

The runner compiles the Android C probe and GLSL shader, uploads them and the diagnostic
library into `/data/local/tmp/x940-bc45-native`, and records six cases: BC3/BC4/BC5 in each mode.
It does not replace the system driver or the libraries in the previous Xclipse test directory.
The probe makes no `vkQueueSubmit` call in this stage. Driver initialization and command-buffer
recording occur on the real device; this is not a claim that the kernel/GPU is wholly unused by
the process or the rest of Android.

The verifier requires the 940 guard, unchanged input blocks, all eight texture descriptor
DWORDs, expected original/hidden plane selection, and absence of a BC decode dispatch in each
native path. Expected existing descriptor FORMAT codes are BC3=113, BC4=115, BC5=117. They come
from `gfx11-rsrc.json`, not a claim that each is a functioning native decoder on MGFX2.

The run stops on a failed check or any GPU counter change. It prints the log directory and
`jim_bc45_record_*_logs.tar.gz` to share. The record alone does not prove sampling works.

## Second stage: explicit real execution

After inspecting an approved first-stage capture:

```sh
python3 -B diagnostics/bc45_native/run.py --submit \
  --record "$HOME/xclipse-diag/jim_bc45_record_REPLACE_WITH_ACTUAL_DIRECTORY"
```

This requires the same source hashes, library, probe and SPIR-V as the approved recording.
It revalidates the six raw logs, then submits BC3 control, native BC4 and native BC5 once each,
stopping at the first failure. Each submit samples 64 texels from an 8x8 image into a host-coherent
SSBO. Four input blocks exercise both endpoint-order modes and every alpha/RGTC index. C and
Python decode references check every returned RGBA component with tolerance 2.1/255. No numerical
sampling test is run for lossy EAC fallback paths; those are recording controls only.

The probe uses a one-second fence wait and a 30-second child watchdog. If the wait fails it
holds resources for six seconds and exits without Vulkan teardown. After each case the runner
observes counters for another six seconds. It collects a bugreport on timeout/reset, and
captures its logcat flow markers. The numerical samples and descriptor words are retained.

A zero/incorrect result is a failure of this tested configuration, not proof that no decoder
can exist with a different, properly established descriptor encoding. A pass only establishes
these UNORM cases: SNORM, BC6H/BC7, filtering, mipmaps, arrays, other block modes and game stability
remain outside this experiment.

## Host checks

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -Iinclude \
  diagnostics/bc45_native/probe.c -o /tmp/bc45-probe -ldl -lm
/tmp/bc45-probe --self-test
python3 -B diagnostics/bc45_native/run.py --self-test
```

No GPU execution or full Android-driver build is implied by these checks.
