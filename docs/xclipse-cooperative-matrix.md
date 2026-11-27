# Experimental cooperative matrices on Xclipse 940

This fork can implement `VK_KHR_cooperative_matrix` using subgroup operations
and FP32 fused multiply-adds. The NIR lowering consumes actual cooperative
matrix SPIR-V operations, so applications can use the KHR API rather than a
custom multiplication shader. No native WMMA/MFMA instruction is requested by
this path. It does not establish whether the hardware supports such instructions.

## Build and enable

Build with the existing `build.sh`, `build.cmd`, or `build-wayland.sh`. The
implementation is part of the driver and needs no external shader package.
Set this environment variable **in the process loading the built driver**, before
instance/physical-device creation:

```sh
RADV_XCLIPSE_COOPMAT_ALU=1 your-vulkan-application
```

The default is disabled; `RADV_XCLIPSE_COOPMAT_ALU=0` explicitly disables it.
There is no Android property alias in this integration. Setting an environment
variable in an ADB shell does not configure an independently launched Android app.
Use the application's/emulator's supported environment settings when available.
The application must query the extension and properties, enable the
`cooperativeMatrix` device feature, and submit cooperative matrix shaders.
Installing the driver alone does not turn ordinary shaders into matrix workloads.

Activation is restricted to Xclipse 940 with a GFX11 shader core, the pre-GFX11.7
compiler layout, and ACO. Other GPUs and the LLVM backend retain their existing
paths. This option does not change the ISA target, hardware identity, kernel,
firmware, register interface, or choice of wave size.

## Advertised support

| M×N×K | A/B components | C/result components | Scope | Saturating accumulation |
|---|---|---|---|---|
| 16×16×16 | FP32 | FP32 | Subgroup | No |
| 16×16×16 | FP16 | FP32 | Subgroup | No |

Both wave32 and wave64 are supported. Integer combinations, FP16 accumulators,
BF16/FP8 cooperative operations, `VK_EXT_cooperative_matrix_maintenance1`, and
`VK_NV_cooperative_matrix2` are not advertised by this experimental path.

The existing GFX11 fragment layout is reused. A/B are replicated per 16 lanes;
C/result elements are distributed over the subgroup. Wave32 keeps the validated
shuffle algorithm. Wave64 reads each of four possible source lanes uniformly,
then selects locally, avoiding ACO's costly generic divergent shuffle fallback
on this target. The tested wave64 compiler printouts contained 256 fixed lane
reads per pipeline, rather than the previous 4,096 reads and comparisons.
These are static compiler counts, not dynamic hardware counters.

The compiler cache key uses a two-bit algorithm tag: 0 disabled, 1 the earlier
ALU experiment, 2 the fixed-broadcast implementation integrated here. The key
remains 20 bytes, and the three tags remain distinct.

## Validation

The eight driver source changes are imported without modification from the v2
experiment based on commit `54d7acd806c7008f9dc3fc1a057a848385dc31a3`.
That commit was also the fork's `main` when this integration was prepared.
Its Android build was run on a Samsung S24 FE SM-S721B, Xclipse 940, Android 14
build `S721BXXS3AYB8`, as Android shell without `su`. The collected driver was
18,048,344 bytes, with SHA256:

```text
1c2d7544d384e8c4af4e7a8b4a2079082abaf0c519f46d95e71d778c1b7c5efc
```

The collection `result-20261123T081549Z-11109.zip` was independently reanalyzed:

- Extension/feature queries with the option disabled and enabled passed.
- 768 API regression cases passed: waves 32/64, FP16/FP32, eight memory layouts,
  two operation profiles, nonzero accumulators, borders, partial K, and K=0.
  All 278,656 final values matched an independently reconstructed reference.
- 864 benchmark dispatches passed, including 576 measured samples. Another
  1,330,956 final values were independently reconstructed and checked.
- Collected compiler printouts confirmed fixed broadcasts on wave64 and no
  `v_wmma_*`/`v_mfma_*` instructions. With this build, these printouts are ACO
  output after lowering, not independent disassembly of the final bytes.

The repository's host regression executes the actual helper through an eager
SIMD adapter, comparing 2,048 cases / 524,288 output values against a separate
dense reference. It checks wave32/wave64, FP16/FP32 inputs, eight layouts, fixed
broadcast indices, and compiler-cache tag separation:

```sh
python3 src/amd/vulkan/tests/cooperative_matrix_alu_regression.py
python3 src/amd/vulkan/tests/cooperative_matrix_alu_regression.py --sanitize
```

These host checks require Python 3 and a C11 compiler, selectable with `CC`.
They model NIR primitives; they do not validate real NIR SSA or ACO codegen.
The numerical patterns are exactly representable finite values and do not
cover all rounding behavior, NaNs, infinities, or subnormals. Vulkan CTS and
broad application compatibility remain unverified. The simple benchmark
control was not an optimized GEMM library, and its speed ratios do not imply
game FPS gains. Keep the implementation experimental and opt-in.
