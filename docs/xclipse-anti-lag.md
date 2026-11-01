# VK_AMD_anti_lag on Xclipse

This change is based on `JimVulkan/radv-xclipse` main at
`fe9b8943aaeec11ed9fccbfcc382da07379afbe5` (28 October 2026).
That base already advertises `VK_AMD_anti_lag`, reports `antiLag = VK_TRUE`,
implements `vkAntiLagUpdateAMD`, and initializes Mesa's frame pacer when the
application enables the feature. Those existing implementations are preserved.

## Changes

* Connect the frame pacer to `QueueSignalReleaseImageANDROID`. Android's
  platform Vulkan loader owns `QueuePresentKHR` and calls the ICD's native
  buffer release entrypoint instead. Previously the pacer did not see those
  frame boundaries, never identified the presenting queue as latency sensitive,
  and could not collect per-frame GPU timing through that path.
* Share the frame-boundary implementation between native Android and the
  existing `QueuePresentKHR` hook used by desktop/Wayland WSI. The release's
  arguments, result and native fence output pass through to the original
  entrypoint. Releases with zero wait semaphores also count as boundaries.
* Keep GPU timestamp differences signed when converting them to nanoseconds.
  Pending queries may be older than a new calibration sample. Treating their
  difference as unsigned could produce enormous queuing delays and excessive
  pacing waits.
* Clear pacing history when the application selects OFF, so subsequent ON
  calls start with fresh measurements.

The Android hook is compiled only with `VK_USE_PLATFORM_ANDROID_KHR` and is
installed only when a native release entrypoint exists. All hooks are installed
only for a device with the `antiLag` feature enabled, as in the original base.
Existing SGPU submission, synchronization and Xclipse hardware workarounds are
preserved.

## Application use

The application or translation layer must enable the `VK_AMD_anti_lag` device
extension and `VkPhysicalDeviceAntiLagFeaturesAMD::antiLag` when creating its
Vulkan device. Load `vkAntiLagUpdateAMD` with `vkGetDeviceProcAddr` and call it
immediately before sampling input. For explicit stage reporting, pass INPUT
there and PRESENT before presentation, using a matching `frameIndex`.

`VK_ANTI_LAG_MODE_ON_AMD` enables pacing. `maxFPS = 0` disables the additional
framerate cap; nonzero values impose that cap while pacing is enabled. OFF
disables pacing. DRIVER_CONTROL retains the base driver's behavior of enabling
pacing. No new environment switch is required.

This does not automatically activate Anti-Lag in applications that never enable
or call the extension. Availability in a particular emulator also depends on its
Vulkan loader and translation layer exposing and using the extension.

## Host regression checks

Run from the repository root:

```sh
python3 src/amd/vulkan/tests/anti_lag_regression.py
git diff --check
```

The standalone harness compiles the production pacing functions with the
repository's Vulkan headers and ringbuffer macros. Vulkan dispatch and clocks
are test doubles. Checks cover Android release with and without waits,
argument/result/fence forwarding, WSI presentation, disabled pacing, queries
older than calibration, maxFPS, INPUT/PRESENT stages, and OFF/ON history reset.
Undefined-behavior and float-to-integer overflow sanitizers are enabled.

These are deterministic host checks, not a complete driver build, Vulkan CTS,
concurrency test, or measurement on an Xclipse GPU.

## Build and device validation

Build the normal or Wayland package with the repository's existing scripts:

```sh
./build.sh /path/to/android-ndk
# Or, for Bannerlator's Wayland mode:
./build-wayland.sh /path/to/android-ndk
```

On the target phone, verify extension enumeration and `antiLag`, create a
device with the feature enabled, and test input/present calls for several
hundred frames. Compare ON/OFF under GPU load, exercise zero-wait native
releases, toggle OFF/ON, and test `maxFPS` at 30 and 60. Repeat for native
Android and Wayland separately. Check for validation errors, hangs and device
loss. Physical input-to-display latency needs an actual device measurement;
no latency reduction is claimed from the host tests alone.

Specification: https://docs.vulkan.org/refpages/latest/refpages/source/VK_AMD_anti_lag.html

Call contract: https://docs.vulkan.org/refpages/latest/refpages/source/vkAntiLagUpdateAMD.html
