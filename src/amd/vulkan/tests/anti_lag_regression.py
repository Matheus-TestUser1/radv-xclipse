#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile real pacing functions with a deterministic Vulkan/clock test double.

No NDK, GPU, Meson or Python dependencies are needed: python3 + a C compiler.
This is a host regression harness, not an Android driver build or GPU test.
"""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "src/vulkan/runtime/vk_frame_pacer.c"
ANTI_LAG = ROOT / "src/amd/vulkan/radv_anti_lag.c"


def function(source, name):
    """Keep the production function body verbatim, including its signature."""
    match = re.search(r"(?m)^" + re.escape(name) + r"\(", source)
    if not match:
        raise RuntimeError(f"Missing production function: {name}")
    start = source.rfind("\n", 0, match.start() - 1) + 1
    opening = source.index("{", match.end())
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "vulkan/vulkan_core.h"
#include "vulkan/vk_android_native_buffer.h"
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define ASSERTED
#define UNUSED __attribute__((unused))
#define ONE_MS_IN_NS INT64_C(1000000)
#define ONE_SECOND_IN_NS INT64_C(1000000000)
#define p_atomic_read(p) (*(p))
#define p_atomic_set(p,v) (*(p) = (v))
#define VK_FROM_HANDLE(type, name, handle) struct type *name = (void *)(handle)
typedef unsigned simple_mtx_t;
enum { mtx_plain };
static void simple_mtx_lock(simple_mtx_t *m) { assert(!*m); *m = 1; }
static void simple_mtx_unlock(simple_mtx_t *m) { assert(*m); *m = 0; }
/* Use Mesa's real ringbuffer macros, with single-threaded test locks. */
#define UTIL_MACROS_H
#define _SIMPLE_MTX_H
static void simple_mtx_init(simple_mtx_t *m, int flags) { (void)flags; *m = 0; }
#include "util/ringbuffer.h"
struct hash_entry { void *key, *data; };
struct hash_table { struct hash_entry entry; unsigned entries; };
#define hash_table_foreach(table, e) \
   for (struct hash_entry *e = (table)->entries ? &(table)->entry : NULL; e; e = NULL)
static struct hash_entry *_mesa_hash_table_search(struct hash_table *t, const void *key)
{ return t->entries && t->entry.key == key ? &t->entry : NULL; }
struct vk_frame_time_info { uint64_t frame_time; int64_t min_delay; };
typedef void (*vk_frame_time_cb)(const struct vk_frame_time_info *, void *);
struct vk_frame_pacer;
struct vk_device {
   struct vk_frame_pacer *frame_pacer;
   struct {
      PFN_vkWaitSemaphores WaitSemaphores;
      PFN_vkGetQueryPoolResults GetQueryPoolResults;
      PFN_vkResetQueryPool ResetQueryPool;
   } dispatch_table;
};
struct vk_queue { struct { struct vk_device *device; } base; };
static struct vk_queue *vk_queue_from_handle(VkQueue q) { return (void *)q; }
static VkDevice vk_device_to_handle(struct vk_device *d) { return (void *)d; }
'''

MOCKS = r'''
static unsigned enable_calls, disable_calls, eval_calls, calibrate_calls, sleep_calls;
static uint64_t now_ns = ONE_SECOND_IN_NS, slept_until;
static void *vk_frame_pacer_get_user_data(struct vk_frame_pacer *p) { return p->user_data; }
static void vk_frame_pacer_enable(struct vk_frame_pacer *p)
{ enable_calls++; p->enabled = true; }
static void vk_frame_pacer_disable(struct vk_frame_pacer *p)
{ disable_calls++; p->enabled = false; }
static void vk_frame_pacer_evaluate(struct vk_frame_pacer *p) { (void)p; eval_calls++; }
static void vk_frame_pacer_calibrate(struct vk_frame_pacer *p) { (void)p; calibrate_calls++; }
static uint64_t os_time_get_nano(void) { return now_ns; }
static void os_time_nanosleep_until(uint64_t deadline)
{ sleep_calls++; slept_until = deadline; }
static VkResult VKAPI_CALL wait_semaphores(VkDevice d, const VkSemaphoreWaitInfo *i, uint64_t t)
{ (void)d; (void)i; (void)t; return VK_SUCCESS; }
static VkResult VKAPI_CALL query_results(VkDevice d, VkQueryPool p, uint32_t first,
   uint32_t count, size_t size, void *data, VkDeviceSize stride, VkQueryResultFlags flags)
{
   (void)d; (void)p; (void)size; (void)flags;
   assert(first == 0 && count == 2);
   *(uint64_t *)data = 99900;
   *(uint64_t *)((char *)data + stride) = 99950;
   return VK_SUCCESS;
}
static void VKAPI_CALL reset_queries(VkDevice d, VkQueryPool p, uint32_t f, uint32_t c)
{ (void)d; (void)p; (void)f; (void)c; }
static unsigned release_calls, present_calls;
static uint32_t expected_wait_count;
static const VkSemaphore *expected_waits;
static VkImage expected_image;
static VkResult VKAPI_CALL release_image(VkQueue q, uint32_t count,
   const VkSemaphore *waits, VkImage image, int *fd)
{
   (void)q;
   assert(count == expected_wait_count && waits == expected_waits && image == expected_image);
   release_calls++; *fd = 37;
   return VK_ERROR_OUT_OF_DATE_KHR;
}
static VkResult VKAPI_CALL present_image(VkQueue q, const VkPresentInfoKHR *info)
{ (void)q; assert(info); present_calls++; return VK_SUBOPTIMAL_KHR; }
'''

TESTS = r'''
static void test_android_and_wsi_boundaries(void)
{
   struct vk_frame_pacer p = { .enabled = true, .queue_release_image = release_image,
                              .queue_present = present_image };
   struct vk_device d = { .frame_pacer = &p };
   struct vk_queue q = { .base.device = &d };
   queue_context ctx = {0};
   p.queues = (struct hash_table){ .entry = { &q, &ctx }, .entries = 1 };
   ringbuffer_init(p.frames);
   int fd = -1;
   /* Native-buffer release with no waits must still advance the frame. */
   assert(frame_pacer_QueueSignalReleaseImageANDROID((VkQueue)&q, 0, NULL, VK_NULL_HANDLE, &fd)
          == VK_ERROR_OUT_OF_DATE_KHR);
   assert(fd == 37 && release_calls == 1 && p.frames.size == 1);
   assert(ctx.latency_sensitive && p.active_frame == &p.frames.data[0]);
   VkSemaphore wait = (VkSemaphore)(uintptr_t)42;
   expected_wait_count = 1; expected_waits = &wait;
   expected_image = (VkImage)(uintptr_t)43;
   frame_pacer_QueueSignalReleaseImageANDROID((VkQueue)&q, 1, &wait, expected_image, &fd);
   assert(p.frames.size == 2 && release_calls == 2);
   VkPresentInfoKHR info = { .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
   assert(frame_pacer_QueuePresentKHR((VkQueue)&q, &info) == VK_SUBOPTIMAL_KHR);
   assert(p.frames.size == 3 && present_calls == 1);
   p.enabled = false;
   frame_pacer_QueueSignalReleaseImageANDROID((VkQueue)&q, 1, &wait, expected_image, &fd);
   assert(p.frames.size == 3 && release_calls == 3);
}

static void test_queries_older_than_calibration(void)
{
   struct vk_frame_pacer p = { .calibration = {
      .base_device_ts = 100000, .base_host_ts = ONE_SECOND_IN_NS, .timestamp_period = 10.0f } };
   struct vk_device d = { .frame_pacer = &p, .dispatch_table = {
      wait_semaphores, query_results, reset_queries } };
   p.device = &d;
   queue_context ctx = {0};
   ringbuffer_init(p.frames); ringbuffer_init(ctx.queries);
   struct vk_frame_time_info *frame = ringbuffer_alloc(p.frames);
   ctx.num_submits[0] = 1;
   struct query *begin = ringbuffer_alloc(ctx.queries);
   begin->submit_ts = ONE_SECOND_IN_NS - 2000;
   assert(ringbuffer_alloc(ctx.queries));
   p.queues = (struct hash_table){ .entry = { NULL, &ctx }, .entries = 1 };
   assert(evaluate_frame(&p, frame, 0));
   assert(frame->frame_time == 500 && frame->min_delay == 1000);
   assert(ctx.queries.size == 0 && ctx.num_submits[0] == 0);
}

static void test_modes_stages_and_fps(void)
{
   struct anti_lag_data data = {0};
   struct vk_frame_pacer p = { .user_data = &data };
   struct vk_device d = { .frame_pacer = &p };
   VkAntiLagDataAMD update = { .sType = VK_STRUCTURE_TYPE_ANTI_LAG_DATA_AMD,
                             .mode = VK_ANTI_LAG_MODE_ON_AMD, .maxFPS = 60 };
   radv_AntiLagUpdateAMD((VkDevice)&d, &update);
   assert(enable_calls == 1 && eval_calls == 1 && data.prev_input_begin == now_ns);
   now_ns += 1000000;
   radv_AntiLagUpdateAMD((VkDevice)&d, &update);
   assert(slept_until == ONE_SECOND_IN_NS + ONE_SECOND_IN_NS / 60);
   VkAntiLagPresentationInfoAMD stage = {
      .sType = VK_STRUCTURE_TYPE_ANTI_LAG_PRESENTATION_INFO_AMD,
      .stage = VK_ANTI_LAG_STAGE_PRESENT_AMD, .frameIndex = 1 };
   update.pPresentationInfo = &stage;
   unsigned prior_sleeps = sleep_calls, prior_evals = eval_calls;
   radv_AntiLagUpdateAMD((VkDevice)&d, &update);
   assert(sleep_calls == prior_sleeps && eval_calls == prior_evals);
   data.avg_frame_time = 10000000; data.queuing_delay = 3000000; data.deviation = 2000000;
   assert(get_wait_time(&data) == 12000000);
   update.mode = VK_ANTI_LAG_MODE_OFF_AMD;
   radv_AntiLagUpdateAMD((VkDevice)&d, &update);
   assert(disable_calls == 1 && !p.enabled && sleep_calls == prior_sleeps);
   assert(!data.avg_frame_time && !data.queuing_delay && !data.deviation);
   assert(!data.prev_input_begin && !data.recalibrate_when);
   update.mode = VK_ANTI_LAG_MODE_DRIVER_CONTROL_AMD;
   update.maxFPS = 0; stage.stage = VK_ANTI_LAG_STAGE_INPUT_AMD;
   radv_AntiLagUpdateAMD((VkDevice)&d, &update);
   assert(p.enabled && data.prev_input_begin == now_ns && slept_until == 0);
   struct vk_frame_time_info sample = { .frame_time = 10000000, .min_delay = 2000000 };
   frame_time_cb(&sample, &data);
   assert(data.avg_frame_time == 5000000 && data.queuing_delay == 1000000);
}
int main(void)
{
   test_android_and_wsi_boundaries();
   test_queries_older_than_calibration();
   test_modes_stages_and_fps();
   puts("PASS: Android/WSI boundaries, timestamp calibration, OFF/ON, stages and maxFPS");
   return 0;
}
'''


def main():
    runtime = RUNTIME.read_text()
    anti_lag = ANTI_LAG.read_text()
    structures = runtime[runtime.index("#define MAX_FRAMES"):runtime.index("\nvoid\nvk_frame_pacer_calibrate")]
    state = anti_lag[anti_lag.index("struct anti_lag_data"):anti_lag.index("\nstatic void\nframe_time_cb")]
    unit = PRELUDE + structures + state + MOCKS
    for name in ("evaluate_frame", "frame_pacer_present", "frame_pacer_QueuePresentKHR",
                 "frame_pacer_QueueSignalReleaseImageANDROID"):
        unit += function(runtime, name)
    for name in ("frame_time_cb", "get_wait_time", "radv_AntiLagUpdateAMD"):
        unit += function(anti_lag, name)
    unit += TESTS
    with tempfile.TemporaryDirectory(prefix="radv-anti-lag-") as tmp:
        source, binary = Path(tmp) / "test.c", Path(tmp) / "test"
        source.write_text(unit)
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) + [
            "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wno-sign-compare",
            "-DVK_USE_PLATFORM_ANDROID_KHR",
            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
            "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
            str(source), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
