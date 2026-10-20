/* SPDX-License-Identifier: MIT
 * Isolated 8x8 BC3/BC4/BC5 UNORM sampling experiment; no window system.
 * The parent watchdog never loads the driver. Default CLI mode is record-only.
 */
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <assert.h>
#include <dlfcn.h>
#include <math.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif

#define INSTANCE_API(X) \
   X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) \
   X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceQueueFamilyProperties) \
   X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceFormatProperties) \
   X(vkCreateDevice) X(vkGetDeviceProcAddr)
#define DEVICE_API(X) \
   X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) \
   X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
   X(vkMapMemory) X(vkUnmapMemory) X(vkCreateImage) X(vkDestroyImage) \
   X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) \
   X(vkCreateSampler) X(vkDestroySampler) X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
   X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
   X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
   X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateCommandPool) X(vkDestroyCommandPool) \
   X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
   X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
   X(vkCmdDispatch) X(vkCreateFence) X(vkDestroyFence) X(vkQueueSubmit) X(vkWaitForFences)
#define DECLARE(n) static PFN_##n n;
INSTANCE_API(DECLARE)
DEVICE_API(DECLARE)
#undef DECLARE

static void flow(const char *event, long value)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   unsigned long long ns = (unsigned long long)t.tv_sec * 1000000000ull + (unsigned long long)t.tv_nsec;
   printf("bc45-flow: pid=%ld mono_ns=%llu event=%s value=%ld\n", (long)getpid(), ns, event, value);
   fflush(stdout);
#ifdef __ANDROID__
   __android_log_print(ANDROID_LOG_INFO, "X940-BC45", "pid=%ld mono_ns=%llu event=%s value=%ld",
                       (long)getpid(), ns, event, value);
#endif
}

#define CHECK(call) do { VkResult _r = (call); if (_r != VK_SUCCESS) { flow(#call, _r); return 1; } } while (0)

/* Four endpoint pairs cover both alpha/RGTC modes, including their explicit 0 and 1 entries. */
static void alpha_block(uint8_t out[8], unsigned block, unsigned channel)
{
   static const uint8_t ends[4][2] = {{240, 16}, {20, 210}, {127, 127}, {0, 255}};
   out[0] = ends[(block + channel) & 3][0];
   out[1] = ends[(block + channel) & 3][1];
   uint64_t bits = 0;
   for (unsigned p = 0; p < 16; p++)
      bits |= (uint64_t)((p + block + channel * 3) & 7) << (3 * p);
   for (unsigned i = 0; i < 6; i++) out[i + 2] = (uint8_t)(bits >> (8 * i));
}

static float alpha_reference(const uint8_t *b, unsigned p)
{
   uint64_t bits = 0;
   for (unsigned i = 0; i < 6; i++) bits |= (uint64_t)b[i + 2] << (8 * i);
   unsigned n = (unsigned)((bits >> (3 * p)) & 7);
   if (n < 2) return b[n] / 255.0f;
   if (b[0] > b[1]) return ((8 - n) * (float)b[0] + (n - 1) * (float)b[1]) / (7 * 255.0f);
   if (n == 6) return 0.0f;
   if (n == 7) return 1.0f;
   return ((6 - n) * (float)b[0] + (n - 1) * (float)b[1]) / (5 * 255.0f);
}

static size_t make_blocks(unsigned bc, uint8_t data[64])
{
   memset(data, 0, 64);
   unsigned stride = bc == 4 ? 8 : 16;
   for (unsigned b = 0; b < 4; b++) {
      uint8_t *p = data + b * stride;
      alpha_block(p, b, 0);
      if (bc == 5) alpha_block(p + 8, b, 1);
      if (bc == 3) {
         /* RGB565 endpoints: exact red and green; all four color indices are exercised. */
         p[8] = 0x00; p[9] = 0xf8; p[10] = 0xe0; p[11] = 0x07;
         uint32_t bits = 0;
         for (unsigned i = 0; i < 16; i++) bits |= ((i + b) & 3) << (2 * i);
         for (unsigned i = 0; i < 4; i++) p[12 + i] = (uint8_t)(bits >> (8 * i));
      }
   }
   return 4 * stride;
}

static void pixel_reference(unsigned bc, const uint8_t *data, unsigned x, unsigned y, float v[4])
{
   unsigned block = (y / 4) * 2 + x / 4;
   unsigned p = (y % 4) * 4 + x % 4;
   const uint8_t *b = data + block * (bc == 4 ? 8 : 16);
   v[0] = alpha_reference(b, p); v[1] = 0; v[2] = 0; v[3] = 1;
   if (bc == 5) v[1] = alpha_reference(b + 8, p);
   if (bc == 3) {
      static const float rgb[4][3] = {{1,0,0}, {0,1,0}, {2.0f/3,1.0f/3,0}, {1.0f/3,2.0f/3,0}};
      unsigned n = (p + block) & 3;
      v[0] = rgb[n][0]; v[1] = rgb[n][1]; v[2] = rgb[n][2]; v[3] = alpha_reference(b, p);
   }
}

static int self_test(void)
{
   uint8_t data[64], b[8] = {255, 0, 0, 0, 0, 0, 0, 0};
   assert(alpha_reference(b, 0) == 1.0f);
   b[2] = 1; assert(alpha_reference(b, 0) == 0.0f);
   b[2] = 2; assert(fabsf(alpha_reference(b, 0) - 6.0f / 7) < 1e-6f);
   b[0] = 0; b[1] = 255; b[2] = 6; assert(alpha_reference(b, 0) == 0.0f);
   b[2] = 7; assert(alpha_reference(b, 0) == 1.0f);
   for (unsigned bc = 3; bc <= 5; bc++) {
      assert(make_blocks(bc, data) == (bc == 4 ? 32u : 64u));
      float lo = 1, hi = 0;
      for (unsigned y = 0; y < 8; y++) for (unsigned x = 0; x < 8; x++) {
         float v[4]; pixel_reference(bc, data, x, y, v);
         for (unsigned c = 0; c < 4; c++) assert(isfinite(v[c]) && v[c] >= 0 && v[c] <= 1);
         if (v[0] < lo) lo = v[0];
         if (v[0] > hi) hi = v[0];
      }
      assert(hi - lo > 0.8f);
   }
   puts("BC45_SELF_TEST_PASS: endpoints, interpolation modes, special entries, indices, nonconstant patterns");
   return 0;
}

static int memory_type(VkPhysicalDeviceMemoryProperties *p, uint32_t bits, VkMemoryPropertyFlags flags)
{
   for (unsigned i = 0; i < p->memoryTypeCount; i++)
      if ((bits & (1u << i)) && (p->memoryTypes[i].propertyFlags & flags) == flags) return (int)i;
   return -1;
}

struct buffer { VkBuffer object; VkDeviceMemory memory; void *map; };
static int make_buffer(VkDevice d, VkPhysicalDeviceMemoryProperties *mp, VkDeviceSize bytes,
                       VkBufferUsageFlags usage, struct buffer *out)
{
   CHECK(vkCreateBuffer(d, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
          .size = bytes, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE}, NULL, &out->object));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(d, out->object, &req);
   int type = memory_type(mp, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (type < 0) { flow("no_host_coherent_memory", -1); return 1; }
   CHECK(vkAllocateMemory(d, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
          .allocationSize = req.size, .memoryTypeIndex = (uint32_t)type}, NULL, &out->memory));
   CHECK(vkBindBufferMemory(d, out->object, out->memory, 0));
   CHECK(vkMapMemory(d, out->memory, 0, VK_WHOLE_SIZE, 0, &out->map));
   return 0;
}

static void free_buffer(VkDevice d, struct buffer *b)
{
   vkUnmapMemory(d, b->memory);
   vkDestroyBuffer(d, b->object, NULL);
   vkFreeMemory(d, b->memory, NULL);
}

static int run(const char *library, const char *shader_path, unsigned bc, bool submit)
{
   flow("start", bc);
   const char *diag = getenv("RADV_XCLIPSE_DIAG_BC45");
   if (!diag || (strcmp(diag, "trace") && strcmp(diag, "native"))) {
      flow("diagnostic_environment_missing", -1); return 1;
   }
   FILE *sf = fopen(shader_path, "rb");
   if (!sf) { perror(shader_path); return 1; }
   if (fseek(sf, 0, SEEK_END)) return 1;
   long shader_size = ftell(sf);
   if (shader_size < 20 || shader_size > 65536 || shader_size % 4) return 1;
   rewind(sf);
   uint32_t *shader = malloc((size_t)shader_size);
   if (!shader || fread(shader, 1, (size_t)shader_size, sf) != (size_t)shader_size || shader[0] != 0x07230203) return 1;
   fclose(sf);
   void *so = dlopen(library, RTLD_NOW | RTLD_LOCAL);
   if (!so) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(so, "vk_icdGetInstanceProcAddr");
   if (!gipa) gipa = (PFN_vkGetInstanceProcAddr)dlsym(so, "vkGetInstanceProcAddr");
   if (!gipa) return 1;
   PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
   if (!create_instance) return 1;
   VkInstance instance;
   CHECK(create_instance(&(VkInstanceCreateInfo){.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
         .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
          .pApplicationName = "x940_bc45_native_v1", .apiVersion = VK_API_VERSION_1_1}}, NULL, &instance));
#define LOAD_INSTANCE(n) n = (PFN_##n)gipa(instance, #n); if (!n) { flow("missing_" #n, -1); return 1; }
   INSTANCE_API(LOAD_INSTANCE)
#undef LOAD_INSTANCE
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   if (count != 1) { flow("physical_device_count", count); return 1; }
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(physical, &props);
   printf("BC45_DEVICE: vendor=%04x device=%04x name=%s\n", props.vendorID, props.deviceID, props.deviceName);
   if (props.vendorID != 0x1002) return 1;
   VkPhysicalDeviceFeatures available;
   vkGetPhysicalDeviceFeatures(physical, &available);
   if (!available.textureCompressionBC) { flow("textureCompressionBC_missing", 0); return 1; }
   VkFormat format = bc == 3 ? VK_FORMAT_BC3_UNORM_BLOCK : bc == 4 ? VK_FORMAT_BC4_UNORM_BLOCK : VK_FORMAT_BC5_UNORM_BLOCK;
   VkFormatProperties fp;
   vkGetPhysicalDeviceFormatProperties(physical, format, &fp);
   if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) { flow("sampled_format_missing", format); return 1; }
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, NULL);
   if (!nq || nq > 64) return 1;
   VkQueueFamilyProperties qprops[64];
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, qprops);
   uint32_t qf = 0;
   while (qf < nq && (!(qprops[qf].queueFlags & VK_QUEUE_COMPUTE_BIT) || !qprops[qf].queueCount)) qf++;
   if (qf == nq) return 1;
   float priority = 1;
   VkDevice device;
   CHECK(vkCreateDevice(physical, &(VkDeviceCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
         .queueCreateInfoCount = 1, .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qf, .queueCount = 1,
          .pQueuePriorities = &priority}, .pEnabledFeatures = &(VkPhysicalDeviceFeatures){.textureCompressionBC = VK_TRUE}}, NULL, &device));
#define LOAD_DEVICE(n) n = (PFN_##n)vkGetDeviceProcAddr(device, #n); if (!n) { flow("missing_" #n, -1); return 1; }
   DEVICE_API(LOAD_DEVICE)
#undef LOAD_DEVICE
   VkQueue queue;
   vkGetDeviceQueue(device, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(physical, &mp);
   struct buffer upload = {0}, output = {0};
   uint8_t blocks[64];
   size_t block_bytes = make_blocks(bc, blocks);
   if (make_buffer(device, &mp, block_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &upload) ||
       make_buffer(device, &mp, 64 * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &output)) return 1;
   memcpy(upload.map, blocks, block_bytes);
   float *pixels = output.map;
   for (unsigned i = 0; i < 256; i++) pixels[i] = NAN;
   printf("BC45_INPUT: format=%u size=%zu hex=", format, block_bytes);
   for (unsigned i = 0; i < block_bytes; i++) printf("%02x", blocks[i]);
   puts("");
   VkImage image;
   CHECK(vkCreateImage(device, &(VkImageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = format, .extent = {8,8,1}, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED}, NULL, &image));
   VkMemoryRequirements ir;
   vkGetImageMemoryRequirements(device, image, &ir);
   int it = memory_type(&mp, ir.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (it < 0) return 1;
   VkDeviceMemory image_memory;
   CHECK(vkAllocateMemory(device, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = ir.size, .memoryTypeIndex = (uint32_t)it}, NULL, &image_memory));
   CHECK(vkBindImageMemory(device, image, image_memory, 0));
   VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
   VkImageView view;
   CHECK(vkCreateImageView(device, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format, .subresourceRange = range}, NULL, &view));
   VkSampler sampler;
   CHECK(vkCreateSampler(device, &(VkSamplerCreateInfo){.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
         .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0}, NULL, &sampler));
   VkDescriptorSetLayoutBinding bindings[2] = {{.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT}, {.binding=1,.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount=1,.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount=2,.pBindings=bindings}, NULL, &dsl));
   VkDescriptorPool pool;
   VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}};
   CHECK(vkCreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
         .maxSets=1,.poolSizeCount=2,.pPoolSizes=sizes}, NULL, &pool));
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool=pool,.descriptorSetCount=1,.pSetLayouts=&dsl}, &set));
   VkDescriptorImageInfo ii = {sampler,view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
   VkDescriptorBufferInfo bi = {output.object,0,VK_WHOLE_SIZE};
   VkWriteDescriptorSet writes[2] = {{.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=0,
         .descriptorCount=1,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.pImageInfo=&ii},
         {.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,.dstSet=set,.dstBinding=1,.descriptorCount=1,
          .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&bi}};
   vkUpdateDescriptorSets(device,2,writes,0,NULL);
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
         .setLayoutCount=1,.pSetLayouts=&dsl},NULL,&pl));
   VkShaderModule module;
   CHECK(vkCreateShaderModule(device, &(VkShaderModuleCreateInfo){.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize=(size_t)shader_size,.pCode=shader},NULL,&module));
   free(shader);
   VkPipeline pipeline;
   CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&(VkComputePipelineCreateInfo){
         .sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,.layout=pl,.stage={
          .sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,.stage=VK_SHADER_STAGE_COMPUTE_BIT,
          .module=module,.pName="main"}},NULL,&pipeline));
   VkCommandPool cp;
   CHECK(vkCreateCommandPool(device,&(VkCommandPoolCreateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
         .queueFamilyIndex=qf},NULL,&cp));
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(device,&(VkCommandBufferAllocateInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
         .commandPool=cp,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1},&cb));
   CHECK(vkBeginCommandBuffer(cb,&(VkCommandBufferBeginInfo){.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   VkImageMemoryBarrier barrier = {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=image,.subresourceRange=range};
   vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&barrier);
   vkCmdCopyBufferToImage(cb,upload.object,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&(VkBufferImageCopy){
         .imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={8,8,1}});
   barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
   barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
   vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,NULL,0,NULL,1,&barrier);
   vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
   vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&set,0,NULL);
   vkCmdDispatch(cb,1,1,1);
   vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&(VkMemoryBarrier){
         .sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT},0,NULL,0,NULL);
   CHECK(vkEndCommandBuffer(cb));
   flow("command_buffer_recorded",bc);
   VkFence fence=VK_NULL_HANDLE;
   int status=0;
   if (!submit) {
      puts("BC45_RECORD_ONLY_PASS: probe vkQueueSubmit calls=0; sampling not executed");
   } else {
      CHECK(vkCreateFence(device,&(VkFenceCreateInfo){.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},NULL,&fence));
      flow("submit_begin",bc);
      VkResult qr=vkQueueSubmit(queue,1,&(VkSubmitInfo){.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount=1,.pCommandBuffers=&cb},fence);
      flow("submit_return",qr);
      if (qr!=VK_SUCCESS) return 70;
      VkResult wr=vkWaitForFences(device,1,&fence,VK_TRUE,1000000000ull);
      flow("fence_return",wr);
      if (wr!=VK_SUCCESS) {
         flow("retaining_resources_after_wait_failure",wr);
         for (unsigned i=0;i<6;i++) { struct timespec delay={1,0}; nanosleep(&delay,NULL); flow("retained_second",i+1); }
         return 70; /* No Vulkan teardown with an uncompleted submission. */
      }
      float worst=0; unsigned bad=0;
      for (unsigned y=0;y<8;y++) for (unsigned x=0;x<8;x++) {
         float expected[4]; pixel_reference(bc,blocks,x,y,expected);
         unsigned p=y*8+x;
         printf("BC45_PIXEL: %u %.9g %.9g %.9g %.9g\n",p,pixels[4*p],pixels[4*p+1],pixels[4*p+2],pixels[4*p+3]);
         for (unsigned c=0;c<4;c++) {
            float error=fabsf(pixels[4*p+c]-expected[c]);
            if (!isfinite(pixels[4*p+c]) || error>2.1f/255.0f) bad++;
            if (error>worst) worst=error;
         }
      }
      printf("BC45_RESULT: bc=%u bad_components=%u max_abs_error=%.9g tolerance=%.9g\n",bc,bad,worst,2.1/255.0);
      puts(bad ? "BC45_SAMPLE_MISMATCH" : "BC45_SAMPLE_PASS");
      status=bad ? 2 : 0;
   }
   if (fence) vkDestroyFence(device,fence,NULL);
   vkDestroyCommandPool(device,cp,NULL);
   vkDestroyPipeline(device,pipeline,NULL);
   vkDestroyShaderModule(device,module,NULL);
   vkDestroyPipelineLayout(device,pl,NULL);
   vkDestroyDescriptorPool(device,pool,NULL);
   vkDestroyDescriptorSetLayout(device,dsl,NULL);
   vkDestroySampler(device,sampler,NULL);
   vkDestroyImageView(device,view,NULL);
   vkDestroyImage(device,image,NULL);
   vkFreeMemory(device,image_memory,NULL);
   free_buffer(device,&output); free_buffer(device,&upload);
   vkDestroyDevice(device,NULL); vkDestroyInstance(instance,NULL);
   flow("teardown_complete",status);
   return status;
}

int main(int argc,char **argv)
{
   setvbuf(stdout,NULL,_IONBF,0); setvbuf(stderr,NULL,_IONBF,0);
   if (argc==2 && !strcmp(argv[1],"--self-test")) return self_test();
   if (argc==3 && !strcmp(argv[1],"--oracle") &&
       (!strcmp(argv[2],"bc3") || !strcmp(argv[2],"bc4") || !strcmp(argv[2],"bc5"))) {
      unsigned bc=(unsigned)(argv[2][2]-'0'); uint8_t data[64]; size_t bytes=make_blocks(bc,data);
      printf("ORACLE_INPUT: "); for (unsigned i=0;i<bytes;i++) printf("%02x",data[i]); puts("");
      for (unsigned y=0;y<8;y++) for (unsigned x=0;x<8;x++) {
         float v[4]; pixel_reference(bc,data,x,y,v);
         printf("ORACLE_PIXEL: %u %.9g %.9g %.9g %.9g\n",y*8+x,v[0],v[1],v[2],v[3]);
      }
      return 0;
   }
   if (argc!=5 || (strcmp(argv[3],"bc3") && strcmp(argv[3],"bc4") && strcmp(argv[3],"bc5")) ||
       (strcmp(argv[4],"record") && strcmp(argv[4],"submit"))) {
      fprintf(stderr,"usage: %s driver.so sample.spv bc3|bc4|bc5 record|submit\n",argv[0]); return 1;
   }
   pid_t child=fork();
   if (child<0) return 1;
   if (!child) _exit(run(argv[1],argv[2],(unsigned)(argv[3][2]-'0'),!strcmp(argv[4],"submit")));
   printf("BC45_CHILD: pid=%ld\n",(long)child);
   for (unsigned i=0;i<300;i++) {
      int status; pid_t r=waitpid(child,&status,WNOHANG);
      if (r==child) { int rc=WIFEXITED(status) ? WEXITSTATUS(status) : 128+WTERMSIG(status); flow("child_exit",rc); return rc; }
      if (r<0) return 1;
      struct timespec delay={0,100000000}; nanosleep(&delay,NULL);
   }
   flow("watchdog_timeout",30);
   kill(child,SIGKILL); waitpid(child,NULL,0); return 124;
}
