/*
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir/radv_meta_nir.h"
#include "radv_meta.h"
#include "radv_sampler.h"
#include "radv_fsr1.h"
#include "ac_xclipse_log.h"
#include "vk_command_pool.h"
#include "vk_shader_module.h"

static const uint32_t fsr1_easu_spv[] = {
#include "fsr1_easu_spv.h"
};
static const uint32_t fsr1_rcas_spv[] = {
#include "fsr1_rcas_spv.h"
};

static enum glsl_sampler_dim
translate_sampler_dim(VkImageType type)
{
   switch (type) {
   case VK_IMAGE_TYPE_1D:
      return GLSL_SAMPLER_DIM_1D;
   case VK_IMAGE_TYPE_2D:
      return GLSL_SAMPLER_DIM_2D;
   case VK_IMAGE_TYPE_3D:
      return GLSL_SAMPLER_DIM_3D;
   default:
      UNREACHABLE("Unhandled image type");
   }
}

static VkResult
get_pipeline_layout(struct radv_device *device, bool fsr1, VkPipelineLayout *layout_out)
{
   enum radv_meta_object_key_type key = fsr1 ? RADV_META_OBJECT_KEY_FSR1_RCAS : RADV_META_OBJECT_KEY_BLIT;

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = fsr1 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };

   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | (fsr1 ? VK_SHADER_STAGE_FRAGMENT_BIT : 0),
      .size = fsr1 ? 32 : 20,
   };

   return vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device, &desc_info, &pc_range, &key, sizeof(key),
                                      layout_out);
}

struct radv_blit_key {
   enum radv_meta_object_key_type type;
   VkFormat format;
   VkImageAspectFlags aspects;
   VkImageType image_type;
};

static VkResult
get_pipeline(struct radv_device *device, const struct radv_image_view *src_iview,
             const struct radv_image_view *dst_iview, bool fsr1,
             VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   const VkImageAspectFlags aspect = src_iview->vk.aspects;
   const struct radv_image *src_image = src_iview->image;
   const enum glsl_sampler_dim tex_dim = translate_sampler_dim(src_image->vk.image_type);
   struct radv_blit_key key;
   VkResult result;

   result = get_pipeline_layout(device, fsr1, layout_out);
   if (result != VK_SUCCESS)
      return result;

   memset(&key, 0, sizeof(key));
   key.type = fsr1 ? RADV_META_OBJECT_KEY_FSR1_RCAS : RADV_META_OBJECT_KEY_BLIT;
   key.aspects = aspect;
   key.image_type = src_image->vk.image_type;

   if (aspect == VK_IMAGE_ASPECT_COLOR_BIT)
      key.format = dst_iview->vk.format;

   VkPipeline pipeline_from_cache = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (pipeline_from_cache != VK_NULL_HANDLE) {
      *pipeline_out = pipeline_from_cache;
      return VK_SUCCESS;
   }

   nir_shader *fs = NULL;
   nir_shader *vs = radv_meta_nir_build_blit_vertex_shader();
   VkShaderModule fsr1_module = VK_NULL_HANDLE;

   if (fsr1) {
      const VkShaderModuleCreateInfo module_info = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = sizeof(fsr1_rcas_spv),
         .pCode = fsr1_rcas_spv,
      };
      result = device->vk.dispatch_table.CreateShaderModule(radv_device_to_handle(device), &module_info,
                                                           &device->meta_state.alloc, &fsr1_module);
      if (result != VK_SUCCESS) {
         ralloc_free(vs);
         return result;
      }
   } else {
      switch (aspect) {
      case VK_IMAGE_ASPECT_COLOR_BIT:
         fs = radv_meta_nir_build_blit_copy_fragment_shader(tex_dim);
         break;
      case VK_IMAGE_ASPECT_DEPTH_BIT:
         fs = radv_meta_nir_build_blit_copy_fragment_shader_depth(tex_dim);
         break;
      case VK_IMAGE_ASPECT_STENCIL_BIT:
         fs = radv_meta_nir_build_blit_copy_fragment_shader_stencil(tex_dim);
         break;
      default:
         UNREACHABLE("Unhandled aspect");
      }
   }

   VkGraphicsPipelineCreateInfo pipeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages =
         (VkPipelineShaderStageCreateInfo[]){
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
             .stage = VK_SHADER_STAGE_VERTEX_BIT,
             .module = vk_shader_module_handle_from_nir(vs),
             .pName = "main",
             .pSpecializationInfo = NULL},
            {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
             .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
             .module = fsr1 ? fsr1_module : vk_shader_module_handle_from_nir(fs),
             .pName = "main",
             .pSpecializationInfo = NULL},
         },
      .pVertexInputState =
         &(VkPipelineVertexInputStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 0,
            .vertexAttributeDescriptionCount = 0,
         },
      .pInputAssemblyState =
         &(VkPipelineInputAssemblyStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_META_RECT_LIST_MESA,
            .primitiveRestartEnable = false,
         },
      .pViewportState =
         &(VkPipelineViewportStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
         },
      .pRasterizationState =
         &(VkPipelineRasterizationStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                   .rasterizerDiscardEnable = false,
                                                   .polygonMode = VK_POLYGON_MODE_FILL,
                                                   .cullMode = VK_CULL_MODE_NONE,
                                                   .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                   .lineWidth = 1.0f},
      .pMultisampleState =
         &(VkPipelineMultisampleStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = 1,
            .sampleShadingEnable = false,
            .pSampleMask = (VkSampleMask[]){UINT32_MAX},
         },
      .pDynamicState =
         &(VkPipelineDynamicStateCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = 2,
            .pDynamicStates =
               (VkDynamicState[]){
                  VK_DYNAMIC_STATE_VIEWPORT,
                  VK_DYNAMIC_STATE_SCISSOR,
               },
         },
      .layout = *layout_out,
   };

   VkPipelineColorBlendStateCreateInfo color_blend_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments =
         (VkPipelineColorBlendAttachmentState[]){
            {.colorWriteMask = VK_COLOR_COMPONENT_A_BIT | VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT},
         },
      .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f}};

   VkPipelineDepthStencilStateCreateInfo depth_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = true,
      .depthWriteEnable = true,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS,
   };

   VkPipelineDepthStencilStateCreateInfo stencil_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = false,
      .depthWriteEnable = false,
      .stencilTestEnable = true,
      .front = {.failOp = VK_STENCIL_OP_REPLACE,
                .passOp = VK_STENCIL_OP_REPLACE,
                .depthFailOp = VK_STENCIL_OP_REPLACE,
                .compareOp = VK_COMPARE_OP_ALWAYS,
                .compareMask = 0xff,
                .writeMask = 0xff,
                .reference = 0},
      .back = {.failOp = VK_STENCIL_OP_REPLACE,
               .passOp = VK_STENCIL_OP_REPLACE,
               .depthFailOp = VK_STENCIL_OP_REPLACE,
               .compareOp = VK_COMPARE_OP_ALWAYS,
               .compareMask = 0xff,
               .writeMask = 0xff,
               .reference = 0},
      .depthCompareOp = VK_COMPARE_OP_ALWAYS,
   };

   struct vk_meta_rendering_info render = {0};

   switch (aspect) {
   case VK_IMAGE_ASPECT_COLOR_BIT:
      pipeline_create_info.pColorBlendState = &color_blend_info;
      render.color_attachment_count = 1;
      render.color_attachment_formats[0] = key.format;
      break;
   case VK_IMAGE_ASPECT_DEPTH_BIT:
      pipeline_create_info.pDepthStencilState = &depth_info;
      render.depth_attachment_format = VK_FORMAT_D32_SFLOAT;
      break;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      pipeline_create_info.pDepthStencilState = &stencil_info;
      render.stencil_attachment_format = VK_FORMAT_S8_UINT;
      break;
   default:
      UNREACHABLE("Unhandled aspect");
   }

   result = vk_meta_create_graphics_pipeline(&device->vk, &device->meta_state.device, &pipeline_create_info, &render,
                                             &key, sizeof(key), pipeline_out);

   ralloc_free(vs);
   ralloc_free(fs);
   if (fsr1_module)
      device->vk.dispatch_table.DestroyShaderModule(radv_device_to_handle(device), fsr1_module,
                                                   &device->meta_state.alloc);
   return result;
}

static void
meta_emit_blit(struct radv_cmd_buffer *cmd_buffer, struct radv_image_view *src_iview, VkImageLayout src_image_layout,
               float src_offset_0[3], float src_offset_1[3], struct radv_image_view *dst_iview,
               VkImageLayout dst_image_layout, VkRect2D dst_box, VkSampler sampler)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_image *src_image = src_iview->image;
   const struct radv_image *dst_image = dst_iview->image;
   uint32_t src_width = u_minify(src_image->vk.extent.width, src_iview->vk.base_mip_level);
   uint32_t src_height = u_minify(src_image->vk.extent.height, src_iview->vk.base_mip_level);
   uint32_t src_depth = u_minify(src_image->vk.extent.depth, src_iview->vk.base_mip_level);
   uint32_t dst_width = u_minify(dst_image->vk.extent.width, dst_iview->vk.base_mip_level);
   uint32_t dst_height = u_minify(dst_image->vk.extent.height, dst_iview->vk.base_mip_level);
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result;

   assert(src_image->vk.samples == dst_image->vk.samples);

   result = get_pipeline(device, src_iview, dst_iview, false, &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
      return;
   }

   float vertex_push_constants[5] = {
      src_offset_0[0] / (float)src_width,  src_offset_0[1] / (float)src_height, src_offset_1[0] / (float)src_width,
      src_offset_1[1] / (float)src_height, src_offset_0[2] / (float)src_depth,
   };

   radv_meta_push_constants(cmd_buffer, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(vertex_push_constants),
                            vertex_push_constants);

   radv_meta_bind_graphics_pipeline(cmd_buffer, pipeline);

   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1,
                              (VkDescriptorGetInfoEXT[]){{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                                                          .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                          .data.pCombinedImageSampler = (VkDescriptorImageInfo[]){
                                                             {
                                                                .sampler = sampler,
                                                                .imageView = radv_image_view_to_handle(src_iview),
                                                                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                                             },
                                                          }}});

   VkRenderingInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags = VK_RENDERING_LOCAL_READ_CONCURRENT_ACCESS_CONTROL_BIT_KHR,
      .renderArea =
         {
            .offset = {0, 0},
            .extent = {dst_width, dst_height},
         },
      .layerCount = 1,
   };

   VkRenderingAttachmentInfo color_att;
   if (src_iview->vk.aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
      color_att = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(dst_iview),
         .imageLayout = dst_image_layout,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      rendering_info.colorAttachmentCount = 1;
      rendering_info.pColorAttachments = &color_att;
   }

   VkRenderingAttachmentInfo depth_att;
   if (src_iview->vk.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
      depth_att = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(dst_iview),
         .imageLayout = dst_image_layout,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      rendering_info.pDepthAttachment = &depth_att;
   }

   VkRenderingAttachmentInfo stencil_att;
   if (src_iview->vk.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
      stencil_att = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = radv_image_view_to_handle(dst_iview),
         .imageLayout = dst_image_layout,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      rendering_info.pStencilAttachment = &stencil_att;
   }

   radv_CmdBeginRendering(radv_cmd_buffer_to_handle(cmd_buffer), &rendering_info);

   radv_CmdDraw(radv_cmd_buffer_to_handle(cmd_buffer), 3, 1, 0, 0);

   const VkRenderingEndInfoKHR end_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_END_INFO_KHR,
   };

   radv_CmdEndRendering2KHR(radv_cmd_buffer_to_handle(cmd_buffer), &end_info);
}

static bool
flip_coords(unsigned *src0, unsigned *src1, unsigned *dst0, unsigned *dst1)
{
   bool flip = false;
   if (*src0 > *src1) {
      unsigned tmp = *src0;
      *src0 = *src1;
      *src1 = tmp;
      flip = !flip;
   }

   if (*dst0 > *dst1) {
      unsigned tmp = *dst0;
      *dst0 = *dst1;
      *dst1 = tmp;
      flip = !flip;
   }
   return flip;
}

static void
blit_image(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src_image, VkImageLayout src_image_layout,
           struct radv_image *dst_image, VkImageLayout dst_image_layout, const VkImageBlit2 *region, VkFilter filter)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const VkImageSubresourceLayers *src_res = &region->srcSubresource;
   const VkImageSubresourceLayers *dst_res = &region->dstSubresource;
   struct radv_sampler sampler;

   /* From the Vulkan 1.0 spec:
    *
    *    vkCmdBlitImage must not be used for multisampled source or
    *    destination images. Use vkCmdResolveImage for this purpose.
    */
   assert(src_image->vk.samples == 1);
   assert(dst_image->vk.samples == 1);

   radv_sampler_init(device, &sampler,
                     &(VkSamplerCreateInfo){
                        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                        .magFilter = filter,
                        .minFilter = filter,
                        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                     });

   unsigned dst_start, dst_end;
   if (dst_image->vk.image_type == VK_IMAGE_TYPE_3D) {
      assert(dst_res->baseArrayLayer == 0);
      dst_start = region->dstOffsets[0].z;
      dst_end = region->dstOffsets[1].z;
   } else {
      dst_start = dst_res->baseArrayLayer;
      dst_end = dst_start + vk_image_subresource_layer_count(&dst_image->vk, dst_res);
   }

   unsigned src_start, src_end;
   if (src_image->vk.image_type == VK_IMAGE_TYPE_3D) {
      assert(src_res->baseArrayLayer == 0);
      src_start = region->srcOffsets[0].z;
      src_end = region->srcOffsets[1].z;
   } else {
      src_start = src_res->baseArrayLayer;
      src_end = src_start + vk_image_subresource_layer_count(&src_image->vk, src_res);
   }

   bool flip_z = flip_coords(&src_start, &src_end, &dst_start, &dst_end);
   float src_z_step = (float)(src_end - src_start) / (float)(dst_end - dst_start);

   /* There is no interpolation to the pixel center during
    * rendering, so add the 0.5 offset ourselves here. */
   float depth_center_offset = 0;
   if (src_image->vk.image_type == VK_IMAGE_TYPE_3D)
      depth_center_offset = 0.5 / (dst_end - dst_start) * (src_end - src_start);

   unsigned src_layer_start = src_start;
   int src_layer_step = 1;

   if (flip_z) {
      src_start = src_end;
      src_z_step *= -1;
      src_layer_start = src_end - 1;
      src_layer_step *= -1;
      depth_center_offset *= -1;
   }

   unsigned src_x0 = region->srcOffsets[0].x;
   unsigned src_x1 = region->srcOffsets[1].x;
   unsigned dst_x0 = region->dstOffsets[0].x;
   unsigned dst_x1 = region->dstOffsets[1].x;

   unsigned src_y0 = region->srcOffsets[0].y;
   unsigned src_y1 = region->srcOffsets[1].y;
   unsigned dst_y0 = region->dstOffsets[0].y;
   unsigned dst_y1 = region->dstOffsets[1].y;

   VkRect2D dst_box;
   dst_box.offset.x = MIN2(dst_x0, dst_x1);
   dst_box.offset.y = MIN2(dst_y0, dst_y1);
   dst_box.extent.width = dst_x1 - dst_x0;
   dst_box.extent.height = dst_y1 - dst_y0;

   const VkOffset2D dst_offset_0 = {
      .x = dst_x0,
      .y = dst_y0,
   };
   const VkOffset2D dst_offset_1 = {
      .x = dst_x1,
      .y = dst_y1,
   };

   radv_meta_set_viewport(cmd_buffer, dst_offset_0.x, dst_offset_0.y, dst_offset_1.x - dst_offset_0.x,
                          dst_offset_1.y - dst_offset_0.y);

   radv_meta_set_scissor(cmd_buffer, MIN2(dst_offset_0.x, dst_offset_1.x), MIN2(dst_offset_0.y, dst_offset_1.y),
                         abs(dst_offset_1.x - dst_offset_0.x), abs(dst_offset_1.y - dst_offset_0.y));

   const unsigned num_layers = dst_end - dst_start;
   for (unsigned i = 0; i < num_layers; i++) {
      struct radv_image_view dst_iview, src_iview;

      float src_offset_0[3] = {
         src_x0,
         src_y0,
         src_start + i * src_z_step + depth_center_offset,
      };
      float src_offset_1[3] = {
         src_x1,
         src_y1,
         src_start + i * src_z_step + depth_center_offset,
      };
      const uint32_t dst_array_slice = dst_start + i;

      /* 3D images have just 1 layer */
      const uint32_t src_array_slice =
         src_image->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : src_layer_start + i * src_layer_step;

      const VkImageViewUsage2CreateInfoKHR dst_iview_usage_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
         .usage = vk_format_is_color(dst_image->vk.format) ? VK_IMAGE_USAGE_2_COLOR_ATTACHMENT_BIT_KHR
                                                           : VK_IMAGE_USAGE_2_DEPTH_STENCIL_ATTACHMENT_BIT_KHR,
      };

      radv_image_view_init(&dst_iview, device,
                           &(VkImageViewCreateInfo){
                              .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                              .pNext = &dst_iview_usage_info,
                              .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                              .image = radv_image_to_handle(dst_image),
                              .viewType = radv_meta_get_view_type(dst_image, true),
                              .format = dst_image->vk.format,
                              .subresourceRange = {.aspectMask = dst_res->aspectMask,
                                                   .baseMipLevel = dst_res->mipLevel,
                                                   .levelCount = 1,
                                                   .baseArrayLayer = dst_array_slice,
                                                   .layerCount = 1},
                           },
                           NULL);

      const VkImageViewUsage2CreateInfoKHR src_iview_usage_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
         .usage = VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR,
      };

      radv_image_view_init(&src_iview, device,
                           &(VkImageViewCreateInfo){
                              .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                              .pNext = &src_iview_usage_info,
                              .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
                              .image = radv_image_to_handle(src_image),
                              .viewType = radv_meta_get_view_type(src_image, false),
                              .format = src_image->vk.format,
                              .subresourceRange = {.aspectMask = src_res->aspectMask,
                                                   .baseMipLevel = src_res->mipLevel,
                                                   .levelCount = 1,
                                                   .baseArrayLayer = src_array_slice,
                                                   .layerCount = 1},
                           },
                           NULL);
      meta_emit_blit(cmd_buffer, &src_iview, src_image_layout, src_offset_0, src_offset_1, &dst_iview, dst_image_layout,
                     dst_box, radv_sampler_to_handle(&sampler));

      radv_image_view_finish(&dst_iview);
      radv_image_view_finish(&src_iview);
   }

   radv_sampler_finish(device, &sampler);
}

/* FSR changes blit filtering, so it is explicitly opt-in and confined to full-frame SDR
 * upscales. No image resizing or changes to the application's advertised capabilities.
 */
static bool
fsr1_eligible(struct radv_cmd_buffer *cmd_buffer, const struct radv_image *src,
              const struct radv_image *dst, const VkBlitImageInfo2 *info, uint8_t mode)
{
   if (!mode || mode > 2 || info->regionCount != 1 || info->filter != VK_FILTER_LINEAR || src == dst ||
       cmd_buffer->qf != RADV_QUEUE_GENERAL ||
       cmd_buffer->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY ||
       (cmd_buffer->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) ||
       (cmd_buffer->vk.pool && (cmd_buffer->vk.pool->flags & VK_COMMAND_POOL_CREATE_PROTECTED_BIT)))
      return false;

   bool presentable = dst->wsi_image;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
   presentable |= dst->vk.android_buffer_type == ANDROID_BUFFER_NATIVE ||
                  dst->vk.android_buffer_type == ANDROID_BUFFER_NATIVE_ALIAS;
#endif
   if (mode == 1 && !presentable)
      return false;

   if (src->vk.format != dst->vk.format)
      return false;
   switch (src->vk.format) {
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_SRGB:
      break;
   default:
      return false;
   }

   const struct radv_image *images[2] = {src, dst};
   const VkImageSubresourceLayers *layers[2] = {
      &info->pRegions[0].srcSubresource, &info->pRegions[0].dstSubresource,
   };
   const VkOffset3D *offsets[2] = {info->pRegions[0].srcOffsets, info->pRegions[0].dstOffsets};
   for (unsigned i = 0; i < 2; i++) {
      const struct radv_image *image = images[i];
      if (image->vk.image_type != VK_IMAGE_TYPE_2D || image->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
          image->vk.mip_levels != 1 || image->vk.array_layers != 1 ||
          (image->vk.create_flags & (VK_IMAGE_CREATE_PROTECTED_BIT | VK_IMAGE_CREATE_SPARSE_BINDING_BIT)) ||
          layers[i]->aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || layers[i]->mipLevel != 0 ||
          layers[i]->baseArrayLayer != 0 || layers[i]->layerCount != 1 ||
          offsets[i][0].x != 0 || offsets[i][0].y != 0 || offsets[i][0].z != 0 ||
          offsets[i][1].x != image->vk.extent.width || offsets[i][1].y != image->vk.extent.height ||
          offsets[i][1].z != 1)
         return false;
   }

   const uint32_t sw = src->vk.extent.width, sh = src->vk.extent.height;
   const uint32_t dw = dst->vk.extent.width, dh = dst->vk.extent.height;
   /* 1x-2x per axis; cap scratch allocation at 64 MiB. Reject degenerate/small surfaces. */
   return sw >= 64 && sh >= 64 && dw > sw && dh > sh &&
          (uint64_t)dw <= 2ull * sw && (uint64_t)dh <= 2ull * sh &&
          (uint64_t)dw * dh <= (64ull << 20) / 4;
}

static VkResult
fsr1_easu_pipeline(struct radv_device *device, VkPipeline *pipeline_out, VkPipelineLayout *layout_out)
{
   const enum radv_meta_object_key_type key = RADV_META_OBJECT_KEY_FSR1_EASU;
   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
   };
   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = ARRAY_SIZE(bindings), .pBindings = bindings,
   };
   const VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 80};
   VkResult result = vk_meta_get_pipeline_layout(&device->vk, &device->meta_state.device,
                                                &desc_info, &range, &key, sizeof(key), layout_out);
   if (result != VK_SUCCESS)
      return result;
   *pipeline_out = vk_meta_lookup_pipeline(&device->meta_state.device, &key, sizeof(key));
   if (*pipeline_out != VK_NULL_HANDLE)
      return VK_SUCCESS;

   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(fsr1_easu_spv), .pCode = fsr1_easu_spv,
   };
   VkShaderModule module;
   result = device->vk.dispatch_table.CreateShaderModule(radv_device_to_handle(device), &module_info,
                                                        &device->meta_state.alloc, &module);
   if (result != VK_SUCCESS)
      return result;
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
      .layout = *layout_out,
   };
   result = vk_meta_create_compute_pipeline(&device->vk, &device->meta_state.device,
                                            &pipeline_info, &key, sizeof(key), pipeline_out);
   device->vk.dispatch_table.DestroyShaderModule(radv_device_to_handle(device), module,
                                                 &device->meta_state.alloc);
   return result;
}

static bool
fsr1_blit(struct radv_cmd_buffer *cmd_buffer, struct radv_image *src, struct radv_image *dst,
           const VkBlitImageInfo2 *info)
{
   struct radv_device *device = radv_cmd_buffer_device(cmd_buffer);
   const struct radv_physical_device *pdev = radv_device_physical(device);
   if (!fsr1_eligible(cmd_buffer, src, dst, info, pdev->xclipse_fsr1_mode))
      return false;

   const VkCommandBuffer cmd = radv_cmd_buffer_to_handle(cmd_buffer);
   const uint32_t sw = src->vk.extent.width, sh = src->vk.extent.height;
   const uint32_t dw = dst->vk.extent.width, dh = dst->vk.extent.height;
   const uint32_t bytes = dw * dh * 4;
   VkPipeline easu, rcas;
   VkPipelineLayout easu_layout, rcas_layout;
   VkResult result = fsr1_easu_pipeline(device, &easu, &easu_layout);
   if (result != VK_SUCCESS)
      goto error;

   struct radv_image_view source_view, target_view;
   struct radv_image *images[2] = {src, dst};
   struct radv_image_view *views[2] = {&source_view, &target_view};
   for (unsigned i = 0; i < 2; i++) {
      const VkImageViewUsage2CreateInfoKHR usage = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_2_CREATE_INFO_KHR,
         .usage = i ? VK_IMAGE_USAGE_2_COLOR_ATTACHMENT_BIT_KHR : VK_IMAGE_USAGE_2_SAMPLED_BIT_KHR,
      };
      const VkImageViewCreateInfo view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &usage,
         .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
         .image = radv_image_to_handle(images[i]), .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
         /* FSR operates on display-encoded SDR values. Avoid automatic sRGB decode/encode. */
         .format = vk_format_srgb_to_linear(images[i]->vk.format),
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
      radv_image_view_init(views[i], device, &view_info, NULL);
   }
   result = get_pipeline(device, &source_view, &target_view, true, &rcas, &rcas_layout);
   if (result != VK_SUCCESS)
      goto finish_views;

   /* Command-buffer-owned scratch survives submission and is reclaimed on reset/destruction.
    * Capture the VA before descriptor uploads can grow/replace the current upload buffer.
    */
   unsigned offset;
   if (!radv_cmd_buffer_upload_alloc_aligned(cmd_buffer, bytes, 256, &offset, NULL)) {
      result = cmd_buffer->vk.record_result;
      goto finish_views;
   }
   const VkDescriptorAddressInfoEXT scratch = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
      .address = radv_buffer_get_va(cmd_buffer->upload.upload_bo) + offset,
      .range = bytes, .format = VK_FORMAT_UNDEFINED,
   };
   struct radv_sampler sampler;
   radv_sampler_init(device, &sampler, &(VkSamplerCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   });
   const VkDescriptorImageInfo sampled = {
      .sampler = radv_sampler_to_handle(&sampler),
      .imageView = radv_image_view_to_handle(&source_view),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDescriptorGetInfoEXT easu_desc[] = {
      {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
       .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .data.pCombinedImageSampler = &sampled},
      {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
       .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .data.pStorageBuffer = &scratch},
   };
   uint32_t constants[20];
   radv_fsr1_easu_constants(constants, sw, sh, dw, dh);
   radv_meta_bind_compute_pipeline(cmd_buffer, easu);
   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, easu_layout,
                               ARRAY_SIZE(easu_desc), easu_desc);
   radv_meta_push_constants(cmd_buffer, easu_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(constants), constants);
   const struct radv_dispatch_info dispatch = {.blocks = {DIV_ROUND_UP(dw, 8), DIV_ROUND_UP(dh, 8), 1}};
   radv_compute_dispatch(cmd_buffer, &dispatch);

   /* The fragment pass must observe every EASU storage-buffer write. */
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier,
   };
   radv_CmdPipelineBarrier2(cmd, &dependency);

   const VkDescriptorGetInfoEXT rcas_desc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .data.pStorageBuffer = &scratch,
   };
   /* FsrRcasCon at 1.0 stops = half of maximum sharpening strength. FP32 path uses con[0]. */
   const uint32_t rcas_constants[8] = {dw, dh, 0, 0, radv_fsr1_float_bits(0.5f), 0, 0, 0};
   radv_meta_bind_graphics_pipeline(cmd_buffer, rcas);
   radv_meta_bind_descriptors(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, rcas_layout, 1, &rcas_desc);
   radv_meta_push_constants(cmd_buffer, rcas_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, sizeof(rcas_constants), rcas_constants);
   radv_meta_set_viewport_and_scissor(cmd_buffer, 0, 0, dw, dh);
   const VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = radv_image_view_to_handle(&target_view), .imageLayout = info->dstImageLayout,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {dw, dh}}, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &attachment,
   };
   radv_CmdBeginRendering(cmd, &rendering);
   radv_CmdDraw(cmd, 3, 1, 0, 0);
   const VkRenderingEndInfoKHR end = {.sType = VK_STRUCTURE_TYPE_RENDERING_END_INFO_KHR};
   radv_CmdEndRendering2KHR(cmd, &end);
   radv_sampler_finish(device, &sampler);

#if DETECT_OS_ANDROID
   if (ac_xclipse_log_level() >= 1) {
      static unsigned reports;
      if (p_atomic_inc_return(&reports) <= 8)
         AC_XCLIPSE_LOGP(ANDROID_LOG_INFO, "RADV_FSR1", "[FSR1] EASU+RCAS %ux%u -> %ux%u mode=%u",
                         sw, sh, dw, dh, pdev->xclipse_fsr1_mode);
   }
#endif

finish_views:
   radv_image_view_finish(&target_view);
   radv_image_view_finish(&source_view);
error:
   if (result != VK_SUCCESS)
      vk_command_buffer_set_error(&cmd_buffer->vk, result);
   return true;
}

VKAPI_ATTR void VKAPI_CALL
radv_CmdBlitImage2(VkCommandBuffer commandBuffer, const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(radv_cmd_buffer, cmd_buffer, commandBuffer);
   VK_FROM_HANDLE(radv_image, src_image, pBlitImageInfo->srcImage);
   VK_FROM_HANDLE(radv_image, dst_image, pBlitImageInfo->dstImage);

   radv_suspend_conditional_rendering(cmd_buffer);

   radv_meta_begin(cmd_buffer);

   if (!fsr1_blit(cmd_buffer, src_image, dst_image, pBlitImageInfo)) {
      for (unsigned r = 0; r < pBlitImageInfo->regionCount; r++) {
         blit_image(cmd_buffer, src_image, pBlitImageInfo->srcImageLayout, dst_image, pBlitImageInfo->dstImageLayout,
                    &pBlitImageInfo->pRegions[r], pBlitImageInfo->filter);
      }
   }

   radv_meta_end(cmd_buffer);

   radv_resume_conditional_rendering(cmd_buffer);
}
