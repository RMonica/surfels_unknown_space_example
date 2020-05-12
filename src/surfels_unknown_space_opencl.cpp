/*
 * Copyright (c) 2019, Riccardo Monica
 *   RIMLab, Department of Engineering and Architecture, University of Parma, Italy
 *   http://www.rimlab.ce.unipr.it/
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions
 * and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 * conditions and the following disclaimer in the documentation and/or other materials provided with
 * the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 * endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <surfels_unknown_space/surfels_unknown_space.h>
#include "surfels_unknown_space.cl.h"

#define LOG_INFO(...) m_logger->LogInfo(__VA_ARGS__)
#define LOG_INFO_STREAM(s) m_logger->LogInfo(OSS() << s)

#define LOG_ERROR(...) m_logger->LogError(__VA_ARGS__)
#define LOG_ERROR_STREAM(s) m_logger->LogError(OSS() << s)

#define LOG_FATAL(...) m_logger->LogFatal(__VA_ARGS__)
#define LOG_FATAL_STREAM(s) m_logger->LogFatal(OSS() << s)

void SurfelsUnknownSpace::initOpenCL(const Config & config)
{
  const std::string platform_name = config.opencl_platform_name;
  const std::string device_name = config.opencl_device_name;
  const bool use_intel = config.opencl_use_intel;

  cl_device_type device_type;
  if (config.opencl_device_type == Config::TOpenCLDeviceType::ALL)
    device_type = CL_DEVICE_TYPE_ALL;
  else if (config.opencl_device_type == Config::TOpenCLDeviceType::CPU)
    device_type = CL_DEVICE_TYPE_CPU;
  else if (config.opencl_device_type == Config::TOpenCLDeviceType::GPU)
    device_type = CL_DEVICE_TYPE_GPU;
  else
  {
    LOG_ERROR_STREAM("invalid config opencl_device_type");
    device_type = CL_DEVICE_TYPE_ALL;
  }

  const int subdevice_size = config.opencl_subdevice_size;

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.empty())
  {
    LOG_FATAL_STREAM("surfels_unknown_space: opencl: no platforms found.");
    exit(1);
  }

  {
    std::string all_platform_names;
    for (uint64 i = 0; i < all_platforms.size(); i++)
      all_platform_names += "\n  -- " + all_platforms[i].getInfo<CL_PLATFORM_NAME>();
    LOG_INFO_STREAM("surfels_unknown_space: opencl: found platforms:" << all_platform_names);
  }
  uint64 platform_id = 0;
  if (!platform_name.empty())
  {
    LOG_INFO("surfels_unknown_space: opencl: looking for matching platform: %s", platform_name.c_str());
    for (uint64 i = 0; i < all_platforms.size(); i++)
    {
      const std::string plat = all_platforms[i].getInfo<CL_PLATFORM_NAME>();
      if (plat.find(platform_name) != std::string::npos)
      {
        LOG_INFO("surfels_unknown_space: opencl: found matching platform: %s", plat.c_str());
        platform_id = i;
        break;
      }
    }
  }

  cl::Platform default_platform = all_platforms[platform_id];
  LOG_INFO_STREAM("surfels_unknown_space: opencl: using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>());

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(device_type, &all_devices);
  if (all_devices.empty())
  {
      LOG_FATAL_STREAM("surfels_unknown_space: opencl: no devices found.");
      exit(1);
  }
  {
    std::string all_device_names;
    for (uint64 i = 0; i < all_devices.size(); i++)
      all_device_names += "\n  -- " + all_devices[i].getInfo<CL_DEVICE_NAME>();
    LOG_INFO_STREAM("surfels_unknown_space: opencl: found devices:" << all_device_names);
  }
  uint64 device_id = 0;
  if (!device_name.empty())
  {
    LOG_INFO("surfels_unknown_space: opencl: looking for matching device: %s", device_name.c_str());
    for (uint64 i = 0; i < all_devices.size(); i++)
    {
      const std::string dev = all_devices[i].getInfo<CL_DEVICE_NAME>();
      if (dev.find(device_name) != std::string::npos)
      {
        LOG_INFO("surfels_unknown_space: opencl: found matching device: %s", dev.c_str());
        device_id = i;
        break;
      }
    }
  }

  cl::Device default_device = all_devices[device_id];
  m_opencl_device = CLDevicePtr(new cl::Device(default_device));
  LOG_INFO_STREAM("surfels_unknown_space: opencl: using device: " << default_device.getInfo<CL_DEVICE_NAME>());

  if (subdevice_size)
  {
#ifdef CL_VERSION_1_2
    LOG_INFO("surfels_unknown_space: opencl: using subdevice of size: %d", int(subdevice_size));
    const cl_device_partition_property properties[4] = {CL_DEVICE_PARTITION_BY_COUNTS,
                                                        subdevice_size,
                                                        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
                                                        0};
    std::vector<cl::Device> subdevices;
    const cl_int err = m_opencl_device->createSubDevices(properties, &subdevices);
    if (err != CL_SUCCESS || subdevices.empty())
      LOG_ERROR("surfels_unknown_space: opencl: could not create subdevice (error: %d), using whole device.", int(err));
    else
      m_opencl_device = CLDevicePtr(new cl::Device(subdevices[0]));
#else
    LOG_ERROR("surfels_unknown_space: opencl: could not create subdevice of size %d: OpenCL 1.2 not supported",
              int(subdevice_size));
#endif
  }

  m_opencl_context = CLContextPtr(new cl::Context({*m_opencl_device}));

  m_opencl_command_queue = CLCommandQueuePtr(new cl::CommandQueue(*m_opencl_context,*m_opencl_device));

  std::string source = SURFELS_UNKNOWN_SPACE_CL;
  if (use_intel)
    source = "#define USE_INTEL_COMPILER 1\n" + source;

  cl::Program::Sources sources;
  sources.push_back({source.c_str(),source.length()});

  LOG_INFO("surfels_unknown_space: opencl: building program... ");
  m_opencl_program = CLProgramPtr(new cl::Program(*m_opencl_context,sources));
  if (m_opencl_program->build({*m_opencl_device}) != CL_SUCCESS)
  {
    LOG_FATAL_STREAM("surfels_unknown_space: opencl: error building opencl_program: " <<
                     m_opencl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_opencl_device));
    exit(1);
  }
  LOG_INFO("surfels_unknown_space: opencl: initialized.");
}

SurfelsUnknownSpace::CLBufferPtr SurfelsUnknownSpace::OpenCLCreateBuffer(const CLContextPtr context,
                                                                       const size_t size,
                                                                       const std::string name) const
{
  cl_int err;
  CLBufferPtr buf = CLBufferPtr(new cl::Buffer(*context,CL_MEM_READ_WRITE,
                                    size, NULL, &err));
  if (err != CL_SUCCESS)
  {
    LOG_ERROR("could not allocate buffer '%s' of size %u, error %d", name.c_str(), unsigned(size), int(err));
  }
  return buf;
}

void SurfelsUnknownSpace::OpenCLFilterKnownStateHullPN(const uint64 count_at_max_range)
{
  if (!m_opencl_filter_known_state_hull_pn_kernel)
    m_opencl_filter_known_state_hull_pn_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program,
                                                                            "filter_known_state_hull_pn"));

  int ret;
  m_opencl_filter_known_state_hull_pn_kernel->setArg(0, cl_ulong(count_at_max_range));
  m_opencl_filter_known_state_hull_pn_kernel->setArg(1, cl_ulong(m_intrinsics->height));
  m_opencl_filter_known_state_hull_pn_kernel->setArg(2, *m_opencl_known_state_hull_xp);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(3, *m_opencl_known_state_hull_xp_filtered);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_state_hull_pn_kernel, cl::NullRange,
                                                     cl::NDRange(count_at_max_range, m_intrinsics->height), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownStateHullPN: error m_opencl_filter_known_state_hull_xp_kernel: %d!", ret);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(2, *m_opencl_known_state_hull_xn);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(3, *m_opencl_known_state_hull_xn_filtered);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_state_hull_pn_kernel, cl::NullRange,
                                                     cl::NDRange(count_at_max_range, m_intrinsics->height), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownStateHullPN: error m_opencl_filter_known_state_hull_xn_kernel: %d!", ret);

  m_opencl_filter_known_state_hull_pn_kernel->setArg(0, cl_ulong(m_intrinsics->width));
  m_opencl_filter_known_state_hull_pn_kernel->setArg(1, cl_ulong(count_at_max_range));
  m_opencl_filter_known_state_hull_pn_kernel->setArg(2, *m_opencl_known_state_hull_yp);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(3, *m_opencl_known_state_hull_yp_filtered);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_state_hull_pn_kernel, cl::NullRange,
                                                     cl::NDRange(m_intrinsics->width, count_at_max_range), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownStateHullPN: error m_opencl_filter_known_state_hull_yp_kernel: %d!", ret);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(2, *m_opencl_known_state_hull_yn);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(3, *m_opencl_known_state_hull_yn_filtered);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_state_hull_pn_kernel, cl::NullRange,
                                                     cl::NDRange(m_intrinsics->width, count_at_max_range), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownStateHullPN: error m_opencl_filter_known_state_hull_yn_kernel: %d!", ret);

  m_opencl_filter_known_state_hull_pn_kernel->setArg(0, cl_ulong(m_intrinsics->height));
  m_opencl_filter_known_state_hull_pn_kernel->setArg(1, cl_ulong(m_intrinsics->width));
  m_opencl_filter_known_state_hull_pn_kernel->setArg(2, *m_opencl_known_state_hull_zp);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(3, *m_opencl_known_state_hull_zp_filtered);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_state_hull_pn_kernel, cl::NullRange,
                                                     cl::NDRange(m_intrinsics->height, m_intrinsics->width), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownStateHullPN: error m_opencl_filter_known_state_hull_zp_kernel: %d!", ret);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(2, *m_opencl_known_state_hull_zn);
  m_opencl_filter_known_state_hull_pn_kernel->setArg(3, *m_opencl_known_state_hull_zn_filtered);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_state_hull_pn_kernel, cl::NullRange,
                                                     cl::NDRange(m_intrinsics->height, m_intrinsics->width), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownStateHullPN: error m_opencl_filter_known_state_hull_zn_kernel: %d!", ret);

  m_opencl_command_queue->finish();
}

void SurfelsUnknownSpace::OpenCLProjectAndDeleteSurfels(const Eigen::Affine3f & pose,
                                                        const SurfelVector & surfels,
                                                        const uint64 count_at_max_range,
                                                        const FloatVector & colors)
{
  {
    if (!m_opencl_surfels)
      m_opencl_surfels = CLBufferPtr(new cl::Buffer(*m_opencl_context,CL_MEM_READ_WRITE,
                                                    m_opencl_max_surfels_in_mem * sizeof(OpenCLSurfel)));
    if (!m_opencl_dot_field)
      m_opencl_dot_field = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width * m_intrinsics->height *
                                              count_at_max_range * 3 * sizeof(cl_int),
                                              "m_opencl_dot_field");
    if (!m_opencl_known_state_hull_xp)
      m_opencl_known_state_hull_xp = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->height *
                                                        count_at_max_range * sizeof(cl_uint),
                                                        "m_opencl_known_state_hull_xp");
    if (!m_opencl_known_state_hull_xn)
      m_opencl_known_state_hull_xn = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->height *
                                                        count_at_max_range * sizeof(cl_uint),
                                                        "m_opencl_known_state_hull_xn");
    if (!m_opencl_known_state_hull_yp)
      m_opencl_known_state_hull_yp = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                        count_at_max_range * sizeof(cl_uint),
                                                        "m_opencl_known_state_hull_yp");
    if (!m_opencl_known_state_hull_yn)
      m_opencl_known_state_hull_yn = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                        count_at_max_range * sizeof(cl_uint),
                                                        "m_opencl_known_state_hull_yn");
    if (!m_opencl_known_state_hull_zp)
      m_opencl_known_state_hull_zp = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                        m_intrinsics->height * sizeof(cl_uint),
                                                        "m_opencl_known_state_hull_zp");
    if (!m_opencl_known_state_hull_zn)
      m_opencl_known_state_hull_zn = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                        m_intrinsics->height * sizeof(cl_uint),
                                                        "m_opencl_known_state_hull_zn");
    if (!m_opencl_known_state_hull_xp_filtered)
      m_opencl_known_state_hull_xp_filtered = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->height *
                                                                 count_at_max_range * sizeof(cl_uint),
                                                                 "m_opencl_known_state_hull_xp_filtered");
    if (!m_opencl_known_state_hull_xn_filtered)
      m_opencl_known_state_hull_xn_filtered = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->height *
                                                                 count_at_max_range * sizeof(cl_uint),
                                                                 "m_opencl_known_state_hull_xn_filtered");
    if (!m_opencl_known_state_hull_yp_filtered)
      m_opencl_known_state_hull_yp_filtered = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                                count_at_max_range * sizeof(cl_uint),
                                                                "m_opencl_known_state_hull_yp_filtered");
    if (!m_opencl_known_state_hull_yn_filtered)
      m_opencl_known_state_hull_yn_filtered = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                                 count_at_max_range * sizeof(cl_uint),
                                                                 "m_opencl_known_state_hull_yn_filtered");
    if (!m_opencl_known_state_hull_zp_filtered)
      m_opencl_known_state_hull_zp_filtered = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                                 m_intrinsics->height * sizeof(cl_uint),
                                                                "m_opencl_known_state_hull_zp_filtered");
    if (!m_opencl_known_state_hull_zn_filtered)
      m_opencl_known_state_hull_zn_filtered = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                                 m_intrinsics->height * sizeof(cl_uint),
                                                                 "m_opencl_known_state_hull_zn_filtered");
    if (!m_opencl_invalid_dot_field)
      m_opencl_invalid_dot_field = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width * m_intrinsics->height *
                                                      count_at_max_range * 3 * sizeof(cl_uchar),
                                                      "m_opencl_invalid_dot_field");
    // the first uint is the number of valid elements in the buffer
    // must be initialized to 0
    if (!m_opencl_ids_to_be_deleted)
      m_opencl_ids_to_be_deleted = OpenCLCreateBuffer(m_opencl_context, (m_opencl_max_surfels_in_mem + 1) * sizeof(cl_uint),
                                                      "m_opencl_ids_to_be_deleted");
    // the first surfel id is the number of valid elements in the buffer
    if (!m_opencl_ids_to_be_recolored)
      m_opencl_ids_to_be_recolored = OpenCLCreateBuffer(m_opencl_context, (m_opencl_max_surfels_in_mem + 1) *
                                                        sizeof(OpenCLRecolor),
                                                        "m_opencl_ids_to_be_recolored");

    if (!m_opencl_occupancy_ids)
      m_opencl_occupancy_ids = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width * m_intrinsics->height *
                                                      count_at_max_range * sizeof(cl_uint),
                                                      "m_opencl_occupancy_ids");

    if (!m_opencl_projected_positions)
      m_opencl_projected_positions = OpenCLCreateBuffer(m_opencl_context, m_opencl_max_surfels_in_mem * sizeof(cl_float3),
                                                           "m_opencl_projected_positions");
    if (!m_opencl_projected_normals)
      m_opencl_projected_normals = OpenCLCreateBuffer(m_opencl_context, m_opencl_max_surfels_in_mem * sizeof(cl_float3),
                                                           "m_opencl_projected_normals");
    if (!m_opencl_projected_image_coords)
      m_opencl_projected_image_coords = OpenCLCreateBuffer(m_opencl_context, m_opencl_max_surfels_in_mem * sizeof(cl_float3),
                                                           "m_opencl_projected_image_coords");
    if (!m_opencl_projected_internal_ids)
      m_opencl_projected_internal_ids = OpenCLCreateBuffer(m_opencl_context, (m_opencl_max_surfels_in_mem + 1) * sizeof(cl_uint),
                                                           "m_opencl_projected_internal_ids");
    if (!m_opencl_projected_external_ids)
      m_opencl_projected_external_ids = OpenCLCreateBuffer(m_opencl_context, (m_opencl_max_surfels_in_mem + 1) * sizeof(cl_uint),
                                                           "m_opencl_projected_external_ids");

    if (!m_opencl_projectsurfels_kernel)
    {
      uint64 pi = 0;
      m_opencl_projectsurfels_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "project_surfels"));
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_intrinsics);
      pi += 4;
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_inverse_pov_matrix);
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_surfels);
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_projected_positions);
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_projected_normals);
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_projected_image_coords);
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_projected_internal_ids);
      m_opencl_projectsurfels_kernel->setArg(pi++,*m_opencl_projected_external_ids);
    }

    if (!m_opencl_project_internal_surfels_kernel)
    {
      uint64 pi = 0;
      m_opencl_project_internal_surfels_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "project_internal_surfels"));
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_intrinsics);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,cl_ulong(m_surfels_projection_threads)); // batch size
      pi++; // offset
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_surfels);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_projected_positions);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_projected_normals);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_projected_image_coords);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_projected_internal_ids);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_observed_space_field);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_bearings);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_depth_image);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_dot_field);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_invalid_dot_field);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_ids_to_be_deleted);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_ids_to_be_recolored);
      m_opencl_project_internal_surfels_kernel->setArg(pi++,*m_opencl_occupancy_ids);
    }

    if (!m_opencl_project_external_surfels_kernel)
    {
      uint64 pi = 0;
      m_opencl_project_external_surfels_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "project_external_surfels"));
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_intrinsics);
      m_opencl_project_external_surfels_kernel->setArg(pi++,cl_ulong(m_surfels_projection_threads)); // batch size
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_surfels);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_projected_positions);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_projected_normals);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_projected_image_coords);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_projected_external_ids);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_bearings);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_known_state_hull_xp);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_known_state_hull_xn);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_known_state_hull_yp);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_known_state_hull_yn);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_known_state_hull_zp);
      m_opencl_project_external_surfels_kernel->setArg(pi++,*m_opencl_known_state_hull_zn);
    }

    if (!m_opencl_zero_dot_field_kernel)
    {
      m_opencl_zero_dot_field_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "zero_dot_field"));
      m_opencl_zero_dot_field_kernel->setArg(0,*m_opencl_intrinsics);
      m_opencl_zero_dot_field_kernel->setArg(1,*m_opencl_dot_field);
      m_opencl_zero_dot_field_kernel->setArg(2,*m_opencl_invalid_dot_field);
    }

    if (!m_opencl_simple_fill_uint_kernel)
    {
      m_opencl_simple_fill_uint_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "simple_fill_uint"));
    }

    if (!m_opencl_zero_occupancy_ids_kernel)
    {
      m_opencl_zero_occupancy_ids_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "zero_occupancy_ids"));
      m_opencl_zero_occupancy_ids_kernel->setArg(0,*m_opencl_intrinsics);
      m_opencl_zero_occupancy_ids_kernel->setArg(1,*m_opencl_occupancy_ids);
    }
  }

  m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_INIT);

  {
    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_zero_dot_field_kernel, cl::NullRange,
                                                           cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                           cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_zero_normal_dot_field_kernel: error enqueue: %d!", ret);
  }

  {
    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_zero_occupancy_ids_kernel, cl::NullRange,
                                                           cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                           cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_zero_occupancy_ids_kernel: error enqueue: %d!", ret);
  }

  {
    int ret;

    m_opencl_simple_fill_uint_kernel->setArg(0,cl_ulong(1));
    m_opencl_simple_fill_uint_kernel->setArg(1,cl_ulong(count_at_max_range));
    m_opencl_simple_fill_uint_kernel->setArg(2,cl_ulong(0));
    m_opencl_simple_fill_uint_kernel->setArg(4,cl_uint(CL_UINT_MAX / 2 * 2));
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_xp);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 1: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_xn);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 2: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_xp_filtered);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 3: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_xn_filtered);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 4: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(0,cl_ulong(1));
    m_opencl_simple_fill_uint_kernel->setArg(1,cl_ulong(count_at_max_range));
    m_opencl_simple_fill_uint_kernel->setArg(2,cl_ulong(0));
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_yp);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 5: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_yn);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 6: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_yp_filtered);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 7: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_yn_filtered);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 8: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(0,cl_ulong(1));
    m_opencl_simple_fill_uint_kernel->setArg(1,cl_ulong(m_intrinsics->width));
    m_opencl_simple_fill_uint_kernel->setArg(2,cl_ulong(0));
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_zp);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(m_intrinsics->width, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 9: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_zn);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(m_intrinsics->width, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 10: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_zp_filtered);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(m_intrinsics->width, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 11: error enqueue: %d!", ret);
    m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_known_state_hull_zn_filtered);
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                       cl::NDRange(m_intrinsics->width, m_intrinsics->height),
                                                       cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  m_opencl_simple_fill_int_kernel 12: error enqueue: %d!", ret);
  }

  const uint64 surfels_size = surfels.size();

  if (surfels_size == 0)
    return; // nothing to do

  OpenCLSurfelVector surfels_buf(m_opencl_max_surfels_in_mem);

  uint64 deleted_counter = 0;
  uint64 recolored_counter = 0;

  m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_INIT);

  int64 counter = -1;
  for (uint64 total_computed = 0; total_computed < surfels_size; )
  {
    counter++;
    m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_UPLOAD, counter);

    const uint64 offset = total_computed;

    uint64 uploaded_count = 0;
    for (uint64 i = offset; i < surfels_size && uploaded_count < m_opencl_max_surfels_in_mem; i++)
    {
      surfels_buf[uploaded_count] = OpenCLSurfel(m_surfels[i]);
      uploaded_count++;
    }
    total_computed += uploaded_count;

    LOG_INFO_STREAM("  iteration: from offset " << offset <<
                 " (total is " << surfels_size <<
                 ", uploaded " << uploaded_count << ")");

    m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_surfels, CL_TRUE, 0,
                                               uploaded_count * sizeof(OpenCLSurfel),
                                               surfels_buf.data());

    // initialize to-be-deleted buffer count to 0
    {
      int ret;
      m_opencl_simple_fill_uint_kernel->setArg(0,cl_ulong(0));
      m_opencl_simple_fill_uint_kernel->setArg(1,cl_ulong(0));
      m_opencl_simple_fill_uint_kernel->setArg(2,cl_ulong(0));
      m_opencl_simple_fill_uint_kernel->setArg(4,cl_uint(0));
      m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_ids_to_be_deleted);
      ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                         cl::NDRange(1),
                                                         cl::NullRange);
      if (ret != CL_SUCCESS)
        LOG_ERROR("  m_opencl_simple_fill_uint_kernel: m_opencl_ids_to_be_deleted error enqueue: %d!", ret);
      m_opencl_simple_fill_uint_kernel->setArg(3,*m_opencl_ids_to_be_recolored);
      ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_simple_fill_uint_kernel, cl::NullRange,
                                                         cl::NDRange(2),
                                                         cl::NullRange);
      if (ret != CL_SUCCESS)
        LOG_ERROR("  m_opencl_simple_fill_uint_kernel: m_opencl_ids_to_be_recolored error enqueue: %d!", ret);
    }

    m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_UPLOAD, counter);

    {
      // project all surfels
      m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_PROJECT, counter);
      const uint64 iterations = uploaded_count / m_surfels_projection_threads + !!(uploaded_count % m_surfels_projection_threads);
      m_opencl_projectsurfels_kernel->setArg(1,cl_ulong(m_surfels_projection_threads)); // batch size
      m_opencl_projectsurfels_kernel->setArg(2,cl_ulong(iterations));// iterations
      m_opencl_projectsurfels_kernel->setArg(3,cl_ulong(uploaded_count)); // total
      m_opencl_projectsurfels_kernel->setArg(4,cl_ulong(offset));

      pcl::console::TicToc tictoc;
      m_opencl_command_queue->finish();
      tictoc.tic();
      cl_uint internal_total = 0;
      cl_uint external_total = 0;
      m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_projected_internal_ids, CL_TRUE,
                                                0, sizeof(cl_uint),
                                                &internal_total);
      m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_projected_external_ids, CL_TRUE,
                                                0, sizeof(cl_uint),
                                                &external_total);
      int ret2 = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_projectsurfels_kernel, cl::NullRange,
                                                   cl::NDRange(m_surfels_projection_threads), cl::NullRange);
      if (ret2 != CL_SUCCESS)
        LOG_ERROR("  OpenCLProjectSurfels: all: error enqueue: %d!", ret2);

//      m_opencl_command_queue->enqueueReadBuffer(*m_opencl_projected_internal_ids, CL_TRUE,
//                                                0, sizeof(cl_uint),
//                                                &internal_total);
//      m_opencl_command_queue->enqueueReadBuffer(*m_opencl_projected_external_ids, CL_TRUE,
//                                                0, sizeof(cl_uint),
//                                                &external_total);
      m_opencl_command_queue->finish();
      m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_PROJECT, counter);
      // project surfels in the NUVG
      m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_INTERNAL, counter);
      {
        m_opencl_project_internal_surfels_kernel->setArg(2,cl_ulong(offset));
        ret2 = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_project_internal_surfels_kernel, cl::NullRange,
                                                            cl::NDRange(m_surfels_projection_threads), cl::NullRange);
        if (ret2 != CL_SUCCESS)
          LOG_ERROR("  OpenCLProjectSurfels: internal: error enqueue: %d!", ret2);
        m_opencl_command_queue->finish();
      }
      m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_INTERNAL, counter);
      // project surfels outside the NUVG
      m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_HULL, counter);
      {
        ret2 = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_project_external_surfels_kernel, cl::NullRange,
                                                            cl::NDRange(m_surfels_projection_threads), cl::NullRange);
        if (ret2 != CL_SUCCESS)
          LOG_ERROR("  OpenCLProjectSurfels: external: error enqueue: %d!", ret2);
        m_opencl_command_queue->finish();
      }

      m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_HULL, counter);
    }

    m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_DELETE, counter);
    CLUInt32Vector to_be_deleted_buf(uploaded_count + 1);
    m_opencl_command_queue->enqueueReadBuffer(*m_opencl_ids_to_be_deleted, CL_TRUE,
                                              0, (uploaded_count + 1) * sizeof(cl_uint),
                                              to_be_deleted_buf.data());
    for (uint32 i = 0; i < to_be_deleted_buf[0]; i++)
    {
      DeleteSurfel(offset + to_be_deleted_buf[i + 1]);
      deleted_counter++;
    }
    m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_DELETE, counter);

    m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD_RECOLOR, counter);
    OpenCLRecolorVector to_be_recolored_buf(uploaded_count + 1);
    m_opencl_command_queue->enqueueReadBuffer(*m_opencl_ids_to_be_recolored, CL_TRUE,
                                              0, (uploaded_count + 1) * sizeof(OpenCLRecolor),
                                              to_be_recolored_buf.data());
    for (uint32 i = 0; i < to_be_recolored_buf[0].surfel_id; i++)
    {
      const OpenCLRecolor & recolor = to_be_recolored_buf[i + 1];
      Surfel & surfel = m_surfels[offset + recolor.surfel_id];
      const uint64 i2 = recolor.coords.x + recolor.coords.y * m_intrinsics->width;
      surfel.is_surfel = true;
      const uint8 r = colors[i2 * 3 + 0] * 255;
      const uint8 g = colors[i2 * 3 + 1] * 255;
      const uint8 b = colors[i2 * 3 + 2] * 255;
      surfel.cr = r;
      surfel.cg = g;
      surfel.cb = b;
      recolored_counter++;
    }
    m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD_RECOLOR, counter);
  }

  LOG_INFO_STREAM("  deleted: " << deleted_counter);
  LOG_INFO_STREAM("  recolored: " << recolored_counter);
  m_opencl_command_queue->finish();
}

void SurfelsUnknownSpace::OpenCLBuildKnownSpaceField(const uint64 count_at_max_range)
{
  if (!m_opencl_known_space_field)
    m_opencl_known_space_field = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width * m_intrinsics->height *
                                                     count_at_max_range * sizeof(cl_uchar),
                                                     "m_opencl_known_space_field");
  if (!m_opencl_zero_known_space_field_kernel)
  {
    m_opencl_zero_known_space_field_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "zero_known_space_field"));
    m_opencl_zero_known_space_field_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_zero_known_space_field_kernel->setArg(1,*m_opencl_known_space_field);
  }
  if (!m_opencl_path_knownxf_kernel)
  {
    m_opencl_path_knownxf_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "path_knownxf"));
    m_opencl_path_knownxf_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_path_knownxf_kernel->setArg(1,*m_opencl_dot_field);
    m_opencl_path_knownxf_kernel->setArg(2,*m_opencl_invalid_dot_field);
    m_opencl_path_knownxf_kernel->setArg(3,*m_opencl_known_state_hull_xn_filtered);
    m_opencl_path_knownxf_kernel->setArg(4,*m_opencl_known_space_field);
  }
  if (!m_opencl_path_knownxb_kernel)
  {
    m_opencl_path_knownxb_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "path_knownxb"));
    m_opencl_path_knownxb_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_path_knownxb_kernel->setArg(1,*m_opencl_dot_field);
    m_opencl_path_knownxb_kernel->setArg(2,*m_opencl_invalid_dot_field);
    m_opencl_path_knownxb_kernel->setArg(3,*m_opencl_known_state_hull_xp_filtered);
    m_opencl_path_knownxb_kernel->setArg(4,*m_opencl_known_space_field);
  }
  if (!m_opencl_path_knownyf_kernel)
  {
    m_opencl_path_knownyf_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "path_knownyf"));
    m_opencl_path_knownyf_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_path_knownyf_kernel->setArg(1,*m_opencl_dot_field);
    m_opencl_path_knownyf_kernel->setArg(2,*m_opencl_invalid_dot_field);
    m_opencl_path_knownyf_kernel->setArg(3,*m_opencl_known_state_hull_yn_filtered);
    m_opencl_path_knownyf_kernel->setArg(4,*m_opencl_known_space_field);
  }
  if (!m_opencl_path_knownyb_kernel)
  {
    m_opencl_path_knownyb_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "path_knownyb"));
    m_opencl_path_knownyb_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_path_knownyb_kernel->setArg(1,*m_opencl_dot_field);
    m_opencl_path_knownyb_kernel->setArg(2,*m_opencl_invalid_dot_field);
    m_opencl_path_knownyb_kernel->setArg(3,*m_opencl_known_state_hull_yp_filtered);
    m_opencl_path_knownyb_kernel->setArg(4,*m_opencl_known_space_field);
  }
  if (!m_opencl_path_knownzf_kernel)
  {
    m_opencl_path_knownzf_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "path_knownzf"));
    m_opencl_path_knownzf_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_path_knownzf_kernel->setArg(1,*m_opencl_dot_field);
    m_opencl_path_knownzf_kernel->setArg(2,*m_opencl_invalid_dot_field);
    m_opencl_path_knownzf_kernel->setArg(3,*m_opencl_known_state_hull_zn_filtered);
    m_opencl_path_knownzf_kernel->setArg(4,*m_opencl_known_space_field);
  }
  if (!m_opencl_path_knownzb_kernel)
  {
    m_opencl_path_knownzb_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "path_knownzb"));
    m_opencl_path_knownzb_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_path_knownzb_kernel->setArg(1,*m_opencl_dot_field);
    m_opencl_path_knownzb_kernel->setArg(2,*m_opencl_invalid_dot_field);
    m_opencl_path_knownzb_kernel->setArg(3,*m_opencl_known_state_hull_zp_filtered);
    m_opencl_path_knownzb_kernel->setArg(4,*m_opencl_known_space_field);
  }

  uint32 width = m_intrinsics->width;
  uint32 height = m_intrinsics->height;
  uint32 depth = count_at_max_range;
  int ret;

  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_zero_known_space_field_kernel, cl::NullRange,
                                                 cl::NDRange(count_at_max_range, width), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField: zero_known_space_frustum: error enqueue: %d!", ret);

  // follow all the paths
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_path_knownxf_kernel, cl::NullRange,
                                               cl::NDRange(height,depth), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField 1: error enqueue: %d!", ret);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_path_knownxb_kernel, cl::NullRange,
                                               cl::NDRange(height,depth), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField 2: error enqueue: %d!", ret);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_path_knownyf_kernel, cl::NullRange,
                                               cl::NDRange(depth,width), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField 3: error enqueue: %d!", ret);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_path_knownyb_kernel, cl::NullRange,
                                               cl::NDRange(depth,width), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField 4: error enqueue: %d!", ret);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_path_knownzf_kernel, cl::NullRange,
                                               cl::NDRange(width,height), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField 5: error enqueue: %d!", ret);
  ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_path_knownzb_kernel, cl::NullRange,
                                               cl::NDRange(width,height), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLBuildKnownSpaceField 6: error enqueue: %d!", ret);

  m_opencl_command_queue->finish();
}

void SurfelsUnknownSpace::OpenCLFilterKnownSpaceField(const uint64 count_at_max_range)
{
  if (!m_opencl_known_space_field_filtered)
    m_opencl_known_space_field_filtered = OpenCLCreateBuffer(m_opencl_context,
                                                             m_intrinsics->width * m_intrinsics->height *
                                                             count_at_max_range * sizeof(cl_uchar),
                                                             "m_opencl_known_space_field_filtered");

  if (!m_opencl_filter_known_space_field_counter)
    m_opencl_filter_known_space_field_counter = OpenCLCreateBuffer(m_opencl_context,
                                                                   sizeof(cl_uint),
                                                                   "m_opencl_filter_known_space_field_counter");

  if (!m_opencl_filter_known_space_field_median_kernel)
  {
    m_opencl_filter_known_space_field_median_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program,
                                                                            "filter_known_space_field_median"));
    m_opencl_filter_known_space_field_median_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_filter_known_space_field_median_kernel->setArg(1,*m_opencl_filter_known_space_field_counter);
    m_opencl_filter_known_space_field_median_kernel->setArg(2,*m_opencl_known_space_field);
    m_opencl_filter_known_space_field_median_kernel->setArg(3,*m_opencl_known_space_field_filtered);
  }

  if (!m_opencl_filter_known_space_field_passthrough_kernel)
  {
    m_opencl_filter_known_space_field_passthrough_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program,
                                                                            "filter_known_space_field_passthrough"));
    m_opencl_filter_known_space_field_passthrough_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_filter_known_space_field_passthrough_kernel->setArg(1,*m_opencl_filter_known_space_field_counter);
    m_opencl_filter_known_space_field_passthrough_kernel->setArg(2,*m_opencl_known_space_field);
    m_opencl_filter_known_space_field_passthrough_kernel->setArg(3,*m_opencl_known_space_field_filtered);
  }

  cl_uint counter = 0;
  m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_filter_known_space_field_counter, CL_TRUE, 0,
                                             sizeof(cl_uint),
                                             &counter);
  int ret;
  if (m_enable_known_space_filter)
  {
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_space_field_median_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                       cl::NullRange);
  }
  else
  {
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_filter_known_space_field_passthrough_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width),
                                                       cl::NullRange);
  }

  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLFilterKnownSpaceField: error enqueue: %d!", ret);

  m_opencl_command_queue->enqueueReadBuffer(*m_opencl_filter_known_space_field_counter, CL_TRUE,
                                            0, sizeof(cl_uint),
                                            &counter);
  LOG_INFO_STREAM("Filtered " << counter << " voxels");
}

void SurfelsUnknownSpace::OpenCLGenerateObservedSpaceField(
                                                         const uint64 count_at_max_range,
                                                         const FloatVector & depths)
{
  if (!m_opencl_depth_image)
  {
    m_opencl_depth_image = OpenCLCreateBuffer(m_opencl_context,
                                              m_intrinsics->width * m_intrinsics->height * sizeof(float),
                                              "m_opencl_depth_image"
                                              );
  }

  if (!m_opencl_observed_space_field)
  {
    m_opencl_observed_space_field = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width * m_intrinsics->height *
                                                      count_at_max_range * sizeof(cl_uchar),
                                                      "m_opencl_observed_space_field");
  }

  if (!m_opencl_generate_observed_space_field_kernel)
  {
    m_opencl_generate_observed_space_field_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "generate_observed_space_field"));
    m_opencl_generate_observed_space_field_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_generate_observed_space_field_kernel->setArg(1,*m_opencl_depth_image);
    m_opencl_generate_observed_space_field_kernel->setArg(2,*m_opencl_observed_space_field);
  }

  if (!m_opencl_zero_observed_space_field_kernel)
  {
    m_opencl_zero_observed_space_field_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "zero_observed_space_field"));
    m_opencl_zero_observed_space_field_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_zero_observed_space_field_kernel->setArg(1,*m_opencl_observed_space_field);
  }

  m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_depth_image, CL_TRUE, 0,
                                             m_intrinsics->width * m_intrinsics->height * sizeof(float),
                                             depths.data());
  m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_zero_observed_space_field_kernel, cl::NullRange,
                                               cl::NDRange(count_at_max_range, m_intrinsics->width), cl::NullRange);
  m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_generate_observed_space_field_kernel, cl::NullRange,
                                               cl::NDRange(m_intrinsics->width, m_intrinsics->height), cl::NullRange);

  m_opencl_command_queue->finish();
}

void SurfelsUnknownSpace::OpenCLCreateSurfels(const FloatVector & colors, const uint64 count_at_max_range)
{
  if (!m_opencl_creation_counters)
    m_opencl_creation_counters = OpenCLCreateBuffer(m_opencl_context, sizeof(OpenCLCreationCounters),
                                                    "m_opencl_creation_counters");
  if (!m_opencl_creation_unfinished_grid)
    m_opencl_creation_unfinished_grid = OpenCLCreateBuffer(m_opencl_context, m_intrinsics->width *
                                                           count_at_max_range * sizeof(cl_uint),
                                                           "m_opencl_creation_unfinished_grid");
  if (!m_opencl_creation_surfel_color_source)
    m_opencl_creation_surfel_color_source = OpenCLCreateBuffer(m_opencl_context, m_opencl_max_surfels_in_mem
                                                               * sizeof(cl_ushort2),
                                                               "m_opencl_creation_surfel_color_source");
  if (!m_opencl_creation_surfel_replace_global_id)
    m_opencl_creation_surfel_replace_global_id = OpenCLCreateBuffer(m_opencl_context, m_opencl_max_surfels_in_mem
                                                               * sizeof(uint),
                                                               "m_opencl_creation_surfel_replace_global_id");

  if (!m_opencl_create_new_surfels_kernel)
  {
    m_opencl_create_new_surfels_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "create_new_surfels"));
    uint64 pt = 0;
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_intrinsics);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_bearings);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_pov_matrix);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_known_space_field_filtered);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_observed_space_field);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_depth_image);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_occupancy_ids);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_creation_counters);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_surfels);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_creation_surfel_color_source);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_creation_surfel_replace_global_id);
    m_opencl_create_new_surfels_kernel->setArg(pt++,*m_opencl_creation_unfinished_grid);
  }

  OpenCLCreationCounters counters;
  counters.max_created = m_opencl_max_surfels_in_mem;// max created at once
  counters.reinit = true;
  uint64 total_created = 0;
  uint64 total_failed = 0;

  do
  {
    counters.created = 0;
    counters.creation_failed = 0;
    counters.unfinished = false;

    m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_creation_counters, CL_TRUE, 0,
                                               sizeof(OpenCLCreationCounters),
                                               &counters);
    int ret;
    ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_create_new_surfels_kernel, cl::NullRange,
                                                       cl::NDRange(count_at_max_range, m_intrinsics->width), cl::NullRange);
    if (ret != CL_SUCCESS)
      LOG_ERROR("  OpenCLCreateSurfels: error enqueue: %d!", ret);
    m_opencl_command_queue->enqueueReadBuffer(*m_opencl_creation_counters, CL_TRUE, 0,
                                              sizeof(OpenCLCreationCounters),
                                              &counters);

    LOG_INFO_STREAM("  iteration: created " << counters.created << ", failed " <<
                    counters.creation_failed << " surfels.");

    if (!counters.created)
      break;

    OpenCLSurfelVector buf(counters.created);
    m_opencl_command_queue->enqueueReadBuffer(*m_opencl_surfels, CL_TRUE, 0,
                                              counters.created * sizeof(OpenCLSurfel),
                                              buf.data());
    CLUShort2Vector color_source_buf(counters.created);
    m_opencl_command_queue->enqueueReadBuffer(*m_opencl_creation_surfel_color_source, CL_TRUE, 0,
                                              counters.created * sizeof(cl_ushort2),
                                              color_source_buf.data());
    CLUInt32Vector replace_buf(counters.created);
    m_opencl_command_queue->enqueueReadBuffer(*m_opencl_creation_surfel_replace_global_id, CL_TRUE, 0,
                                              counters.created * sizeof(cl_uint),
                                              replace_buf.data());

    for (uint64 i = 0; i < counters.created; i++)
    {
      Surfel surfel = buf[i].toSurfel();
      if (surfel.is_surfel)
      {
        const cl_ushort2 color_source = color_source_buf[i];
        const uint64 i2 = color_source.x + uint64(color_source.y) * m_intrinsics->width;
        const uint8 r = colors[i2 * 3 + 0] * 255;
        const uint8 g = colors[i2 * 3 + 1] * 255;
        const uint8 b = colors[i2 * 3 + 2] * 255;
        surfel.cr = r;
        surfel.cg = g;
        surfel.cb = b;
      }

      if (replace_buf[i] > 0)
      {
        const uint64 id = replace_buf[i] - 1;
        Surfel & old_surfel = m_surfels[id];
        const bool replace = !old_surfel.erased && (!old_surfel.is_surfel || surfel.is_surfel);
                             // never replace a surfel with a frontel
        if (replace)
        {
          const Eigen::Vector3f new_position = (old_surfel.position + surfel.position) / 2.0f;
          const Eigen::Vector3f new_normal = old_surfel.normal + surfel.normal;

          if (old_surfel.is_surfel && surfel.is_surfel)
          {
            const Eigen::Vector3i new_color = (Eigen::Vector3i(old_surfel.cr, old_surfel.cg, old_surfel.cb) +
                                              Eigen::Vector3i(surfel.cr, surfel.cg, surfel.cb)) / 2;
            surfel.cr = new_color.x();
            surfel.cg = new_color.y();
            surfel.cb = new_color.z();
          }

          old_surfel = surfel;
          if (new_normal.squaredNorm() > SQR(0.01f))
            old_surfel.normal = new_normal.normalized();
          old_surfel.position = new_position;
        }
      }
      else
        CreateSurfel(surfel);
    }

    total_created += counters.created;
    total_failed += counters.creation_failed;
    counters.reinit = false; // do not reinit in next iteration
  }
  while (counters.unfinished);

  LOG_INFO_STREAM("  Total created: " << total_created << " failed: " << total_failed);
}

void SurfelsUnknownSpace::OpenCLGetAllBearings(Vector3fVector & bearings)
{
  const uint64 image_size = m_intrinsics->width * m_intrinsics->height;
  const uint64 buffer_size = image_size * sizeof(cl_float3);

  if (!m_opencl_bearings)
    m_opencl_bearings = CLBufferPtr(new cl::Buffer(*m_opencl_context,CL_MEM_READ_WRITE,
                                    buffer_size));

  if (!m_opencl_get_all_bearings_kernel)
  {
    m_opencl_get_all_bearings_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "get_all_bearings"));
    m_opencl_get_all_bearings_kernel->setArg(0,*m_opencl_intrinsics);
    m_opencl_get_all_bearings_kernel->setArg(1,*m_opencl_pov_matrix);
    m_opencl_get_all_bearings_kernel->setArg(2,*m_opencl_bearings);
  }

  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_opencl_get_all_bearings_kernel, cl::NullRange,
                                                         cl::NDRange(m_intrinsics->width, m_intrinsics->height), cl::NullRange);
  if (ret != CL_SUCCESS)
    LOG_ERROR("  OpenCLGetAllBearings: error enqueue: %d!", ret);

  CLFloat3Vector bearings_buf(image_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_opencl_bearings, CL_TRUE,
                                            0, buffer_size,
                                            bearings_buf.data());
  bearings.resize(image_size);
  bool at_least_one_wrong_bearing = false;
  for (uint64 i = 0; i < image_size; i++)
  {
    Eigen::Vector3f b(bearings_buf[i].x, bearings_buf[i].y, bearings_buf[i].z);
    Eigen::Vector3f bb = bearings[i];
    if ((bb - b).norm() > 0.01)
    {
      //LOG_ERROR_STREAM("Error: bearing should be " << bb.transpose() << " it is " << b.transpose());
      at_least_one_wrong_bearing = true;
    }
    else
    {
      bearings[i].x() = bearings_buf[i].x;
      bearings[i].y() = bearings_buf[i].y;
      bearings[i].z() = bearings_buf[i].z;
    }
  }
  if (at_least_one_wrong_bearing)
    LOG_ERROR("Error: at_least_one_wrong_bearing");
}

void SurfelsUnknownSpace::OpenCLUpdateCurrentPoseAndIntrinsics(const Eigen::Affine3f & pose,
                                                              const float min_range, const float max_range,
                                                              const float dot_field_valid_th, const float dot_field_safety_th,
                                                              const uint64 back_ignore_margin)
{
  if (!m_opencl_pov_matrix)
    m_opencl_pov_matrix = CLBufferPtr(new cl::Buffer(*m_opencl_context,CL_MEM_READ_WRITE,
                                      12 * sizeof(float)));
  if (!m_opencl_inverse_pov_matrix)
    m_opencl_inverse_pov_matrix = CLBufferPtr(new cl::Buffer(*m_opencl_context,CL_MEM_READ_WRITE,
                                                             12 * sizeof(float)));
  if (!m_opencl_intrinsics)
    m_opencl_intrinsics = CLBufferPtr(new cl::Buffer(*m_opencl_context,CL_MEM_READ_WRITE,
                                      sizeof(OpenCLIntrinsics)));

  const Eigen::Affine3f pov_matrix = pose;
  CLFloatVector pov_matrix_buf(12);
  for (uint64 y = 0; y < 3; y++)
    for (uint64 x = 0; x < 4; x++)
      pov_matrix_buf[x + y * 4] = pov_matrix.matrix()(y,x);
  m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_pov_matrix, CL_TRUE, 0,
                                             12 * sizeof(float),
                                             pov_matrix_buf.data());

  const Eigen::Affine3f inverse_pov_matrix = pose.inverse();
  CLFloatVector inverse_pov_matrix_buf(12);
  for (uint64 y = 0; y < 3; y++)
    for (uint64 x = 0; x < 4; x++)
      inverse_pov_matrix_buf[x + y * 4] = inverse_pov_matrix.matrix()(y,x);
  m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_inverse_pov_matrix, CL_TRUE, 0,
                                             12 * sizeof(float),
                                             inverse_pov_matrix_buf.data());

  OpenCLIntrinsics opencl_intrinsics(*m_intrinsics, min_range, max_range,
                                     getVoxelCountAtDistance(0.0f) * getVoxelSideAtDistance(0.0f),
                                     getVoxelCountAtDistance(max_range),
                                     getVoxelCountAtDistance(min_range),
                                     getVoxelCountAtDistance(0.0f),
                                     dot_field_valid_th, dot_field_safety_th,
                                     back_ignore_margin, m_surfel_thickness, m_surfel_radius_mult,
                                     m_unknown_surfels_radius_mult_pn, m_side_padding);
  m_opencl_command_queue->enqueueWriteBuffer(*m_opencl_intrinsics, CL_TRUE, 0,
                                             sizeof(OpenCLIntrinsics),
                                             &opencl_intrinsics);
}
