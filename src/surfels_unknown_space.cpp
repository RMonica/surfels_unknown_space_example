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

#define LOG_INFO(...) m_logger->LogInfo(__VA_ARGS__)
#define LOG_INFO_STREAM(s) m_logger->LogInfo(OSS() << s)

SurfelsUnknownSpace::SurfelsUnknownSpace(const Config & config,
                                         const ILoggerPtr logger,
                                         const IVisualListenerPtr visual_listener,
                                         const ITimerListenerPtr timer_listener)
{
  if (logger)
    m_logger = logger;
  else
    m_logger.reset(new ILogger);

  if (visual_listener)
    m_visual_listener = visual_listener;
  else
    m_visual_listener.reset(new IVisualListener);

  if (timer_listener)
    m_timer_listener = timer_listener;
  else
    m_timer_listener.reset(new ITimerListener);

  m_max_range = config.max_range;
  m_min_range = config.min_range;
  m_unknown_surfels_radius_mult_pn = config.unknown_surfels_radius_mult_pn;
  m_surfel_thickness = config.surfel_thickness;
  m_surfel_radius_mult = config.surfel_radius_mult;
  m_enable_known_space_filter = config.enable_known_space_filter;
  m_dot_field_safety_th = config.dot_field_valid_th;
  m_dot_field_valid_th = config.dot_field_valid_th;
  m_back_padding = config.back_padding;
  m_side_padding = config.side_padding;
  m_opencl_max_surfels_in_mem = config.opencl_max_surfels_in_mem;
  m_surfels_projection_threads = config.surfels_projection_threads;
  m_downsample_factor = config.downsample_factor;

  // start threads
  initOpenCL(config);
}

SurfelsUnknownSpace::~SurfelsUnknownSpace()
{
}

float SurfelsUnknownSpace::getVoxelSideAtDistance(const float distance, float min_range) const
{
  if (distance < min_range)
    return min_range * 1.0f / m_intrinsics->focal_avg;

  return distance * 1.0f / m_intrinsics->focal_avg;
}

float SurfelsUnknownSpace::getVoxelSideAtDistance(const float distance) const
{
  return getVoxelSideAtDistance(distance, m_min_range);
}

float SurfelsUnknownSpace::getVoxelCountAtDistanceF(const float distance) const
{
  if (distance < m_min_range)
    return (distance / getVoxelSideAtDistance(0)) + m_back_padding;

  const float result = (m_min_range / getVoxelSideAtDistance(0) +
      (std::log(distance) - std::log(m_min_range)) * m_intrinsics->focal_avg) + m_back_padding;

  return result;
}

SurfelsUnknownSpace::int64 SurfelsUnknownSpace::getVoxelCountAtDistance(const float distance) const
{
  const float rounded = std::round(getVoxelCountAtDistanceF(distance));
  if (rounded < 0)
    return -1;

  return rounded;
}

float SurfelsUnknownSpace::getDistanceFromVoxelCount(const uint64 count)
{
  const float zero_side = getVoxelSideAtDistance(0);
  const float diff = (float(count) - m_back_padding) * zero_side;
  if (diff < m_min_range)
    return diff;

  const float K = (1.0 / m_intrinsics->focal_avg);

  const float result = std::exp(K * (count - m_back_padding) + std::log(m_min_range) - K * (m_min_range / zero_side));
  return result;
}

Eigen::Vector3f SurfelsUnknownSpace::getBearingForPixel(const uint64 x,const uint64 y)
{
  const Eigen::Vector3f result(float(x) - m_intrinsics->center_x,float(y) - m_intrinsics->center_y,m_intrinsics->focal_avg);
  return result.normalized();
}

SurfelsUnknownSpace::Vector3fVector SurfelsUnknownSpace::getAllBearings(const uint64 width,const uint64 height)
{
  Vector3fVector result(width * height);
  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      result[x + y * width] = getBearingForPixel(x,y);
  return result;
}

void SurfelsUnknownSpace::SplitNearSurfels(const Eigen::Affine3f & pose)
{
  SurfelVector new_surfels;
  const uint64 surfels_size = m_surfels.size();
  uint64 counter = 0;

  for (uint64 i = 0; i < surfels_size; i++)
  {
    const Surfel & surfel = m_surfels[i];
    if (surfel.erased)
      continue;

    const float distance = (pose.translation() - surfel.position).norm();
    const float radius = surfel.radius;
    const Eigen::Vector3f normal = surfel.normal;
    const float new_radius = std::sqrt(3.0f) * getVoxelSideAtDistance(distance) / 2.0;
    if (radius > new_radius * 2.0)
    {
      // if surfel near enough
      // split the surfel into four new surfels

      uint64 lower_i;
      // build reference frame on surfel with normal = axis z
      for (uint64 ai = 0; ai < 3; ai++) // find lower normal coord
        if (ai == 0 || (std::abs(normal[ai]) < std::abs(normal[lower_i])))
          lower_i = ai;
      const Eigen::Vector3f best_axis = Eigen::Vector3f::Unit(lower_i);
      const Eigen::Vector3f axis_z = normal;
      const Eigen::Vector3f axis_x = best_axis - best_axis.dot(normal) * normal;
      const Eigen::Vector3f axis_y = axis_z.cross(axis_x);

      Surfel nf[4];
      for (int64 dy = 0; dy <= 1; dy++)
        for (int64 dx = 0; dx <= 1; dx++)
        {
          const float fdx = dx ? 1 : -1;
          const float fdy = dy ? 1 : -1;
          const Eigen::Vector3f transl = (axis_x * fdx + axis_y * fdy) * new_radius / sqrt(2.0f);
          const uint64 di = dy * 2 + dx;
          nf[di] = surfel;
          nf[di].position = surfel.position + transl;
          nf[di].radius = radius / 2.0f;
        }

      m_surfels[i] = nf[0];
      for (uint64 h = 1; h < 4; h++)
        new_surfels.push_back(nf[h]);

      counter++;
    }
  }
  for (const Surfel & nsurf : new_surfels)
    CreateSurfel(nsurf);

  LOG_INFO_STREAM("Split " << counter << " surfels.");
}

template <typename T>
std::vector<T> SurfelsUnknownSpace::DownSample(const uint64 width,
                                               const uint64 height,
                                               const uint64 towidth,
                                               const uint64 toheight,
                                               const uint64 step,
                                               const T mult,
                                               const std::vector<T> & vec
                                               )
{
  const uint64 tosize = towidth * toheight;
  std::vector<T> result(tosize * step);

  for (uint64 y = 0; y < toheight; y++)
    for (uint64 x = 0; x < towidth; x++)
    {
      const uint64 toi = x + y * towidth;
      const uint64 fromx = x * width / towidth;
      const uint64 fromy = y * height / toheight;
      const uint64 fromi = fromx + fromy * width;
      for (uint64 si = 0; si < step; si++)
        result[toi * step + si] = mult * vec[fromi * step + si];
    }
  return result;
}

SurfelsUnknownSpace::uint64 SurfelsUnknownSpace::CreateSurfel(const Surfel & surfel)
{
  uint64 pos;
  if (m_deleted_surfels.empty())
  {
    m_surfels.push_back(surfel);
    pos = m_surfels.size() - 1;
  }
  else
  {
    pos = *m_deleted_surfels.begin();
    m_surfels[pos] = surfel;
    m_deleted_surfels.erase(m_deleted_surfels.begin());
  }

  return pos;
}

void SurfelsUnknownSpace::DeleteSurfel(const uint64 index)
{
  if (m_surfels[index].erased)
    return;
  if (index + 1 == m_surfels.size())
  {
    m_surfels.pop_back();
    while (!m_surfels.empty() && m_surfels.back().erased)
    {
      m_deleted_surfels.erase(m_surfels.size() - 1);
      m_surfels.pop_back();
    }
  }
  else
  {
    m_surfels[index].erased = true;
    m_deleted_surfels.insert(index);
  }
}

SurfelsUnknownSpace::SurfelVector SurfelsUnknownSpace::GetSurfels()
{
  SurfelVector result;
  result.reserve(m_surfels.size());

  for (Surfel s : m_surfels)
    if (!s.erased)
      result.push_back(s);

  return result;
}

void SurfelsUnknownSpace::ComputeApproximateFrustumBoundingBox(const Eigen::Affine3f & pose,
                                                              Eigen::Vector3f & bounding_box_min,
                                                              Eigen::Vector3f & bounding_box_max)
{
  const float center_dist = m_intrinsics->max_range / 2.0f;
  const float lateral_dist_xp = (m_intrinsics->width - m_intrinsics->center_x) *
                                (m_intrinsics->max_range / m_intrinsics->focal_x);
  const float lateral_dist_yp = (m_intrinsics->height - m_intrinsics->center_y) *
                                (m_intrinsics->max_range / m_intrinsics->focal_y);
  const float lateral_dist_xn = (m_intrinsics->center_x) *
                                (m_intrinsics->max_range / m_intrinsics->focal_x);
  const float lateral_dist_yn = (m_intrinsics->center_y) *
                                (m_intrinsics->max_range / m_intrinsics->focal_y);
  const Eigen::Vector3f center(0.0f, 0.0f, center_dist);
  bounding_box_max = bounding_box_min = (pose * center);
  // bounding box
  for (int64 x = -1; x <= 1; x++)
    for (int64 y = -1; y <= 1; y++)
      for (int64 z = -1; z <= 1; z++)
      {
        if (!x || !y || !z)
          continue;

        Eigen::Vector3f vertex;
        vertex.x() = (x > 0) ? lateral_dist_xp : -lateral_dist_xn;
        vertex.y() = (y > 0) ? lateral_dist_yp : -lateral_dist_yn;
        vertex.z() = center_dist + center_dist * z;

        const Eigen::Vector3f world_vertex = pose * vertex;
        bounding_box_max = bounding_box_max.array().max(world_vertex.array());
        bounding_box_min = bounding_box_min.array().min(world_vertex.array());
      }
  bounding_box_max += Eigen::Vector3f::Ones() * 0.1f;
  bounding_box_min -= Eigen::Vector3f::Ones() * 0.1f;
}

void SurfelsUnknownSpace::ProcessFrame(const uint64 input_width, const uint64 input_height,
                                       const FloatVector & raw_depths, const FloatVector & raw_colors,
                                       const Eigen::Affine3f & pose, const Intrinsics & input_intrinsics)
{
  pcl::console::TicToc tictoc;
  pcl::console::TicToc total_tictoc;

  m_timer_listener->NewFrame();
  m_timer_listener->StartTimer(ITimerListener::TPhase::SUBSAMPLE);
  m_timer_listener->StartTimer(ITimerListener::TPhase::TOTAL);

  const uint64 width = input_width / m_downsample_factor;
  const uint64 height = input_height / m_downsample_factor;
  m_intrinsics = std::make_shared<Intrinsics>(input_intrinsics.RescaleInt(m_downsample_factor));
  m_intrinsics->min_range = m_min_range;
  m_intrinsics->max_range = m_max_range;

  FloatVector max_range_raw_depths(raw_depths);
  for (uint64 y = 0; y < input_height; y++)
    for (uint64 x = 0; x < input_width; x++)
    {
      if (max_range_raw_depths[x + y * input_width] > m_max_range - 0.1f)
        max_range_raw_depths[x + y * input_width] = 0.0f;
    }

  const FloatVector depths = DownSample<float>(input_width,input_height,width,height,1,1.0,max_range_raw_depths);
  const FloatVector colors = DownSample<float>(input_width,input_height,width,height,3,1.0,raw_colors);

  m_timer_listener->StopTimer(ITimerListener::TPhase::SUBSAMPLE);
  m_timer_listener->StartTimer(ITimerListener::TPhase::INIT);

  OpenCLUpdateCurrentPoseAndIntrinsics(pose, m_min_range, m_max_range,
                                       m_dot_field_valid_th, m_dot_field_safety_th, m_back_padding);

  total_tictoc.tic();

  const uint64 count_at_max_range = getVoxelCountAtDistance(m_max_range);
  const uint64 count_at_min_range = getVoxelCountAtDistance(m_min_range);
  const int64 count_at_zero = getVoxelCountAtDistance(0.0f);

  m_timer_listener->StopTimer(ITimerListener::TPhase::INIT);

  LOG_INFO_STREAM("Computing bearings.");
  tictoc.tic();
  Vector3fVector bearings = getAllBearings(width,height);
  tictoc.toc_print();

  m_timer_listener->StartTimer(ITimerListener::TPhase::BEARINGS);
  LOG_INFO_STREAM("Computing bearings OpenCL.");
  tictoc.tic();
  OpenCLGetAllBearings(bearings);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::BEARINGS);

  m_timer_listener->StartTimer(ITimerListener::TPhase::SPLITTING);
  LOG_INFO_STREAM("Splitting surfels.");
  tictoc.tic();
  SplitNearSurfels(pose);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::SPLITTING);

  m_timer_listener->StartTimer(ITimerListener::TPhase::OBSERVED_FIELD);
  LOG_INFO_STREAM("Generating observed space field OpenCL.");
  tictoc.tic();
  OpenCLGenerateObservedSpaceField(count_at_max_range, depths);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::OBSERVED_FIELD);

  m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_FIELD);
  LOG_INFO_STREAM("Projecting surfels OpenCL.");
  tictoc.tic();
  OpenCLProjectAndDeleteSurfels(pose,m_surfels,count_at_max_range,colors);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_FIELD);

  m_timer_listener->StartTimer(ITimerListener::TPhase::DOT_HULL);
  LOG_INFO_STREAM("Filtering known state hull OpenCL.");
  tictoc.tic();
  OpenCLFilterKnownStateHullPN(count_at_max_range);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::DOT_HULL);

  m_timer_listener->StartTimer(ITimerListener::TPhase::KNOWN_SPACE);
  LOG_INFO_STREAM("Building known space field OpenCL.");
  tictoc.tic();
  OpenCLBuildKnownSpaceField(count_at_max_range);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::KNOWN_SPACE);

  m_timer_listener->StartTimer(ITimerListener::TPhase::KNOWN_SPACE_FILTERING);
  LOG_INFO_STREAM("Filtering known space field OpenCL.");
  tictoc.tic();
  OpenCLFilterKnownSpaceField(count_at_max_range);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::KNOWN_SPACE_FILTERING);

  m_timer_listener->StartTimer(ITimerListener::TPhase::CREATION);
  LOG_INFO_STREAM("Creating new surfels OpenCL.");
  tictoc.tic();
  OpenCLCreateSurfels(colors, count_at_max_range);
  tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::CREATION);

  LOG_INFO_STREAM("Total processing: ");
  total_tictoc.toc_print();
  m_timer_listener->StopTimer(ITimerListener::TPhase::TOTAL);

  LOG_INFO_STREAM("Publishing: ");
  tictoc.tic();

  m_visual_listener->ShowStableImage(width, height, depths, colors, bearings, pose);
  m_visual_listener->ShowNUVG(pose, count_at_max_range, count_at_min_range, count_at_zero);
  m_visual_listener->ShowCamera(pose, m_side_padding, m_side_padding, m_min_range);
  ShowKnownStateHull(count_at_max_range,height,count_at_min_range,-1,false,
                     IVisualListener::TKnownStateHullIndex::XN,m_opencl_known_state_hull_xn);
  ShowKnownStateHull(count_at_max_range,height,count_at_min_range,-1,false,
                     IVisualListener::TKnownStateHullIndex::XP,m_opencl_known_state_hull_xp);
  ShowKnownStateHull(count_at_max_range,width,count_at_min_range,-1,true,
                     IVisualListener::TKnownStateHullIndex::YN,m_opencl_known_state_hull_yn);
  ShowKnownStateHull(count_at_max_range,width,count_at_min_range,-1,true,
                     IVisualListener::TKnownStateHullIndex::YP,m_opencl_known_state_hull_yp);
  ShowKnownStateHull(width,height,-1,-1,true,
                     IVisualListener::TKnownStateHullIndex::ZN,m_opencl_known_state_hull_zn);
  ShowKnownStateHull(width,height,-1,-1,true,
                     IVisualListener::TKnownStateHullIndex::ZP,m_opencl_known_state_hull_zp);
  ShowKnownStateHull(count_at_max_range,height,count_at_min_range,-1,false,
                     IVisualListener::TKnownStateHullIndex::XNF,m_opencl_known_state_hull_xn_filtered);
  ShowKnownStateHull(count_at_max_range,height,count_at_min_range,-1,false,
                     IVisualListener::TKnownStateHullIndex::XPF,m_opencl_known_state_hull_xp_filtered);
  ShowKnownStateHull(count_at_max_range,width,count_at_min_range,-1,true,
                     IVisualListener::TKnownStateHullIndex::YNF,m_opencl_known_state_hull_yn_filtered);
  ShowKnownStateHull(count_at_max_range,width,count_at_min_range,-1,true,
                     IVisualListener::TKnownStateHullIndex::YPF,m_opencl_known_state_hull_yp_filtered);
  ShowKnownStateHull(width,height,-1,-1,true,
                     IVisualListener::TKnownStateHullIndex::ZNF,m_opencl_known_state_hull_zn_filtered);
  ShowKnownStateHull(width,height,-1,-1,true,
                     IVisualListener::TKnownStateHullIndex::ZPF,m_opencl_known_state_hull_zp_filtered);
  m_visual_listener->ShowSurfelCloud(m_surfels);
  tictoc.toc_print();

  const uint64 mem_usage = m_surfels.size() * sizeof(Surfel);
  LOG_INFO_STREAM("Surfels memory usage: " << mem_usage << " (" << mem_usage / 1000000 << " MB)");
}

void SurfelsUnknownSpace::ShowKnownStateHull(const uint64 width,
                                            const uint64 height,
                                            const uint64 special_color_width,
                                            const uint64 special_color_height,
                                            const bool transpose,
                                            IVisualListener::TKnownStateHullIndex index,
                                            CLBufferPtr buf
                                            )
{
  if (!m_visual_listener->HasShowKnownStateHull(index))
    return; // listener not interested (save processing time)
  if (!buf)
    return;

  CLUInt32Vector vec(width * height);
  m_opencl_command_queue->enqueueReadBuffer(*buf, CL_TRUE,
                                            0, width * height * sizeof(cl_uint),
                                            vec.data());

  Uint32Vector data;
  data.reserve(width * height);
  for (cl_uint v : vec)
    data.push_back(v);

  if (transpose)
  {
    Uint32Vector data_transpose(width * height);
    for (uint64 x = 0; x < width; x++)
      for (uint64 y = 0; y < height; y++)
        data_transpose[x * height + y] = vec[y * width + x];
    data_transpose.swap(data);
  }

  m_visual_listener->ShowKnownStateHull(width, height, special_color_width, special_color_height, index, data);
}

