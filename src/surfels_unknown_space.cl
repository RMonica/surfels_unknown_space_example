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

#ifndef __OPENCL_VERSION__
  #define global
  #define __global
  #define kernel
#endif

#define SQR(x) ((x)*(x))

#ifndef NULL
  #define NULL (0)
#endif

#define DOT_FIELD_MULTIPLIER (1000.0f * 100.0f)

void Float3ToArray(const float3 inv, float outv[3])
{
  outv[0] = inv.x;
  outv[1] = inv.y;
  outv[2] = inv.z;
}

float3 ArrayToFloat3(const float inv[3])
{
  float3 result;
  result.x = inv[0];
  result.y = inv[1];
  result.z = inv[2];
  return result;
}

float GetFloat3ArrayElement(const float3 inv, int i)
{
  switch (i)
  {
    case 0: return inv.x;
    case 1: return inv.y;
    case 2: return inv.z;
    default:
      return inv.x; // error
  }
}

void SetFloat3ArrayElement(float3 * inv, int i, float v)
{
  switch (i)
  {
    case 0: (*inv).x = v; return;
    case 1: (*inv).y = v; return;
    case 2: (*inv).z = v; return;
    default: (*inv).z = v; return;
  }
}

int3 Float3ToInt3(const float3 inv)
{
  int3 result;
  result.x = inv.x;
  result.y = inv.y;
  result.z = inv.z;
  return result;
}

int2 Float2ToInt2(const float2 inv)
{
  int2 result;
  result.x = inv.x;
  result.y = inv.y;
  return result;
}

bool Int3EqualsInt3(const int3 a, const int3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

float3 Int3ToFloat3(int3 inv)
{
  float3 result;
  result.x = inv.x;
  result.y = inv.y;
  result.z = inv.z;
  return result;
}

void Int3ToArray(int3 inv, int outv[3])
{
  outv[0] = inv.x;
  outv[1] = inv.y;
  outv[2] = inv.z;
}

int GetInt3ArrayElement(int3 inv, int i)
{
  switch (i)
  {
    case 0: return inv.x;
    case 1: return inv.y;
    case 2: return inv.z;
    default:
      return inv.x; // error
  }
}

int GetInt3Dot(int3 a, int3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

void SetInt3ArrayElement(int3 * inv, int i, int v)
{
  switch (i)
  {
    case 0: (*inv).x = v; return;
    case 1: (*inv).y = v; return;
    case 2: (*inv).z = v; return;
    default: (*inv).z = v; return;
  }
}

int3 ArrayToInt3(int inv[3])
{
  int3 result;
  result.x = inv[0];
  result.y = inv[1];
  result.z = inv[2];
  return result;
}

void MulMatrix3x3_3x3(const float a[9], const float b[9], float result[9])
{
  float result_copy[9];
  for (int aa = 0; aa < 3; aa++)
    for (int bb = 0; bb < 3; bb++)
    {
      result_copy[aa * 3 + bb] = 0;
      for (int cc = 0; cc < 3; cc++)
        result_copy[aa * 3 + bb] += a[aa * 3 + cc] * b[cc * 3 + bb];
    }
  for (int aa = 0; aa < 9; aa++)
    result[aa] = result_copy[aa];
}

float3 MulMatrix3x3_Float3(const float a[9], const float3 b)
{
  float3 result = (float3)(0,0,0);
  for (int aa = 0; aa < 3; aa++)
    for (int bb = 0; bb < 3; bb++)
      SetFloat3ArrayElement(&result,aa,GetFloat3ArrayElement(result,aa) +
                            GetFloat3ArrayElement(b,bb) * a[aa * 3 + bb]);
  return result;
}

float3 TransformPointPOVMatrix(global const float * pov_matrix, const float3 lposition)
{
  float position[3];
  for (int aa = 0; aa < 3; aa++)
  {
    position[aa] = 0;

    #pragma unroll
    for (int bb = 0; bb < 3; bb++)
      position[aa] += pov_matrix[aa * 4 + bb] * GetFloat3ArrayElement(lposition, bb);
    position[aa] += pov_matrix[aa * 4 + 3];
  }

  return ArrayToFloat3(position);
}

float3 TransformNormalPOVMatrix(global const float * pov_matrix, const float3 lnormal)
{
  float normal[3];
  for (int aa = 0; aa < 3; aa++)
  {
    normal[aa] = 0;

    #pragma unroll
    for (int bb = 0; bb < 3; bb++)
      normal[aa] += pov_matrix[aa * 4 + bb] * GetFloat3ArrayElement(lnormal, bb);
  }

  return ArrayToFloat3(normal);
}

void TransposeMatrix3x3(const float a[9], float result[9])
{
  for (int aa = 0; aa < 3; aa++)
    for (int bb = 0; bb < 3; bb++)
      result[aa * 3 + bb] = a[bb * 3 + aa];
}

struct OpenCLSurfel
{
  float position[3];
  float radius;
  float normal[3];
  uchar erased;
  uchar is_surfel;
} __attribute__((packed));

struct OpenCLIntrinsics
{
  float center_x, center_y;
  float focal_x, focal_y, focal_avg;
  int width,height;

  float min_range, max_range, back_range;
  uint count_at_max_range, count_at_min_range, count_at_zero;
  uint back_ignore_margin;

  // parameters
  float dot_field_valid_th, dot_field_safety_th;
  float surfel_thickness;
  float surfel_radius_mult;
  float unknown_surfels_radius_mult_pn;
  uint side_padding;
} __attribute__((packed));

struct OpenCLRecolor
{
  uint surfel_id;
  ushort2 coords;
} __attribute__((packed));

struct OpenCLCreationCounters
{
  uint created;
  uint creation_failed;
  uint max_created;
  uchar reinit;
  uchar unfinished;
} __attribute__((packed));

float getVoxelSideAtDistance(global const struct OpenCLIntrinsics * intrinsics, const float distance)
{
  if (distance < intrinsics->min_range)
    return intrinsics->min_range * 1.0f / intrinsics->focal_avg;

  return distance * 1.0f / intrinsics->focal_avg;
}

float getVoxelCountAtDistanceF(global const struct OpenCLIntrinsics * intrinsics, const float distance)
{
  if (distance < intrinsics->min_range)
    return (distance / getVoxelSideAtDistance(intrinsics, 0)) + intrinsics->back_ignore_margin;

  const float result = (intrinsics->min_range / getVoxelSideAtDistance(intrinsics, 0) +
      (log(distance) - log(intrinsics->min_range)) * intrinsics->focal_avg) + intrinsics->back_ignore_margin;

  return result;
}

int getVoxelCountAtDistance(global const struct OpenCLIntrinsics * intrinsics, const float distance)
{
  const float rounded = round(getVoxelCountAtDistanceF(intrinsics, distance));
  if (rounded < 0)
    return -1;

  return rounded;
}

float3 getVoxelCountAtPositionF(global const struct OpenCLIntrinsics * intrinsics, const float3 position)
{
  float3 result;
  if (position.z < intrinsics->min_range)
  {
    const float side = getVoxelSideAtDistance(intrinsics, intrinsics->min_range);
    result.xy = (position.xy / side) +
      (float2)(intrinsics->center_x,intrinsics->center_y);
  }
  else
  {
    result.xy = ((position.xy * (float2)(intrinsics->focal_x, intrinsics->focal_y)) / position.z) +
      (float2)(intrinsics->center_x,intrinsics->center_y);
  }
  result.z = getVoxelCountAtDistanceF(intrinsics, position.z);
  return result;
}

float getDistanceFromVoxelCount(global const struct OpenCLIntrinsics * intrinsics, const float count)
{
  const float zero_side = getVoxelSideAtDistance(intrinsics, 0);
  const float diff = (count - intrinsics->count_at_zero) * zero_side;
  if (diff < intrinsics->min_range)
    return diff;

  const float K = (1.0 / intrinsics->focal_avg);

  const float result = exp(K * (count - (float)(intrinsics->count_at_zero)) + log(intrinsics->min_range) -
                           K * (intrinsics->min_range / zero_side));
  return result;
}

float3 getPositionAtVoxelCount(global const struct OpenCLIntrinsics * intrinsics,
                               global const float3 * bearings,
                               const float3 voxel_count)
{
  const float zf = getDistanceFromVoxelCount(intrinsics, voxel_count.z);
  const float diameter_at_min_range = getVoxelSideAtDistance(intrinsics, intrinsics->min_range);

  if (zf < intrinsics->min_range)
  {
    const float3 position_near = (float3)((voxel_count.x - intrinsics->center_x) * diameter_at_min_range,
                                          (voxel_count.y - intrinsics->center_y) * diameter_at_min_range,
                                          zf); // if nearer than min range
    return position_near;
  }
  const float3 bearing = bearings[(int)(round(voxel_count.x)) + (int)(round(voxel_count.y)) * intrinsics->width];
  return zf * bearing / bearing.z;
}

// ************** IS SIGNIFICANT BEGIN **************

bool IsSignificantVoxelN(global const struct OpenCLIntrinsics * intrinsics,
                         const uint x,const uint y,const uint n)
{
  if (y + n < intrinsics->side_padding || x + n < intrinsics->side_padding ||
      y >= (intrinsics->height - intrinsics->side_padding + n) ||
      x >= (intrinsics->width - intrinsics->side_padding + n))
    return false;
  return true;
}

bool IsSignificantVoxel(global const struct OpenCLIntrinsics * intrinsics,const uint x,const uint y)
  { return IsSignificantVoxelN(intrinsics,x,y,0); }
bool IsSignificantVoxel1(global const struct OpenCLIntrinsics * intrinsics,const uint x,const uint y)
  { return IsSignificantVoxelN(intrinsics,x,y,1); }

bool IsSignificantVoxelZN(global const struct OpenCLIntrinsics * intrinsics,
                          const int x,const int y,const int z, const uint n)
{
  if (!IsSignificantVoxelN(intrinsics, x, y, n))
    return false;
  if (z + (int)n < intrinsics->count_at_zero)
    return false;
  const int z_scaler = z - intrinsics->count_at_zero;

  if (z < (int)(intrinsics->count_at_min_range) && z_scaler > 0)
  {
    const int nx = round((x - intrinsics->center_x) * intrinsics->focal_x / (float)z_scaler + intrinsics->center_x);
    const int ny = round((y - intrinsics->center_y) * intrinsics->focal_y / (float)z_scaler + intrinsics->center_y);
    const int scaled_n = n * intrinsics->focal_avg / z_scaler;
    if (nx < 0 || ny < 0 || nx >= (int)(intrinsics->width) || ny >= (int)(intrinsics->height))
      return false;
    if (!IsSignificantVoxelN(intrinsics, nx, ny, scaled_n))
      return false;
  }

  return true;
}

bool IsSignificantVoxelZ(global const struct OpenCLIntrinsics * intrinsics, const int x, const int y, const int z)
  { return IsSignificantVoxelZN(intrinsics, x, y, z, 0); }
bool IsSignificantVoxelZ1(global const struct OpenCLIntrinsics * intrinsics, const int x, const int y, const int z)
  { return IsSignificantVoxelZN(intrinsics, x, y, z, 1); }

// ************** IS SIGNIFICANT END **************

float3 getBearingForPixel(global const struct OpenCLIntrinsics * intrinsics, const uint x, const uint y)
{
  const float3 result = (float3)((float)(x) - intrinsics->center_x, (float)(y) - intrinsics->center_y, intrinsics->focal_avg);
  return normalize(result);
}

void kernel get_all_bearings(global const struct OpenCLIntrinsics * intrinsics,
                             global const float * pov_matrix,
                             global float3 * bearings)
{
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  float3 bearing = getBearingForPixel(intrinsics,x,y);
  bearings[x + y * intrinsics->width] = bearing;
}

void kernel zero_dot_field(
  global const struct OpenCLIntrinsics * intrinsics,
  global int * dot_field,
  global uchar * invalid_dot_field
  )
{
  size_t z = get_global_id(0);
  size_t x = get_global_id(1);

  size_t count_at_max_range = intrinsics->count_at_max_range;
  size_t width = intrinsics->width;
  size_t height = intrinsics->height;

  for (size_t y = 0; y < height; y++)
  {
    size_t i2 = x + y * width;
    size_t i3 = z + i2 * count_at_max_range;
    for (int aa = 0; aa < 3; aa++)
      dot_field[i3 * 3 + aa] = 0;
    for (int aa = 0; aa < 3; aa++)
      invalid_dot_field[i3 * 3 + aa] = false;
  }
}

void kernel zero_known_space_field(
  global const struct OpenCLIntrinsics * intrinsics,
  global uchar * known_space_field_known
  )
{
  const size_t z = get_global_id(0);
  const size_t x = get_global_id(1);

  for (size_t y = 0; y < intrinsics->height; y++)
  {
    const size_t i2 = x + y * intrinsics->width;
    const size_t i3 = z + i2 * intrinsics->count_at_max_range;
    known_space_field_known[i3] = false;
  }
}

void kernel zero_occupancy_ids(
  global const struct OpenCLIntrinsics * intrinsics,
  global uint * occupancy_ids
  )
{
  size_t z = get_global_id(0);
  size_t x = get_global_id(1);

  size_t count_at_max_range = intrinsics->count_at_max_range;
  size_t width = intrinsics->width;
  size_t height = intrinsics->height;

  for (size_t y = 0; y < height; y++)
  {
    size_t i2 = x + y * width;
    size_t i3 = z + i2 * count_at_max_range;
    occupancy_ids[i3] = 0;
  }
}

void kernel simple_fill_uint(
  ulong x_mult,
  ulong y_mult,
  ulong z_mult,
  global uint * matrix,
  uint value
  )
{
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);
  size_t z = get_global_id(2);
  matrix[x * x_mult + y * y_mult + z * z_mult] = value;
}

void kernel filter_known_state_hull_pn(
  ulong width,
  ulong height,
  global const uint * matrix_in,
  global uint * matrix_out
  )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const size_t i2 = x + y * width;

  matrix_out[i2] = matrix_in[i2];

  if (matrix_in[i2] % 2 == 0) // already unknown, keep unknown
    return;

  for (int dx = -1; dx <= 1; dx++)
    for (int dy = -1; dy <= 1; dy++)
    {
      if (!dx && !dy)
        continue;
      if (dx && dy)
        continue;
      const int nx = x + dx;
      const int ny = y + dy;
      if (nx < 0 || nx >= width || y < 0 || ny >= height)
        continue;
      const size_t ni2 = nx + ny * width;
      if (matrix_in[ni2] == (UINT_MAX / 2 * 2)) // unknown found
      {
        matrix_out[i2] = matrix_in[ni2]; // set to unknown
        return;
      }
    }
}

// true if point was in range to be projected
bool ProjectKnownStateHullHelper(
  global const struct OpenCLIntrinsics * intrinsics,
  global const float3 * bearings,
  const int3 int_coords,
  const float3 local_position,
  const float3 local_normal,
  const float3 image_coords,
  const float radius,
  const uint coord0_i, // 0, 1, 2 -> x, y, z
  global uint * known_state_hull_p,
  global uint * known_state_hull_n
  )
{
  const uint coord1_i = (coord0_i + 1) % 3;
  const uint coord2_i = (coord0_i + 2) % 3;

  const int3 max_coords = (int3)(intrinsics->width, intrinsics->height, intrinsics->count_at_max_range);
    // max values for x, y, z

  const float radius_at_distance = getVoxelSideAtDistance(intrinsics, local_position.z);
  const float large_dist = GetFloat3ArrayElement(local_position, coord0_i) / radius_at_distance;
  if (fabs(large_dist * 2.0f) >= INT_MAX)
    return false;
  const float ratio = radius * 2.0f / radius_at_distance;
  const float half_ratio = ratio / 2.0f;

  // coord0 must be out of range
  if (GetFloat3ArrayElement(image_coords, coord0_i) >= half_ratio &&
      GetFloat3ArrayElement(image_coords, coord0_i) < GetInt3ArrayElement(max_coords, coord0_i) - half_ratio)
    return false;

  // only coord0 can be out of range
  if (GetInt3ArrayElement(int_coords, coord1_i) < 0 ||
      GetInt3ArrayElement(int_coords, coord1_i) >= GetInt3ArrayElement(max_coords, coord1_i))
    return false;
  if (GetInt3ArrayElement(int_coords, coord2_i) < 0 ||
      GetInt3ArrayElement(int_coords, coord2_i) >= GetInt3ArrayElement(max_coords, coord2_i))
    return false;

  const int3 clamped_int_coords = min(max(int_coords, 0), max_coords);

  const float3 dots = local_normal;
//  const float dotx = dot(local_normal,(float3)(1,0,0));
//  const float doty = dot(local_normal,(float3)(0,1,0));
//  const float dotz = dot(local_normal,(float3)(0,0,1));

  const bool left = large_dist < 0;
  int known = 0;
  if (left && GetFloat3ArrayElement(dots, coord0_i) > intrinsics->dot_field_safety_th)
    known = 1;
  if (!left && (coord0_i != 2) && GetFloat3ArrayElement(dots, coord0_i) < -intrinsics->dot_field_safety_th)
    known = 1;
  if (!left && (coord0_i == 2)) // pyramidal region, negative z direction
  {
    const uint i2 = GetInt3ArrayElement(clamped_int_coords, coord2_i) +
      GetInt3ArrayElement(clamped_int_coords, coord1_i) * GetInt3ArrayElement(max_coords, coord2_i);
    const float3 bearing = bearings[i2];
    if (dot(local_normal, bearing) < -intrinsics->dot_field_safety_th)
      known = 1;
  }
  const uint maybe_min = abs((int)round(large_dist)) * 2 + known;

  const float known_radius_mult = known ? 1.0f : intrinsics->unknown_surfels_radius_mult_pn;
  float size_1 = sqrt(1.0 - SQR(GetFloat3ArrayElement(dots, coord1_i))) * ratio * known_radius_mult;
  float size_2 = sqrt(1.0 - SQR(GetFloat3ArrayElement(dots, coord2_i))) * ratio * known_radius_mult;
  const int size_1_i = (int)(size_1 + 1.0f);
  const int size_2_i = (int)(size_2 + 1.0f);

  for (int d1 = -size_1_i; d1 <= size_1_i; d1++)
    for (int d2 = -size_2_i; d2 <= size_2_i; d2++)
    {
      const int n1 = d1 + GetInt3ArrayElement(int_coords, coord1_i);
      const int n2 = d2 + GetInt3ArrayElement(int_coords, coord2_i);
      if (n1 < 0 || n2 < 0 ||
          n1 >= GetInt3ArrayElement(max_coords, coord1_i) || n2 >= GetInt3ArrayElement(max_coords, coord2_i))
        continue;

      const float fdy = n1 - GetFloat3ArrayElement(image_coords, coord1_i);
      const float fdz = n2 - GetFloat3ArrayElement(image_coords, coord2_i);

      if (SQR(fdy) * SQR(size_2) + SQR(fdz) * SQR(size_1) > SQR(size_1) * SQR(size_2))
        continue; // exact ellipse

      const uint ni2 = n2 + n1 * GetInt3ArrayElement(max_coords, coord2_i);
      if (left)
        atomic_min(&(known_state_hull_n[ni2]), maybe_min);
      else
        atomic_min(&(known_state_hull_p[ni2]), maybe_min);
    }

  return true;
}

size_t ComputeDotFieldIndex(
  const int3 sizes,
  const int path_family,
  const int3 coords
  )
{
  const size_t coord_plane_size = sizes.x * sizes.y * sizes.z;
  const int coord1 = path_family;
  const int coord2 = (path_family + 1) % 3;
  const int coord3 = (path_family + 2) % 3;

  const size_t i3 = GetInt3ArrayElement(coords, coord1) *
    GetInt3ArrayElement(sizes, coord2) * GetInt3ArrayElement(sizes, coord3) +
    GetInt3ArrayElement(coords, coord3) * GetInt3ArrayElement(sizes, coord2) +
    GetInt3ArrayElement(coords, coord2);

  return coord_plane_size * path_family + i3;
}

void GenerateDotField(
  global const struct OpenCLIntrinsics * intrinsics,
  global const float3 * bearings,
  const float3 image_coords,
  const int3 int_coords,
  const float3 local_normal,
  const float3 local_position,
  const float radius,
  global int * dot_field,
  global uchar * invalid_dot_field
  )
{
  const float3 local_position_bearing = normalize(local_position);

  const int x = int_coords.x;
  const int y = int_coords.y;
  const int z = int_coords.z;
  if (x <= 0 || y <= 0 || x + 1 >= intrinsics->width || y + 1 >= intrinsics->height)
    return;

  const uint i2 = x + y * intrinsics->width;
  if (z <= 0 || z + 1 >= intrinsics->count_at_max_range)
    return;
  const uint i3 = i2 * intrinsics->count_at_max_range + z;

  const int3 sizes = (int3)(intrinsics->width, intrinsics->height, intrinsics->count_at_max_range);

  const float radius_at_distance = getVoxelSideAtDistance(intrinsics, local_position.z) / 2.0f;
  const float ratio = radius / radius_at_distance;

  const float dotx = dot(local_normal,(float3)(1,0,0));
  const float doty = dot(local_normal,(float3)(0,1,0));
  const float dotznear = dot(local_normal,(float3)(0,0,1));
  const float dotzfar = dot(local_normal,local_position_bearing);
  const float dotz = (image_coords.z < intrinsics->count_at_min_range) ? dotznear : dotzfar;

  // projection of the surfel border
  const float3 ratios = (float3)(ratio * sqrt(1.0f - SQR(dotx)),
                                 ratio * sqrt(1.0f - SQR(doty)),
                                 ratio * sqrt(1.0f - SQR(dotz))); // ellipsoid axes
  // surfels have 1 thickness: projection of the tip of the normal
  const float thickness = intrinsics->surfel_thickness;
  const float3 ratios_min = (float3)(fabs(dotx),fabs(doty),fabs(dotz)) * thickness;
  // x = max(ratio * sqrt(1 - <x,n>^2), <x,n>)
  const float3 Rxyzf = max(ratios,ratios_min);

  const int Rx = (int)(Rxyzf.x + 1.0f);
  const int Ry = (int)(Rxyzf.y + 1.0f);
  const int Rz = (int)(Rxyzf.z + 1.0f);

  int dy = -Ry;
  int dx = -Rx;
  int dz = -Rz-1;
//  for (int dy = -Ry; dy <= Ry; dy++)
//    for (int dx = -Rx; dx <= Rx; dx++)
//      for (int dz = -Rz; dz <= Rz; dz++)
  while (true)
      {
        dz++;
        if (dz > Rz) { dx++; dz = -Rz; }
        if (dx > Rx) { dy++; dx = -Rx; }
        if (dy > Ry) break;

        if (y + dy < 0 || x + dx < 0 || z + dz < 0)
          continue;
        if (y + dy >= intrinsics->height || x + dx >= intrinsics->width || z + dz >= intrinsics->count_at_max_range)
          continue;

        const size_t di2 = (y + dy) * intrinsics->width + (x + dx);
        const size_t di3 = di2 * intrinsics->count_at_max_range + (z + dz);

        const bool is_near = (z) < (int)(intrinsics->count_at_min_range);

//#define APPROXIMATE_CUBIC_VOXELS
#ifndef APPROXIMATE_CUBIC_VOXELS
        const float3 bearing = is_near ? (float3)(0,0,1) : (bearings[di2] / bearings[di2].z);
        const float3 cube_pos = (float3)(x + dx, y + dy, z) + dz * bearing;

        const float3 cube_pos_proj = cube_pos - dot(cube_pos - image_coords, local_normal) * local_normal;
        const float dist_vert = length(cube_pos_proj - cube_pos);
        const float dist_hor = length(cube_pos_proj - image_coords);
        const float sqr_dist_hor = SQR(dist_hor);
        const float sqr_distance = SQR(dist_vert) / SQR(thickness) + SQR(dist_hor) / SQR(ratio);
#else
        const float3 cube_pos = (float3)(x + dx, y + dy, z + dz);
        // project cube center to frontel
        const float sqr_dist = dot(cube_pos - image_coords, cube_pos - image_coords);
        const float signed_dist_vert = dot(cube_pos - image_coords, local_normal);
        // distance cube to projection
        const float sqr_dist_vert = SQR(signed_dist_vert);
        // distance projection to frontel center
        const float sqr_dist_hor = sqr_dist - sqr_dist_vert;
        const float sqr_distance = sqr_dist_vert / SQR(thickness) + sqr_dist_hor / SQR(ratio);
#endif

        if (sqr_distance >= 1.0f)
          continue;

        const float distance = sqrt(sqr_distance);
        const float3 dot3 = (float3)(dotx,doty,is_near ? dotznear : dotzfar);

        const float3 normal_dot = sign(dot3) * (1.0f - 1.0f * distance);
        const int3 i_normal_dot = Float3ToInt3(round(normal_dot * DOT_FIELD_MULTIPLIER));

        const int3 coords = (int3)(x + dx, y + dy, z + dz);

        // ATOMIC SUM
        {
          #pragma unroll
          for (int aa = 0; aa < 3; aa++)
          {
            if (fabs(GetFloat3ArrayElement(dot3, aa)) < intrinsics->dot_field_valid_th)
              continue;

            const size_t dot_field_index = ComputeDotFieldIndex(sizes, aa, coords);

            const int v = GetInt3ArrayElement(i_normal_dot, aa);
            const int old_value = atomic_add(&(dot_field[dot_field_index]), v);
            // prevent overflow
            if (v > 0 && old_value > INT_MAX - v)
              atomic_sub(&(dot_field[dot_field_index]), v);
            else if (v < 0 && old_value < INT_MIN - v)
              atomic_sub(&(dot_field[dot_field_index]), v);
          }
        }
        // END ATOMIC SUM

        //if (dist_vert < 0.5f * thickness && dist_hor < 0.5f / sqrt(2.0f) * ratio)
        if (sqr_dist_hor < SQR(0.5f * ratio))
        {
          #pragma unroll
          for (int aa = 0; aa < 3; aa++)
            if (fabs(GetFloat3ArrayElement(dot3, aa)) < intrinsics->dot_field_valid_th)
            {
              const size_t dot_field_index = ComputeDotFieldIndex(sizes, aa, coords);
              invalid_dot_field[dot_field_index] = true;
            }
        }
      }
}

void FindSurfelsToBeDeleted(
  global const struct OpenCLIntrinsics * intrinsics,
  const size_t i,
  const size_t global_i,
  const int3 int_coords,
  const float3 image_coords,
  const bool is_surfel,
  global const uchar * observed_space_field,
  global const float * depths,
  global uint * ids_to_be_deleted,
  global struct OpenCLRecolor * ids_to_be_recolored,
  global uint * occupancy_ids
  )
{
  const int x = int_coords.x;
  const int y = int_coords.y;
  const uint i2 = x + y * intrinsics->width;

  const int z = int_coords.z;
  if (z <= 0 || (uint)(z + 1) >= intrinsics->count_at_max_range)
    return;

  const uint i3 = i2 * intrinsics->count_at_max_range + z;

  occupancy_ids[i3] = global_i + 1;

  if (!IsSignificantVoxelZ(intrinsics, x, y, z))
    return;

  if (!observed_space_field[i3])
  {
    // check for recolor
    if (!is_surfel && z >= intrinsics->count_at_min_range && depths[i2])
    {
      const uint z_depth = getVoxelCountAtDistance(intrinsics, depths[i2]);
      if (z <= z_depth + 1)
      {
        const uint prev = atomic_inc(&(ids_to_be_recolored[0].surfel_id));
        ids_to_be_recolored[prev + 1].surfel_id = i;
        ids_to_be_recolored[prev + 1].coords = (ushort2)(x, y);
      }
    }

    return;
  }

  // check also nearby voxel in the direction of the approximation
  float3 dxy = image_coords - Int3ToFloat3(int_coords);
  int3 idxy = (int3)(0,0,0);
  if (fabs(dxy.x) > fabs(dxy.y) && fabs(dxy.x) > fabs(dxy.z))
    idxy.x = (dxy.x > 0.0) ? 1 : -1;
  else if (fabs(dxy.y) > fabs(dxy.x) && fabs(dxy.y) > fabs(dxy.z))
    idxy.y = (dxy.y > 0.0) ? 1 : -1;
  else
    idxy.z = (dxy.z > 0.0) ? 1 : -1;
  const uint di2 = (x + idxy.x) + (y + idxy.y) * intrinsics->width;
  const uint di3 = di2 * intrinsics->count_at_max_range + z + idxy.z;
  if (!observed_space_field[di3])
    return;

  const uint prev = atomic_inc(ids_to_be_deleted);
  ids_to_be_deleted[prev + 1] = i;

  occupancy_ids[i3] = 0;
}

void kernel project_surfels(
  global const struct OpenCLIntrinsics * intrinsics,
  unsigned long batch_size,
  unsigned long iterations,
  unsigned long total,
  unsigned long offset,
  global const float * pov_matrix,
  global const struct OpenCLSurfel * surfels,
  global float3 * local_position,
  global float3 * local_normal,
  global float3 * local_image_coords,
  global uint * internal_ids,
  global uint * external_ids
  )
{
  size_t tid = get_global_id(0);

  for (uint iter = 0; iter < iterations; iter++)
  {
    size_t i = iter * batch_size + tid;
    if (i >= total)
      break;

    const struct OpenCLSurfel surfel = surfels[i];
    if (surfel.erased)
      continue;

    const float3 local_p = TransformPointPOVMatrix(pov_matrix, ArrayToFloat3(surfel.position));
    local_position[i] = local_p;

    const float3 local_n = TransformNormalPOVMatrix(pov_matrix, ArrayToFloat3(surfel.normal));
    local_normal[i] = local_n;

    const float3 image_coords = getVoxelCountAtPositionF(intrinsics, local_p);
    local_image_coords[i] = image_coords;

    if (!isfinite(image_coords.x) || fabs(image_coords.x) >= INT_MAX ||
        !isfinite(image_coords.y) || fabs(image_coords.y) >= INT_MAX ||
        !isfinite(image_coords.z) || fabs(image_coords.z) >= INT_MAX)
      continue;

    const float diameter_at_distance = getVoxelSideAtDistance(intrinsics, local_p.z);
    const float half_ratio = surfel.radius / diameter_at_distance;

    const int3 int_coords = Float3ToInt3(round(image_coords));
    const int3 coord_max = (int3)(intrinsics->width, intrinsics->height, intrinsics->count_at_max_range);

    uint maybe_external = 0;
    uint external = 0;

    #pragma unroll
    for (uint ci = 0; ci < 3; ci++)
    {
      const float c = GetFloat3ArrayElement(image_coords, ci);
      const int ic = GetInt3ArrayElement(int_coords, ci);
      const int max_ic = GetInt3ArrayElement(coord_max, ci);
      if (c < half_ratio || c > max_ic - half_ratio)
      {
        maybe_external++;
        if (ic < 0 || ic >= max_ic)
          external++;
      }
    }

    if (external == 0)
    {
      const uint old_value = atomic_inc(internal_ids);
      internal_ids[old_value + 1] = i;
    }
    if (maybe_external == 1)
    {
      const uint old_value = atomic_inc(external_ids);
      external_ids[old_value + 1] = i;
    }
  }
}

void kernel project_internal_surfels(
  global const struct OpenCLIntrinsics * intrinsics,
  unsigned long batch_size,
  unsigned long offset,
  global const struct OpenCLSurfel * surfels,
  global const float3 * local_position,
  global const float3 * local_normal,
  global const float3 * local_image_coords,
  global const uint * internal_ids,
  global const uchar * observed_space_field,
  global const float3 * bearings,
  global const float * depths,
  global int * dot_field,
  global uchar * invalid_dot_field,
  global uint * ids_to_be_deleted,
  global struct OpenCLRecolor * ids_to_be_recolored,
  global uint * occupancy_ids
  )
{
  size_t tid = get_global_id(0);

  const unsigned long total = internal_ids[0];
  const uint iterations = total / batch_size + (((total % batch_size) != 0) ? 1 : 0);

  for (uint iter = 0; iter < iterations; iter++)
  {
    size_t i = iter * batch_size + tid;
    if (i >= total)
      break;
    i = internal_ids[i + 1];
    const size_t global_i = i + offset;

    const struct OpenCLSurfel surfel = surfels[i];

    const float3 image_coords = local_image_coords[i];
    const int3 int_coords = Float3ToInt3(round(image_coords));

    /* **************  GENERATE DOT FIELD *************** */
    GenerateDotField(intrinsics, bearings, image_coords, int_coords, local_normal[i], local_position[i],
                     surfel.radius, dot_field, invalid_dot_field);

    /* ***** DELETE SURFELS ***** */
    FindSurfelsToBeDeleted(intrinsics, i, global_i, int_coords, image_coords, surfel.is_surfel, observed_space_field,
                           depths, ids_to_be_deleted, ids_to_be_recolored, occupancy_ids);
  }
}

void kernel project_external_surfels(
  global const struct OpenCLIntrinsics * intrinsics,
  unsigned long batch_size,
  global const struct OpenCLSurfel * surfels,
  global const float3 * local_positions,
  global const float3 * local_normals,
  global const float3 * local_image_coords,
  global const uint * external_ids,
  global const float3 * bearings,
  global uint * known_state_hull_xp,
  global uint * known_state_hull_xn,
  global uint * known_state_hull_yp,
  global uint * known_state_hull_yn,
  global uint * known_state_hull_zp,
  global uint * known_state_hull_zn
  )
{
  size_t tid = get_global_id(0);

  const unsigned long total = external_ids[0];
  const uint iterations = total / batch_size + (((total % batch_size) != 0) ? 1 : 0);

  for (uint iter = 0; iter < iterations; iter++)
  {
    size_t i = iter * batch_size + tid;
    if (i >= total)
      break;
    i = external_ids[i + 1];

    const struct OpenCLSurfel surfel = surfels[i];
    if (surfel.erased)
      continue;

    const float3 local_position = local_positions[i];
    const float3 local_normal = local_normals[i];

    float3 image_coords = local_image_coords[i];

    const int3 int_coords = Float3ToInt3(round(image_coords));

    // project along x onto known_state_hull_xn/known_state_hull_xp
    if (ProjectKnownStateHullHelper(intrinsics, bearings, int_coords, local_position, local_normal, image_coords, surfel.radius,
                                    0, known_state_hull_xp, known_state_hull_xn))
      continue;
    // project along y onto known_state_hull_yn/known_state_hull_yp
    if (ProjectKnownStateHullHelper(intrinsics, bearings, int_coords, local_position, local_normal, image_coords, surfel.radius,
                                    1, known_state_hull_yp, known_state_hull_yn))
      continue;
    // project along z onto known_state_hull_zn/known_state_hull_zp
    if (ProjectKnownStateHullHelper(intrinsics, bearings, int_coords, local_position, local_normal, image_coords, surfel.radius,
                                    2, known_state_hull_zp, known_state_hull_zn))
      continue;
  }
}

// generic, for any path
void path_known(int coord1, // 0, 1, 2
                bool backwards,
                global const struct OpenCLIntrinsics * intrinsics,
                global const int * dot_field,
                global const uchar * invalid_dot_field,
                global const uint * known_state_hull,
                global uchar * known_space_field
                )
{
  size_t i_coord2 = get_global_id(0);
  size_t i_coord3 = get_global_id(1);

  int sizes[3];
  sizes[0] = intrinsics->width;
  sizes[1] = intrinsics->height;
  sizes[2] = intrinsics->count_at_max_range;
  const int3 sizesv = (int3)(intrinsics->width, intrinsics->height, intrinsics->count_at_max_range);

  const int3 cumulative_sizes = (int3)(sizes[2],sizes[0] * sizes[2],1);

  const int coord2 = (coord1 + 1) % 3;
  const int coord3 = (coord2 + 1) % 3;

  const int max = backwards ? -1                  : (sizes[coord1]);
  const int min = backwards ? (sizes[coord1] - 1) : 0;
  const int inc = backwards ? -1                  : 1;
  const float dot_mult = backwards ? -1.0         : 1.0;

  bool first_found = false;
  bool in_known_state = false;
  if (known_state_hull && (known_state_hull[sizes[coord3] * i_coord2 + i_coord3] % 2))
    first_found = in_known_state = true;

  int3 i;
  SetInt3ArrayElement(&i, coord1, min);
  SetInt3ArrayElement(&i, coord2, i_coord2);
  SetInt3ArrayElement(&i, coord3, i_coord3);

  int3 unit_coord1 = (int3)(0,0,0);
  SetInt3ArrayElement(&unit_coord1, coord1, 1);

  const int int_safety_th = round(0.50f * DOT_FIELD_MULTIPLIER);

  int dot;
  {
    const size_t normal_dot_field_index = ComputeDotFieldIndex(sizesv, coord1, i);
    dot = dot_field[normal_dot_field_index] * dot_mult;
  }

  for (SetInt3ArrayElement(&i, coord1, min); GetInt3ArrayElement(i, coord1) != max;
       SetInt3ArrayElement(&i, coord1, GetInt3ArrayElement(i, coord1) + inc))
  {
    if (GetInt3ArrayElement(i, coord1) == (max - inc))
      continue;

    const int3 next_i = i + unit_coord1 * inc;
    const size_t normal_dot_field_index = ComputeDotFieldIndex(sizesv, coord1, next_i);
    const int next_dot = dot_field[normal_dot_field_index] * dot_mult;

    const uint i3 = GetInt3Dot(i, cumulative_sizes);

    const uchar short_dot = invalid_dot_field[normal_dot_field_index];
    if (short_dot)
      first_found = false;
    else
    {
      // we want known space to be between the two Xs, included
      // this is the dot field, varying coord1
      //          ....
      //      .../    \.X.
      // ..../            \........             ....... <--- 0
      //                           \.X.     .../
      //                               \.../

      if (dot < -int_safety_th && next_dot > dot)
      {
        in_known_state = false;
        first_found = true;
      }

      if (first_found)
      {
        if (in_known_state)
          known_space_field[i3] = true;
      }

      if (dot > int_safety_th && next_dot < dot)
      {
        in_known_state = true;
        first_found = true;
      }
    }

    dot = next_dot;
  }
}

void kernel path_knownxf(global const struct OpenCLIntrinsics * intrinsics,
                         global const int * dot_field,
                         global const uchar * invalid_dot_field,
                         global const uint * known_state_hull,
                         global uchar * known_space_field)
  {path_known(0,false,intrinsics,dot_field,invalid_dot_field,known_state_hull,
             known_space_field); }
void kernel path_knownxb(global const struct OpenCLIntrinsics * intrinsics,
                         global const int * dot_field,
                         global const uchar * invalid_dot_field,
                         global const uint * known_state_hull,
                         global uchar * known_space_field)
  {path_known(0,true,intrinsics,dot_field,invalid_dot_field,known_state_hull,
             known_space_field); }
void kernel path_knownyf(global const struct OpenCLIntrinsics * intrinsics,
                         global const int * dot_field,
                         global const uchar * invalid_dot_field,
                         global const uint * known_state_hull,
                         global uchar * known_space_field)
  {path_known(1,false,intrinsics,dot_field,invalid_dot_field,known_state_hull,
             known_space_field); }
void kernel path_knownyb(global const struct OpenCLIntrinsics * intrinsics,
                         global const int * dot_field,
                         global const uchar * invalid_dot_field,
                         global const uint * known_state_hull,
                         global uchar * known_space_field)
  {path_known(1,true,intrinsics,dot_field,invalid_dot_field,known_state_hull,
             known_space_field); }
void kernel path_knownzf(global const struct OpenCLIntrinsics * intrinsics,
                         global const int * dot_field,
                         global const uchar * invalid_dot_field,
                         global const uint * known_state_hull,
                         global uchar * known_space_field)
  {path_known(2,false,intrinsics,dot_field,invalid_dot_field,known_state_hull,
             known_space_field); }
void kernel path_knownzb(global const struct OpenCLIntrinsics * intrinsics,
                         global const int * dot_field,
                         global const uchar * invalid_dot_field,
                         global const uint * known_state_hull,
                         global uchar * known_space_field)
  {path_known(2,true,intrinsics,dot_field,invalid_dot_field,known_state_hull,
             known_space_field); }

void kernel filter_known_space_field_passthrough(global const struct OpenCLIntrinsics * intrinsics,
                                                 global uint * counter, // no. of filtered voxels, for debug
                                                 global const uchar * old_known_space_field,
                                                 global uchar * known_space_field)
{
  const int width = intrinsics->width;
  const int height = intrinsics->height;
  const int count_at_max_range = intrinsics->count_at_max_range;

  const size_t z = get_global_id(0);
  const size_t x = get_global_id(1);

  for (size_t y = 0; y < height; y++)
  {
    const size_t i2 = x + y * width;
    const size_t i3 = i2 * count_at_max_range + z;
    known_space_field[i3] = old_known_space_field[i3];
  }
}

void kernel filter_known_space_field_median(global const struct OpenCLIntrinsics * intrinsics,
                                            global uint * counter, // no. of filtered voxels, for debug
                                            global const uchar * old_known_space_field,
                                            global uchar * known_space_field)
{
  const int width = intrinsics->width;
  const int height = intrinsics->height;
  const int count_at_max_range = intrinsics->count_at_max_range;

  const size_t z = get_global_id(0);
  const size_t x = get_global_id(1);

  for (size_t y = 0; y < height; y++)
  {
    const size_t i2 = x + y * width;
    const size_t i3 = i2 * count_at_max_range + z;
    const bool k = old_known_space_field[i3];

    known_space_field[i3] = old_known_space_field[i3];

    uint count = 1;
    uint total = 1;

    const int WINDOW = 1;

    for (int dx = -WINDOW; dx <= WINDOW; dx++)
      for (int dy = -WINDOW; dy <= WINDOW; dy++)
        for (int dz = -WINDOW; dz <= WINDOW; dz++)
        {
          if (!dx && !dy && !dz)
            continue;
          total++;

          int nx = x + dx;
          int ny = y + dy;
          int nz = z + dz;
          if (nz < 0 || nz >= count_at_max_range ||
              nx < 0 || nx >= width ||
              ny < 0 || ny >= height)
          {
            count++;
            continue;
          }

          const uint ni2 = nx + ny * width;
          const uint ni3 = ni2 * count_at_max_range + nz;
          const bool nk = old_known_space_field[ni3];
          if (nk == k)
            count++;
        }

    const bool to_be_changed = count * 2 < total;
    known_space_field[i3] = to_be_changed ? (!k) : k;

    if (to_be_changed)
      atomic_inc(counter);
  }
}

void kernel zero_observed_space_field(global const struct OpenCLIntrinsics * intrinsics,
                                      global uchar * observed_space_field)
{
  const size_t z = get_global_id(0);
  const size_t x = get_global_id(1);

  for (uint y = 0; y < intrinsics->height; y++)
  {
    const uint i2 = x + y * intrinsics->width;
    const uint i3 = i2 * intrinsics->count_at_max_range + z;

    observed_space_field[i3] = false;
  }
}

void kernel generate_observed_space_field(global const struct OpenCLIntrinsics * intrinsics,
                                          global float * depths,
                                          global uchar * observed_space_field)
{
  const size_t x = get_global_id(0);
  const size_t y = get_global_id(1);

  if (!IsSignificantVoxel(intrinsics, x, y))
    return;

  // pyramidal part
  for (int fake_i = 0; fake_i < 1; fake_i++)
  {
    const uint i2 = x + y * intrinsics->width;
    const float depth = depths[i2];
    if (depth == 0.0f)
      continue;

    int depth_z = getVoxelCountAtDistance(intrinsics, depth);
    if (depth_z < (int)(intrinsics->count_at_min_range))
      continue;

    if ((uint)depth_z >= intrinsics->count_at_max_range - 1)
      depth_z = intrinsics->count_at_max_range - 2;

    for (uint z = intrinsics->count_at_min_range; z < (uint)depth_z; z++)
    {
      const uint i3 = i2 * intrinsics->count_at_max_range + z;
      observed_space_field[i3] = true;
    }
  }

  // cubic part
  for (int fake_i = 0; fake_i < 1; fake_i++)
  {
    if (!IsSignificantVoxel(intrinsics, x, y))
      break;
    if (depths[x + y * intrinsics->width] < intrinsics->min_range)
      break;
    for (uint z = intrinsics->count_at_zero; z < intrinsics->count_at_min_range; z++)
    {
      const float zw = z - (int)(intrinsics->back_ignore_margin);
      const float2 float_coords = (float2)((x - intrinsics->center_x) * zw / intrinsics->focal_x,
                                           (y - intrinsics->center_y) * zw / intrinsics->focal_y);

      const int2 int_coords = Float2ToInt2(round(float_coords)) + (int2)(intrinsics->center_x, intrinsics->center_y);
      if (int_coords.x >= intrinsics->width ||
          int_coords.y >= intrinsics->height ||
          int_coords.x < 0 ||
          int_coords.y < 0)
        break;

      const uint i2 = int_coords.x + int_coords.y * intrinsics->width;
      const uint i3 = i2 * intrinsics->count_at_max_range + z;
      observed_space_field[i3] = true;
    }
  }
}

void kernel create_new_surfels(global const struct OpenCLIntrinsics * intrinsics,
                               global const float3 * bearings,
                               global const float * pov_matrix,
                               global const uchar * known_space_field,
                               global const uchar * observed_space_field,
                               global const float * depths,
                               global const uint * occupancy_ids,
                               global struct OpenCLCreationCounters * counters,
                               global struct OpenCLSurfel * surfels,
                               global ushort2 * surfels_color_source,
                               global uint * replace_global_id,
                               global uint * unfinished_grid
                               )
{
  const int z = get_global_id(0);
  const int x = get_global_id(1);

  if (x == 0 || z == 0 || z + 1 >= (int)(intrinsics->count_at_max_range) || x + 1 >= (int)(intrinsics->width))
    return;
//  if (!IsSignificantVoxel1(intrinsics, x, y))
//    return;

  const uint gridi = z + x * intrinsics->count_at_max_range;
  if (counters->reinit)
    unfinished_grid[gridi] = 1;

  for (int y = unfinished_grid[gridi]; y < (int)(intrinsics->height - 1); y++)
  {
    const uint i2 = x + y * intrinsics->width;
    const float3 bearing = bearings[i2];

    const float normal_base[9] = {1,0,0,0,1,0,bearing.x,bearing.y,bearing.z};
    // inverting the shear
    const float inv_normal_base[9] = {1, 0, 0,
                                      0, 1, 0,
                                      -bearing.x / bearing.z, -bearing.y / bearing.z, 1.0f / bearing.z};
    const uint i3 = i2 * intrinsics->count_at_max_range + z;
    if (observed_space_field[i3] || known_space_field[i3])
      continue; // in known space

//    if (occupancy_ids[i3])
//      continue; // already created

    bool found_any = false;
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
          if (!dx && !dy && !dz)
            continue;
          if (dx * dx + dy * dy + dz * dz > 1)
            continue;

          const uint di2 = (x + dx) + (y + dy) * intrinsics->width;
          const uint di3 = di2 * intrinsics->count_at_max_range + (z + dz);
          if (observed_space_field[di3]) // unknown found known space
            found_any = true;
        }

    if (!found_any)
      continue;

    float3 normal = (float3)(0,0,0);
    // accurate normal computation
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dz = -1; dz <= 1; dz++)
        {
          if (!dx && !dy && !dz)
            continue;

          float3 normal_diff = normalize((float3)(dx,dy,dz));
          if (z + dz >= (int)(intrinsics->count_at_min_range))
            normal_diff = MulMatrix3x3_Float3(inv_normal_base, normal_diff);

          const uint di2 = (x + dx) + (y + dy) * intrinsics->width;
          const uint di3 = di2 * intrinsics->count_at_max_range + (z + dz);
          if (observed_space_field[di3])
            normal += normal_diff;
        }

    if (length(normal) < 0.01)
    {
#ifndef USE_INTEL_COMPILER // for some reason, this line causes a compiler crash on intel
      atomic_inc(&(counters->creation_failed));
#endif
      continue;
    }

    const float zf = getDistanceFromVoxelCount(intrinsics, z);
    const float diameter = getVoxelSideAtDistance(intrinsics, zf);

    bool found_surfel = false;
    if ((uint)(z) >= intrinsics->count_at_min_range && depths[i2] && IsSignificantVoxel(intrinsics, x, y))
    {
      if (fabs(depths[i2] - zf) <= diameter)
        found_surfel = true;
    }

    const float3 new_local_pose = getPositionAtVoxelCount(intrinsics, bearings, (float3)(x,y,z));

    normal = normalize(normal);

    float radius_incline_mult;
    {
      const float3 abs_normal = fabs(normal);
      float maybe_max = 0.0f;
      #pragma unroll
      for (int order = 0; order < 3; order++)
      {
        float3 reorder_normal;
        reorder_normal.x = GetFloat3ArrayElement(abs_normal, (order + 0) % 3);
        reorder_normal.y = GetFloat3ArrayElement(abs_normal, (order + 1) % 3);
        reorder_normal.z = GetFloat3ArrayElement(abs_normal, (order + 2) % 3);
        if (reorder_normal.z < 0.01)
          continue;
        const float zeta_sum = (reorder_normal.x + reorder_normal.y) / reorder_normal.z;
        const float zeta_diff = fabs(reorder_normal.x - reorder_normal.y) / reorder_normal.z;
        if (zeta_sum <= 1.0f && zeta_sum > maybe_max)
          maybe_max = zeta_sum;
        if (zeta_diff <= 1.0f && zeta_diff > maybe_max)
          maybe_max = zeta_diff;
      }
      radius_incline_mult = sqrt(SQR(maybe_max) + SQR(1) + SQR(1));
    }

    normal = TransformNormalPOVMatrix(pov_matrix, normal);
    const float3 new_pose = TransformPointPOVMatrix(pov_matrix, new_local_pose);

    const uint slot = atomic_inc(&(counters->created));
    if (slot >= counters->max_created)
    {
      atomic_dec(&(counters->created));
      unfinished_grid[gridi] = y;
      counters->unfinished = true;
      return;
    }

    global struct OpenCLSurfel * surfel = &(surfels[slot]);
    #pragma unroll
    for (int i = 0; i < 3; i++)
      surfel->position[i] = GetFloat3ArrayElement(new_pose, i);
    surfel->radius = diameter / 2.0 * intrinsics->surfel_radius_mult * radius_incline_mult;
    #pragma unroll
    for (int i = 0; i < 3; i++)
      surfel->normal[i] = GetFloat3ArrayElement(normal, i);
    surfel->erased = false;
    surfel->is_surfel = found_surfel;
    if (found_surfel)
      surfels_color_source[slot] = (ushort2)(x,y);
    if (occupancy_ids[i3])
      replace_global_id[slot] = occupancy_ids[i3];
    else
      replace_global_id[slot] = 0;
  }
  unfinished_grid[gridi] = intrinsics->height;
}
