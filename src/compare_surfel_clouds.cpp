/*
 * Copyright (c) 2020, Riccardo Monica
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

#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <stdint.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

typedef uint64_t uint64;
typedef pcl::PointCloud<pcl::PointSurfel> PointSurfelCloud;
typedef pcl::PointCloud<pcl::PointXYZ> PointXYZCloud;

int main(int argc, char ** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: compare_surfel_clouds cloud.pcd ground_truth.pcd [threshold]" << std::endl;
    exit(1);
  }

  float threshold = 0.05f;

  const std::string cloud_filename(argv[1]);
  const std::string ground_truth_filename(argv[2]);

  if (argc >= 4)
  {
    std::istringstream istr(argv[3]);
    istr >> threshold;
    if (!istr)
    {
      std::cerr << "invalid threshold: " << argv[3] << std::endl;
      exit(1);
    }
  }

  PointSurfelCloud::Ptr cloud(new PointSurfelCloud);
  if (pcl::io::loadPCDFile(cloud_filename, *cloud))
  {
    std::cerr << "could not load file " << cloud_filename << std::endl;
    exit(1);
  }

  PointXYZCloud::Ptr cloud_xyz(new PointXYZCloud);
  pcl::copyPointCloud(*cloud, *cloud_xyz);

  PointSurfelCloud::Ptr groundtruth(new PointSurfelCloud);
  if (pcl::io::loadPCDFile(ground_truth_filename, *groundtruth))
  {
    std::cerr << "could not load ground truth file: " << ground_truth_filename << std::endl;
    exit(1);
  }

  PointXYZCloud::Ptr groundtruth_xyz(new PointXYZCloud);
  pcl::copyPointCloud(*groundtruth, *groundtruth_xyz);

  const uint64 groundtruth_size = groundtruth->size();
  const uint64 cloud_size = cloud->size();

  pcl::KdTreeFLANN<pcl::PointXYZ> cloud_kdtree;
  cloud_kdtree.setInputCloud(cloud_xyz);
  pcl::KdTreeFLANN<pcl::PointXYZ> groundtruth_kdtree;
  groundtruth_kdtree.setInputCloud(groundtruth_xyz);

  uint64 fp = 0;
  uint64 fn = 0;

  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  for (uint64 i = 0; i < cloud_size; i++)
  {
    const pcl::PointXYZ searchpoint = (*cloud_xyz)[i];
    if (!groundtruth_kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance))
    {
      fp++;
      continue;
    }

    const float radius_cloud = (*cloud)[i].radius;
    const float radius_groundtruth = (*groundtruth)[pointIdxNKNSearch[0]].radius;
    const float distance = std::sqrt(pointNKNSquaredDistance[0]);

    if (distance > radius_cloud + radius_groundtruth + threshold)
      fp++;
  }

  for (uint64 i = 0; i < groundtruth_size; i++)
  {
    const pcl::PointXYZ searchpoint = (*groundtruth_xyz)[i];
    if (!cloud_kdtree.nearestKSearch(searchpoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance))
    {
      fn++;
      continue;
    }

    const float radius_cloud = (*cloud)[pointIdxNKNSearch[0]].radius;
    const float radius_groundtruth = (*groundtruth)[i].radius;
    const float distance = std::sqrt(pointNKNSquaredDistance[0]);

    if (distance > radius_cloud + radius_groundtruth + threshold)
      fn++;
  }

  const float fp_frac = float(fp) / float(cloud->size());
  const float fn_frac = float(fn) / float(groundtruth->size());

  std::cout << fp << "\t\t" << cloud->size() << "\t\t" << fp_frac << std::endl;
  std::cout << fn << "\t\t" << groundtruth->size() << "\t\t" << fn_frac << std::endl;

  return 0;
}
