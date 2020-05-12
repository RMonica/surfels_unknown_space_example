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

#include <surfels_unknown_space/surfels_unknown_space.h>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// STL
#include <stdint.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "state_point_type.h"

typedef uint64_t uint64;
typedef std::vector<float> FloatVector;

typedef SurfelsUnknownSpace::SurfelVector SurfelVector;
typedef SurfelsUnknownSpace::Surfel Surfel;

typedef pcl::PointCloud<pcl::PointSurfel> PointSurfelCloud;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointXYZRGBNormalCloud;

pcl::PolygonMesh GenerateMesh(const PointSurfelCloud & cloud)
{
  std::cout << "Generating mesh..." << std::endl;
  const uint64 SAMPLES = 6;

  pcl::PolygonMesh mesh;
  mesh.polygons.reserve(cloud.size() * SAMPLES);
  PointXYZRGBNormalCloud mesh_cloud;
  mesh_cloud.reserve(cloud.size() * SAMPLES);
  for (const pcl::PointSurfel & pt : cloud)
  {
    const Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
    const Eigen::Vector3f origin(pt.x, pt.y, pt.z);
    Eigen::Vector3f reference;
    for (uint64 i = 0; i < 3; i++)
      if (i == 0 ||
          std::abs(Eigen::Vector3f::Unit(i).dot(normal)) < std::abs(reference.dot(normal)))
        reference = Eigen::Vector3f::Unit(i);
    reference = (reference - reference.dot(normal) * normal).normalized();
    const Eigen::Vector3f binormal = normal.cross(reference);
    Eigen::Affine3f transf;
    transf.translation() = origin;
    transf.linear().col(0) = reference;
    transf.linear().col(1) = binormal;
    transf.linear().col(2) = normal;

    const uint64 first_index = mesh_cloud.size();

    for (uint64 i = 0; i < SAMPLES; i++)
    {
      const float x = pt.radius * std::cos(float(i) / SAMPLES * M_PI * 2.0f);
      const float y = pt.radius * std::sin(float(i) / SAMPLES * M_PI * 2.0f);
      const float z = 0.0;
      Eigen::Vector3f ev = transf * Eigen::Vector3f(x, y, z);
      pcl::PointXYZRGBNormal vertex;
      vertex.x = ev.x();
      vertex.y = ev.y();
      vertex.z = ev.z();
      vertex.r = pt.r;
      vertex.g = pt.g;
      vertex.b = pt.b;
      vertex.a = pt.a;
      vertex.normal_x = pt.normal_x;
      vertex.normal_y = pt.normal_y;
      vertex.normal_z = pt.normal_z;
      mesh_cloud.push_back(vertex);

      if (i > 1)
      {
        pcl::Vertices tri;
        tri.vertices.resize(3);
        tri.vertices[0] = first_index;
        tri.vertices[1] = mesh_cloud.size() - 2;
        tri.vertices[2] = mesh_cloud.size() - 1;
        mesh.polygons.push_back(tri);
      }
    }
  }

  pcl::toPCLPointCloud2(mesh_cloud, mesh.cloud);

  return mesh;
}

PointSurfelCloud GenerateCloud(const SurfelVector & sv)
{
  std::cout << "Generating cloud..." << std::endl;
  PointSurfelCloud cloud;
  cloud.reserve(sv.size());

  for (const Surfel & s : sv)
  {
    pcl::PointSurfel ps;
    ps.radius = s.radius;
    ps.x = s.position.x();
    ps.y = s.position.y();
    ps.z = s.position.z();
    ps.normal_x = s.normal.x();
    ps.normal_y = s.normal.y();
    ps.normal_z = s.normal.z();

    if (s.is_surfel)
    {
      ps.r = s.cr;
      ps.g = s.cg;
      ps.b = s.cb;
      ps.a = 255;
      ps.confidence = 1.0;
    }
    else
    {
      ps.r = 0;
      ps.g = 0;
      ps.b = 255;
      ps.confidence = 0.0;
    }

    cloud.push_back(ps);
  }

  return cloud;
}

void SaveCloud(const PointSurfelCloud & cloud, const std::string & prefix, const std::string & filename_prefix)
{
  std::cout << "saving point cloud." << std::endl;

  const std::string filename = prefix + filename_prefix + ".pcd";
  std::cout << "saving pointcloud: " << filename << std::endl;

  if (cloud.empty())
  {
    std::cout << "point cloud not saved, it is empty." << std::endl;
  }
  else
  {
    if (!pcl::io::savePCDFileBinary(filename, cloud))
      std::cout << "saved file " << filename << std::endl;
    else
      std::cout << "could not save file " << filename << std::endl;
  }
}

void SaveMesh(const pcl::PolygonMesh & mesh, const std::string & prefix, const std::string & filename_prefix)
{
  std::cout << "saving mesh." << std::endl;

  const std::string filename = prefix + filename_prefix + ".ply";
  std::cout << "saving mesh: " << filename << std::endl;

  if (mesh.polygons.empty())
  {
    std::cout << "mesh not saved, it is empty." << std::endl;
  }
  else
  {
    if (!pcl::io::savePLYFileBinary(filename, mesh))
      std::cout << "saved file " << filename << std::endl;
    else
      std::cout << "could not save file " << filename << std::endl;
  }
}

int main(int argc, char ** argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: surfels_unknown_space_example /path/to/data/ [output_file] [stop_at_frame]" << std::endl;
    return 1;
  }
  const std::string prefix(argv[1]);
  std::cout << "Data prefix is: " << prefix << std::endl;

  std::string filename_prefix = "output_cloud";
  if (argc >= 3)
  {
    filename_prefix = std::string(argv[2]);
    std::cout << "Using filename prefix " << filename_prefix << std::endl;
  }

  uint64 max_counter = 1000 * 1000;
  if (argc >= 4)
  {
    std::istringstream istr;
    istr.str(std::string(argv[3]));
    istr >> max_counter;
    if (!istr)
    {
      std::cerr << "Invalid stop_at_frame: " << std::string(argv[3]);
      exit(1);
    }
    std::cout << "Stop at frame " << max_counter << std::endl;
  }

  SurfelsUnknownSpace::Config config;
  //config.opencl_use_intel = true;
  config.opencl_platform_name = ""; //"NVIDIA";
  config.max_range = 3.0; // measurements beyond this are discardeded

  SurfelsUnknownSpace sus(config, NULL, NULL, NULL);

  uint64 counter = 0;
  while (true)
  {
    const std::string counter_string = boost::lexical_cast<std::string>(counter);
    const std::string full_prefix = prefix + counter_string + "_";

    std::string discard;

    uint64 width, height;
    float focal_x, focal_y, center_x, center_y;

    // ******** load intrinsics *******
    std::ifstream mfile(full_prefix + "intrinsics.txt");
    mfile >> discard;
    mfile >> width >> height;
    mfile >> focal_x >> focal_y;
    mfile >> center_x >> center_y;

    SurfelsUnknownSpace::Intrinsics intrinsics(center_x, center_y, focal_x, focal_y, width, height,
                                               config.min_range, config.max_range);

    if (!mfile)
    {
      std::cerr << "Error while loading " << full_prefix + "intrinsics.txt" << std::endl;
      break;
    }

    // ******** load pose *******
    std::ifstream pfile(full_prefix + "pose.matrix");
    Eigen::Affine3f pose;
    for (uint64 y = 0; y < 3; y++)
    {
      float v;
      for (uint64 x = 0; x < 3; x++)
      {
        pfile >> v;
        pose.linear()(y, x) = v;
      }
      pfile >> v;
      pose.translation()[y] = v;
    }

    if (!pfile)
    {
      std::cerr << "Error while loading " << full_prefix + "pose.matrix" << std::endl;
      break;
    }

    // ****** load cloud ******
    pcl::PointCloud<StatePointType> cloud;
    const std::string filename = full_prefix + "cloud.pcd";
    std::cout << "Loading cloud " << filename << std::endl;
    if (pcl::io::loadPCDFile(filename, cloud))
    {
      std::cerr << "could not load file " << filename << std::endl;
      break;
    }

    FloatVector input_depth(width * height);
    FloatVector input_color(width * height * 3); // RGB

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const uint64 i2 = y * width + x;
        const StatePointType & point = cloud[i2];

        input_color[i2 * 3 + 0] = point.input_r / 255.0;
        input_color[i2 * 3 + 1] = point.input_g / 255.0;
        input_color[i2 * 3 + 2] = point.input_b / 255.0;

        input_depth[i2] = point.input_depth / 1000.0;
      }

    std::cout << "Processing cloud " << (unsigned)counter << std::endl;

    sus.ProcessFrame(width, height, input_depth, input_color, pose, intrinsics);

    counter++;

    if (counter >= max_counter)
      break;
  }

  const SurfelVector sv = sus.GetSurfels();
  const PointSurfelCloud cloud = GenerateCloud(sv);
  const pcl::PolygonMesh mesh = GenerateMesh(cloud);

  SaveCloud(cloud, prefix, filename_prefix);
  SaveMesh(mesh, prefix, filename_prefix);
}
