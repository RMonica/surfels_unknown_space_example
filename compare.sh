#!/bin/bash

echo "Comparing (1)"
SCEN1=`./build/compare_surfel_clouds data/Scenario_1/output_cloud.pcd data/Scenario_1_octomap_gt/2000_octomap_surfels_hr.pcd`
echo "Comparing (2)"
SCEN2=`./build/compare_surfel_clouds data/Scenario_2/output_cloud.pcd data/Scenario_2_octomap_gt/1450_octomap_surfels_hr.pcd`

echo $'Scen.\t\tFP\t\t|Psi_n|\t\tFDR'
echo -n $'1\t\t'
echo "$SCEN1" | head -n1
echo -n $'2\t\t'
echo "$SCEN2" | head -n1
echo $'Scen.\t\tFN\t\t|M_n|\t\tFOR'
echo -n $'1\t\t'
echo "$SCEN1" | tail -n1
echo -n $'2\t\t'
echo "$SCEN2" | tail -n1
