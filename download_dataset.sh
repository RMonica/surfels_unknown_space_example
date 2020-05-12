#!/bin/bash

mkdir -p data
cd data

echo "Downloading..."
wget -c http://rimlab.ce.unipr.it/~rmonica/dataset_surfels_unknown_space/Scenario_1.zip
wget -c http://rimlab.ce.unipr.it/~rmonica/dataset_surfels_unknown_space/Scenario_2.zip

wget -c http://rimlab.ce.unipr.it/~rmonica/dataset_surfels_unknown_space/Scenario_1_octomap_gt.zip
wget -c http://rimlab.ce.unipr.it/~rmonica/dataset_surfels_unknown_space/Scenario_2_octomap_gt.zip

echo "Unzipping Scenario_1.zip..."
rm -r data
rm -r Scenario_1
unzip Scenario_1.zip
mv data Scenario_1
echo "Unzipping Scenario_2.zip..."
rm -r Scenario_2
unzip Scenario_2.zip
mv data Scenario_2
echo "Unzipping Scenario_1_octomap_gt.zip..."
unzip -o Scenario_1_octomap_gt.zip
echo "Unzipping Scenario_2_octomap_gt.zip..."
unzip -o Scenario_2_octomap_gt.zip

echo "Done."

cd ..


