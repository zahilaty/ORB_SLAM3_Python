"""Extractor invariants and an optional differential oracle from the baseline."""
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from orbslam3.features import ORBExtractor, distribute_octree, hamming


def test_hamming_all_bits_and_broadcasting():
    a = np.zeros(32,np.uint8)
    b = np.full((2,32),255,np.uint8)
    b[1] = 1
    np.testing.assert_array_equal(hamming(a,b),[256,32])
    assert hamming(a,a) == 0


def test_blank_image_and_small_pyramid_are_empty():
    for shape in [(480,640),(40,40)]:
        features = ORBExtractor().extract(np.zeros(shape,np.uint8))
        assert features.points.shape == (0,2)
        assert features.descriptors.shape == (0,32)


def test_octree_distributes_before_choosing_response():
    points = np.array([[10,10],[11,10],[90,10],[10,90],[90,90]],np.float32)
    response = np.array([100,99,1,2,3])
    indices = distribute_octree(points,response,100,100,4)
    assert set(indices) == {0,2,3,4}


def test_orientation_and_brief_rotate_with_image():
    image = np.random.default_rng(8).integers(0,256,(100,120),dtype=np.uint8)
    extractor = ORBExtractor(nlevels=1)
    points = np.array([[40,35],[70,60]],np.float32)
    angles = extractor.orientation(image,points)
    rotated = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    rotated_points = np.c_[image.shape[0]-1-points[:,1],points[:,0]]
    rotated_angles = extractor.orientation(rotated,rotated_points)
    difference = (rotated_angles-angles+180)%360-180
    np.testing.assert_allclose(difference,90,atol=0.02)
    blur = cv2.GaussianBlur(image,(7,7),2,2)
    rotated_blur = cv2.rotate(blur,cv2.ROTATE_90_CLOCKWISE)
    np.testing.assert_array_equal(extractor.descriptors(blur,points,angles),
                                  extractor.descriptors(rotated_blur,rotated_points,rotated_angles))


def test_overlap_partition_preserves_association_and_reverses_tail():
    image = np.random.default_rng(11).integers(0,256,(240,320),dtype=np.uint8)
    extractor = ORBExtractor(nlevels=4)
    plain = extractor.extract(image)
    overlap = extractor.extract(image,overlap=(100,200))
    inside = (plain.points[:,0] >= 100) & (plain.points[:,0] <= 200)
    order = np.r_[np.flatnonzero(~inside),np.flatnonzero(inside)[::-1]]
    assert overlap.mono_count == sum(~inside)
    np.testing.assert_array_equal(overlap.points,plain.points[order])
    np.testing.assert_array_equal(overlap.descriptors,plain.descriptors[order])


# The oracle is test-only, never imported or linked by the Python package.
_CPP_ORACLE = r'''
#include "ORBextractor.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iomanip>
int main(int argc, char** argv) {
    cv::Mat image = cv::imread(argv[1],cv::IMREAD_GRAYSCALE);
    ORB_SLAM3::ORBextractor extractor(1000,1.2f,8,20,7);
    std::vector<cv::KeyPoint> keys; cv::Mat descriptors;
    std::vector<int> overlap={0,0};
    extractor(image,cv::noArray(),keys,descriptors,overlap);
    std::ofstream out(argv[2]); out << std::setprecision(10);
    for (int i=0;i<keys.size();++i) {
        const auto& k=keys[i];
        out << k.pt.x << ' ' << k.pt.y << ' ' << k.octave << ' ' << k.angle;
        for (int j=0;j<32;++j) out << ' ' << int(descriptors.at<uchar>(i,j));
        out << '\n';
    }
}
'''


@pytest.mark.skipif(not os.environ.get("ORBSLAM3_REFERENCE"),
                    reason="set ORBSLAM3_REFERENCE to the original checkout for C++ comparison")
def test_original_cpp_extractor_on_identical_image(tmp_path):
    reference = Path(os.environ["ORBSLAM3_REFERENCE"]).resolve()
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required for the test-only source oracle")
    prefix = reference/".venv"
    source, executable = tmp_path/"oracle.cc", tmp_path/"oracle"
    source.write_text(_CPP_ORACLE)
    subprocess.run([compiler,"-std=c++17","-O2",f"-I{reference/'include'}",
                    f"-I{prefix/'include/opencv4'}",str(source),
                    str(reference/"src/ORBextractor.cc"),f"-L{prefix/'lib'}",
                    f"-Wl,-rpath,{prefix/'lib'}","-lopencv_core","-lopencv_imgproc",
                    "-lopencv_features2d","-lopencv_imgcodecs","-o",str(executable)],check=True)
    image = np.random.default_rng(1).integers(0,256,(480,640),dtype=np.uint8)
    image_path, result_path = tmp_path/"image.pgm", tmp_path/"features.txt"
    assert cv2.imwrite(str(image_path),image)
    subprocess.run([str(executable),str(image_path),str(result_path)],check=True)
    original = np.loadtxt(result_path)
    features = ORBExtractor().extract(image)
    assert len(features) == len(original)
    np.testing.assert_array_equal(np.bincount(features.octaves),
                                  np.bincount(original[:,2].astype(int)))
    # The C++ comparator has no y tie-breaker. A few tied octree leaves can
    # select different candidates, while the selection operation is identical.
    lookup = {(round(x,3),round(y,3),int(level)):i
              for i,(x,y,level) in enumerate(original[:,:3])}
    shared = []
    for j,(point,level) in enumerate(zip(features.points,features.octaves)):
        key = (round(float(point[0]),3),round(float(point[1]),3),int(level))
        if key in lookup:
            shared.append((lookup[key],j))
    assert len(shared) >= 0.99*len(original)
    ii,jj = np.array(shared).T
    np.testing.assert_allclose(features.angles[jj],original[ii,3],atol=1e-5)
    np.testing.assert_array_equal(features.descriptors[jj],original[ii,4:].astype(np.uint8))


def test_saved_original_cpp_extractor_fixture():
    fixture = np.load(Path(__file__).parent/"fixtures/extractor_cpp.npz",allow_pickle=False)
    features = ORBExtractor().extract(fixture["image"])
    assert len(features) == len(fixture["points"])
    np.testing.assert_array_equal(np.bincount(features.octaves),np.bincount(fixture["octaves"]))
    lookup = {(round(float(x),3),round(float(y),3),int(level)):i
              for i,((x,y),level) in enumerate(zip(fixture["points"],fixture["octaves"]))}
    shared = []
    for j,((x,y),level) in enumerate(zip(features.points,features.octaves)):
        key = (round(float(x),3),round(float(y),3),int(level))
        if key in lookup:
            shared.append((lookup[key],j))
    assert len(shared) >= 0.99*len(features)
    ii,jj = np.array(shared).T
    np.testing.assert_allclose(features.angles[jj],fixture["angles"][ii],atol=1e-5)
    np.testing.assert_array_equal(features.descriptors[jj],fixture["descriptors"][ii])
    np.testing.assert_array_equal(features.responses[jj],fixture["responses"][ii])
    np.testing.assert_array_equal(features.sizes[jj],fixture["sizes"][ii])


def test_empty_high_threshold_cells_retry_low_fast_threshold():
    image = np.random.default_rng(25).integers(100,121,(240,320),dtype=np.uint8)
    high_only = ORBExtractor(nlevels=3,initial_fast=20,minimum_fast=20).extract(image)
    fallback = ORBExtractor(nlevels=3,initial_fast=20,minimum_fast=7).extract(image)
    assert len(high_only) == 0
    assert len(fallback) > 100
