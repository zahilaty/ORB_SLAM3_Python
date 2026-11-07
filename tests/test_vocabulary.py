from pathlib import Path
import numpy as np
import pytest

from orbslam3.vocabulary import Vocabulary, InvertedDatabase


def make_vocabulary(tmp_path, scoring=0, weighting=0):
    # Two first-level branches with two leaves each, exact text asset format.
    def node(parent,leaf,byte,weight):
        return f"{parent} {leaf} "+" ".join([str(byte)]*32)+f" {weight}\n"
    path = tmp_path/"tiny_vocabulary.txt"
    path.write_text(f"2 2 {scoring} {weighting}\n"+
                    node(0,0,0,0)+node(0,0,255,0)+
                    node(1,1,0,1)+node(1,1,15,2)+
                    node(2,1,240,3)+node(2,1,255,0))
    return Vocabulary.from_text(path)


def test_quantization_tie_breaking_and_ancestor_grouping(tmp_path):
    vocabulary = make_vocabulary(tmp_path)
    assert len(vocabulary) == 4
    assert vocabulary.quantize(np.zeros(32,np.uint8),levelsup=1) == (0,1,1)
    # 0x0f is equally far from both root children: first child wins.
    assert vocabulary.quantize(np.full(32,15,np.uint8),levelsup=1) == (1,2,1)
    assert vocabulary.quantize(np.full(32,240,np.uint8),levelsup=1) == (0,1,1)
    assert vocabulary.quantize(np.full(32,255,np.uint8),levelsup=1) == (3,0,2)


def test_tfidf_normalization_stopped_words_and_feature_indices(tmp_path):
    vocabulary = make_vocabulary(tmp_path)
    descriptors = np.array([[0]*32,[15]*32,[15]*32,[255]*32],np.uint8)
    bow,nodes = vocabulary.transform(descriptors,levelsup=1)
    assert bow == {0:0.2,1:0.8}
    assert nodes == {1:[0,1,2]}
    assert vocabulary.score(bow,bow) == pytest.approx(1)
    assert vocabulary.score(bow,{0:1.0}) == pytest.approx(0.2)
    assert vocabulary.transform(np.empty((0,32),np.uint8)) == ({},{})


@pytest.mark.parametrize("scoring",range(6))
def test_all_dbow2_scoring_formulas(tmp_path,scoring):
    vocabulary = make_vocabulary(tmp_path,scoring=scoring)
    descriptors = np.array([[0]*32,[15]*32],np.uint8)
    bow,_ = vocabulary.transform(descriptors)
    score = vocabulary.score(bow,bow)
    if scoring == 3:
        assert score == pytest.approx(0)
    elif scoring == 5:
        assert score == pytest.approx(1.25)  # weights / distinct word count
    else:
        assert score == pytest.approx(1)


def test_idf_counts_each_word_once(tmp_path):
    vocabulary = make_vocabulary(tmp_path,weighting=2)
    bow,_ = vocabulary.transform(np.array([[0]*32,[15]*32,[15]*32],np.uint8))
    assert bow == {0:pytest.approx(1/3),1:pytest.approx(2/3)}


def test_inverted_candidates_accumulate_covisibility_and_split_maps(tmp_path):
    database = InvertedDatabase(make_vocabulary(tmp_path))
    database.add(0,{0:0.6,1:0.4},0,neighbors=[1])
    database.add(1,{0:0.3,1:0.7},0,neighbors=[0])
    database.add(2,{0:0.5,1:0.5},1)
    database.add(3,{0:1.0},0)
    loops,merges = database.detect_n_best({0:0.5,1:0.5},0,count=3)
    assert loops == [0]  # duplicate representative from both covisible clusters
    assert merges == [2]
    assert database.relocalization_candidates({0:0.5,1:0.5},0) == [0]
    database.entries[2].bad = True
    assert database.detect_n_best({0:0.5,1:0.5},0,connected=[0,1])[1] == []
    database.clear_map(0)
    assert list(database.entries) == [2]
    database.erase(2)
    assert not database.inverted


def test_invalid_tree_rejected(tmp_path):
    path = tmp_path/"broken.txt"
    path.write_text("2 2 0 0\n0 0 "+" ".join(["0"]*32)+" 1\n")
    with pytest.raises(ValueError,match="no words"):
        Vocabulary.from_text(path)


# These C++ routines are independent test oracles, never package dependencies.
import os
import shutil
import subprocess

_CPP_VOCABULARY_ORACLE = r'''
#include "ORBVocabulary.h"
#include <fstream>
#include <iomanip>
int main(int argc,char**argv) {
    ORB_SLAM3::ORBVocabulary vocabulary;
    if(!vocabulary.loadFromTextFile(argv[1])) return 2;
    std::ifstream input(argv[2]); int count; input >> count;
    std::vector<cv::Mat> descriptors;
    for(int i=0;i<count;++i) {
        cv::Mat descriptor(1,32,CV_8U);
        for(int j=0;j<32;++j) {int v; input >> v; descriptor.at<uchar>(0,j)=v;}
        descriptors.push_back(descriptor);
    }
    DBoW2::BowVector bow; DBoW2::FeatureVector nodes;
    vocabulary.transform(descriptors,bow,nodes,4);
    std::ofstream out(argv[3]); out << std::setprecision(17);
    out << vocabulary.size() << '\n';
    for(auto& p:bow) out << "B " << p.first << ' ' << p.second << '\n';
    for(auto& p:nodes) {
        out << "F " << p.first;
        for(auto i:p.second) out << ' ' << i;
        out << '\n';
    }
}
'''


@pytest.mark.skipif(not os.environ.get("ORBSLAM3_REFERENCE"),
                    reason="set ORBSLAM3_REFERENCE for original trained-vocabulary comparison")
def test_original_cpp_trained_vocabulary_on_identical_descriptors(tmp_path):
    reference = Path(os.environ["ORBSLAM3_REFERENCE"]).resolve()
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required for the test-only source oracle")
    prefix = reference/".venv"
    source, executable = tmp_path/"oracle.cc", tmp_path/"oracle"
    source.write_text(_CPP_VOCABULARY_ORACLE)
    subprocess.run([compiler,"-std=c++17","-O2",f"-I{reference/'include'}",
                    f"-I{reference}",f"-I{prefix/'include'}",f"-I{prefix/'include/opencv4'}",
                    str(source),f"-L{reference/'Thirdparty/DBoW2/lib'}",f"-L{prefix/'lib'}",
                    f"-Wl,-rpath,{reference/'Thirdparty/DBoW2/lib'}",f"-Wl,-rpath,{prefix/'lib'}",
                    "-lDBoW2","-lopencv_core","-o",str(executable)],check=True)
    descriptors = np.random.default_rng(12).integers(0,256,(1000,32),dtype=np.uint8)
    descriptor_path, result_path = tmp_path/"descriptors.txt", tmp_path/"results.txt"
    np.savetxt(descriptor_path,descriptors,fmt="%d",header=str(len(descriptors)),comments="")
    asset = reference/"Vocabulary/ORBvoc.txt"
    subprocess.run([str(executable),str(asset),str(descriptor_path),str(result_path)],check=True)
    reference_bow,reference_nodes = {},{}
    lines = result_path.read_text().splitlines()
    for line in lines[1:]:
        fields = line.split()
        if fields[0] == "B":
            reference_bow[int(fields[1])] = float(fields[2])
        else:
            reference_nodes[int(fields[1])] = list(map(int,fields[2:]))
    vocabulary = Vocabulary.from_text(asset)
    bow,nodes = vocabulary.transform(descriptors)
    # The source EOF loop appends a phantom node for a terminal newline.
    # Count valid leaves, rather than reproducing uninitialized C++ fields.
    assert len(vocabulary)+1 == int(lines[0])
    assert bow.keys() == reference_bow.keys()
    np.testing.assert_allclose(list(bow.values()),list(reference_bow.values()),atol=1e-15,rtol=1e-13)
    assert nodes == reference_nodes


def test_saved_original_cpp_vocabulary_fixture():
    fixture = np.load(Path(__file__).parent/"fixtures/vocabulary_cpp.npz",allow_pickle=False)
    asset = Path(__file__).resolve().parents[1]/"assets/ORBvoc.txt"
    vocabulary = Vocabulary.from_text(asset)
    bow,nodes = vocabulary.transform(fixture["descriptors"])
    assert len(vocabulary) == fixture["valid_leaf_count"]
    assert fixture["reported_cpp_word_count"] == len(vocabulary)+1  # EOF-parser defect, documented
    np.testing.assert_array_equal(list(bow),fixture["word_ids"])
    np.testing.assert_allclose(list(bow.values()),fixture["weights"],atol=1e-15,rtol=1e-13)
    np.testing.assert_array_equal(list(nodes),fixture["node_ids"])
    np.testing.assert_array_equal([len(indices) for indices in nodes.values()],fixture["node_counts"])
    np.testing.assert_array_equal(np.concatenate(list(nodes.values())),fixture["feature_indices"])
