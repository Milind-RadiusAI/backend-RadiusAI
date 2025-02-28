#ifndef HMM_MODEL_H
#define HMM_MODEL_H

#include <utils.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>
#include <chrono>
#include <map>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <geos_c.h>
#include <HMM_Compiler.h>
#include <armadillo>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <StreamingMovingAverage.h>

class hmmModel {
public:
    StreamingMovingAverage averager;
    int del_idx = 0;
    //std::vector<torch::Tensor> proposals;
    torch::Tensor proposals = torch::zeros({5,1080, 1920}).to(torch::kCUDA);
    torch::Tensor frame_prior = torch::zeros({1,1000});
    hmmModel();
    std::map<std::string, torch::Tensor> feats = {
        {"Feats", torch::zeros({1,1000})}
    }  ;
    std::chrono::steady_clock::time_point ref_time = std::chrono::steady_clock::now();
    void init_emptytabledict();
    torch::Tensor process_hmm_frame(
    const torch::Tensor& feature,
    const torch::Tensor& det_boxes_hand_const,
    const torch::Tensor& num_det_hand,
    const torch::Tensor& det_boxes_rpn, 
    const torch::Tensor& num_det_rpn
    ) ;
    int proposal_idx = 0;
    void add_past_to_refdict();
    arma::mat get_largest_cluster(const arma::Row<size_t>& labels, const arma::mat& centroids);
    void rotate_emptydict(arma::mat added_data);
    std::map<std::string, torch::Tensor> trim_memory_data(std::map<std::string, torch::Tensor> datadict);
    float mse();
    std::map<std::string, torch::Tensor> calc_move_dist_hmm(std::map<std::string, torch::Tensor> framedata);
    torch::Tensor get_averaged_movement(torch::Tensor feat);
    torch::Tensor get_movement(torch::Tensor feat);
    torch::Tensor pairwise_distance(const torch::Tensor& a, const torch::Tensor& b);
    torch::Tensor get_distance(const torch::Tensor& feat);

private:
    int ref_camera;
    std::vector<std::pair<int, int>> cam_poly_coords;
    std::map<std::string, torch::Tensor> refdict;
    std::vector<std::string> emptykeyoptions= {"0", "t1", "t2", "t3", "t4", "t5"};
    HMM_Compiler hmm;
    double table_update_time = 300;
    float metadata_memory = 3600;
};

#endif 

