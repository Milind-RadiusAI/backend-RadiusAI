#ifndef HMM_COMPILER_H
#define HMM_COMPILER_H

#include <torch/torch.h>
#include <torch/script.h>
#include "ViterbiModel.h"
#include <deque>
#include <map>
#include <vector>

class HMM_Compiler {
public:
    HMM_Compiler();
    torch::Tensor get_state(std::map<std::string, torch::Tensor> datadict);
    std::tuple<torch::Tensor, torch::Tensor> get_state_and_edge(const std::map<std::string, torch::Tensor> datadict);
    torch::Tensor datadict_to_sequence_discrete(const std::map<std::string, torch::Tensor> datadict);
    Eigen::MatrixXi torchToEigen(const torch::Tensor& tensor);
    torch::Tensor eigenToTorch(Eigen::VectorXi& eigen_matrix);

private:
    ViterbiModel load_hmm();
    int del_idx;
    //void add_to_list(std::deque<torch::Tensor>& src_list, torch::Tensor data);
    int movement_average_window;
    double movement_thresh;
    double distance_thresh;
    size_t memory;
    torch::Tensor prior_states;
    ViterbiModel hmm;
    std::map<std::string, torch::Tensor> prior_datalist;
    void add_to_list(torch::Tensor& src_tensor, torch::Tensor data);

};

#endif // HMM_COMPILER_H
