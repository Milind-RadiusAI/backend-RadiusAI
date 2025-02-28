#include "HMM_Compiler.h"

HMM_Compiler::HMM_Compiler() {
    movement_average_window = 1;
    movement_thresh = 0.02;
    distance_thresh = 0.12;
    memory = 3600;
    hmm = load_hmm();
    del_idx = 0;
    prior_datalist = {
        {"Movement", torch::tensor({0})},
        {"Distances", torch::tensor({0})},
        {"Hands", torch::tensor({0})},
        {"Proposal Count", torch::tensor({0})},
        {"Proposal", torch::tensor({0})}
    };
    prior_states = torch::tensor({0});

}

ViterbiModel HMM_Compiler::load_hmm() {
     ViterbiModel model;
    try {
        ViterbiModel model;
        return model;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        exit(-1);  // Exit the program with an error code
    }
}

torch::Tensor HMM_Compiler::get_state(std::map<std::string, torch::Tensor> data) {
    for (const auto& key_val : data) {
        //std::cout <<  key_val.first << " " << key_val.second << std::endl;
        const std::string& key = key_val.first;
        torch::Tensor last_element = key_val.second;//prior_datalist.at(key).index({-1});
        //std::cout <<  prior_datalist.at(key) << std::endl;
        add_to_list(prior_datalist.at(key), last_element);
    }
    if(del_idx == 0){
        for (auto& pair : prior_datalist) {
            pair.second = pair.second.narrow(0, 1, pair.second.size(0) - 1);
        }
        del_idx = 1;
    }
    auto seq = datadict_to_sequence_discrete(prior_datalist);
    std::chrono::steady_clock::time_point forward_time = std::chrono::steady_clock::now();
    auto seq_input = torchToEigen(seq);
    Eigen::VectorXi output_ivalue = hmm.viterbi(seq_input);
    std::chrono::steady_clock::time_point forward_time_end_time = std::chrono::steady_clock::now();
    auto duration_forward = std::chrono::duration_cast<std::chrono::milliseconds>(forward_time_end_time - forward_time);
    
    auto output_tensor = eigenToTorch(output_ivalue);
    auto state = output_tensor[-1].unsqueeze(0);
    add_to_list(prior_states, state);
    
    return state;
}

std::tuple<torch::Tensor, torch::Tensor> HMM_Compiler::get_state_and_edge(const std::map<std::string, torch::Tensor> datadict){
    torch::Tensor state = get_state(datadict);  
    bool edge = false;

    if (prior_states.size(0) > 1) {
        if (prior_states[prior_states.size(0) - 2].item<int64_t>() != 1 && 
            prior_states[prior_states.size(0) - 1].item<int64_t>() == 1) {
            edge = true;
        }
    }
    torch::Tensor edge_ret = torch::tensor(edge, torch::dtype(torch::kBool));
    torch::Tensor last_state = state.index({torch::indexing::Slice(-1)});
    return std::make_tuple(last_state, edge_ret);
}

void HMM_Compiler::add_to_list(torch::Tensor& src_tensor, torch::Tensor data) {
    data = data.view({-1});

    src_tensor = torch::cat({src_tensor, data}, 0);
    if ( src_tensor.size(0) > (int64_t) memory) {
        src_tensor = src_tensor.slice(0, 1);  // Remove the first element by slicing
    }
}

torch::Tensor HMM_Compiler::datadict_to_sequence_discrete(const std::map<std::string, torch::Tensor> datadict) {
    torch::Tensor movement, distances, hands, proposal_cnt, proposal;
    movement = torch::where(datadict.at("Movement") > movement_thresh, torch::full_like(datadict.at("Movement"), 0, torch::kInt), torch::full_like(datadict.at("Movement"), 1, torch::kInt));
    distances = torch::where(datadict.at("Distances") > distance_thresh, torch::full_like(datadict.at("Distances"), 0, torch::kInt), torch::full_like(datadict.at("Distances"), 1, torch::kInt));
    hands = torch::where(datadict.at("Hands") > 0.5, torch::full_like(datadict.at("Hands"), 0, torch::kInt), torch::full_like(datadict.at("Hands"), 1, torch::kInt));
    proposal_cnt = torch::where(datadict.at("Proposal Count") > 0.5, torch::full_like(datadict.at("Proposal Count"), 0, torch::kInt), torch::full_like(datadict.at("Proposal Count"), 1, torch::kInt));
    proposal = torch::where(datadict.at("Proposal") == 0, torch::full_like(datadict.at("Proposal"), 0, torch::kInt), torch::full_like(datadict.at("Proposal"), 1, torch::kInt));
    
    std::vector<torch::Tensor> tensors = {hands, movement, distances, proposal_cnt, proposal};
    torch::Tensor out = torch::stack(tensors, 1);

    return out;
}

Eigen::MatrixXi HMM_Compiler::torchToEigen(const torch::Tensor& tensor) {
    auto tensor_cpu_int = tensor.to(torch::kCPU, torch::kInt).contiguous().t();
    const int rows = tensor_cpu_int.size(0);
    const int cols = tensor_cpu_int.size(1);
    Eigen::MatrixXi eigen_matrix(rows, cols);

    memcpy(eigen_matrix.data(), tensor_cpu_int.data_ptr<int>(), rows * cols * sizeof(int));
    return eigen_matrix.transpose();
}

torch::Tensor HMM_Compiler::eigenToTorch(Eigen::VectorXi& eigen_matrix) {
    // Get the dimensions of the Eigen matrix
    const auto rows = eigen_matrix.rows();
    const auto cols = eigen_matrix.cols();

    // This tensor will share data with the Eigen matrix
    auto tensor = torch::from_blob(eigen_matrix.data(), {rows, cols}, torch::kInt);

    // If you want to create a tensor that owns its data, you can use torch::clone
    auto owning_tensor = tensor.clone();

    return owning_tensor;
}
