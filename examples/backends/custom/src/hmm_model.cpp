#include <hmm_model.h>

hmmModel::hmmModel() : averager(static_cast<int64_t>(1)) {
    init_emptytabledict();
}

void hmmModel::init_emptytabledict() {
    del_idx = 0;
    std::ifstream json_file("/opt/triton_custom/backends/recommended/empty_desk_dict_store17-orin1data.json");
    nlohmann::json refdict_json;
    json_file >> refdict_json;
    json_file.close();
    std::string first_key = refdict_json.begin().key();
    for (const auto& key : emptykeyoptions) {
        auto data = refdict_json[first_key].get<std::vector<std::vector<float>>>();

        torch::Tensor tensor = torch::empty({static_cast<int64_t>(data.size()), static_cast<int64_t>(data[0].size())});
        for (size_t i = 0; i < data.size(); ++i) {
            for (size_t j = 0; j < data[i].size(); ++j) {
                tensor[i][j] = data[i][j];
            }
        }
        refdict[key] = tensor;
        
    }
}

torch::Tensor hmmModel::process_hmm_frame(
    const torch::Tensor& feature,
    const torch::Tensor& det_boxes_hand_const,
    const torch::Tensor& num_det_hand,
    const torch::Tensor& det_boxes_rpn_const, 
    const torch::Tensor& num_det_rpn
) {
    

    std::vector<int64_t> original_image_shape = {1080, 1920};
    std::vector<int64_t> resized_image_shape = {384, 640};
    float gain = std::min(
        static_cast<float>(resized_image_shape[0]) / original_image_shape[0],
        static_cast<float>(resized_image_shape[1]) / original_image_shape[1]
    ); 
    std::vector<float> pad = {
        (resized_image_shape[1] - original_image_shape[1] * gain) / 2,
        (resized_image_shape[0] - original_image_shape[0] * gain) / 2
    }; // wh padding
    bool presence_bool = num_det_hand.item<int64_t>() > 0;
    torch::Tensor presence = torch::tensor(presence_bool, torch::dtype(torch::kBool));
    
    if (num_det_hand.item<int64_t>() > 0) {
        torch::Tensor det_boxes_hand = det_boxes_hand_const.clone().slice(0, 0, num_det_hand.item<int64_t>());
    }


    torch::Device device(torch::kCUDA);

    torch::Tensor proposal_frame = torch::zeros({1080, 1920}, torch::TensorOptions().device(device));


    torch::Tensor det_boxes_rpn = det_boxes_rpn_const.clone().slice(0, 0, num_det_rpn.item<int64_t>());
    torch::Tensor index_0 = torch::tensor({0}, torch::dtype(torch::kFloat32));
    torch::Tensor index_1 = torch::tensor({1}, torch::dtype(torch::kFloat32));
    torch::Tensor index_2 = torch::tensor({2}, torch::dtype(torch::kFloat32));
    torch::Tensor index_3 = torch::tensor({3}, torch::dtype(torch::kFloat32));

    det_boxes_rpn.index({torch::indexing::Slice(), 0}) -= pad[0];
    det_boxes_rpn.index({torch::indexing::Slice(), 1}) -= pad[1];
    det_boxes_rpn.index({torch::indexing::Slice(), 2}) -= pad[0];
    det_boxes_rpn.index({torch::indexing::Slice(), 3}) -= pad[1];

    torch::Tensor det_boxes_rescaled_norm = det_boxes_rpn.clone();
    det_boxes_rescaled_norm.index({torch::indexing::Slice(), 0}) /= gain;
    det_boxes_rescaled_norm.index({torch::indexing::Slice(), 1}) /= gain;
    det_boxes_rescaled_norm.index({torch::indexing::Slice(), 2}) /= gain;
    det_boxes_rescaled_norm.index({torch::indexing::Slice(), 3}) /= gain;
    // Initialize the GEOS library
    initGEOS(nullptr, nullptr);
    GEOSWKTReader* reader = GEOSWKTReader_create();

    for (int i = 0; i < det_boxes_rescaled_norm.size(0); ++i) {
        int x1 = det_boxes_rescaled_norm[i][0].item<int>();
        int y1 = det_boxes_rescaled_norm[i][1].item<int>();
        int x2 = det_boxes_rescaled_norm[i][2].item<int>();
        int y2 = det_boxes_rescaled_norm[i][3].item<int>();
        const char* wkt_polygon = "POLYGON((972 1068, 656 1067, 404 520, 971 300, 1475 589, 972 1068))";
        GEOSGeometry* polygon = GEOSWKTReader_read(reader, wkt_polygon);
        double polygon_area;
        GEOSArea(polygon, &polygon_area);  
        
        std::stringstream rect_stream;
        rect_stream << "POLYGON((" << x1 << " " << y1 << ", "
                    << x2 << " " << y1 << ", "
                    << x2 << " " << y2 << ", "
                    << x1 << " " << y2 << ", "
                    << x1 << " " << y1 << "))";
        std::string rect_wkt = rect_stream.str();

        GEOSGeometry* rect = GEOSWKTReader_read(reader, rect_wkt.c_str());
        GEOSGeometry* intersectionGeometry = GEOSIntersection(polygon, rect);

        double intersection_area;
        GEOSArea(intersectionGeometry, &intersection_area);
        double rect_area;
        GEOSArea(rect, &rect_area);   
        double iou = intersection_area / rect_area;
        
        if (iou > 0) {
            auto y1_int = static_cast<int>(y1);
            auto y2_int = static_cast<int>(y2);
            auto x1_int = static_cast<int>(x1);
            auto x2_int = static_cast<int>(x2);
            proposal_frame.slice(0, y1_int, y2_int).slice(1, x1_int, x2_int).fill_(255);
        }
        // Clean up
        GEOSGeom_destroy(polygon);
        GEOSGeom_destroy(rect);
        GEOSGeom_destroy(intersectionGeometry);

    }

    std::map<std::string, torch::Tensor> data = {
            {"Hands", presence},
            {"Feats", feature},
            {"Proposal Count", num_det_rpn},
            {"Proposal", proposal_frame},
    };


    

    std::map<std::string, torch::Tensor> calc_data = calc_move_dist_hmm(data);




    feats["Feats"] = torch::cat({feats["Feats"],feature},0);
    if(del_idx == 0){
        feats["Feats"] = feats["Feats"].narrow(0, 1, feats["Feats"].size(0) - 1);
        del_idx = 1;
    }
    if(feats["Feats"].size(0) > metadata_memory){
        feats = trim_memory_data(feats);
    }
    auto time_since_update = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - ref_time).count();
    if(time_since_update > table_update_time){
        add_past_to_refdict();
        ref_time = std::chrono::steady_clock::now();
    }

    return calc_data["State"];
}

void hmmModel::add_past_to_refdict() {
    // Convert the tensor to an OpenCV matrix
    auto tensor_data = feats["Feats"];
    cv::Mat points(tensor_data.size(0), tensor_data.size(1), CV_32F, tensor_data.data_ptr<float>());
    int K = 10;
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(points, K, labels, 
        cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0), 
        3, cv::KMEANS_PP_CENTERS, centers);
    // Transpose back the centers matrix to fit Armadillo's expectation
    // Convert centers to Armadillo mat for further processing
    arma::mat arma_centers(static_cast<int>(centers.rows), static_cast<int>(centers.cols));
    for(int i = 0; i < centers.rows; ++i)
        for(int j = 0; j < centers.cols; ++j)
            arma_centers(i, j) = centers.at<float>(i, j);

    // Convert labels to Armadillo row vector
    arma::Row<size_t> assignments(labels.rows);
    for (int i = 0; i < labels.rows; ++i) {
        assignments[i] = labels.at<int>(i, 0);
    }

    // Proceed to find the largest cluster and rotate emptydict
    arma::mat center = get_largest_cluster(assignments, arma_centers);
    rotate_emptydict(center);
}


arma::mat hmmModel::get_largest_cluster(const arma::Row<size_t>& labels, const arma::mat& centroids) {
    // Create a map to count occurrences of each label
    std::map<size_t, size_t> counter;
    for (size_t label : labels) {
        counter[label]++;
    }

    // Find the label with the maximum count
    size_t maxCount = 0;
    size_t maxIdx = 0;  
    for (const auto& pair : counter) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            maxIdx = pair.first;
        }
    }

    return centroids.row(maxIdx);
}

void hmmModel::rotate_emptydict(arma::mat added_data) {
    arma::fmat added_data_float = arma::conv_to<arma::fmat>::from(added_data);
    torch::Tensor temp = torch::from_blob(added_data_float.memptr(), 
                                            {1, 1000}, 
                                            torch::kFloat32);
    for (size_t i = 1; i < emptykeyoptions.size(); ++i) {
        const std::string& key = emptykeyoptions[i];
        torch::Tensor temp2 = refdict[key].clone();
        refdict[key] = temp.clone();
        temp = temp2.clone();
    }
}

std::map<std::string, torch::Tensor> hmmModel::trim_memory_data(std::map<std::string, torch::Tensor> datadict) {
    for (auto& pair : datadict) {
        pair.second = pair.second.narrow(0, 1, pair.second.size(0) - 1);
    }
    return datadict;
}

float hmmModel::mse() {
    auto base_img = proposals[proposal_idx].unsqueeze(0);

    auto mse_values = torch::mse_loss(base_img, proposals, torch::Reduction::None);

    auto mse_sum = mse_values.sum().item<float>();

    auto element_count = base_img.numel();
    float mse_value = static_cast<float>(mse_sum / static_cast<double>(element_count));

    return mse_value;
}


std::map<std::string, torch::Tensor> hmmModel::calc_move_dist_hmm(std::map<std::string, torch::Tensor> framedata){

    framedata["Proposal"] = framedata["Proposal"].to(torch::kCUDA);

    torch::Tensor feat = framedata["Feats"].index({0});
    proposals[proposal_idx] =  framedata["Proposal"];

    auto sum_bool = proposals.index({torch::indexing::Slice(-1)}).sum().item<float>() > 0.0;
    bool proposal_static = false;

    if (proposals.size(0) == 5 && sum_bool) {
         float total_mse = mse();
        
     if (total_mse < (300 * framedata["Proposal Count"].item<float>())) {
            proposal_static = true;
        }
    }

    proposal_idx = (proposal_idx + 1) % 5; 
    torch::Tensor proposal_static_tensor = torch::tensor(proposal_static, torch::dtype(torch::kBool));
    std::map<std::string, torch::Tensor> m = {
        {"Movement", get_averaged_movement(feat).squeeze(0)},
        {"Distances", get_distance(feat)},
        {"Hands", framedata["Hands"]},
        {"Proposal Count", framedata["Proposal Count"]},
        {"Proposal", proposal_static_tensor},
    };

    std::tuple<torch::Tensor, torch::Tensor> state_edge = hmm.get_state_and_edge(m);

    m["State"] = std::get<0>(state_edge); 
    m["Edges"] = std::get<1>(state_edge);
    return m;
}


torch::Tensor hmmModel::get_averaged_movement(torch::Tensor feat){
    torch::Tensor movement = get_movement(feat);    
    movement = averager.process(movement);

    return movement;
}

torch::Tensor hmmModel::get_movement(torch::Tensor feat){
    if(frame_prior.sum().item<int64_t>() == 0) {
        torch::Tensor movement = torch::zeros(1);
        frame_prior = feat.clone();
        return movement;
    }
    torch::Tensor movement = pairwise_distance(frame_prior.unsqueeze(0), feat.unsqueeze(0));
    frame_prior = feat.clone();
    
    return movement;
}

torch::Tensor hmmModel::pairwise_distance(const torch::Tensor& X, const torch::Tensor& Y) {

    torch::Tensor x_double = X.to(torch::kDouble);
    torch::Tensor y_double = Y.to(torch::kDouble);
    torch::Tensor X_norm = torch::nn::functional::normalize(x_double, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    torch::Tensor Y_norm = torch::nn::functional::normalize(y_double, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    torch::Tensor similarity = torch::matmul(X_norm, Y_norm.transpose(0, 1));

    torch::Tensor distance = 1 - similarity;
    distance = torch::clamp(distance, 0, 2);

    // If comparing vectors with themselves, set the diagonal to 0.0 to remove floating point errors
    if (x_double.is_same(Y) || !y_double.defined()) {
        distance.fill_diagonal_(0.0);
    }
    return distance.flatten(0);
}

torch::Tensor hmmModel::get_distance(const torch::Tensor& feat) {
    torch::Tensor dist_stack;
    for (const auto& pair : refdict) {
        auto distance = pairwise_distance(pair.second, feat.unsqueeze(0));
        if (dist_stack.numel() == 0) {
            dist_stack = distance;
        } else {
            dist_stack = torch::cat({dist_stack, distance}, 0);
        }
    }

    auto min_distance = std::get<0>(torch::min(dist_stack, 0));

    return min_distance;
}
