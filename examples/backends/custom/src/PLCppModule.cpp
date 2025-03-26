#include "PLCppModule.h"
#include "hmm.h"
#include "utils.h"
#include <iostream>
#include <ctime>
#include <chrono>

void PLCppModule::profile(std::string name, bool status) {
    if(name == "print") {
        std::ofstream logFile("/data/repos/profile_projection_logic.txt", std::ios::app);
        if (logFile.is_open()) {
            for(const auto &p:profile_info) {
                std::string log = "";
                log += p.first + ": ";
                log += std::to_string(p.second.second - p.second.first) + "ms";
                logFile << log << std::endl;
            }

            logFile.close();
        } else {
            std::cerr << "Unable to open log file" << std::endl;
        }

        return;
    }
    
    auto itr = profile_info.find(name);
    if(itr == profile_info.end()) {
        profile_info[name] = {0, 0};
    }

    long long milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if(status == 0) {
        profile_info[name].first = milliseconds_since_epoch;
    } else {
        profile_info[name].second = milliseconds_since_epoch;
    }
}

template<typename T>
T* copyMatToHeap(const cv::Mat& mat) {
    // Ensure the matrix is continuous
    cv::Mat continuous = mat.isContinuous() ? mat : mat.clone();
    
    // Allocate memory on the heap
    T* heapData = new T[continuous.total() * continuous.channels()];
    
    // Copy data from matrix to heap
    std::memcpy(heapData, continuous.data, continuous.total() * continuous.channels() * sizeof(T));
    
    return heapData;
}

std::vector<int64_t> getMatShapes(cv::Mat mat) {
    std::vector<int64_t> shapes;
    for(int i=0; i<mat.dims; i++) {
        shapes.push_back(static_cast<int64_t>(mat.size[i]));
    } return shapes;
}

void PLCppModule::deserialize_inputs_to_cv(
    const std::vector<float*>& dataPointers,
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string> &data_types,
    std::vector<cv::Mat> &deserializedDatasets
) {
    for(std::size_t i=0; i<dataPointers.size(); i++) {
        const std::vector<std::int64_t> &currentShape = shapes[i];
        std::vector<int> cv_shape_vec(currentShape.begin(), currentShape.end());
        const int* cv_shape = cv_shape_vec.data();
        int cv_dim = cv_shape_vec.size();

        int cv_type = CV_8U; // Default to 8-bit Unsigned
        if (data_types[i] == "TYPE_UINT8") {
            cv_type = CV_8U; // Unsigned 8-bit
        } else if (data_types[i] == "TYPE_INT8") {
            cv_type = CV_8S; // Signed 8-bit
        } else if (data_types[i] == "TYPE_INT32") {
            cv_type = CV_32S; // Signed 32-bit
        } else if (data_types[i] == "TYPE_FP32") {
            cv_type = CV_32F; // Float 32-bit
        } else if (data_types[i] == "TYPE_INT64") {
            throw std::invalid_argument("OpenCV does not support 64-bit integer");
        }

        // creat n-dimensional opencv mat from raw pointer
        profile("chw_read_"+std::to_string(i), 0);
        cv::Mat bchw_mat(cv_dim, cv_shape, cv_type, static_cast<void*>(dataPointers[i]));
        profile("chw_read_"+std::to_string(i), 1);
        cv::Mat bhwc_mat;

        // std::vector<int> order = {0, 2, 3, 1};

        // profile("hwc_transpose_"+std::to_string(i), 0);
        // cv::transposeND(bchw_mat, order, bhwc_mat);
        // profile("hwc_transpose_"+std::to_string(i), 1);
        deserializedDatasets.push_back(bchw_mat);
    }
}

void PLCppModule::execute(const std::vector<float*>& dataPointers, const std::vector<std::vector<std::int64_t>>& shapes, const std::vector <std::string>& data_types, std::vector<const void*>& outputPointers, std::vector<std::vector<std::int64_t>>& output_shapes) {
    std::vector<cv::Mat> all_inputs;
    profile("deserialize_inputs", 0);
    deserialize_inputs_to_cv(dataPointers,shapes,data_types,all_inputs);
    profile("deserialize_inputs", 1);

    int num_images = all_inputs[0].size[0] - 1;
    std::vector<cv::cuda::Stream> streams(num_images);
    std::vector<cv::cuda::GpuMat> images_gpu(num_images);
    std::vector<cv::cuda::GpuMat> images_reformatted(num_images);
    for(int i=0;i<num_images;i++) {
        images_reformatted[i] = cv::cuda::GpuMat(img_h, img_w, CV_8UC3);
    }
    std::vector<cv::cuda::GpuMat> resized_images(num_images);
    std::vector<cv::cuda::GpuMat> polymask_gpu(num_images);
    std::vector<cv::cuda::GpuMat> bg_masks_gpu(num_images);
    std::vector<cv::cuda::GpuMat> thresh_images_gpu(num_images);
    std::vector<cv::cuda::GpuMat> filter_images_gpu(num_images);
    // std::vector<cv::Mat> thresholded_images;
    // std::vector<float> contour_areas;
    cv::Mat batch_input = all_inputs[0];
    // int batch_size = batch_input.size[0];
    int height = batch_input.size[1];
    int width = batch_input.size[2];
    int channels = batch_input.size[3];

    // std::string log = "";
    // log += "BATCH: " + std::to_string(num_images) + ", HEIGHT: " + std::to_string(height) + ", WEIGHT: " + std::to_string(width) + ", CHANNELS: " + std::to_string(channels);
    // writeToLog(log);

    profile("start_all_streams", 0);
    for(int i=0;i<num_images;i++) {
        // get single image
        cv::Mat single_image(height, width, CV_8UC(channels), batch_input.ptr<uchar>(i));

        // transfer image to gpu
        images_gpu[i].upload(single_image, streams[i]);    
        // print_shape(images_gpu[i], "images_gpu");

        // convert from HWC -> CHW
        size_t width = images_gpu[i].cols * images_gpu[i].rows;
        std::vector<cv::cuda::GpuMat> input_channels(3);
        for(int k=0;k<3;k++) {
            input_channels[i] = cv::cuda::GpuMat(images_gpu[i].rows, images_gpu[i].cols, CV_8U, images_reformatted[i].ptr()[width * k]);
        }
        cv::cuda::split(images_gpu[i], input_channels, streams[i]);

        // resize
        cv::cuda::resize(images_reformatted[i], resized_images[i], cv::Size(), 0.25, 0.25, cv::INTER_LINEAR, streams[i]);
        // print_shape(resized_images[i], "resized_images");

        // print_shape(mask_gpu[i], "mask_gpu");
        // apply mask
        // std::string log = "";
        // log += "resized_images type: " + cv::typeToString(resized_images[i].depth());
        // log += ", mask_gpu type: " + cv::typeToString(mask_gpu[i].depth());
        // log += ", resized_images channels: " + std::to_string(resized_images[i].channels());
        // log += ", mask_gpu channels: " + std::to_string(mask_gpu[i].channels());
        // writeToLog(log);
        cv::cuda::multiply(resized_images[i], mask_gpu[i], polymask_gpu[i], 1, -1, streams[i]);
        // print_shape(polymask_gpu[i], "polymask_gpu");

        // apply background subtraction
        bg_subs[i]->apply(polymask_gpu[i], bg_masks_gpu[i], -1, streams[i]);
        // print_shape(bg_masks_gpu[i], "bg_masks_gpu");

        // thresholding
        cv::cuda::threshold(bg_masks_gpu[i], thresh_images_gpu[i], 40, 255, cv::THRESH_BINARY, streams[i]);
        // print_shape(thresh_images_gpu[i], "thresh_images_gpu");

        // filter morphology
        filter_morphology->apply(thresh_images_gpu[i], filter_images_gpu[i], streams[i]);
        // print_shape(filter_images_gpu[i], "filter_images_gpu");
    } profile("start_all_streams", 1);

    std::vector<cv::Mat> results;
    std::vector<double> countour_areas;
    profile("stream_completion_contour", 0);
    for(int i=0;i<num_images;i++) {
        streams[i].waitForCompletion();

        cv::Mat result;
        filter_images_gpu[i].download(result);
        results.push_back(result);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt) * 4; // Multiplying by 4 as in Python code
            countour_areas.push_back(area);
        }       
    } profile("stream_completion_contour", 1);

    double max_area = 0;
    if (!countour_areas.empty()) {
        max_area = *std::max_element(countour_areas.begin(), countour_areas.end());
    }
    
    profile("output_formatting", 0);
    int sizes[3] = {num_images, (int)img_h/4, (int)img_w/4};
    cv::Mat thresholded_images(3, sizes, results[0].type());
    for(int i=0;i<num_images;i++) {
        cv::Mat slice((int)img_h/4, (int)img_w/4, results[i].type(), thresholded_images.ptr(i));
        results[i].copyTo(slice);
    }

    output_shapes.push_back({1});
    output_shapes.push_back(getMatShapes(thresholded_images));

    // for(auto shapes:output_shapes) {
    //     std::string log = "";
    //     for(auto dim:shapes) {
    //         log += std::to_string(dim) + ",";
    //     } writeToLog(log);
    // }

    double* out_area = new double;
    *out_area = max_area;
    outputPointers.push_back(static_cast<const void*>(out_area));
    outputPointers.push_back(copyMatToHeap<uint8_t>(thresholded_images));
    profile("output_formatting", 1);
    profile("print", 0);
}
