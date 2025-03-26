#ifndef PLCPPMODULE_H
#define PLCPPMODULE_H
#include <iostream>
#include <fstream>
#include <chrono>
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

class PLCppModule: public BaseModel {
public:
    void writeToLog(const std::string& message) {
        std::ofstream logFile("/data/repos/log_projection_logic.txt", std::ios::app);
        if (logFile.is_open()) {
            time_t now = time(0);
            char* dt = ctime(&now);
            logFile << dt << ": " << message << std::endl;
            logFile.close();
        } else {
            std::cerr << "Unable to open log file" << std::endl;
        }
    }

    void print_shape(const cv::cuda::GpuMat &mat, std::string name) {
        cv::Size matSize = mat.size();
        int channels = mat.channels();

        std::vector<int> shape = {matSize.height, matSize.width, channels};
        std::string log = name + ": ";
        for(auto dim:shape) {
            log += std::to_string(dim) + ", ";
        } writeToLog(log);
    }

    PLCppModule(std::vector<int64_t> &input_shape): conf_yaml(PipelineConf()) {
        img_h = input_shape[2];
        img_w = input_shape[3];

        // get poly coordinates from camera set up
        for(std::size_t i=0;i<conf_yaml.input_cam_sources.size();i++) {
            std::vector<std::pair<int, int>> cam_coords;
            for(auto itr = conf_yaml.input_cam_sources[i].begin(); itr != conf_yaml.input_cam_sources[i].end(); itr++) {
                for(const auto& point: itr->second["poly_coords"]) {
                    cam_coords.push_back({point[0].as<int>(), point[1].as<int>()});
                }
            } poly_coords.push_back(cam_coords);
        }

        // initialise background subtractors
        for(std::size_t i=0; i<conf_yaml.input_cam_sources.size(); i++) {
            bg_subs.push_back(cv::cuda::createBackgroundSubtractorMOG2(5));
        }

        // initialise mask from cam coordinates
        mask_gpu.resize(poly_coords.size());
        for(std::size_t i=0; i<poly_coords.size(); i++) {
            std::vector<cv::Point> poly_points;
            for(const auto& point:poly_coords[i]) {
                poly_points.emplace_back(point.first, point.second);
            }

            cv::Mat cam_mask = cv::Mat::zeros(img_h, img_w, CV_8UC3);
            cv::fillPoly(cam_mask, {poly_points}, cv::Scalar(1));

            cv::Mat cam_mask_resized;
            cv::resize(cam_mask, cam_mask_resized, cv::Size(), 0.25, 0.25);

            mask_gpu[i].upload(cam_mask_resized);
        }

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); // Adjust size as needed
        filter_morphology = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN, CV_8UC1, kernel);

    };
    void initialize() {};
    void execute(const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes, const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,std::vector<std::vector<std::int64_t>>&  output_shapes);
    void profile(std::string name, bool status);
private:
    PipelineConf conf_yaml;
    int64_t img_h;
    int64_t img_w;
    std::vector<std::vector<std::pair<int, int>>> poly_coords;
    std::vector<cv::Ptr<cv::cuda::BackgroundSubtractorMOG2>> bg_subs;
    cv::Ptr<cv::cuda::Filter> filter_morphology;
    std::vector<cv::cuda::GpuMat> mask_gpu;
    std::map<std::string, std::pair<long long int, long long int>> profile_info;
};

#endif 

