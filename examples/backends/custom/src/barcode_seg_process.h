#ifndef BARCODE_SEG_PROCESS_H
#define BARCODE_SEG_PROCESS_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <array>
#include <unordered_set>
#include <chrono>
#include "utils.h"


class BarcodeSegProcess : public BaseModel{
    public:
        BarcodeSegProcess();

        void initialize() override;

        void execute(
                const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes,
                const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,
                std::vector<std::vector<std::int64_t>>&  output_shapes) override;

    private:

        cv::Point2i rotate(const cv::Point2i& p, const cv::Point2i& origin = cv::Point2i(0, 0), float degrees = 0);

        std::array<cv::Point2i, 4> rotate_box(const std::array<cv::Point2i, 4>& box, float angle, int pad, const cv::Point2i& origin = cv::Point2i(0,0));

        torch::Tensor ocr_transform(const cv::Mat& mat_image);

        cv::Mat tensorToMat(const torch::Tensor& tensor);

        cv::Mat crop_rectangle(const cv::Mat& image, const std::tuple<cv::Point, cv::Size, int>& rect);

        cv::Point2f findCentroid(const std::array<cv::Point2f, 4>& box);

        cv::Vec4f shortestCalculator(const std::array<cv::Point2f, 4>& box);

        std::array<cv::Point2i, 4> shift_box_to_crop(std::array<cv::Point2f, 4>& box, int x=0, int y=0);

        std::tuple<cv::Point, cv::Size, int> rect_bbx(const std::array<cv::Point2i, 4>& box);

        cv::Mat rotate_image(cv::Mat& image, double angle, int pad);

        std::array<cv::Point2f, 4> rescale_box(const std::array<cv::Point2f, 4>& bbox, const torch::Tensor& barcode_box, const std::array<int, 2>& src_dim);

        std::pair<float, std::array<cv::Point2f, 4>>  binaryMaskProcess(const cv::Mat& bMask);

        void batchedRotation(torch::Tensor& liveImg, std::vector<cv::Mat>& seg, torch::Tensor&  barcode_indices, torch::Tensor& scaled_barcode_boxes, std::vector<int64_t>& src_image_index,torch::Tensor& images_4k, torch::Tensor& rotList,std::vector<int>& seg_index);

        std::vector<int64_t> get_src_image_list(const torch::Tensor& num_dets);

        std::vector<cv::Mat> process_det_outputs(torch::Tensor org_boxes, torch::Tensor boxes, at::Tensor mask_coefficients, at::Tensor proto);

};

#endif 