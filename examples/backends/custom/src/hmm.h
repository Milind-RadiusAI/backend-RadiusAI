#include <opencv2/opencv.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xmanipulation.hpp>
#include <array>
#include <unordered_set>
#include <chrono>
#include <xtensor/xadapt.hpp>
#include <variant>
#include <hmm_model.h>
#include "utils.h"

using namespace torch::indexing;
using ArrayVariant = std::variant<xt::xarray<float>, xt::xarray<uint8_t> , xt::xarray<int64_t> , xt::xarray<int8_t>, xt::xarray<int32_t>>;

class HMMCppModule : public BaseModel{
public:
    hmmModel model;
    torch::Tensor output;
    HMMCppModule() {
        // Constructor logic if needed
    }

    void initialize() {
        // Initialization logic here
    }

    void execute(const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes, const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,std::vector<std::vector<std::int64_t>>&  output_shapes ) {
        std::vector<torch::Tensor> all_inputs;
        deserialize_inputs(dataPointers,shapes,data_types,all_inputs);
        torch::Tensor feature = all_inputs[0];
        torch::Tensor detection_boxes_box_outputs_hand = all_inputs[1];
        torch::Tensor num_detections_box_outputs_hand = all_inputs[2];
        torch::Tensor detection_boxes_box_outputs_rpn = all_inputs[3];
        torch::Tensor num_detections_box_outputs_rpn = all_inputs[4];

        torch::Tensor det_boxes_hand = detection_boxes_box_outputs_hand[0];
        torch::Tensor num_det_hand = num_detections_box_outputs_hand.view(-1)[0];

        // Assuming self.ref_camera is an available index,
        torch::Tensor det_boxes_rpn = detection_boxes_box_outputs_rpn[0];
        torch::Tensor num_det_rpn = num_detections_box_outputs_rpn[0].view(-1)[0];
        
        output = model.process_hmm_frame(feature, det_boxes_hand, num_det_hand, det_boxes_rpn, num_det_rpn);


        torch::IntArrayRef sizes = output.sizes();
        std::vector<int64_t> output_sizes(sizes.begin(), sizes.end());
        output_shapes.push_back(output_sizes);
        
        uint32_t* heapMatData = new uint32_t[ output.numel()];

        std::memcpy(heapMatData, output.data_ptr(), output.numel() * sizeof(uint32_t));

        outputPointers.push_back(heapMatData);
    }
private:

void deserialize_inputs(
                    const std::vector<float*>& dataPointers,
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string>& data_types,
    std::vector<torch::Tensor>& deserializedDatasets)
    {
      torch::TensorOptions options_uint = torch::TensorOptions().dtype(torch::kUInt8);
      torch::TensorOptions options_float32 = torch::TensorOptions().dtype(torch::kFloat32);
      torch::TensorOptions options_int32 = torch::TensorOptions().dtype(torch::kInt32);
      torch::TensorOptions options_int64 = torch::TensorOptions().dtype(torch::kInt64);
      torch::TensorOptions options_int8 = torch::TensorOptions().dtype(torch::kInt8);

      for (std::size_t i = 0; i < dataPointers.size(); ++i) {
          const std::vector<std::int64_t>& currentShape = shapes[i];
          std::vector<std::int64_t> new_shape;
          // Calculate the total number of elements based on the shape
          std::int64_t num_elements = 1;
          for (auto& dim : currentShape) {
              num_elements *= dim; new_shape.push_back(dim);
          }
          if (data_types[i] == "TYPE_UINT8") {

            deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape,options_uint));
          }
          else if (data_types[i] == "TYPE_INT32") {
            deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape, options_int32));
          }
          else if (data_types[i] == "TYPE_FP32") {
            deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape, options_float32));
          }
          else if (data_types[i] == "TYPE_INT64") {
            deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape, options_int64));
          }
          else if (data_types[i] == "TYPE_INT8") {
            deserializedDatasets.push_back(torch::from_blob(dataPointers[i], new_shape, options_int8));
          }
          else{
                  continue;
          }
      }
    }

    void deserialize_float(
    const std::vector<float*>& dataPointers,
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string>& data_types,
    std::vector<xt::xarray<float>>& deserializedDatasets)
    {
	    for (std::size_t i = 0; i < dataPointers.size(); ++i) {
        const std::vector<std::int64_t>& currentShape = shapes[i];
        std::vector<std::int64_t> new_shape;
        // Calculate the total number of elements based on the shape
        std::int64_t num_elements = 1;
        for (auto& dim : currentShape) {
            if (dim == -1)
            {  num_elements *= 6; new_shape.push_back(6);}
            else
            {num_elements *= dim; new_shape.push_back(dim);}
        }
	if (data_types[i] == "TYPE_FP32") {
            //auto array = xt::adapt(reinterpret_cast<float*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            //deserializedDatasets.push_back(std::move(array));
            //deserializedDatasets.push_back(xt::adapt(reinterpret_cast<float*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape));
	    deserializedDatasets.push_back(xt::adapt(dataPointers[i], num_elements, xt::no_ownership(), new_shape));
	}
	else{
                continue;
        }
	    }

    }

    void deserialize_torchuint8(
		    const std::vector<float*>& dataPointers,
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string>& data_types,
    torch::Tensor& deserializedDatasets)
    {
	    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kUInt8);
	    for (std::size_t i = 0; i < dataPointers.size(); ++i) {
        const std::vector<std::int64_t>& currentShape = shapes[i];
        std::vector<std::int64_t> new_shape;
        // Calculate the total number of elements based on the shape
        std::int64_t num_elements = 1;
        for (auto& dim : currentShape) {
            if (dim == -1)
            {  num_elements *= 6; new_shape.push_back(6);}
            else
            {num_elements *= dim; new_shape.push_back(dim);}
        }
        if (data_types[i] == "TYPE_UINT8") {
		deserializedDatasets  = torch::from_blob(dataPointers[i], new_shape,options);
		//deserializedDatasets = tensor;
		break;
	}
        else{
                continue;
        }
	    }
    }

    void deserialize_uint8(
    const std::vector<float*>& dataPointers,
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string>& data_types,
    std::vector<xt::xarray<uint8_t>>& deserializedDatasets)
    {
    //        std::vector<xt::xarray<uint8_t>> deserializedDatasets;
            for (std::size_t i = 0; i < dataPointers.size(); ++i) {
        const std::vector<std::int64_t>& currentShape = shapes[i];
        std::vector<std::int64_t> new_shape;
        // Calculate the total number of elements based on the shape
        std::int64_t num_elements = 1;
        for (auto& dim : currentShape) {
            if (dim == -1)
            {  num_elements *= 6; new_shape.push_back(6);}
            else
            {num_elements *= dim; new_shape.push_back(dim);}
        }
	if (data_types[i] == "TYPE_UINT8") {
            auto array = xt::adapt(reinterpret_cast<uint8_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            deserializedDatasets.push_back(std::move(array));
//            deserializedDatasets.push_back(xt::adapt(reinterpret_cast<uint8_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape));
	}
	else{
		continue;
	}

            }
  //          return deserializedDatasets;

    }

    void deserialize_int32(
    const std::vector<float*>& dataPointers,
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string>& data_types,
    std::vector<xt::xarray<int32_t>>& deserializedDatasets)
    {
//            std::vector<xt::xarray<int32_t>> deserializedDatasets;
            for (std::size_t i = 0; i < dataPointers.size(); ++i) {
        const std::vector<std::int64_t>& currentShape = shapes[i];
        std::vector<std::int64_t> new_shape;
        // Calculate the total number of elements based on the shape
        std::int64_t num_elements = 1;
        for (auto& dim : currentShape) {
            if (dim == -1)
            {  num_elements *= 6; new_shape.push_back(6);}
            else
            {num_elements *= dim; new_shape.push_back(dim);}
        }
         if (data_types[i] == "TYPE_INT32") {
            //auto array = xt::adapt(reinterpret_cast<int32_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            //deserializedDatasets.push_back(std::move(array));
            deserializedDatasets.push_back(xt::adapt(reinterpret_cast<int32_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape));
	 }
	else{
                continue;
        }

            }
    //        return deserializedDatasets;

    }

    std::vector<ArrayVariant> deserialize_datasets(
    const std::vector<float*>& dataPointers, 
    const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector<std::string>& data_types) 
{
    std::vector<ArrayVariant> deserializedDatasets;

    // Ensure that the number of data pointers matches the number of shapes and data types
    assert(dataPointers.size() == shapes.size() && dataPointers.size() == data_types.size());

    for (std::size_t i = 0; i < dataPointers.size(); ++i) {
        const std::vector<std::int64_t>& currentShape = shapes[i];
        std::vector<std::int64_t> new_shape;
        // Calculate the total number of elements based on the shape
        std::int64_t num_elements = 1;
        for (auto& dim : currentShape) {
	    if (dim == -1)
	    {  num_elements *= 6; new_shape.push_back(6);}
	    else
	    {num_elements *= dim; new_shape.push_back(dim);}
        }
        // Depending on the data type, adapt the pointer to the corresponding xtensor type
        if (data_types[i] == "TYPE_FP32") {
            auto array = xt::adapt(reinterpret_cast<float*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            deserializedDatasets.push_back(std::move(xt::xarray<float>(array)));
        } 
        else if (data_types[i] == "TYPE_UINT8") {
            auto array = xt::adapt(reinterpret_cast<uint8_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            deserializedDatasets.push_back(std::move(xt::xarray<uint8_t>(array)));
        }
        else if (data_types[i] == "TYPE_INT64") {
            auto array = xt::adapt(reinterpret_cast<int64_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            deserializedDatasets.push_back(std::move(xt::xarray<int64_t>(array)));
        }
        else if (data_types[i] == "TYPE_INT8") {
            auto array = xt::adapt(reinterpret_cast<int8_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            deserializedDatasets.push_back(std::move(xt::xarray<int8_t>(array)));
        }
        else if (data_types[i] == "TYPE_INT32") {
            auto array = xt::adapt(reinterpret_cast<int32_t*>(dataPointers[i]), num_elements, xt::no_ownership(), new_shape);
            deserializedDatasets.push_back(std::move(xt::xarray<int32_t>(array)));
        }
        // ... add more data types as needed
        else {
            throw std::runtime_error("Unknown data type: " + data_types[i]);
        }
    }

    return deserializedDatasets;
}

    


    cv::Point2i rotate(const cv::Point2i& p, const cv::Point2i& origin = cv::Point2i(0, 0), float degrees = 0) {
    float angle = degrees * CV_PI / 180.0; // Convert degrees to radians
    cv::Matx22f R(std::cos(angle), -std::sin(angle),
                  std::sin(angle),  std::cos(angle));

    cv::Point2f pf(static_cast<float>(p.x), static_cast<float>(p.y));
    cv::Point2f originf(static_cast<float>(origin.x), static_cast<float>(origin.y));
    
    cv::Point2f rotatedPointf = R * (pf - originf) + originf;
    return cv::Point2i(static_cast<int>(rotatedPointf.x), static_cast<int>(rotatedPointf.y));
}

std::array<cv::Point2i, 4> rotate_box(const std::array<cv::Point2i, 4>& box, float angle, int pad, const cv::Point2i& origin = cv::Point2i(0,0)) {
    std::array<cv::Point2i, 4> new_box;
    
    for(int i = 0; i < 4; i++) {
        cv::Point2i paddedPoint = box[i] + cv::Point2i(pad, pad);
        new_box[i] = rotate(paddedPoint, origin, angle);
    }
    
    return new_box;
}

/*xt::xarray<float> ocr_transform(const cv::Mat& mat_image) {
    // Resize the image
    cv::Mat resized_image_mat;
    cv::resize(mat_image, resized_image_mat, cv::Size(128, 32), 0, 0, cv::INTER_CUBIC);
    
    // Convert resized cv::Mat to xt::xarray
    xt::xarray<uint8_t> resized_image = xt::adapt(static_cast<uint8_t*>(resized_image_mat.data), {3,32, 128});

    // Normalization (assuming the image is in the range 0-255)
    xt::xarray<float> normalized_image = (resized_image / 255.0 - 0.5) / 0.5;
    
    return normalized_image;
}*/

torch::Tensor ocr_transform(const cv::Mat& mat_image) {
    // Resize the image
    cv::Mat resized_image_mat;
    cv::resize(mat_image, resized_image_mat, cv::Size(128, 32), 0, 0, cv::INTER_CUBIC);

    // Convert resized cv::Mat to torch::Tensor
    torch::Tensor resized_image = torch::from_blob(resized_image_mat.data, {resized_image_mat.rows, resized_image_mat.cols, 3}, torch::kUInt8);

    // Convert to float, normalize and permute the tensor to CxHxW
    torch::Tensor normalized_image = (resized_image.permute({2, 0, 1}).contiguous().to(torch::kFloat32) / 255.0 - 0.5) / 0.5;

    return normalized_image;
}

cv::Mat tensorToMat(const torch::Tensor& tensor) {
    torch::Tensor tensor_tmp = tensor.cpu().to(torch::kFloat32);
    cv::Mat mat(tensor_tmp.sizes()[0], tensor_tmp.sizes()[1], CV_32F);
    std::memcpy(mat.data, tensor_tmp.data_ptr<float>(), tensor_tmp.numel() * sizeof(float));

    return mat;
}




cv::Mat crop_rectangle(const cv::Mat& image, const std::tuple<cv::Point, cv::Size, int>& rect) {
    int num_rows = image.rows;
    int num_cols = image.cols;

    cv::Point rect_center = std::get<0>(rect);
    int rect_center_x = rect_center.x;
    int rect_center_y = rect_center.y;
    int rect_width = std::get<1>(rect).width;
    int rect_height = std::get<1>(rect).height;

    int left = std::max(0, rect_center_y - rect_height / 2);
    int right = std::min(num_rows, rect_center_y + rect_height - rect_height / 2);
    int bottom = std::max(0, rect_center_x - rect_width / 2);
    int top = std::min(num_cols, rect_center_x + rect_width - rect_width / 2);


    top = std::max(0, top);
    top = std::min(image.cols, top);
    bottom = std::min(image.cols, bottom);
    bottom = std::max(0, bottom);
    left = std::min(image.rows, left);
    left = std::max(0, left);
    right = std::min(image.rows, right);
    right = std::max(0, right);
    

    cv::Rect roi(bottom, left,top - bottom, right - left);
    return image(roi);
}





/*
cv::Point2f findCentroid(cv::Point2f box[4]) {
    float x = (box[0].x + box[2].x) / 2.0f;
    float y = (box[0].y + box[2].y) / 2.0f;
    return cv::Point2f(x, y);
}
*/

cv::Point2f findCentroid(const std::array<cv::Point2f, 4>& box) {
    float x = (box[0].x + box[2].x) / 2.0f;
    float y = (box[0].y + box[2].y) / 2.0f;
    return cv::Point2f(x, y);
}

cv::Vec4f shortestCalculator(const std::array<cv::Point2f, 4>& box) {
    // Calculate sides using Euclidean distance
    double d1 = cv::norm(box[0] - box[1]);
    double d2 = cv::norm(box[1] - box[2]);

    if (d1 < d2) {
        return cv::Vec4f(box[0].x, box[0].y, box[1].x, box[1].y);
    } else {
        return cv::Vec4f(box[1].x, box[1].y, box[2].x, box[2].y);
    }
}

std::array<cv::Point2i, 4> shift_box_to_crop(std::array<cv::Point2f, 4>& box, int x=0, int y=0)
{
    std::array<cv::Point2i, 4> new_box;
    int count = 0;
    for (cv::Point2f& point : box) {
        new_box[count].x = (int)point.x - x;
        new_box[count].y = (int)point.y - y;
	count += 1;
    }
    return new_box;
}


std::tuple<cv::Point, cv::Size, int> rect_bbx(const std::array<cv::Point2i, 4>& box) {
    int x_max = static_cast<int>(std::max({box[0].x, box[1].x, box[2].x, box[3].x}));
    int x_min = static_cast<int>(std::min({box[0].x, box[1].x, box[2].x, box[3].x}));
    int y_max = static_cast<int>(std::max({box[0].y, box[1].y, box[2].y, box[3].y}));
    int y_min = static_cast<int>(std::min({box[0].y, box[1].y, box[2].y, box[3].y}));

    cv::Point center((x_min + x_max) / 2, (y_min + y_max) / 2);
    cv::Size dimensions(x_max - x_min, y_max - y_min);
    int angle = 0;
    return std::make_tuple(center, dimensions, angle);
}



cv::Mat rotate_image(cv::Mat& image, double angle, int pad)
{
    // If the image is empty, create an image of size 2*pad x 2*pad
    if(image.empty()) {
        cv::copyMakeBorder(image, image, pad, pad, pad, pad,cv::BORDER_CONSTANT);
    }else{

    cv::copyMakeBorder(image, image, pad, pad, pad, pad, cv::BORDER_REPLICATE);
    }
    cv::Point2f image_center(image.cols / 2.0, image.rows / 2.0);
    cv::Mat rot_mat = cv::getRotationMatrix2D(image_center, angle, 1.0);
    cv::Mat result;
    cv::warpAffine(image, result, rot_mat, image.size(), cv::INTER_CUBIC);
    return result;
}

std::array<cv::Point2f, 4> rescale_box(const std::array<cv::Point2f, 4>& bbox, const torch::Tensor& barcode_box, const std::array<int, 2>& src_dim)
{
    float dst_dim_w = barcode_box[2].item<float>() - barcode_box[0].item<float>();
    float dst_dim_h = barcode_box[3].item<float>() - barcode_box[1].item<float>();


    std::array<cv::Point2f, 4> transformed_bbox;

    for (int i = 0; i < 4; i++)
    {
        transformed_bbox[i].x = (bbox[i].x / src_dim[0]) * dst_dim_w + barcode_box[0].item<int>();
        transformed_bbox[i].y = (bbox[i].y / src_dim[1]) * dst_dim_h + barcode_box[1].item<int>();
    }

    return transformed_bbox;
}



std::pair<float, std::array<cv::Point2f, 4>>  binaryMaskProcess(const cv::Mat& bMask) {
    cv::Mat resized = bMask.clone();
    cv::Mat blurred;
    cv::blur(resized, blurred, cv::Size(3, 3));
    blurred = blurred * 255.0f ; 
    blurred.convertTo(blurred, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(blurred, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


    // Check if we found any contour
    if (!contours.empty()) {
        // Find the largest contour by area
        double maxArea = 0;
        int maxIdx = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIdx = i;
            }
        }
        
        // Get the minimum area rectangle
        cv::RotatedRect rect = cv::minAreaRect(contours[maxIdx]);

        // Get OBB points
        std::array<cv::Point2f, 4> box;
        rect.points(box.data()); 

        //cv::Vec4f shortestPoints = shortestCalculator(box); 
        cv::Point2f centroid = findCentroid(box);
        cv::Point2f imCenter(bMask.cols / 2.0f, bMask.rows / 2.0f);

        float mult = centroid.x < imCenter.x ? -1.0f : 1.0f;

        cv::Vec4f shortestSide = shortestCalculator(box);

        cv::Point2f firstPoint(shortestSide[0], shortestSide[1]);
        cv::Point2f secondPoint(shortestSide[2], shortestSide[3]);

        float rise = secondPoint.y - firstPoint.y;
        float run = secondPoint.x - firstPoint.x;
        float angle = std::atan2(rise, run) * 180.0f / CV_PI;
        float rotAngle = angle + (-90.0f) * mult;
        if (centroid.y < imCenter.y && std::abs(angle) >= 45.0f) {
            rotAngle += 180.0f * mult;
         }
             return std::make_pair(rotAngle, box);
     }
            std::array<cv::Point2f, 4> invalidBox = {};
            return std::make_pair(std::numeric_limits<float>::quiet_NaN(), invalidBox); 
}



void batchedRotation(torch::Tensor& liveImg, std::vector<cv::Mat>& seg, torch::Tensor&  barcode_indices, torch::Tensor& scaled_barcode_boxes, std::vector<int64_t>& src_image_index,torch::Tensor& images_4k, torch::Tensor& rotList,std::vector<int>& seg_index){

//auto start_orig = std::chrono::high_resolution_clock::now();
    // Use a vector of pairs to store the result
    int min_bbox_dim = 65 ;
//torch::Tensor rotList;
    //xt::xarray<float> rotList;
    //std::vector<int> seg_index;
bool first_crop = true;
rotList = torch::zeros({1, 3, 32, 128});
for (size_t i = 0; i < seg.size(); i++) 
    {
        auto [rotAngle, segbox] = binaryMaskProcess(seg[i]);

        // If the rotation angle is NaN, we continue to the next iteration
        if (std::isnan(rotAngle)) {
            continue;
        }
        //auto barcode_box = xt::view(scaled_barcode_boxes, i, xt::all());
        auto barcode_box = scaled_barcode_boxes[i];
        //float w = barcode_box(2) - barcode_box(0);
        //float h = barcode_box(3) - barcode_box(1);
        float w = barcode_box[2].item<float>() - barcode_box[0].item<float>();
        float h = barcode_box[3].item<float>() - barcode_box[1].item<float>();

        if ((h < min_bbox_dim) && (w < min_bbox_dim)) {
            continue;
        }

        if ((h < 10.0f) || (w < 10.0f)) {
            continue;
        }
        int src_idx = src_image_index[static_cast<int> (barcode_indices[i].item<int>())];
	uint8_t* start_ptr = images_4k.data_ptr<uint8_t>() + src_idx * 3 * 2160 * 3840;
	cv::Mat orig_img = cv::Mat(2160,3840, CV_8UC3, start_ptr);
//	if (i == 0){
	//cv::Mat orig_img = cv::Mat(2160,3840, CV_8UC3, &images_4k[src_idx * 3 * 2160 * 3840]);
        //cv::Mat orig_img_tmp = cv::Mat(2160, 3840, CV_8UC3, (&images_4k(src_idx * 3 * 2160 * 3840)));
	/*cv::Mat rgbchannel[3];
	cv::split(orig_img_tmp, rgbchannel);
	cv::Mat orig_img;
	std::vector<cv::Mat> channels;
	channels.push_back(rgbchannel[1]);channels.push_back(rgbchannel[2]);channels.push_back(rgbchannel[0]);
	cv::merge(channels, orig_img);*/
        //cv::transpose(orig_img, orig_img); // Equivalent to the transpose operation in Python

        // Add additional logic here for processing 'orig_img' or any further operations
        
        // For now, adding the transposed image to rotList
        //auto transposed = xt::transpose(xt::view(liveImg, i, xt::all()), {1, 2, 0});
        std::array<int, 2> src_dim = {static_cast<int>(liveImg[i].sizes()[1]), static_cast<int>(liveImg[i].sizes()[2])};
        auto seg_ocr_box = rescale_box(segbox, barcode_box, src_dim);

	float minX = std::numeric_limits<float>::max();
	float maxX = std::numeric_limits<float>::lowest();
	float minY = std::numeric_limits<float>::max();
	float maxY = std::numeric_limits<float>::lowest();

	for (const auto& pt : seg_ocr_box) {
		if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
	}

	minX = std::max(0.0f, minX);
	minX = std::min(3840.0f, minX);
        maxX = std::min(3840.0f, maxX);
        maxX = std::max(0.0f, maxX);
        minY = std::max(0.0f, minY);
        minY = std::min(2160.0f, minY);
        maxY = std::max(0.0f, maxY);
        maxY = std::min(2160.0f, maxY);
       int crop_width = static_cast<int>(maxX) - static_cast<int>(minX);
       int crop_height = static_cast<int>(maxY) -static_cast<int>(minY);
       cv::Rect roi(static_cast<int>(minX), static_cast<int>(minY), crop_width, crop_height);
       cv::Mat seg_ocr_crop = orig_img(roi);
       // Convert the cropped region to uint8 (if orig_img is not already of type CV_8U)
       if(seg_ocr_crop.type() != CV_8U) {
       seg_ocr_crop.convertTo(seg_ocr_crop, CV_8U);
       }
       //for(auto poinn: seg_ocr_box)
       auto seg_ocr_crop_box = shift_box_to_crop(seg_ocr_box,minX,minY);
       
       //for(auto point : seg_ocr_crop_box) 

       auto rotated = rotate_image(seg_ocr_crop, rotAngle, 20) ;

       cv::Point2i origin;
       origin.x = rotated.size().width / 2;
       origin.y = rotated.size().height /2 ;
       
       auto new_box = rotate_box(seg_ocr_crop_box, -rotAngle, 20, origin);

       auto rectified_rect  = rect_bbx(new_box);
       auto parseq_crop =  crop_rectangle(rotated, rectified_rect);
       if (parseq_crop.size().width < 5 || parseq_crop.size().height < 5) {
        continue;
       }
       auto transformed_crop = ocr_transform(parseq_crop);
       if (first_crop)
       {
/*        for (std::size_t k = 0; k < transformed_crop.shape()[0]; ++k) {
        for (std::size_t i = 0; i < transformed_crop.shape()[1]; ++i) {
            for (std::size_t j = 0; j < transformed_crop.shape()[2]; ++j) {
            }
        }
    }*/
	       first_crop = false;
  //     transformed_crop = transformed_crop[xt::newaxis(), xt::all(), xt::all()];
       rotList = transformed_crop.unsqueeze(0).clone();
	      // rotList = xt::expand_dims(transformed_crop,0);
       }
       else
       {
       	rotList = torch::cat({rotList, transformed_crop.unsqueeze(0).clone()}, 0);		
//       transformed_crop = transformed_crop[xt::newaxis(), xt::all(), xt::all()];
       //rotList = xt::concatenate(xt::xtuple(rotList,xt::expand_dims(transformed_crop,0)),0);
       }
       seg_index.push_back(i);
       //auto end_orig = std::chrono::high_resolution_clock::now();
/*
       using namespace std; 
  */
 }

/*while (rotList.shape()[0] < (size_t) 4) {
    xt::xarray<float> zeros = xt::zeros<float>({1,3, 32, 128});
    rotList = xt::concatenate(xt::xtuple(rotList,zeros),0);
    seg_index.push_back(-1);
}*/
while (rotList.size(0) < 4) {
    torch::Tensor zeros = torch::zeros({1, 3, 32, 128});
    rotList = torch::cat({rotList, zeros}, 0);
    seg_index.push_back(-1);
}
/*const float* begin = rotList.data();
const float* end = begin + rotList.size();
std::vector<float> rot_vec(begin,end);
*/
//std::vector<float> rot_vec(rotList.data_ptr<float>(), rotList.data_ptr<float>() + rotList.numel());

//return  seg_index;

}


std::vector<int64_t> get_src_image_list(const torch::Tensor& num_dets) {
    std::vector<int64_t> src_image_list;

    // Ensure num_dets is a 1D tensor
    assert(num_dets.dim() == 1);

    // Get max value from num_dets tensor
    int64_t max_num_dets = num_dets.max().item<int64_t>();
    // Iterate over num_dets tensor
    if (max_num_dets > 16){ max_num_dets = 16;}
    for (int64_t i = 0; i < num_dets.size(0); ++i) {
        for (int64_t j = 0; j < max_num_dets; ++j) {
            src_image_list.push_back(i);
        }
    }

    return src_image_list;
}

std::vector<int64_t> get_src_image_list_old(const xt::xtensor<int32_t, 1>& num_dets) {
    std::vector<int64_t> src_image_list;

    // Get max value from num_dets tensor
    int64_t max_num_dets = xt::amax(num_dets)();

    // Iterate over num_dets tensor
    for (std::size_t i = 0; i < num_dets.size(); ++i) {
        for (int64_t j = 0; j < max_num_dets; ++j) {
            src_image_list.push_back(static_cast<int64_t>(i));
        }
    }

    return src_image_list;
}



    std::vector<cv::Mat> process_det_outputs(torch::Tensor org_boxes, torch::Tensor boxes, at::Tensor mask_coefficients, at::Tensor proto) {
        int h_org = 224;
  int w_org = 224;

  mask_coefficients = mask_coefficients.transpose(1, 2);

  //torch::Tensor det_masks = torch::zeros({proto.size(0), h_org, w_org});
  std::vector<cv::Mat> det_masks;
  float rpn_mask_threshold = 0.3;
//#pragma omp parallel num_threads(32)
//#pragma omp for
  for (int64_t index = 0; index < boxes.size(0); ++index)
  {
    int flag = 0;
    auto mask_single = mask_coefficients[index];
    auto indices_mask = torch::zeros_like(mask_single, torch::kBool);
    //auto box = xt::view(boxes,index);
    auto box = boxes[index];
    int c = proto[index].size(0);
    int mh = proto[index].size(1);
    int mw = proto[index].size(2);
    //auto box_single = xt::view(xt::view(boxes,index, xt::all()), 0,xt::all());
    //auto box_single = box.slice(0,0,1);
    auto box_single = boxes[index].index({0, torch::indexing::Slice()});
    //auto res_index =  xt::where(box_single==xt::view(org_boxes,index,xt::all()),box_single,xt::view(org_boxes,index,xt::all()));
    //auto condition = xt::equal(xt::view(org_boxes,index,xt::all()), box_single);
    //auto res_index = xt::where(condition);
    //auto selected_rows = org_boxes.index({index});
    //auto condition = torch::eq(selected_rows, box_single.unsqueeze(0).expand_as(selected_rows));
    //auto res_index = torch::where(org_boxes[index] == box_single);
    //auto sliced_org_boxes = org_boxes.index({torch::indexing::Slice(index, index + 1), torch::indexing::Slice(), torch::indexing::Slice()});
    //auto condition = (sliced_org_boxes == box_single);
    auto condition = (org_boxes[index] == box_single);
    auto res_index = torch::nonzero(condition);
    //std::vector<torch::Tensor> res_index = torch::nonzero(condition);
    //int64_t index_value = static_cast<int64_t>(res_index[0][0]);
    if (res_index.size(0) > 0) {
    int64_t index_value = res_index[0][0].item<int64_t>();
    indices_mask.index_put_({index_value},1);
    flag = 1;
    }
    else
    {
	det_masks.push_back(cv::Mat::zeros(224, 224, CV_32F));
        continue;
    }
    if (flag ==1){
      //auto sliced_box = xt::xarray<float>(xt::view(box, xt::range(0, 1), xt::all()));
      //auto box_tensor = torch::from_blob(sliced_box.data(), {1, static_cast<long int>(sliced_box.shape()[1])}, torch::kFloat32).clone();
      auto sliced_box = box.slice(0, 0, 1).clone();
      auto box_tensor = sliced_box.clone();
	auto det_mask = mask_single.index({indices_mask}).view({-1, 32});
      det_mask = det_mask.mm(proto[index].to(torch::kFloat32).view({c, -1})).sigmoid().view({-1, mh, mw});
      det_mask = torch::nn::functional::interpolate(det_mask.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{h_org, w_org}).mode(torch::kBilinear).align_corners(false));
      int64_t h = det_mask.sizes()[2];
      int64_t w = det_mask.sizes()[3];
      auto x = torch::chunk(box_tensor, 4, 1);
      auto x1 = x[0];
      auto y1 = x[1];
      auto x2 = x[2];
      auto y2 = x[3];
      auto r = torch::arange(w, box_tensor.options()).unsqueeze(0).unsqueeze(1);
      auto c = torch::arange(h, box_tensor.options()).unsqueeze(0).unsqueeze(-1);

      det_mask = det_mask * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2));
      det_mask = (det_mask > rpn_mask_threshold).to(torch::kFloat);
      det_mask = torch::nn::functional::interpolate(det_mask, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{h_org, w_org}).mode(torch::kBilinear).align_corners(false));
      det_masks.push_back(tensorToMat(det_mask.clone().index({0, 0}).clone()));
      
}
}

return det_masks;
    }
};
