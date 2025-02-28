#include "barcode_seg_process.h"
using namespace torch::indexing;


BarcodeSegProcess::BarcodeSegProcess() {}


void BarcodeSegProcess::initialize() {}


void BarcodeSegProcess::execute(
    const std::vector<float*>& dataPointers , const std::vector<std::vector<std::int64_t>>& shapes,
    const std::vector <std::string>& data_types , std::vector<const void*>& outputPointers,
    std::vector<std::vector<std::int64_t>>&  output_shapes
)
{
    std::vector<torch::Tensor> all_inputs;
    deserialize_inputs(dataPointers,shapes,data_types,all_inputs);
    torch::Tensor input_images_4k_arr = all_inputs[7].permute({0, 2,3,1}).contiguous();
    torch::Tensor org_boxes_arr = all_inputs[0];
    torch::Tensor det_boxes_arr = all_inputs[2];
    torch::Tensor mask_coefficients_tensor = all_inputs[3];
    torch::Tensor proto_tensor = all_inputs[4];
    torch::Tensor input_barcode_indices_arr  = all_inputs[6];
    torch::Tensor input_scaled_barcode_boxes_arr  = all_inputs[5];
    torch::Tensor input_barcode_crops_4k_arr = all_inputs[1];
    torch::Tensor input_rpn_obj_num_dets_arr = all_inputs[8];
    auto src_image_index = get_src_image_list(input_rpn_obj_num_dets_arr);
    auto result = process_det_outputs(org_boxes_arr, det_boxes_arr, mask_coefficients_tensor, proto_tensor);
    torch::Tensor rotList;
    std::vector<int> segIdx;
    batchedRotation(input_barcode_crops_4k_arr, result, input_barcode_indices_arr, input_scaled_barcode_boxes_arr, src_image_index, input_images_4k_arr, rotList, segIdx);
    size_t total_size = 0;
    std::vector<int64_t> shape_mat = {static_cast<int64_t>(result.size()), result[0].rows, result[0].cols};
    torch::IntArrayRef sizes = rotList.sizes();
    std::vector<int64_t> shape_org_boxes(sizes.begin(), sizes.end());
    std::vector<int64_t> shape_segIdx = {static_cast<int64_t>(segIdx.size())};
    output_shapes.push_back(shape_mat); output_shapes.push_back(shape_org_boxes); output_shapes.push_back(shape_segIdx);
    for (const cv::Mat& mat : result) 
    {
        total_size += mat.total();
    }
    float* heapMatData = new float[total_size];
    size_t offset = 0;
    for (const cv::Mat& mat : result)
    {
        std::memcpy(heapMatData + offset, mat.data, mat.total() * sizeof(float));
        offset += mat.total();
    }
    outputPointers.push_back(heapMatData);

    float* heapTensorData = new float[rotList.numel()];
    std::memcpy(heapTensorData, rotList.data_ptr<float>(), rotList.numel() * sizeof(float));
    outputPointers.push_back(heapTensorData);
    int* heapVecData = new int[segIdx.size()];
    std::memcpy(heapVecData, segIdx.data(), segIdx.size() * sizeof(int));
    outputPointers.push_back(heapVecData);
}








cv::Point2i BarcodeSegProcess::rotate(const cv::Point2i& p, const cv::Point2i& origin, float degrees) 
{
    float angle = degrees * CV_PI / 180.0; // Convert degrees to radians
    cv::Matx22f R(std::cos(angle), -std::sin(angle),
                  std::sin(angle),  std::cos(angle));

    cv::Point2f pf(static_cast<float>(p.x), static_cast<float>(p.y));
    cv::Point2f originf(static_cast<float>(origin.x), static_cast<float>(origin.y));
    
    cv::Point2f rotatedPointf = R * (pf - originf) + originf;
    return cv::Point2i(static_cast<int>(rotatedPointf.x), static_cast<int>(rotatedPointf.y));
}

std::array<cv::Point2i, 4> BarcodeSegProcess::rotate_box(const std::array<cv::Point2i, 4>& box, float angle, int pad, const cv::Point2i& origin) 
{
    std::array<cv::Point2i, 4> new_box;
    
    for(int i = 0; i < 4; i++) {
        cv::Point2i paddedPoint = box[i] + cv::Point2i(pad, pad);
        new_box[i] = rotate(paddedPoint, origin, angle);
    }
    
    return new_box;
}


torch::Tensor BarcodeSegProcess::ocr_transform(const cv::Mat& mat_image) 
{
    // Resize the image
    cv::Mat resized_image_mat;
    cv::resize(mat_image, resized_image_mat, cv::Size(128, 32), 0, 0, cv::INTER_CUBIC);

    // Convert resized cv::Mat to torch::Tensor
    torch::Tensor resized_image = torch::from_blob(resized_image_mat.data, {resized_image_mat.rows, resized_image_mat.cols, 3}, torch::kUInt8);

    // Convert to float, normalize and permute the tensor to CxHxW
    torch::Tensor normalized_image = (resized_image.permute({2, 0, 1}).contiguous().to(torch::kFloat32) / 255.0 - 0.5) / 0.5;

    return normalized_image;
}

cv::Mat BarcodeSegProcess::tensorToMat(const torch::Tensor& tensor) 
{
    torch::Tensor tensor_tmp = tensor.cpu().to(torch::kFloat32);
    cv::Mat mat(tensor_tmp.sizes()[0], tensor_tmp.sizes()[1], CV_32F);
    std::memcpy(mat.data, tensor_tmp.data_ptr<float>(), tensor_tmp.numel() * sizeof(float));

    return mat;
}




cv::Mat BarcodeSegProcess::crop_rectangle(const cv::Mat& image, const std::tuple<cv::Point, cv::Size, int>& rect) 
{
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





cv::Point2f BarcodeSegProcess::findCentroid(const std::array<cv::Point2f, 4>& box) 
{
    float x = (box[0].x + box[2].x) / 2.0f;
    float y = (box[0].y + box[2].y) / 2.0f;
    return cv::Point2f(x, y);
}

cv::Vec4f BarcodeSegProcess::shortestCalculator(const std::array<cv::Point2f, 4>& box) 
{
    // Calculate sides using Euclidean distance
    double d1 = cv::norm(box[0] - box[1]);
    double d2 = cv::norm(box[1] - box[2]);

    if (d1 < d2) {
        return cv::Vec4f(box[0].x, box[0].y, box[1].x, box[1].y);
    } else {
        return cv::Vec4f(box[1].x, box[1].y, box[2].x, box[2].y);
    }
}

std::array<cv::Point2i, 4> BarcodeSegProcess::shift_box_to_crop(std::array<cv::Point2f, 4>& box, int x, int y)
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


std::tuple<cv::Point, cv::Size, int> BarcodeSegProcess::rect_bbx(const std::array<cv::Point2i, 4>& box) 
{
    int x_max = static_cast<int>(std::max({box[0].x, box[1].x, box[2].x, box[3].x}));
    int x_min = static_cast<int>(std::min({box[0].x, box[1].x, box[2].x, box[3].x}));
    int y_max = static_cast<int>(std::max({box[0].y, box[1].y, box[2].y, box[3].y}));
    int y_min = static_cast<int>(std::min({box[0].y, box[1].y, box[2].y, box[3].y}));

    cv::Point center((x_min + x_max) / 2, (y_min + y_max) / 2);
    cv::Size dimensions(x_max - x_min, y_max - y_min);
    int angle = 0;
    return std::make_tuple(center, dimensions, angle);
}



cv::Mat BarcodeSegProcess::rotate_image(cv::Mat& image, double angle, int pad)
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

std::array<cv::Point2f, 4> BarcodeSegProcess::rescale_box(const std::array<cv::Point2f, 4>& bbox, const torch::Tensor& barcode_box, const std::array<int, 2>& src_dim)
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



std::pair<float, std::array<cv::Point2f, 4>>  BarcodeSegProcess::binaryMaskProcess(const cv::Mat& bMask) {
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



void BarcodeSegProcess::batchedRotation(
    torch::Tensor& liveImg, std::vector<cv::Mat>& seg, 
    torch::Tensor&  barcode_indices, torch::Tensor& scaled_barcode_boxes, 
    std::vector<int64_t>& src_image_index,torch::Tensor& images_4k, 
    torch::Tensor& rotList,std::vector<int>& seg_index)
{
    int min_bbox_dim = 65 ;
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
        std::array<int, 2> src_dim = {static_cast<int>(liveImg[i].sizes()[1]), static_cast<int>(liveImg[i].sizes()[2])};
        auto seg_ocr_box = rescale_box(segbox, barcode_box, src_dim);

        float minX = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max();
        float maxY = std::numeric_limits<float>::lowest();

        for (const auto& pt : seg_ocr_box) 
        {
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
        auto seg_ocr_crop_box = shift_box_to_crop(seg_ocr_box,minX,minY);
        

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
            first_crop = false;
        rotList = transformed_crop.unsqueeze(0).clone();
        }
        else
        {
            rotList = torch::cat({rotList, transformed_crop.unsqueeze(0).clone()}, 0);		
        }
        seg_index.push_back(i);
 }

while (rotList.size(0) < 4) {
    torch::Tensor zeros = torch::zeros({1, 3, 32, 128});
    rotList = torch::cat({rotList, zeros}, 0);
    seg_index.push_back(-1);
}
if (static_cast<size_t>(rotList.size(0)) != seg_index.size()){ seg_index.push_back(-1);}

}


std::vector<int64_t> BarcodeSegProcess::get_src_image_list(const torch::Tensor& num_dets) 
{
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




std::vector<cv::Mat> BarcodeSegProcess::process_det_outputs(torch::Tensor org_boxes, torch::Tensor boxes, at::Tensor mask_coefficients, at::Tensor proto) 
{
    int h_org = 224;
    int w_org = 224;

    mask_coefficients = mask_coefficients.transpose(1, 2);

    std::vector<cv::Mat> det_masks;
    float rpn_mask_threshold = 0.3;
//#pragma omp parallel num_threads(32)
//#pragma omp for
    for (int64_t index = 0; index < boxes.size(0); ++index)
    {
        int flag = 0;
        auto mask_single = mask_coefficients[index];
        auto indices_mask = torch::zeros_like(mask_single, torch::kBool);
        auto box = boxes[index];
        int c = proto[index].size(0);
        int mh = proto[index].size(1);
        int mw = proto[index].size(2);
        auto box_single = boxes[index].index({0, torch::indexing::Slice()});
        auto condition = (org_boxes[index] == box_single);
        auto res_index = torch::nonzero(condition);
        if (res_index.size(0) > 0) 
        {
            int64_t index_value = res_index[0][0].item<int64_t>();
            indices_mask.index_put_({index_value},1);
            flag = 1;
        }
        else
        {
            det_masks.push_back(cv::Mat::zeros(224, 224, CV_32F));
            continue;
        }
        if (flag ==1)
        {
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