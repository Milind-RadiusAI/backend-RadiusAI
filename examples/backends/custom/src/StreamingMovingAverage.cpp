#include "StreamingMovingAverage.h"

StreamingMovingAverage::StreamingMovingAverage(int64_t window_size)
        : window_size(window_size){
        sum = torch::zeros({1});
        values = torch::zeros({1,window_size});
        count = 0;
}
torch::Tensor StreamingMovingAverage::process(torch::Tensor value) {      
    // Ensure value is a single-element tensor
    if (value.numel() != 1) {
        throw std::runtime_error("Input tensor must have exactly one element.");
    }

    // Add new value to the sum
    sum += value;

    // Replace the oldest value in the window with the new one
    int64_t index_to_replace = count % window_size;
    if (count >= window_size) { // We're now overwriting old values
        // Subtract the value that is going to be replaced
        sum -= values.index({index_to_replace});
    }

    // Place the new value
    values.index({index_to_replace}) = value;

    // Increment the count of values processed
    count++;

    // Calculate the average based on the minimum of the count and window size
    int64_t current_window_size = std::min(count, window_size);
    return sum / current_window_size;

}
   