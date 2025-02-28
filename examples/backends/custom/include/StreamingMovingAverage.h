#ifndef STREAMING_MOVING_AVERAGE_H
#define STREAMING_MOVING_AVERAGE_H

#include <deque>
#include <torch/torch.h>
#include <torch/nn.h>

class StreamingMovingAverage {
public:
    StreamingMovingAverage(int64_t window_size);
    torch::Tensor process(torch::Tensor value);
private:
    torch::Tensor values;
    torch::Tensor sum;
    int64_t window_size;
    int64_t count; 
};

#endif // STREAMING_MOVING_AVERAGE_H
