//
// Created by robert on 24/01/20.
//

#ifndef DEEP_TRAFFIC_COUNTER_KALMAN_H
#define DEEP_TRAFFIC_COUNTER_KALMAN_H

#include <torch/torch.h>

/**
 * Motion and observation uncertainty are chosen relative to the current
 * state estimate. These weights control the amount of uncertainty in
 * the model. This is a bit hacky.
 * self._std_weight_position = 1. / 20
 * self._std_weight_velocity = 1. / 160
 */
struct TrackerBbox {
    float x;
    float y;
    float aspect_ratio;
    float height;
};

class KalmanFilter {
public:


    float chi2inv95[9] = {
            3.8415,
            5.9915,
            7.8147,
            9.4877,
            11.070,
            12.592,
            14.067,
            15.507,
            16.919
    };

    KalmanFilter();

    int ndim = 4;
    float dt = 1.;
    torch::Tensor motion_mat = torch::eye(2 * ndim, 2 * ndim);
    torch::Tensor update_mat = torch::eye(ndim, 2 * ndim);
    float std_weight_position = 1. / 20;
    float std_weight_velocity = 1. / 160;
    std::tuple<torch::Tensor,torch::Tensor> initiate(TrackerBbox measurment);

    std::tuple<torch::Tensor, torch::Tensor> project(torch::Tensor mean, torch::Tensor covariance);
    std::tuple<torch::Tensor, torch::Tensor>
    update(torch::Tensor mean, torch::Tensor covariance, TrackerBbox measurement);

    std::tuple<torch::Tensor, torch::Tensor> predict(torch::Tensor mean, torch::Tensor covariance);

    torch::Tensor

    gating_distance(torch::Tensor mean, torch::Tensor covariance, TrackerBbox measurments, bool only_position);
};


#endif //DEEP_TRAFFIC_COUNTER_KALMAN_H
