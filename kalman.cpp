//
// Created by robert on 24/01/20.
//
#include "../../include/detector/kalman.h"

KalmanFilter::KalmanFilter() {
    //TODO : Makesure the code below does not cause memoryleak
    // or other issue
    for (int i = 0; i < ndim; i++) {
        motion_mat[i][ndim + i] = dt;
    }
}

/**
 * Create track from unassociate measurement
 * @param measurment  format (x, y, aspect_ratio, height) (x, y) is the bounding box center
 *          position
 */
std::tuple<torch::Tensor, torch::Tensor> KalmanFilter::initiate(TrackerBbox measurment) {
    std::tuple<torch::Tensor, torch::Tensor> mean_covariance;
    float mean_data[] = {measurment.x, measurment.y, measurment.aspect_ratio, measurment.height, 0, 0, 0, 0};
    float std_data[] = {
            2 * std_weight_position * measurment.height,
            2 * std_weight_position * measurment.height,
            1e-2,
            2 * std_weight_position * measurment.height,
            10 * std_weight_velocity * measurment.height,
            10 * std_weight_velocity * measurment.height,
            1e-5,
            10 * std_weight_velocity * measurment.height
    };
    torch::Tensor mean = torch::from_blob(mean_data, {8}).clone();
    torch::Tensor standar_deviation = torch::from_blob(std_data, {8}).clone();
    torch::Tensor covariance = torch::diag(standar_deviation * standar_deviation);
    mean_covariance = std::make_tuple(mean, covariance);
    return mean_covariance;
}

/**
 * Project state distribution to measurement space
 * @return projected_mean, projected_variance
 */
std::tuple<torch::Tensor, torch::Tensor> KalmanFilter::project(torch::Tensor mean, torch::Tensor covariance) {
    std::tuple<torch::Tensor, torch::Tensor> projectedMean_projectedCovariance;
    auto mean_accessor = mean.accessor<float, 1>();
    float std_data[] = {
            std_weight_position * mean_accessor[3],
            std_weight_position * mean_accessor[3],
            1e-1,
            std_weight_position * mean_accessor[3]
    };
    torch::Tensor standard_deviation = torch::from_blob(std_data, {4}).clone();
    torch::Tensor innovation_cov = torch::diag(standard_deviation * standard_deviation);
    torch::Tensor projected_mean = torch::matmul(update_mat, mean);
    torch::Tensor projected_covariance = torch::matmul(update_mat,
                                                       torch::matmul(covariance, update_mat.transpose(0, 1)));
    projectedMean_projectedCovariance = std::make_tuple(projected_mean, projected_covariance + innovation_cov);
    return projectedMean_projectedCovariance;
}

std::tuple<torch::Tensor, torch::Tensor>
KalmanFilter::update(torch::Tensor mean, torch::Tensor covariance, TrackerBbox measurments) {
    float measurment_data[] = {measurments.x, measurments.y, measurments.aspect_ratio, measurments.height};
    auto measurment_tensor = torch::from_blob(measurment_data, {4}).clone();
    auto projected = KalmanFilter::project(mean, covariance);
    auto projected_mean = std::get<0>(projected);
    auto projected_cov = std::get<1>(projected);
//    Do cholesky factorization
    auto chofactor_lower = torch::cholesky(projected_cov);
    auto mm = torch::matmul(covariance, update_mat
            .transpose(0, 1)).transpose(0, 1);

    auto kalman_gain = torch::cholesky_solve(mm, chofactor_lower).transpose(0, 1);
    auto innovation = measurment_tensor - projected_mean;

    auto new_mean = mean + torch::matmul(innovation, kalman_gain.transpose(0, 1));
    auto new_covariance =
            covariance - torch::matmul(kalman_gain, torch::matmul(projected_cov, kalman_gain.transpose(0, 1)));
    return std::make_tuple(new_mean, new_covariance);
}

/**
 * Run Kalman filter prediction step
 * @param mean
 * @param covariance
 * @return
 */
std::tuple<torch::Tensor, torch::Tensor> KalmanFilter::predict(torch::Tensor mean, torch::Tensor covariance) {
    auto mean_accessor = mean.accessor<float, 1>();
    float std_data[] = {
            std_weight_position * mean_accessor[3],
            std_weight_position * mean_accessor[3],
            1e-2,
            std_weight_position * mean_accessor[3],
            std_weight_velocity * mean_accessor[3],
            std_weight_velocity * mean_accessor[3],
            1e-5,
            std_weight_velocity * mean_accessor[3]
    };
    torch::Tensor standard_deviation = torch::from_blob(std_data, {8}).clone();
    torch::Tensor motion_cov = torch::diag(standard_deviation * standard_deviation);
    torch::Tensor predicted_mean = torch::matmul(motion_mat, mean);
    torch::Tensor predicted_covariance = torch::matmul(motion_mat,
                                                       torch::matmul(covariance, motion_mat.transpose(0, 1))) +
                                         motion_cov;
    return std::make_tuple(predicted_mean, predicted_covariance);
}

//def gating_distance(self, mean, covariance, measurements,
//                    only_position=False):
torch::Tensor
KalmanFilter::gating_distance(torch::Tensor mean, torch::Tensor covariance, TrackerBbox measurments,
                              bool only_position) {
    torch::Tensor measurment_tensor;
    if (only_position) {
        float measurment_data_position[] = {measurments.x, measurments.y};
        measurment_tensor = torch::from_blob(measurment_data_position, {2}).clone();
    } else {
        float measurment_data[] = {measurments.x, measurments.y, measurments.aspect_ratio, measurments.height};
        measurment_tensor = torch::from_blob(measurment_data, {4}).clone();
    }
    auto projected = KalmanFilter::project(mean, covariance);
    auto projected_mean = std::get<0>(projected);
    auto projected_covariance = std::get<1>(projected);
    auto cholesky_factor = torch::cholesky(projected_covariance);
    torch::Tensor d = measurment_tensor - projected_mean;
    at::print(std::cout, cholesky_factor, 99);
    auto tuple_solution_cloned = torch::triangular_solve( d.unsqueeze(0).transpose(0,1),cholesky_factor, false);
    auto z = std::get<0>(tuple_solution_cloned);
    auto squared_maholobis_distance = torch::sum(z * z, 0);
    return squared_maholobis_distance;
}
