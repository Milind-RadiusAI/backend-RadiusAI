#ifndef VITERBIMODEL_H
#define VITERBIMODEL_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <chrono>

class ViterbiModel {
public:
    Eigen::MatrixXd transition_matrix;
    std::vector<Eigen::MatrixXd> distributions;
    Eigen::VectorXd start_probabilities;
    int64_t duration_em = 0;

    ViterbiModel();
    double calculate_emission_log_prob(int state, const Eigen::RowVectorXi& observation);
    Eigen::VectorXi viterbi(const Eigen::MatrixXi& sequence);
};

#endif // VITERBIMODEL_H
