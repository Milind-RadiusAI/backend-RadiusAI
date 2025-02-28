#include "ViterbiModel.h"
#include <iostream>

using namespace std;
using namespace Eigen;

ViterbiModel::ViterbiModel() {
    // Initialize your matrices and vectors here
    transition_matrix = MatrixXd(3, 3);
    transition_matrix << 0.9880239520958084, 0.001996007984031937, 0.009980039920159688,
                            0.0, 0.9, 0.1,
                            0.008980039920159688, 0.000980039920159688, 0.9880239520958084;

    distributions.push_back(MatrixXd(5, 2));
    distributions[0] << 0.1, 0.9,
                            0.1, 0.9,
                            0.1, 0.9,
                            0.1, 0.9,
                            0.1, 0.9;

    distributions.push_back(MatrixXd(5, 2));
    distributions[1] << 0.1, 0.9,
                            0.1, 0.9,
                            0.9, 0.1,
                            0.9, 0.1,
                            0.1, 0.9;

    distributions.push_back(MatrixXd(5, 2));
    distributions[2] << 0.9, 0.3,
                            0.9, 0.1,
                            0.9, 0.1,
                            0.9, 0.1,
                            0.9, 0.1;

    start_probabilities = VectorXd(3);
    start_probabilities << 0.4, 0.27, 0.33;
}

double ViterbiModel::calculate_emission_log_prob(int state, const RowVectorXi& observation) {
    auto start_time_emission = chrono::high_resolution_clock::now();
    double log_prob = 0.0;
    for (int i = 0; i < observation.size(); ++i) {
        log_prob += log(distributions[state](i, observation(i)));
    }
    auto stop = chrono::high_resolution_clock::now();
    duration_em += chrono::duration_cast<chrono::milliseconds>(stop - start_time_emission).count();
    return log_prob;
}

VectorXi ViterbiModel::viterbi(const MatrixXi& sequence) {
    int num_states = 3;
    int num_observations = sequence.rows();
    MatrixXd dp = MatrixXd::Constant(num_states, num_observations, -std::numeric_limits<double>::infinity());  // initialize to negative infinity
    MatrixXi ptr(num_states, num_observations);

    // Initialization
    for (int s = 0; s < num_states; ++s) {
        dp(s, 0) = log(start_probabilities(s)) + calculate_emission_log_prob(s, sequence.row(0));
    }

    // Recursion
    for (int t = 1; t < num_observations; ++t) {
        //pragma omp parallel for
        for (int s = 0; s < num_states; ++s) {
            for (int prev_s = 0; prev_s < num_states; ++prev_s) {
                double score = dp(prev_s, t - 1) + log(transition_matrix(prev_s, s)) + calculate_emission_log_prob(s, sequence.row(t));
                if (score > dp(s, t)) {
                    dp(s, t) = score;
                    ptr(s, t) = prev_s;
                }
            }
        }
    }

    // Termination and path backtracking
    VectorXi best_path(num_observations);
    int best_state;
    dp.col(num_observations - 1).maxCoeff(&best_state);
    best_path(num_observations - 1) = best_state;
    for (int t = num_observations - 2; t >= 0; --t) {
        best_state = ptr(best_state, t + 1);
        best_path(t) = best_state;
    }
    return best_path;
}


