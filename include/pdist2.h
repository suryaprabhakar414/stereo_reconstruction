//
// Created by Zhijie Yang on 16/12/2021.
//

#ifndef STEREO_RECONSTRUCTION_PDIST2_H
#define STEREO_RECONSTRUCTION_PDIST2_H

#include <Eigen/Dense>
#include <numeric>
#include "types.h"

struct Dist {
    std::vector<float> dist;
    std::vector<int> idx;
};

bool comparePairs(const std::pair<float, int>&i, const std::pair<float,int>&j) {
    return i.first < j.first;
}

/**
 * Pairwise distance between two sets of observations. Implementation of MATLAB pdist2 function.
 *@tparam Derived A matrix type
 *@param X first matrix
 *@param Y second matrix
 *@return the pairwise distance (L2 norm) matrix
 *@details D = pdist2(X,Y) returns a matrix D containing the Euclidean distances between each pair of observations in the mx-by-n data matrix X and my-by-n data matrix Y.
 * Rows of X and Y correspond to observations, columns correspond to variables. D is an mx-by-my matrix, with the (i,j) entry equal to distance between observation i in X and observation j in Y. The (i,j) entry will be NaN if observation i in X or observation j in Y contain NaNs.
 *@note see https://www.mathworks.com/help/stats/pdist2.html
 */
template<typename Derived>
typename Derived::PlainObject pdist2(const Eigen::MatrixBase<Derived>& X, const Eigen::MatrixBase<Derived>& Y)
{
    typename Derived::PlainObject D(X.rows(), Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        D.row(i) = (Y.rowwise() - X.row(i)).rowwise().norm();
    }

    return D;
}

/**
 * return K smallest dist and their index of image X and Y, notice that K must be larger than 0
 */
template<typename Derived>
std::vector<std::pair<float,int>> pdist2SmallestK(const Eigen::MatrixBase<Derived>& X, const Eigen::MatrixBase<Derived>& Y, int k)
{
    typename Derived::PlainObject D(X.rows(), Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        D.row(i) = (Y.rowwise() - X.row(i)).rowwise().norm();
    }

    std::vector<std::pair<float,int>> distPairs;
    for (int i = 0; i < D.rows(); i++) {
        std::pair<float, int> dist;
        dist.first = D(i, 0);
        dist.second = i;
        distPairs.push_back(dist);
    }
    int size = distPairs.size();
    std::sort(distPairs.begin(), distPairs.end(), comparePairs);

    std::vector<std::pair<float,int>> smallestK;
    for (int i = 0; i < k; i++) {
        smallestK.push_back(distPairs[i]);
    }

    return smallestK;
}

/**
* Squared pairwise distance between two sets of observations.
*@see pdist2
*/
template<typename Derived>
typename Derived::PlainObject pdist2Squared(const Eigen::MatrixBase<Derived>& X, const Eigen::MatrixBase<Derived>& Y)
{
    typename Derived::PlainObject D(X.rows(), Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        D.row(i) = (Y.rowwise() - X.row(i)).rowwise().squaredNorm();
    }
    return D;
}


#endif //STEREO_RECONSTRUCTION_PDIST2_H
