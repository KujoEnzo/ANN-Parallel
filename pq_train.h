#ifndef PQ_TRAIN_H
#define PQ_TRAIN_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <limits>
#include <omp.h>

struct PQIndex {
    std::vector<uint8_t> codes;     
    std::vector<float> centroids;   
    size_t n, m, k_star, sub_dim;
};

inline float l2_sqr(const float* a, const float* b, size_t d) {
    float dist = 0;
    for (size_t i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

std::vector<float> train_kmeans(const std::vector<float>& sub_data, size_t n, size_t sub_dim, size_t k_star, int max_iters = 30) {
    std::vector<float> centers(k_star * sub_dim);
    

    std::mt19937 rng(9); // 固定baka种子
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    for (size_t i = 0; i < k_star; ++i) {
        size_t rand_idx = dist(rng);
        for (size_t d = 0; d < sub_dim; ++d) {
            centers[i * sub_dim + d] = sub_data[rand_idx * sub_dim + d];
        }
    }

    std::vector<int> assign(n, 0); 
    for (int iter = 0; iter < max_iters; ++iter) {
        int changed = 0;

        #pragma omp parallel for reduction(+:changed)
        for (size_t i = 0; i < n; ++i) {
            const float* pt = &sub_data[i * sub_dim];
            float min_dist = std::numeric_limits<float>::max();
            int best_c = 0;
            
            for (size_t c = 0; c < k_star; ++c) {
                float d = l2_sqr(pt, &centers[c * sub_dim], sub_dim);
                if (d < min_dist) {
                    min_dist = d;
                    best_c = c;
                }
            }
            if (assign[i] != best_c) {
                assign[i] = best_c;
                changed++;
            }
        }

        if (changed == 0) break;

        std::vector<float> new_centers(k_star * sub_dim, 0.0f);
        std::vector<int> counts(k_star, 0);

        for (size_t i = 0; i < n; ++i) {
            int c = assign[i];
            counts[c]++;
            for (size_t d = 0; d < sub_dim; ++d) {
                new_centers[c * sub_dim + d] += sub_data[i * sub_dim + d];
            }
        }

        for (size_t c = 0; c < k_star; ++c) {
            if (counts[c] > 0) {
                for (size_t d = 0; d < sub_dim; ++d) {
                    centers[c * sub_dim + d] = new_centers[c * sub_dim + d] / counts[c];
                }
            }
        }
    }
    return centers;
}

PQIndex build_pq_index(const float* base, size_t n, size_t dim, size_t m = 16, size_t k_star = 256) {
    PQIndex index;
    index.n = n;
    index.m = m;
    index.k_star = k_star;
    index.sub_dim = dim / m;
    
    index.centroids.resize(m * k_star * index.sub_dim);
    index.codes.resize(n * m);

    std::cout << "[PQ] Start training " << m << " subspaces..." << std::endl;

    for (size_t j = 0; j < m; ++j) {
        std::vector<float> sub_data(n * index.sub_dim);
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < index.sub_dim; ++d) {
                sub_data[i * index.sub_dim + d] = base[i * dim + j * index.sub_dim + d];
            }
        }

        std::vector<float> sub_centers = train_kmeans(sub_data, n, index.sub_dim, k_star);

        for (size_t c = 0; c < k_star * index.sub_dim; ++c) {
            index.centroids[j * k_star * index.sub_dim + c] = sub_centers[c];
        }
        std::cout << "  - Subspace " << j + 1 << "/" << m << " trained." << std::endl;
    }

    std::cout << "[PQ] Training done. Start encoding..." << std::endl;

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            const float* sub_vec = &base[i * dim + j * index.sub_dim];
            const float* sub_centers = &index.centroids[j * k_star * index.sub_dim];
            
            float min_dist = std::numeric_limits<float>::max();
            uint8_t best_c = 0;
            
            for (size_t c = 0; c < k_star; ++c) {
                float d = l2_sqr(sub_vec, sub_centers + c * index.sub_dim, index.sub_dim);
                if (d < min_dist) {
                    min_dist = d;
                    best_c = (uint8_t)c;
                }
            }
            index.codes[i * m + j] = best_c; 
        }
    }
    
    std::cout << "[PQ] Encoding done. Code size: " << (index.codes.size() / 1024.0 / 1024.0) << " MB" << std::endl;
    return index;
}

#endif