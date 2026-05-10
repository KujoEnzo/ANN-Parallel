#ifndef PQ_SEARCH_H
#define PQ_SEARCH_H

#include <vector>
#include <queue>
#include <arm_neon.h>
#include "pq_train.h"

void build_lut(const PQIndex& index, const float* query, float* lut) {
    for (size_t m = 0; m < index.m; ++m) {
        const float* sub_q = query + m * index.sub_dim;
        const float* sub_centroids = &index.centroids[m * index.k_star * index.sub_dim];
        
        for (size_t ks = 0; ks < index.k_star; ++ks) {
            const float* center = sub_centroids + ks * index.sub_dim;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            
            for (size_t d = 0; d < index.sub_dim; d += 4) {
                float32x4_t q_vec = vld1q_f32(sub_q + d);
                float32x4_t c_vec = vld1q_f32(center + d);
                float32x4_t diff = vsubq_f32(q_vec, c_vec);
                sum_vec = vmlaq_f32(sum_vec, diff, diff);
            }
            lut[m * index.k_star + ks] = vaddvq_f32(sum_vec);
        }
    }
}
inline std::priority_queue<std::pair<float, int>> pq_search_fast(
    const PQIndex& index, const float* query, size_t k) {

    std::vector<float> lut(index.m * index.k_star);
    build_lut(index, query, lut.data());

    std::priority_queue<std::pair<float, int>> global_top_k;

    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, int>> local_top_k;
        float threshold = std::numeric_limits<float>::max();

        #pragma omp for nowait
        for (size_t i = 0; i < index.n; ++i) {
            float total_dist = 0;
            const uint8_t* code = &index.codes[i * index.m];
            

            for (size_t m = 0; m < index.m; ++m) {
                total_dist += lut[m * index.k_star + code[m]];
            }

            if (total_dist < threshold) {
                local_top_k.push({total_dist, (int)i});
                if (local_top_k.size() > k) {
                    local_top_k.pop();
                    threshold = local_top_k.top().first; // 更新阈值
                }
            }
        }

        #pragma omp critical
        {
            while (!local_top_k.empty()) {
                if (global_top_k.size() < k) {
                    global_top_k.push(local_top_k.top());
                } else if (local_top_k.top().first < global_top_k.top().first) {
                    global_top_k.pop();
                    global_top_k.push(local_top_k.top());
                }
                local_top_k.pop();
            }
        }
    }
    return global_top_k;
}
inline std::priority_queue<std::pair<float, int>> pq_search(
    const PQIndex& index, const float* query, size_t k) {

    std::vector<float> lut(index.m * index.k_star);
    
    for (size_t m = 0; m < index.m; ++m) {
        for (size_t ks = 0; ks < index.k_star; ++ks) {
            float dist = 0;
            const float* sub_q = query + m * index.sub_dim;
            const float* center = &index.centroids[(m * index.k_star + ks) * index.sub_dim];
            for(size_t d=0; d<index.sub_dim; ++d) {
                float diff = sub_q[d] - center[d];
                dist += diff * diff;
            }
            lut[m * index.k_star + ks] = dist;
        }
    }

    std::priority_queue<std::pair<float, int>> global_top_k;

    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, int>> local_top_k;
        #pragma omp for nowait
        for (size_t i = 0; i < index.n; ++i) {
            float total_dist = 0;
            const uint8_t* code = &index.codes[i * index.m];
            
            for (size_t m = 0; m < index.m; ++m) {
                total_dist += lut[m * index.k_star + code[m]];
            }

            if (local_top_k.size() < k) {
                local_top_k.push({total_dist, (int)i});
            } else if (total_dist < local_top_k.top().first) {
                local_top_k.pop();
                local_top_k.push({total_dist, (int)i});
            }
        }
         #pragma omp critical
        {
            while (!local_top_k.empty()) {
                if (global_top_k.size() < k) {
                    global_top_k.push(local_top_k.top());
                } else if (local_top_k.top().first < global_top_k.top().first) {
                    global_top_k.pop();
                    global_top_k.push(local_top_k.top());
                }
                local_top_k.pop();
            }
        }
    }
    return global_top_k;
}

inline std::priority_queue<std::pair<float, int>> pq_rerank_search(
    const PQIndex& index, const float* base_data, const float* query, size_t k) {

    size_t k_search = k * 4; 
    if (k_search > index.n) k_search = index.n;

    std::vector<float> lut(index.m * index.k_star);
    for (size_t j = 0; j < index.m; ++j) {
        const float* sub_q = query + j * index.sub_dim;
        for (size_t ks = 0; ks < index.k_star; ++ks) {
            lut[j * index.k_star + ks] = l2_sqr(sub_q, &index.centroids[(j * index.k_star + ks) * index.sub_dim], index.sub_dim);
        }
    }

    //PQ 粗排
    std::priority_queue<std::pair<float, int>> candidate_queue;
    for (size_t i = 0; i < index.n; ++i) {
        float pq_dist = 0;
        const uint8_t* code = &index.codes[i * index.m];
        for (size_t j = 0; j < index.m; ++j) {
            pq_dist += lut[j * index.k_star + code[j]];
        }
        
        if (candidate_queue.size() < k_search) {
            candidate_queue.push({pq_dist, (int)i});
        } else if (pq_dist < candidate_queue.top().first) {
            candidate_queue.pop();
            candidate_queue.push({pq_dist, (int)i});
        }
    }

    //重排
    std::priority_queue<std::pair<float, int>> final_top_k;
    while (!candidate_queue.empty()) {
        int idx = candidate_queue.top().second;
        candidate_queue.pop();

        float real_dist = l2_sqr(query, base_data + idx * 96, 96);

        if (final_top_k.size() < k) {
            final_top_k.push({real_dist, idx});
        } else if (real_dist < final_top_k.top().first) {
            final_top_k.pop();
            final_top_k.push({real_dist, idx});
        }
    }

    return final_top_k;
}
#endif