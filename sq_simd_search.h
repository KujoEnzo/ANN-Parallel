#ifndef SQ_SIMD_SEARCH_H
#define SQ_SIMD_SEARCH_H

#include <vector>
#include <queue>
#include <arm_neon.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
using namespace std;

struct SQIndex {

    size_t n;
    size_t dim;

    std::vector<uint8_t> data;

    std::vector<float> min_vals;
    std::vector<float> diff_vals;
};


SQIndex build_sq_index(const float* base, size_t n, size_t dim) {
    SQIndex index;
    index.n = n;
    index.dim = dim;
    index.data.resize(n * dim);

    index.min_vals.resize(dim);
    index.diff_vals.resize(dim);

    vector<float> max_vals(dim, -FLT_MAX);
    for (size_t d = 0; d < dim; ++d)
        index.min_vals[d] = FLT_MAX;
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {

            float v = base[i * dim + d];

            if (v < index.min_vals[d])
                index.min_vals[d] = v;

            if (v > max_vals[d])
                max_vals[d] = v;
        }
    }
    for (size_t d = 0; d < dim; ++d) {

        index.diff_vals[d] =
            (max_vals[d] - index.min_vals[d]) / 255.0f;

        if (index.diff_vals[d] == 0)
            index.diff_vals[d] = 1e-8f;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {

        for (size_t d = 0; d < dim; ++d) {

            float v = base[i * dim + d];

            float q =
                (v - index.min_vals[d])
                / index.diff_vals[d];

            index.data[i * dim + d] =
                (uint8_t)roundf(q);
        }
    }
    return index;
}


inline float sq_l2_distance_per_dim_simd(const float* q_minus_min, const uint8_t* b, const float* diff_ptr, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < dim; i += 8) {
        uint8x8_t vb_u8 = vld1_u8(b + i);
        uint16x8_t vb_u16 = vmovl_u8(vb_u8);
        
        float32x4_t vb_f32_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vb_u16)));
        float32x4_t v_diff_low = vld1q_f32(diff_ptr + i);
        float32x4_t v_q_min_low = vld1q_f32(q_minus_min + i);

        float32x4_t vb0 = vmulq_f32(vb_f32_low, v_diff_low);
        float32x4_t d0 = vsubq_f32(v_q_min_low, vb0);
        sum_vec = vmlaq_f32(sum_vec, d0, d0);

        float32x4_t vb_f32_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vb_u16)));
        float32x4_t v_diff_high = vld1q_f32(diff_ptr + i + 4);
        float32x4_t v_q_min_high = vld1q_f32(q_minus_min + i + 4);
        
        float32x4_t vb1 = vmulq_f32(vb_f32_high, v_diff_high);
        float32x4_t d1 = vsubq_f32(v_q_min_high, vb1);
        sum_vec = vmlaq_f32(sum_vec, d1, d1);
    }

    return vaddvq_f32(sum_vec);
}

inline std::priority_queue<std::pair<float, int>> sq_search(
    const SQIndex& index, const float* query, size_t k) {
    
    std::vector<float> q_minus_min(index.dim);
    for (size_t d = 0; d < index.dim; ++d) {
        q_minus_min[d] = query[d] - index.min_vals[d];
    }

    std::priority_queue<std::pair<float, int>> global_top_k;

    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, int>> local_top_k;
        #pragma omp for nowait
        for (size_t i = 0; i < index.n; ++i) {
            float dist = sq_l2_distance_per_dim_simd(
                q_minus_min.data(), 
                &index.data[i * index.dim], 
                index.diff_vals.data(), 
                index.dim
            );
            
            if (local_top_k.size() < k) {
                local_top_k.push({dist, (int)i});
            } else if (dist < local_top_k.top().first) {
                local_top_k.pop();
                local_top_k.push({dist, (int)i});
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

#endif