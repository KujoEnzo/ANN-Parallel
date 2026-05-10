#ifndef FLAT_SIMD_SEARCH_H
#define FLAT_SIMD_SEARCH_H

#include <vector>
#include <queue>
#include <arm_neon.h>

inline float simd_l2_distance(const float* a, const float* b, size_t dim) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (size_t i = 0; i < dim; i += 16) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        float32x4_t diff0 = vsubq_f32(va0, vb0);
        sum0 = vmlaq_f32(sum0, diff0, diff0);

        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        float32x4_t diff1 = vsubq_f32(va1, vb1);
        sum1 = vmlaq_f32(sum1, diff1, diff1);

        float32x4_t va2 = vld1q_f32(a + i + 8);
        float32x4_t vb2 = vld1q_f32(b + i + 8);
        float32x4_t diff2 = vsubq_f32(va2, vb2);
        sum2 = vmlaq_f32(sum2, diff2, diff2);

        float32x4_t va3 = vld1q_f32(a + i + 12);
        float32x4_t vb3 = vld1q_f32(b + i + 12);
        float32x4_t diff3 = vsubq_f32(va3, vb3);
        sum3 = vmlaq_f32(sum3, diff3, diff3);
    }

    float32x4_t res_vec = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    
    return vaddvq_f32(res_vec);
}

inline std::priority_queue<std::pair<float, int>> flat_simd_search(
    const float* base, const float* query, size_t base_number, size_t vecdim, size_t k) {
    
    std::priority_queue<std::pair<float, int>> global_top_k;
    
    #pragma omp parallel
    {
        std::priority_queue<std::pair<float, int>> local_top_k;
        
        #pragma omp for nowait
        for (size_t i = 0; i < base_number; ++i) {
            float dist = simd_l2_distance(base + i * vecdim, query, vecdim);
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
                auto item = local_top_k.top();
                local_top_k.pop();
                if (global_top_k.size() < k) {
                    global_top_k.push(item);
                } else if (item.first < global_top_k.top().first) {
                    global_top_k.pop();
                    global_top_k.push(item);
                }
            }
        }
    }
    return global_top_k;
}
#endif