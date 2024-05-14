#include <stdio.h>
#include <algorithm>
#include <pthread.h>
#include <math.h>
#include <immintrin.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

static const float kThreshold = 0.00001f; 

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

void print_avx2(__m256 vec) {
    float arr[8];
    _mm256_storeu_ps(arr, vec);

    for (int i = 0; i < 8; ++i) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}

__m256 _mm256_abs_ps(__m256 x) {
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    // make sign bit zero
    return _mm256_and_ps(x, mask);
}

// Function to compute the reciprocal of the square root using AVX2 and Newton's method
__m256 avx2_rsqrt_newton(__m256 x, float initialGuess) {
    // Constants used in the Newton-Raphson iteration
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 oneHalf = _mm256_set1_ps(1.5f);
    const __m256 one = _mm256_set1_ps(1.0f); 
    const __m256 threshold = _mm256_set1_ps(kThreshold);
    const __m256 half_x = _mm256_mul_ps(x, half);

    __m256 y = _mm256_set1_ps(initialGuess);

    // Newton-Raphson iteration to refine the approximation
    __m256 error = _mm256_sub_ps(one, _mm256_mul_ps(x, _mm256_mul_ps(y, y)));
    while (_mm256_movemask_ps(_mm256_cmp_ps(_mm256_abs_ps(error), threshold, _CMP_GT_OS)) != 0) {
        // Newton-Raphson iteration: y = y * (1.5 - 0.5 * x * y * y)
        y = _mm256_mul_ps(y, _mm256_sub_ps(oneHalf, _mm256_mul_ps(half_x, _mm256_mul_ps(y, y))));
        // Recalculate the error
        error = _mm256_sub_ps(one, _mm256_mul_ps(x, _mm256_mul_ps(y, y)));
    }

    return y;
}

__m256 avx2_sqrt_newtion(__m256 x, float initialGuess){
    return _mm256_mul_ps(x, avx2_rsqrt_newton(x, initialGuess));
}

void sqrtAVX2(int N, float initialGuess, float values[], float output[]) {
    for (int i = 0; i < N; i += 8) {
        __m256 input_values = _mm256_loadu_ps(values + i);

        // printf("[before avx2]i:%d\n", i);
        __m256 sqrt_result = avx2_sqrt_newtion(input_values, initialGuess);

        // 存储结果到输出数组
        _mm256_storeu_ps(output + i, sqrt_result);
    }
}


static void verifyResult(int N, float* result, float* gold) {
    for (int i=0; i<N; i++) {
        if (fabs(result[i] - gold[i]) > 1e-4) {
            printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
        }
    }
}

int main() {

    const unsigned int N = 20 * 1000 * 1000;
    const float initialGuess = 1.0f;

    float* values = new float[N];
    float* output = new float[N];
    float* gold = new float[N];

    for (unsigned int i=0; i<N; i++)
    {
        // TODO: CS149 students.  Attempt to change the values in the
        // array here to meet the instructions in the handout: we want
        // to you generate best and worse-case speedups
        
        // starter code populates array with random input values
        values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;
        
        // maximize speedup
        // values[i] = 2.999f; 
        // minimize speedup
        // values[i] = (i % 8 == 0) ? 2.999f : 1.0f; 
    }

    // generate a gold version to check results
    for (unsigned int i=0; i<N; i++)
        gold[i] = sqrt(values[i]);

    //
    // And run the serial implementation 3 times, again reporting the
    // minimum time.
    //
    double minSerial = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtSerial(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the sqrt using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minISPC = std::min(minISPC, endTime - startTime);
    }

    printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

    verifyResult(N, output, gold);

    // Clear out the buffer
    for (unsigned int i = 0; i < N; ++i)
        output[i] = 0;

    //
    // Tasking version of the ISPC code
    //
    double minTaskISPC = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrt_ispc_withtasks(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minTaskISPC = std::min(minTaskISPC, endTime - startTime);
    }

    printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

    verifyResult(N, output, gold);

    //
    // Compute the sqrt using the ispc implementation; report the minimum
    // time of three runs.
    //
    double minAVX2 = 1e30;
    for (int i = 0; i < 3; ++i) {
        double startTime = CycleTimer::currentSeconds();
        sqrtAVX2(N, initialGuess, values, output);
        double endTime = CycleTimer::currentSeconds();
        minAVX2 = std::min(minAVX2, endTime - startTime);
    }

    printf("[sqrt avx2]:\t\t[%.3f] ms\n", minAVX2 * 1000);

    verifyResult(N, output, gold);

    printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial/minISPC);
    printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial/minTaskISPC);
    printf("\t\t\t\t(%.2fx speedup from AVX2)\n", minSerial/minAVX2);

    delete [] values;
    delete [] output;
    delete [] gold;

    return 0;
}
