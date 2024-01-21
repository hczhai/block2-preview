
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2024 Huanchen Zhai <hczhai.ok@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 */

#pragma once

#ifdef _HAS_SIMD_OPT
#pragma GCC push_options
#pragma GCC target("sse4,avx,avx2,fma")

#include <bits/stdc++.h>
#include <immintrin.h>

using namespace std;

namespace block2 {

// Matrix operations using SIMD parallelization
template <typename FL, typename = void> struct SimdMatrixFunctions;

template <typename FL>
struct SimdMatrixFunctions<
    FL, typename enable_if<is_same<FL, complex<double>>::value>::type> {
    template <size_t nk, size_t ni>
    static void avx_full_copy(uint8_t ta, size_t pk, size_t pi,
                              const complex<double> *a, size_t lda,
                              complex<double> *__restrict__ b) {
        throw runtime_error("!");
    }
    template <size_t nk, size_t ni>
    static void avx_rev_full_copy(uint8_t ta, size_t pk, size_t pi,
                                  const complex<double> *a,
                                  complex<double> *__restrict__ b, size_t ldb) {
        throw runtime_error("!");
    }
    template <uint8_t full_copy, size_t ni, size_t nj, size_t nk>
    static void avx_gemm(uint8_t ta, uint8_t tb, size_t m, size_t n, size_t k,
                         complex<double> alpha, const complex<double> *a,
                         size_t lda, const complex<double> *b, size_t ldb,
                         complex<double> beta, complex<double> *c, size_t ldc) {
        throw runtime_error("!");
    }
};

template <typename FL>
struct SimdMatrixFunctions<
    FL, typename enable_if<is_same<FL, double>::value>::type> {
    // b[m, n] = beta * b[m, n] + alpha * a[m, n]
    static void avx_matrix_nscale(size_t m, size_t n, size_t ni, size_t nj,
                                  const double *a, size_t lda,
                                  double *__restrict__ b, size_t ldb,
                                  double alpha, double beta) {
        for (size_t ii = 0; ii < m; ii += ni) {
            const size_t nni = min(m - ii, ni);
            for (size_t jj = 0; jj < n; jj += nj) {
                const size_t nnj = min(n - jj, nj);
                const double *pa = &a[ii * lda + jj];
                double *pb = &b[ii * ldb + jj];
                size_t xnnj = nnj >> 2 << 2;
                if (nnj >= 4) {
                    __m256d x0, x1, x2, x3, z0, z1;
                    z0 = _mm256_broadcast_sd(&alpha);
                    z1 = _mm256_broadcast_sd(&beta);
                    size_t xnni = nni >> 2 << 2;
                    for (size_t xi = 0; xi < xnni; xi += 4)
                        for (size_t xj = 0; xj < xnnj; xj += 4) {
                            x0 = z0 * _mm256_loadu_pd(&pa[(xi + 0) * lda + xj]);
                            x1 = z0 * _mm256_loadu_pd(&pa[(xi + 1) * lda + xj]);
                            x2 = z0 * _mm256_loadu_pd(&pa[(xi + 2) * lda + xj]);
                            x3 = z0 * _mm256_loadu_pd(&pa[(xi + 3) * lda + xj]);
                            if (beta != 0.0) {
                                x0 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 0) * ldb + xj]);
                                x1 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 1) * ldb + xj]);
                                x2 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 2) * ldb + xj]);
                                x3 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 3) * ldb + xj]);
                            }
                            _mm256_storeu_pd(&pb[(xi + 0) * ldb + xj], x0);
                            _mm256_storeu_pd(&pb[(xi + 1) * ldb + xj], x1);
                            _mm256_storeu_pd(&pb[(xi + 2) * ldb + xj], x2);
                            _mm256_storeu_pd(&pb[(xi + 3) * ldb + xj], x3);
                        }
                    if ((nni - xnni) & 2) {
                        for (size_t xj = 0; xj < xnnj; xj += 4) {
                            x0 = z0 *
                                 _mm256_loadu_pd(&pa[(xnni + 0) * lda + xj]);
                            x1 = z0 *
                                 _mm256_loadu_pd(&pa[(xnni + 1) * lda + xj]);
                            if (beta != 0.0) {
                                x0 += z1 * _mm256_loadu_pd(
                                               &pb[(xnni + 0) * ldb + xj]);
                                x1 += z1 * _mm256_loadu_pd(
                                               &pb[(xnni + 1) * ldb + xj]);
                            }
                            _mm256_storeu_pd(&pb[(xnni + 0) * ldb + xj], x0);
                            _mm256_storeu_pd(&pb[(xnni + 1) * ldb + xj], x1);
                        }
                        xnni += 2;
                    }
                    if ((nni - xnni) & 1) {
                        for (size_t xj = 0; xj < xnnj; xj += 4) {
                            x0 = z0 *
                                 _mm256_loadu_pd(&pa[(xnni + 0) * lda + xj]);
                            if (beta != 0.0)
                                x0 += z1 * _mm256_loadu_pd(
                                               &pb[(xnni + 0) * ldb + xj]);
                            _mm256_storeu_pd(&pb[(xnni + 0) * ldb + xj], x0);
                        }
                    }
                }
                if ((nnj - xnnj) & 2) {
                    __m128d x0, x1, x2, x3, z0, z1;
                    z0 = _mm_set1_pd(alpha);
                    z1 = _mm_set1_pd(beta);
                    size_t xnni = nni >> 2 << 2;
                    for (size_t xi = 0; xi < xnni; xi += 4) {
                        x0 = z0 * _mm_loadu_pd(&pa[(xi + 0) * lda + xnnj]);
                        x1 = z0 * _mm_loadu_pd(&pa[(xi + 1) * lda + xnnj]);
                        x2 = z0 * _mm_loadu_pd(&pa[(xi + 2) * lda + xnnj]);
                        x3 = z0 * _mm_loadu_pd(&pa[(xi + 3) * lda + xnnj]);
                        if (beta != 0.0) {
                            x0 += z1 * _mm_loadu_pd(&pb[(xi + 0) * ldb + xnnj]);
                            x1 += z1 * _mm_loadu_pd(&pb[(xi + 1) * ldb + xnnj]);
                            x2 += z1 * _mm_loadu_pd(&pb[(xi + 2) * ldb + xnnj]);
                            x3 += z1 * _mm_loadu_pd(&pb[(xi + 3) * ldb + xnnj]);
                        }
                        _mm_storeu_pd(&pb[(xi + 0) * ldb + xnnj], x0);
                        _mm_storeu_pd(&pb[(xi + 1) * ldb + xnnj], x1);
                        _mm_storeu_pd(&pb[(xi + 2) * ldb + xnnj], x2);
                        _mm_storeu_pd(&pb[(xi + 3) * ldb + xnnj], x3);
                    }
                    if ((nni - xnni) & 2) {
                        x0 = z0 * _mm_loadu_pd(&pa[(xnni + 0) * lda + xnnj]);
                        x1 = z0 * _mm_loadu_pd(&pa[(xnni + 1) * lda + xnnj]);
                        if (beta != 0.0) {
                            x0 +=
                                z1 * _mm_loadu_pd(&pb[(xnni + 0) * ldb + xnnj]);
                            x1 +=
                                z1 * _mm_loadu_pd(&pb[(xnni + 1) * ldb + xnnj]);
                        }
                        _mm_storeu_pd(&pb[(xnni + 0) * ldb + xnnj], x0);
                        _mm_storeu_pd(&pb[(xnni + 1) * ldb + xnnj], x1);
                        xnni += 2;
                    }
                    if ((nni - xnni) & 1) {
                        x0 = z0 * _mm_loadu_pd(&pa[(xnni + 0) * lda + xnnj]);
                        if (beta != 0.0)
                            x0 +=
                                z1 * _mm_loadu_pd(&pb[(xnni + 0) * ldb + xnnj]);
                        _mm_storeu_pd(&pb[(xnni + 0) * ldb + xnnj], x0);
                    }
                    xnnj += 2;
                }
                if ((nnj - xnnj) & 1) {
                    size_t xnni = nni >> 2 << 2;
                    if (beta != 0.0) {
                        for (size_t xi = 0; xi < xnni; xi += 4) {
                            pb[(xi + 0) * ldb + xnnj] =
                                beta * pb[(xi + 0) * ldb + xnnj] +
                                alpha * pa[(xi + 0) * lda + xnnj];
                            pb[(xi + 1) * ldb + xnnj] =
                                beta * pb[(xi + 1) * ldb + xnnj] +
                                alpha * pa[(xi + 1) * lda + xnnj];
                            pb[(xi + 2) * ldb + xnnj] =
                                beta * pb[(xi + 2) * ldb + xnnj] +
                                alpha * pa[(xi + 2) * lda + xnnj];
                            pb[(xi + 3) * ldb + xnnj] =
                                beta * pb[(xi + 3) * ldb + xnnj] +
                                alpha * pa[(xi + 3) * lda + xnnj];
                        }
                        if ((nni - xnni) & 2) {
                            pb[(xnni + 0) * ldb + xnnj] =
                                beta * pb[(xnni + 0) * ldb + xnnj] +
                                alpha * pa[(xnni + 0) * lda + xnnj];
                            pb[(xnni + 1) * ldb + xnnj] =
                                beta * pb[(xnni + 1) * ldb + xnnj] +
                                alpha * pa[(xnni + 1) * lda + xnnj];
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1)
                            pb[(xnni + 0) * ldb + xnnj] =
                                beta * pb[(xnni + 0) * ldb + xnnj] +
                                alpha * pa[(xnni + 0) * lda + xnnj];
                    } else {
                        for (size_t xi = 0; xi < xnni; xi += 4) {
                            pb[(xi + 0) * ldb + xnnj] =
                                alpha * pa[(xi + 0) * lda + xnnj];
                            pb[(xi + 1) * ldb + xnnj] =
                                alpha * pa[(xi + 1) * lda + xnnj];
                            pb[(xi + 2) * ldb + xnnj] =
                                alpha * pa[(xi + 2) * lda + xnnj];
                            pb[(xi + 3) * ldb + xnnj] =
                                alpha * pa[(xi + 3) * lda + xnnj];
                        }
                        if ((nni - xnni) & 2) {
                            pb[(xnni + 0) * ldb + xnnj] =
                                alpha * pa[(xnni + 0) * lda + xnnj];
                            pb[(xnni + 1) * ldb + xnnj] =
                                alpha * pa[(xnni + 1) * lda + xnnj];
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1)
                            pb[(xnni + 0) * ldb + xnnj] =
                                alpha * pa[(xnni + 0) * lda + xnnj];
                    }
                }
            }
        }
    }
    // b[m, n] = beta * b[m, n] + alpha * a[n, m]
    static void avx_matrix_tscale(size_t m, size_t n, size_t ni, size_t nj,
                                  const double *a, size_t lda,
                                  double *__restrict__ b, size_t ldb,
                                  double alpha, double beta) {
        for (size_t ii = 0; ii < m; ii += ni) {
            const size_t nni = min(m - ii, ni);
            for (size_t jj = 0; jj < n; jj += nj) {
                const size_t nnj = min(n - jj, nj);
                const double *pa = &a[jj * lda + ii];
                double *pb = &b[ii * ldb + jj];
                size_t xnnj = nnj >> 2 << 2;
                if (nnj >= 4) {
                    __m256d x0, x1, x2, x3, r3, r33, r4, r34, z0, z1;
                    z0 = _mm256_broadcast_sd(&alpha);
                    z1 = _mm256_broadcast_sd(&beta);
                    size_t xnni = nni >> 2 << 2;
                    for (size_t xi = 0; xi < xnni; xi += 4)
                        for (size_t xj = 0; xj < xnnj; xj += 4) {
                            x0 = z0 * _mm256_loadu_pd(&pa[(xj + 0) * lda + xi]);
                            x1 = z0 * _mm256_loadu_pd(&pa[(xj + 1) * lda + xi]);
                            x2 = z0 * _mm256_loadu_pd(&pa[(xj + 2) * lda + xi]);
                            x3 = z0 * _mm256_loadu_pd(&pa[(xj + 3) * lda + xi]);
                            r3 = _mm256_shuffle_pd(x0, x1, 0x3);
                            r4 = _mm256_shuffle_pd(x0, x1, 0xc);
                            r33 = _mm256_shuffle_pd(x2, x3, 0x3);
                            r34 = _mm256_shuffle_pd(x2, x3, 0xc);
                            x0 = _mm256_permute2f128_pd(r34, r4, 0x2);
                            x1 = _mm256_permute2f128_pd(r33, r3, 0x2);
                            x2 = _mm256_permute2f128_pd(r33, r3, 0x13);
                            x3 = _mm256_permute2f128_pd(r34, r4, 0x13);
                            if (beta != 0.0) {
                                x0 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 0) * ldb + xj]);
                                x1 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 1) * ldb + xj]);
                                x2 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 2) * ldb + xj]);
                                x3 += z1 *
                                      _mm256_loadu_pd(&pb[(xi + 3) * ldb + xj]);
                            }
                            _mm256_storeu_pd(&pb[(xi + 0) * ldb + xj], x0);
                            _mm256_storeu_pd(&pb[(xi + 1) * ldb + xj], x1);
                            _mm256_storeu_pd(&pb[(xi + 2) * ldb + xj], x2);
                            _mm256_storeu_pd(&pb[(xi + 3) * ldb + xj], x3);
                        }
                    if ((nni - xnni) & 2) {
                        __m128d w0, w1, w2, w3, w00, w10, w01, w11, wz0, wz1;
                        wz0 = _mm_set1_pd(alpha);
                        wz1 = _mm_set1_pd(beta);
                        for (size_t xj = 0; xj < xnnj; xj += 4) {
                            w0 = wz0 * _mm_loadu_pd(&pa[(xj + 0) * lda + xnni]);
                            w1 = wz0 * _mm_loadu_pd(&pa[(xj + 1) * lda + xnni]);
                            w2 = wz0 * _mm_loadu_pd(&pa[(xj + 2) * lda + xnni]);
                            w3 = wz0 * _mm_loadu_pd(&pa[(xj + 3) * lda + xnni]);
                            w00 = _mm_shuffle_pd(w0, w1, 0x0);
                            w10 = _mm_shuffle_pd(w0, w1, 0x3);
                            w01 = _mm_shuffle_pd(w2, w3, 0x0);
                            w11 = _mm_shuffle_pd(w2, w3, 0x3);
                            if (beta != 0.0) {
                                w00 +=
                                    wz1 * _mm_loadu_pd(
                                              &pb[(xnni + 0) * ldb + xj + 0]);
                                w01 +=
                                    wz1 * _mm_loadu_pd(
                                              &pb[(xnni + 0) * ldb + xj + 2]);
                                w10 +=
                                    wz1 * _mm_loadu_pd(
                                              &pb[(xnni + 1) * ldb + xj + 0]);
                                w11 +=
                                    wz1 * _mm_loadu_pd(
                                              &pb[(xnni + 1) * ldb + xj + 2]);
                            }
                            _mm_storeu_pd(&pb[(xnni + 0) * ldb + xj + 0], w00);
                            _mm_storeu_pd(&pb[(xnni + 0) * ldb + xj + 2], w01);
                            _mm_storeu_pd(&pb[(xnni + 1) * ldb + xj + 0], w10);
                            _mm_storeu_pd(&pb[(xnni + 1) * ldb + xj + 2], w11);
                        }
                        xnni += 2;
                    }
                    if ((nni - xnni) & 1) {
                        for (size_t xj = 0; xj < xnnj; xj += 4)
                            if (beta != 0.0) {
                                pb[xnni * ldb + xj + 0] =
                                    beta * pb[xnni * ldb + xj + 0] +
                                    alpha * pa[(xj + 0) * lda + xnni];
                                pb[xnni * ldb + xj + 1] =
                                    beta * pb[xnni * ldb + xj + 1] +
                                    alpha * pa[(xj + 1) * lda + xnni];
                                pb[xnni * ldb + xj + 2] =
                                    beta * pb[xnni * ldb + xj + 2] +
                                    alpha * pa[(xj + 2) * lda + xnni];
                                pb[xnni * ldb + xj + 3] =
                                    beta * pb[xnni * ldb + xj + 3] +
                                    alpha * pa[(xj + 3) * lda + xnni];
                            } else {
                                pb[xnni * ldb + xj + 0] =
                                    alpha * pa[(xj + 0) * lda + xnni];
                                pb[xnni * ldb + xj + 1] =
                                    alpha * pa[(xj + 1) * lda + xnni];
                                pb[xnni * ldb + xj + 2] =
                                    alpha * pa[(xj + 2) * lda + xnni];
                                pb[xnni * ldb + xj + 3] =
                                    alpha * pa[(xj + 3) * lda + xnni];
                            }
                    }
                }
                if ((nnj - xnnj) & 2) {
                    __m128d x0, x1, x2, x3, x00, x10, x01, x11, z0, z1;
                    z0 = _mm_set1_pd(alpha);
                    z1 = _mm_set1_pd(beta);
                    size_t xnni = nni >> 2 << 2;
                    for (size_t xi = 0; xi < xnni; xi += 4) {
                        x0 = z0 * _mm_loadu_pd(&pa[(xnnj + 0) * lda + xi + 0]);
                        x1 = z0 * _mm_loadu_pd(&pa[(xnnj + 0) * lda + xi + 2]);
                        x2 = z0 * _mm_loadu_pd(&pa[(xnnj + 1) * lda + xi + 0]);
                        x3 = z0 * _mm_loadu_pd(&pa[(xnnj + 1) * lda + xi + 2]);
                        x00 = _mm_shuffle_pd(x0, x2, 0x0);
                        x10 = _mm_shuffle_pd(x0, x2, 0x3);
                        x01 = _mm_shuffle_pd(x1, x3, 0x0);
                        x11 = _mm_shuffle_pd(x1, x3, 0x3);
                        if (beta != 0.0) {
                            x00 +=
                                z1 * _mm_loadu_pd(&pb[(xi + 0) * ldb + xnnj]);
                            x10 +=
                                z1 * _mm_loadu_pd(&pb[(xi + 1) * ldb + xnnj]);
                            x01 +=
                                z1 * _mm_loadu_pd(&pb[(xi + 2) * ldb + xnnj]);
                            x11 +=
                                z1 * _mm_loadu_pd(&pb[(xi + 3) * ldb + xnnj]);
                        }
                        _mm_storeu_pd(&pb[(xi + 0) * ldb + xnnj], x00);
                        _mm_storeu_pd(&pb[(xi + 1) * ldb + xnnj], x10);
                        _mm_storeu_pd(&pb[(xi + 2) * ldb + xnnj], x01);
                        _mm_storeu_pd(&pb[(xi + 3) * ldb + xnnj], x11);
                    }
                    if ((nni - xnni) & 2) {
                        x0 = z0 * _mm_loadu_pd(&pa[(xnnj + 0) * lda + xnni]);
                        x1 = z0 * _mm_loadu_pd(&pa[(xnnj + 1) * lda + xnni]);
                        x00 = _mm_shuffle_pd(x0, x1, 0x0);
                        x10 = _mm_shuffle_pd(x0, x1, 0x3);
                        if (beta != 0.0) {
                            x00 +=
                                z1 * _mm_loadu_pd(&pb[(xnni + 0) * ldb + xnnj]);
                            x10 +=
                                z1 * _mm_loadu_pd(&pb[(xnni + 1) * ldb + xnnj]);
                        }
                        _mm_storeu_pd(&pb[(xnni + 0) * ldb + xnnj], x00);
                        _mm_storeu_pd(&pb[(xnni + 1) * ldb + xnnj], x10);
                        xnni += 2;
                    }
                    if ((nni - xnni) & 1) {
                        if (beta != 0.0) {
                            pb[xnni * ldb + xnnj + 0] =
                                beta * pb[xnni * ldb + xnnj + 0] +
                                alpha * pa[(xnnj + 0) * lda + xnni];
                            pb[xnni * ldb + xnnj + 1] =
                                beta * pb[xnni * ldb + xnnj + 1] +
                                alpha * pa[(xnnj + 1) * lda + xnni];
                        } else {
                            pb[xnni * ldb + xnnj + 0] =
                                alpha * pa[(xnnj + 0) * lda + xnni];
                            pb[xnni * ldb + xnnj + 1] =
                                alpha * pa[(xnnj + 1) * lda + xnni];
                        }
                    }
                    xnnj += 2;
                }
                if ((nnj - xnnj) & 1) {
                    size_t xnni = nni >> 2 << 2;
                    if (beta != 0.0) {
                        for (size_t xi = 0; xi < xnni; xi += 4) {
                            pb[(xi + 0) * ldb + xnnj] =
                                beta * pb[(xi + 0) * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xi + 0];
                            pb[(xi + 1) * ldb + xnnj] =
                                beta * pb[(xi + 1) * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xi + 1];
                            pb[(xi + 2) * ldb + xnnj] =
                                beta * pb[(xi + 2) * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xi + 2];
                            pb[(xi + 3) * ldb + xnnj] =
                                beta * pb[(xi + 3) * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xi + 3];
                        }
                        if ((nni - xnni) & 2) {
                            pb[(xnni + 0) * ldb + xnnj] =
                                beta * pb[(xnni + 0) * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xnni + 0];
                            pb[(xnni + 1) * ldb + xnnj] =
                                beta * pb[(xnni + 1) * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xnni + 1];
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1)
                            pb[xnni * ldb + xnnj] =
                                beta * pb[xnni * ldb + xnnj] +
                                alpha * pa[xnnj * lda + xnni];
                    } else {
                        for (size_t xi = 0; xi < xnni; xi += 4) {
                            pb[(xi + 0) * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xi + 0];
                            pb[(xi + 1) * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xi + 1];
                            pb[(xi + 2) * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xi + 2];
                            pb[(xi + 3) * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xi + 3];
                        }
                        if ((nni - xnni) & 2) {
                            pb[(xnni + 0) * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xnni + 0];
                            pb[(xnni + 1) * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xnni + 1];
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1)
                            pb[xnni * ldb + xnnj] =
                                alpha * pa[xnnj * lda + xnni];
                    }
                }
            }
        }
    }
    // copy a[nk, ni] -> b[ni / 4, nk, 4]
    static void avx_tcopy_4(size_t ni, size_t nk, const double *a, size_t lda,
                            double *__restrict__ b) {
        for (size_t mi = ni >> 2 << 2; mi >= 4; mi -= 4) {
            __m256d x0, x1, x2, x3;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                x0 = _mm256_loadu_pd(&a[mi - 4 + (mk - 4) * lda]);
                x1 = _mm256_loadu_pd(&a[mi - 4 + (mk - 3) * lda]);
                x2 = _mm256_loadu_pd(&a[mi - 4 + (mk - 2) * lda]);
                x3 = _mm256_loadu_pd(&a[mi - 4 + (mk - 1) * lda]);
                _mm256_storeu_pd(&b[(mk - 4) * 4 + (mi - 4) * nk], x0);
                _mm256_storeu_pd(&b[(mk - 3) * 4 + (mi - 4) * nk], x1);
                _mm256_storeu_pd(&b[(mk - 2) * 4 + (mi - 4) * nk], x2);
                _mm256_storeu_pd(&b[(mk - 1) * 4 + (mi - 4) * nk], x3);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                x0 = _mm256_loadu_pd(&a[mi - 4 + (mk - 2) * lda]);
                x1 = _mm256_loadu_pd(&a[mi - 4 + (mk - 1) * lda]);
                _mm256_storeu_pd(&b[(mk - 2) * 4 + (mi - 4) * nk], x0);
                _mm256_storeu_pd(&b[(mk - 1) * 4 + (mi - 4) * nk], x1);
            }
            if (nk & 1) {
                x0 = _mm256_loadu_pd(&a[mi - 4 + (nk - 1) * lda]);
                _mm256_storeu_pd(&b[(nk - 1) * 4 + (mi - 4) * nk], x0);
            }
        }
        if (ni & 2) {
            const size_t mi = (ni >> 1 << 1) - 2;
            __m128d x0, x1, x2, x3;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                x0 = _mm_loadu_pd(&a[mi + (mk - 4) * lda]);
                x1 = _mm_loadu_pd(&a[mi + (mk - 3) * lda]);
                x2 = _mm_loadu_pd(&a[mi + (mk - 2) * lda]);
                x3 = _mm_loadu_pd(&a[mi + (mk - 1) * lda]);
                _mm_storeu_pd(&b[(mk - 4) * 2 + mi * nk], x0);
                _mm_storeu_pd(&b[(mk - 3) * 2 + mi * nk], x1);
                _mm_storeu_pd(&b[(mk - 2) * 2 + mi * nk], x2);
                _mm_storeu_pd(&b[(mk - 1) * 2 + mi * nk], x3);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                x0 = _mm_loadu_pd(&a[mi + (mk - 2) * lda]);
                x1 = _mm_loadu_pd(&a[mi + (mk - 1) * lda]);
                _mm_storeu_pd(&b[(mk - 2) * 2 + mi * nk], x0);
                _mm_storeu_pd(&b[(mk - 1) * 2 + mi * nk], x1);
            }
            if (nk & 1) {
                x0 = _mm_loadu_pd(&a[mi + (nk - 1) * lda]);
                _mm_storeu_pd(&b[(nk - 1) * 2 + mi * nk], x0);
            }
        }
        if (ni & 1) {
            const size_t mi = ni - 1;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                b[mk - 4 + mi * nk] = a[mi + (mk - 4) * lda];
                b[mk - 3 + mi * nk] = a[mi + (mk - 3) * lda];
                b[mk - 2 + mi * nk] = a[mi + (mk - 2) * lda];
                b[mk - 1 + mi * nk] = a[mi + (mk - 1) * lda];
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                b[mk - 2 + mi * nk] = a[mi + (mk - 2) * lda];
                b[mk - 1 + mi * nk] = a[mi + (mk - 1) * lda];
            }
            if (nk & 1)
                b[nk - 1 + mi * nk] = a[mi + (nk - 1) * lda];
        }
    }
    // copy a[ni, nk] -> b[ni / 4, nk, 4]
    static void avx_ncopy_4(size_t ni, size_t nk, const double *a, size_t lda,
                            double *__restrict__ b) {
        for (size_t mi = ni >> 2 << 2; mi >= 4; mi -= 4) {
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                __m256d x0, x1, x2, x3, r3, r33, r4, r34;
                x0 = _mm256_loadu_pd(&a[mk - 4 + (mi - 4) * lda]);
                x1 = _mm256_loadu_pd(&a[mk - 4 + (mi - 3) * lda]);
                x2 = _mm256_loadu_pd(&a[mk - 4 + (mi - 2) * lda]);
                x3 = _mm256_loadu_pd(&a[mk - 4 + (mi - 1) * lda]);
                r3 = _mm256_shuffle_pd(x0, x1, 0x3);
                r4 = _mm256_shuffle_pd(x0, x1, 0xc);
                r33 = _mm256_shuffle_pd(x2, x3, 0x3);
                r34 = _mm256_shuffle_pd(x2, x3, 0xc);
                x0 = _mm256_permute2f128_pd(r34, r4, 0x2);
                x1 = _mm256_permute2f128_pd(r33, r3, 0x2);
                x2 = _mm256_permute2f128_pd(r33, r3, 0x13);
                x3 = _mm256_permute2f128_pd(r34, r4, 0x13);
                _mm256_storeu_pd(&b[(mk - 4) * 4 + (mi - 4) * nk], x0);
                _mm256_storeu_pd(&b[(mk - 3) * 4 + (mi - 4) * nk], x1);
                _mm256_storeu_pd(&b[(mk - 2) * 4 + (mi - 4) * nk], x2);
                _mm256_storeu_pd(&b[(mk - 1) * 4 + (mi - 4) * nk], x3);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                __m128d x0, x1, x2, x3, x00, x10, x01, x11;
                x0 = _mm_loadu_pd(&a[mk - 2 + (mi - 4) * lda]);
                x1 = _mm_loadu_pd(&a[mk - 2 + (mi - 3) * lda]);
                x2 = _mm_loadu_pd(&a[mk - 2 + (mi - 2) * lda]);
                x3 = _mm_loadu_pd(&a[mk - 2 + (mi - 1) * lda]);
                x00 = _mm_shuffle_pd(x0, x1, 0x0);
                x10 = _mm_shuffle_pd(x0, x1, 0x3);
                x01 = _mm_shuffle_pd(x2, x3, 0x0);
                x11 = _mm_shuffle_pd(x2, x3, 0x3);
                _mm_storeu_pd(&b[(mk - 2) * 4 + 0 + (mi - 4) * nk], x00);
                _mm_storeu_pd(&b[(mk - 2) * 4 + 2 + (mi - 4) * nk], x01);
                _mm_storeu_pd(&b[(mk - 1) * 4 + 0 + (mi - 4) * nk], x10);
                _mm_storeu_pd(&b[(mk - 1) * 4 + 2 + (mi - 4) * nk], x11);
            }
            if (nk & 1) {
                b[(nk - 1) * 4 + 0 + (mi - 4) * nk] =
                    a[nk - 1 + (mi - 4) * lda];
                b[(nk - 1) * 4 + 1 + (mi - 4) * nk] =
                    a[nk - 1 + (mi - 3) * lda];
                b[(nk - 1) * 4 + 2 + (mi - 4) * nk] =
                    a[nk - 1 + (mi - 2) * lda];
                b[(nk - 1) * 4 + 3 + (mi - 4) * nk] =
                    a[nk - 1 + (mi - 1) * lda];
            }
        }
        if (ni & 2) {
            const size_t mi = ni >> 1 << 1;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                __m128d x0, x1, x2, x3, x00, x10, x01, x11;
                x0 = _mm_loadu_pd(&a[mk - 4 + (mi - 2) * lda]);
                x1 = _mm_loadu_pd(&a[mk - 2 + (mi - 2) * lda]);
                x2 = _mm_loadu_pd(&a[mk - 4 + (mi - 1) * lda]);
                x3 = _mm_loadu_pd(&a[mk - 2 + (mi - 1) * lda]);
                x00 = _mm_shuffle_pd(x0, x2, 0x0);
                x10 = _mm_shuffle_pd(x0, x2, 0x3);
                x01 = _mm_shuffle_pd(x1, x3, 0x0);
                x11 = _mm_shuffle_pd(x1, x3, 0x3);
                _mm_storeu_pd(&b[(mk - 4) * 2 + (mi - 2) * nk], x00);
                _mm_storeu_pd(&b[(mk - 3) * 2 + (mi - 2) * nk], x10);
                _mm_storeu_pd(&b[(mk - 2) * 2 + (mi - 2) * nk], x01);
                _mm_storeu_pd(&b[(mk - 1) * 2 + (mi - 2) * nk], x11);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                __m128d x0, x1, x00, x10;
                x0 = _mm_loadu_pd(&a[mk - 2 + (mi - 2) * lda]);
                x1 = _mm_loadu_pd(&a[mk - 2 + (mi - 1) * lda]);
                x00 = _mm_shuffle_pd(x0, x1, 0x0);
                x10 = _mm_shuffle_pd(x0, x1, 0x3);
                _mm_storeu_pd(&b[(mk - 2) * 2 + (mi - 2) * nk], x00);
                _mm_storeu_pd(&b[(mk - 1) * 2 + (mi - 2) * nk], x10);
            }
            if (nk & 1) {
                b[(nk - 1) * 2 + 0 + (mi - 2) * nk] =
                    a[nk - 1 + (mi - 2) * lda];
                b[(nk - 1) * 2 + 1 + (mi - 2) * nk] =
                    a[nk - 1 + (mi - 1) * lda];
            }
        }
        if (ni & 1) {
            const size_t mi = ni - 1;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                b[mk - 4 + mi * nk] = a[mk - 4 + mi * lda];
                b[mk - 3 + mi * nk] = a[mk - 3 + mi * lda];
                b[mk - 2 + mi * nk] = a[mk - 2 + mi * lda];
                b[mk - 1 + mi * nk] = a[mk - 1 + mi * lda];
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                b[mk - 2 + mi * nk] = a[mk - 2 + mi * lda];
                b[mk - 1 + mi * nk] = a[mk - 1 + mi * lda];
            }
            if (nk & 1)
                b[nk - 1 + mi * nk] = a[nk - 1 + mi * lda];
        }
    }
    // copy a[nk, ni] <- b[ni / 4, nk, 4]
    static void avx_rev_tcopy_4(size_t ni, size_t nk, const double *b,
                                double *__restrict__ a, size_t lda) {
        for (size_t mi = ni >> 2 << 2; mi >= 4; mi -= 4) {
            __m256d x0, x1, x2, x3;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                x0 = _mm256_loadu_pd(&b[(mk - 4) * 4 + (mi - 4) * nk]);
                x1 = _mm256_loadu_pd(&b[(mk - 3) * 4 + (mi - 4) * nk]);
                x2 = _mm256_loadu_pd(&b[(mk - 2) * 4 + (mi - 4) * nk]);
                x3 = _mm256_loadu_pd(&b[(mk - 1) * 4 + (mi - 4) * nk]);
                _mm256_storeu_pd(&a[mi - 4 + (mk - 4) * lda], x0);
                _mm256_storeu_pd(&a[mi - 4 + (mk - 3) * lda], x1);
                _mm256_storeu_pd(&a[mi - 4 + (mk - 2) * lda], x2);
                _mm256_storeu_pd(&a[mi - 4 + (mk - 1) * lda], x3);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                x0 = _mm256_loadu_pd(&b[(mk - 2) * 4 + (mi - 4) * nk]);
                x1 = _mm256_loadu_pd(&b[(mk - 1) * 4 + (mi - 4) * nk]);
                _mm256_storeu_pd(&a[mi - 4 + (mk - 2) * lda], x0);
                _mm256_storeu_pd(&a[mi - 4 + (mk - 1) * lda], x1);
            }
            if (nk & 1) {
                x0 = _mm256_loadu_pd(&b[(nk - 1) * 4 + (mi - 4) * nk]);
                _mm256_storeu_pd(&a[mi - 4 + (nk - 1) * lda], x0);
            }
        }
        if (ni & 2) {
            const size_t mi = (ni >> 1 << 1) - 2;
            __m128d x0, x1, x2, x3;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                x0 = _mm_loadu_pd(&b[(mk - 4) * 2 + mi * nk]);
                x1 = _mm_loadu_pd(&b[(mk - 3) * 2 + mi * nk]);
                x2 = _mm_loadu_pd(&b[(mk - 2) * 2 + mi * nk]);
                x3 = _mm_loadu_pd(&b[(mk - 1) * 2 + mi * nk]);
                _mm_storeu_pd(&a[mi + (mk - 4) * lda], x0);
                _mm_storeu_pd(&a[mi + (mk - 3) * lda], x1);
                _mm_storeu_pd(&a[mi + (mk - 2) * lda], x2);
                _mm_storeu_pd(&a[mi + (mk - 1) * lda], x3);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                x0 = _mm_loadu_pd(&b[(mk - 2) * 2 + mi * nk]);
                x1 = _mm_loadu_pd(&b[(mk - 1) * 2 + mi * nk]);
                _mm_storeu_pd(&a[mi + (mk - 2) * lda], x0);
                _mm_storeu_pd(&a[mi + (mk - 1) * lda], x1);
            }
            if (nk & 1) {
                x0 = _mm_loadu_pd(&b[(nk - 1) * 2 + mi * nk]);
                _mm_storeu_pd(&a[mi + (nk - 1) * lda], x0);
            }
        }
        if (ni & 1) {
            const size_t mi = ni - 1;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                a[mi + (mk - 4) * lda] = b[mk - 4 + mi * nk];
                a[mi + (mk - 3) * lda] = b[mk - 3 + mi * nk];
                a[mi + (mk - 2) * lda] = b[mk - 2 + mi * nk];
                a[mi + (mk - 1) * lda] = b[mk - 1 + mi * nk];
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                a[mi + (mk - 2) * lda] = b[mk - 2 + mi * nk];
                a[mi + (mk - 1) * lda] = b[mk - 1 + mi * nk];
            }
            if (nk & 1)
                a[mi + (nk - 1) * lda] = b[nk - 1 + mi * nk];
        }
    }
    // copy a[ni, nk] <- b[ni / 4, nk, 4]
    static void avx_rev_ncopy_4(size_t ni, size_t nk, const double *b,
                                double *__restrict__ a, size_t lda) {
        for (size_t mi = ni >> 2 << 2; mi >= 4; mi -= 4) {
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                __m256d x0, x1, x2, x3, r3, r33, r4, r34;
                x0 = _mm256_loadu_pd(&b[(mk - 4) * 4 + (mi - 4) * nk]);
                x1 = _mm256_loadu_pd(&b[(mk - 3) * 4 + (mi - 4) * nk]);
                x2 = _mm256_loadu_pd(&b[(mk - 2) * 4 + (mi - 4) * nk]);
                x3 = _mm256_loadu_pd(&b[(mk - 1) * 4 + (mi - 4) * nk]);
                r3 = _mm256_shuffle_pd(x0, x1, 0x3);
                r4 = _mm256_shuffle_pd(x0, x1, 0xc);
                r33 = _mm256_shuffle_pd(x2, x3, 0x3);
                r34 = _mm256_shuffle_pd(x2, x3, 0xc);
                x0 = _mm256_permute2f128_pd(r34, r4, 0x2);
                x1 = _mm256_permute2f128_pd(r33, r3, 0x2);
                x2 = _mm256_permute2f128_pd(r33, r3, 0x13);
                x3 = _mm256_permute2f128_pd(r34, r4, 0x13);
                _mm256_storeu_pd(&a[mk - 4 + (mi - 4) * lda], x0);
                _mm256_storeu_pd(&a[mk - 4 + (mi - 3) * lda], x1);
                _mm256_storeu_pd(&a[mk - 4 + (mi - 2) * lda], x2);
                _mm256_storeu_pd(&a[mk - 4 + (mi - 1) * lda], x3);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                __m128d x0, x1, x2, x3, x00, x10, x01, x11;
                x0 = _mm_loadu_pd(&b[(mk - 2) * 4 + 0 + (mi - 4) * nk]);
                x1 = _mm_loadu_pd(&b[(mk - 2) * 4 + 2 + (mi - 4) * nk]);
                x2 = _mm_loadu_pd(&b[(mk - 1) * 4 + 0 + (mi - 4) * nk]);
                x3 = _mm_loadu_pd(&b[(mk - 1) * 4 + 2 + (mi - 4) * nk]);
                x00 = _mm_shuffle_pd(x0, x2, 0x0);
                x10 = _mm_shuffle_pd(x0, x2, 0x3);
                x01 = _mm_shuffle_pd(x1, x3, 0x0);
                x11 = _mm_shuffle_pd(x1, x3, 0x3);
                _mm_storeu_pd(&a[mk - 2 + (mi - 4) * lda], x00);
                _mm_storeu_pd(&a[mk - 2 + (mi - 3) * lda], x10);
                _mm_storeu_pd(&a[mk - 2 + (mi - 2) * lda], x01);
                _mm_storeu_pd(&a[mk - 2 + (mi - 1) * lda], x11);
            }
            if (nk & 1) {
                a[nk - 1 + (mi - 4) * lda] =
                    b[(nk - 1) * 4 + 0 + (mi - 4) * nk];
                a[nk - 1 + (mi - 3) * lda] =
                    b[(nk - 1) * 4 + 1 + (mi - 4) * nk];
                a[nk - 1 + (mi - 2) * lda] =
                    b[(nk - 1) * 4 + 2 + (mi - 4) * nk];
                a[nk - 1 + (mi - 1) * lda] =
                    b[(nk - 1) * 4 + 3 + (mi - 4) * nk];
            }
        }
        if (ni & 2) {
            const size_t mi = ni >> 1 << 1;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                __m128d x0, x1, x2, x3, x00, x10, x01, x11;
                x0 = _mm_loadu_pd(&b[(mk - 4) * 2 + (mi - 2) * nk]);
                x1 = _mm_loadu_pd(&b[(mk - 3) * 2 + (mi - 2) * nk]);
                x2 = _mm_loadu_pd(&b[(mk - 2) * 2 + (mi - 2) * nk]);
                x3 = _mm_loadu_pd(&b[(mk - 1) * 2 + (mi - 2) * nk]);
                x00 = _mm_shuffle_pd(x0, x1, 0x0);
                x10 = _mm_shuffle_pd(x0, x1, 0x3);
                x01 = _mm_shuffle_pd(x2, x3, 0x0);
                x11 = _mm_shuffle_pd(x2, x3, 0x3);
                _mm_storeu_pd(&a[mk - 4 + (mi - 2) * lda], x00);
                _mm_storeu_pd(&a[mk - 2 + (mi - 2) * lda], x01);
                _mm_storeu_pd(&a[mk - 4 + (mi - 1) * lda], x10);
                _mm_storeu_pd(&a[mk - 2 + (mi - 1) * lda], x11);
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                __m128d x0, x1, x00, x10;
                x0 = _mm_loadu_pd(&b[(mk - 2) * 2 + (mi - 2) * nk]);
                x1 = _mm_loadu_pd(&b[(mk - 1) * 2 + (mi - 2) * nk]);
                x00 = _mm_shuffle_pd(x0, x1, 0x0);
                x10 = _mm_shuffle_pd(x0, x1, 0x3);
                _mm_storeu_pd(&a[mk - 2 + (mi - 2) * lda], x00);
                _mm_storeu_pd(&a[mk - 2 + (mi - 1) * lda], x10);
            }
            if (nk & 1) {
                a[nk - 1 + (mi - 2) * lda] =
                    b[(nk - 1) * 2 + 0 + (mi - 2) * nk];
                a[nk - 1 + (mi - 1) * lda] =
                    b[(nk - 1) * 2 + 1 + (mi - 2) * nk];
            }
        }
        if (ni & 1) {
            const size_t mi = ni - 1;
            for (size_t mk = nk >> 2 << 2; mk >= 4; mk -= 4) {
                a[mk - 4 + mi * lda] = b[mk - 4 + mi * nk];
                a[mk - 3 + mi * lda] = b[mk - 3 + mi * nk];
                a[mk - 2 + mi * lda] = b[mk - 2 + mi * nk];
                a[mk - 1 + mi * lda] = b[mk - 1 + mi * nk];
            }
            if (nk & 2) {
                const size_t mk = nk >> 1 << 1;
                a[mk - 2 + mi * lda] = b[mk - 2 + mi * nk];
                a[mk - 1 + mi * lda] = b[mk - 1 + mi * nk];
            }
            if (nk & 1)
                a[nk - 1 + mi * lda] = b[nk - 1 + mi * nk];
        }
    }
    template <size_t nk, size_t ni>
    static void avx_full_copy(uint8_t ta, size_t pk, size_t pi, const double *a,
                              size_t lda, double *__restrict__ b) {
        // b[pk, pi, i / 4, k, 4]
        for (size_t kk = 0; kk < pk; kk += nk) {
            const size_t nnk = min(pk - kk, nk);
            for (size_t ii = 0; ii < pi; ii += ni) {
                const size_t nni = min(pi - ii, ni);
                if (ta)
                    avx_ncopy_4(nni, nnk, &a[kk + ii * lda], lda,
                                &b[kk * pi + ii * nnk]);
                else
                    avx_tcopy_4(nni, nnk, &a[ii + kk * lda], lda,
                                &b[kk * pi + ii * nnk]);
            }
        }
    }
    template <size_t nk, size_t ni>
    static void avx_rev_full_copy(uint8_t ta, size_t pk, size_t pi,
                                  const double *a, double *__restrict__ b,
                                  size_t ldb) {
        // b[pk, pi, i / 4, k, 4]
        for (size_t kk = 0; kk < pk; kk += nk) {
            const size_t nnk = min(pk - kk, nk);
            for (size_t ii = 0; ii < pi; ii += ni) {
                const size_t nni = min(pi - ii, ni);
                if (ta)
                    avx_rev_ncopy_4(nni, nnk, &a[kk * pi + ii * nnk],
                                    &b[kk + ii * ldb], ldb);
                else
                    avx_rev_tcopy_4(nni, nnk, &a[kk * pi + ii * nnk],
                                    &b[ii + kk * ldb], ldb);
            }
        }
    }
    template <uint8_t full_copy, size_t ni, size_t nj, size_t nk>
    static void avx_gemm(uint8_t ta, uint8_t tb, size_t m, size_t n, size_t k,
                         double alpha, const double *a, size_t lda,
                         const double *b, size_t ldb, double beta, double *c,
                         size_t ldc) {
        double *__restrict__ xa = nullptr, *__restrict__ xb = nullptr;
        if (full_copy == 2) {
            // assert((ta == 1 && tb == 0));
            xa = (double *)a, xb = (double *)b;
        } else if (full_copy == 3) { // alloc b
            assert(ta == 1);
            xa = (double *)a;
            xb = (double *)aligned_alloc(32, k * n * sizeof(double));
            avx_full_copy<nk, ni>(!tb, k, n, b, ldb, xb);
        } else if (full_copy == 4) { // alloc a
            assert(tb == 0);
            xa = (double *)aligned_alloc(32, k * m * sizeof(double));
            xb = (double *)b;
            avx_full_copy<nk, ni>(ta, k, m, a, lda, xa);
        } else {
            xa = (double *)aligned_alloc(32, min(k, nk) * min(m, ni) *
                                                 sizeof(double));
            xb = (double *)aligned_alloc(32, min(k, nk) * n * sizeof(double));
        }
        for (size_t kk = 0; kk < k; kk += nk) {
            const size_t nnk = min(k - kk, nk);
            for (size_t ii = 0; ii < m; ii += ni) {
                const size_t nni = min(m - ii, ni);
                double *__restrict__ pa =
                    full_copy ? &xa[kk * m + ii * nnk] : &xa[0];
                if (!full_copy) {
                    if (ta)
                        avx_ncopy_4(nni, nnk, &a[kk + ii * lda], lda, pa);
                    else
                        avx_tcopy_4(nni, nnk, &a[ii + kk * lda], lda, pa);
                }
                for (size_t jj = 0; jj < n; jj += nj) {
                    const size_t nnj = min(n - jj, nj);
                    double *__restrict__ pb =
                        full_copy ? &xb[kk * n + jj * nnk] : &xb[jj * nnk];
                    if (!full_copy && ii == 0) {
                        if (tb)
                            avx_tcopy_4(nnj, nnk, &b[jj + kk * ldb], ldb, pb);
                        else
                            avx_ncopy_4(nnj, nnk, &b[kk + jj * ldb], ldb, pb);
                    }
                    __m256d x0, x1, x2, y0, z0, z1, z2, z3, z4, z5, z6, z7, z8,
                        z9, z10, z11;
                    size_t xnnj = nnj >> 2 << 2;
                    if (nnj >= 4) {
                        size_t xnni = nni / 12 * 12;
                        for (size_t xi = 0; xi < xnni; xi += 12)
                            for (size_t xj = 0; xj < xnnj; xj += 4) {
                                const double *px0 = &pa[xi * nnk];
                                const double *px1 = &pa[(xi + 4) * nnk];
                                const double *px2 = &pa[(xi + 8) * nnk];
                                const double *py = &pb[xj * nnk];
                                double *pz = &c[(ii + xi) + (jj + xj) * ldc];
                                z0 = _mm256_setzero_pd();
                                z1 = _mm256_setzero_pd();
                                z2 = _mm256_setzero_pd();
                                z3 = _mm256_setzero_pd();
                                z4 = _mm256_setzero_pd();
                                z5 = _mm256_setzero_pd();
                                z6 = _mm256_setzero_pd();
                                z7 = _mm256_setzero_pd();
                                z8 = _mm256_setzero_pd();
                                z9 = _mm256_setzero_pd();
                                z10 = _mm256_setzero_pd();
                                z11 = _mm256_setzero_pd();
                                for (size_t xk = 0; xk < nnk; xk++) {
                                    x0 = _mm256_loadu_pd(px0);
                                    x1 = _mm256_loadu_pd(px1);
                                    x2 = _mm256_loadu_pd(px2);
                                    y0 = _mm256_broadcast_sd(&py[0]);
                                    z0 = _mm256_fmadd_pd(x0, y0, z0);
                                    z1 = _mm256_fmadd_pd(x1, y0, z1);
                                    z2 = _mm256_fmadd_pd(x2, y0, z2);
                                    y0 = _mm256_broadcast_sd(&py[1]);
                                    z3 = _mm256_fmadd_pd(x0, y0, z3);
                                    z4 = _mm256_fmadd_pd(x1, y0, z4);
                                    z5 = _mm256_fmadd_pd(x2, y0, z5);
                                    y0 = _mm256_broadcast_sd(&py[2]);
                                    z6 = _mm256_fmadd_pd(x0, y0, z6);
                                    z7 = _mm256_fmadd_pd(x1, y0, z7);
                                    z8 = _mm256_fmadd_pd(x2, y0, z8);
                                    y0 = _mm256_broadcast_sd(&py[3]);
                                    z9 = _mm256_fmadd_pd(x0, y0, z9);
                                    z10 = _mm256_fmadd_pd(x1, y0, z10);
                                    z11 = _mm256_fmadd_pd(x2, y0, z11);
                                    px0 += 4, px1 += 4, px2 += 4, py += 4;
                                    _mm_prefetch(&px0[64], _MM_HINT_T0);
                                    _mm_prefetch(&py[64], _MM_HINT_T0);
                                }
                                if (alpha != 1.0) {
                                    y0 = _mm256_broadcast_sd(&alpha);
                                    z0 *= y0, z1 *= y0, z2 *= y0;
                                    z3 *= y0, z4 *= y0, z5 *= y0;
                                    z6 *= y0, z7 *= y0, z8 *= y0;
                                    z9 *= y0, z10 *= y0, z11 *= y0;
                                }
                                if (kk != 0) {
                                    z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                    z1 += _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                                    z2 += _mm256_loadu_pd(&pz[8 + 0 * ldc]);
                                    z3 += _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                    z4 += _mm256_loadu_pd(&pz[4 + 1 * ldc]);
                                    z5 += _mm256_loadu_pd(&pz[8 + 1 * ldc]);
                                    z6 += _mm256_loadu_pd(&pz[0 + 2 * ldc]);
                                    z7 += _mm256_loadu_pd(&pz[4 + 2 * ldc]);
                                    z8 += _mm256_loadu_pd(&pz[8 + 2 * ldc]);
                                    z9 += _mm256_loadu_pd(&pz[0 + 3 * ldc]);
                                    z10 += _mm256_loadu_pd(&pz[4 + 3 * ldc]);
                                    z11 += _mm256_loadu_pd(&pz[8 + 3 * ldc]);
                                } else if (beta != 0.0) {
                                    y0 = _mm256_broadcast_sd(&beta);
                                    z0 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                    z1 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                                    z2 +=
                                        y0 * _mm256_loadu_pd(&pz[8 + 0 * ldc]);
                                    z3 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                    z4 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 1 * ldc]);
                                    z5 +=
                                        y0 * _mm256_loadu_pd(&pz[8 + 1 * ldc]);
                                    z6 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 2 * ldc]);
                                    z7 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 2 * ldc]);
                                    z8 +=
                                        y0 * _mm256_loadu_pd(&pz[8 + 2 * ldc]);
                                    z9 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 3 * ldc]);
                                    z10 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 3 * ldc]);
                                    z11 +=
                                        y0 * _mm256_loadu_pd(&pz[8 + 3 * ldc]);
                                }
                                _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                                _mm256_storeu_pd(&pz[4 + 0 * ldc], z1);
                                _mm256_storeu_pd(&pz[8 + 0 * ldc], z2);
                                _mm256_storeu_pd(&pz[0 + 1 * ldc], z3);
                                _mm256_storeu_pd(&pz[4 + 1 * ldc], z4);
                                _mm256_storeu_pd(&pz[8 + 1 * ldc], z5);
                                _mm256_storeu_pd(&pz[0 + 2 * ldc], z6);
                                _mm256_storeu_pd(&pz[4 + 2 * ldc], z7);
                                _mm256_storeu_pd(&pz[8 + 2 * ldc], z8);
                                _mm256_storeu_pd(&pz[0 + 3 * ldc], z9);
                                _mm256_storeu_pd(&pz[4 + 3 * ldc], z10);
                                _mm256_storeu_pd(&pz[8 + 3 * ldc], z11);
                            }
                        if ((nni - xnni) & 8) {
                            for (size_t xj = 0; xj < xnnj; xj += 4) {
                                const double *px0 = &pa[xnni * nnk];
                                const double *px1 = &pa[(xnni + 4) * nnk];
                                const double *py = &pb[xj * nnk];
                                double *pz = &c[(ii + xnni) + (jj + xj) * ldc];
                                z0 = _mm256_setzero_pd();
                                z1 = _mm256_setzero_pd();
                                z3 = _mm256_setzero_pd();
                                z4 = _mm256_setzero_pd();
                                z6 = _mm256_setzero_pd();
                                z7 = _mm256_setzero_pd();
                                z9 = _mm256_setzero_pd();
                                z10 = _mm256_setzero_pd();
                                for (size_t xk = 0; xk < nnk; xk++) {
                                    x0 = _mm256_loadu_pd(px0);
                                    x1 = _mm256_loadu_pd(px1);
                                    z2 = _mm256_broadcast_sd(&py[0]);
                                    z0 = _mm256_fmadd_pd(x0, z2, z0);
                                    z1 = _mm256_fmadd_pd(x1, z2, z1);
                                    z5 = _mm256_broadcast_sd(&py[1]);
                                    z3 = _mm256_fmadd_pd(x0, z5, z3);
                                    z4 = _mm256_fmadd_pd(x1, z5, z4);
                                    z8 = _mm256_broadcast_sd(&py[2]);
                                    z6 = _mm256_fmadd_pd(x0, z8, z6);
                                    z7 = _mm256_fmadd_pd(x1, z8, z7);
                                    z11 = _mm256_broadcast_sd(&py[3]);
                                    z9 = _mm256_fmadd_pd(x0, z11, z9);
                                    z10 = _mm256_fmadd_pd(x1, z11, z10);
                                    px0 += 4, px1 += 4, py += 4;
                                    _mm_prefetch(&px0[64], _MM_HINT_T0);
                                    _mm_prefetch(&py[64], _MM_HINT_T0);
                                }
                                if (alpha != 1.0) {
                                    y0 = _mm256_broadcast_sd(&alpha);
                                    z0 *= y0, z1 *= y0;
                                    z3 *= y0, z4 *= y0;
                                    z6 *= y0, z7 *= y0;
                                    z9 *= y0, z10 *= y0;
                                }
                                if (kk != 0) {
                                    z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                    z1 += _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                                    z3 += _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                    z4 += _mm256_loadu_pd(&pz[4 + 1 * ldc]);
                                    z6 += _mm256_loadu_pd(&pz[0 + 2 * ldc]);
                                    z7 += _mm256_loadu_pd(&pz[4 + 2 * ldc]);
                                    z9 += _mm256_loadu_pd(&pz[0 + 3 * ldc]);
                                    z10 += _mm256_loadu_pd(&pz[4 + 3 * ldc]);
                                } else if (beta != 0.0) {
                                    y0 = _mm256_broadcast_sd(&beta);
                                    z0 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                    z1 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                                    z3 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                    z4 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 1 * ldc]);
                                    z6 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 2 * ldc]);
                                    z7 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 2 * ldc]);
                                    z9 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 3 * ldc]);
                                    z10 +=
                                        y0 * _mm256_loadu_pd(&pz[4 + 3 * ldc]);
                                }
                                _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                                _mm256_storeu_pd(&pz[4 + 0 * ldc], z1);
                                _mm256_storeu_pd(&pz[0 + 1 * ldc], z3);
                                _mm256_storeu_pd(&pz[4 + 1 * ldc], z4);
                                _mm256_storeu_pd(&pz[0 + 2 * ldc], z6);
                                _mm256_storeu_pd(&pz[4 + 2 * ldc], z7);
                                _mm256_storeu_pd(&pz[0 + 3 * ldc], z9);
                                _mm256_storeu_pd(&pz[4 + 3 * ldc], z10);
                            }
                            xnni += 8;
                        }
                        if ((nni - xnni) & 4) {
                            for (size_t xj = 0; xj < xnnj; xj += 4) {
                                const double *px0 = &pa[xnni * nnk];
                                const double *py = &pb[xj * nnk];
                                double *pz = &c[(ii + xnni) + (jj + xj) * ldc];
                                z0 = _mm256_setzero_pd();
                                z3 = _mm256_setzero_pd();
                                z6 = _mm256_setzero_pd();
                                z9 = _mm256_setzero_pd();
                                for (size_t xk = 0; xk < nnk; xk++) {
                                    x0 = _mm256_loadu_pd(px0);
                                    z2 = _mm256_broadcast_sd(&py[0]);
                                    z0 = _mm256_fmadd_pd(x0, z2, z0);
                                    z5 = _mm256_broadcast_sd(&py[1]);
                                    z3 = _mm256_fmadd_pd(x0, z5, z3);
                                    z8 = _mm256_broadcast_sd(&py[2]);
                                    z6 = _mm256_fmadd_pd(x0, z8, z6);
                                    z11 = _mm256_broadcast_sd(&py[3]);
                                    z9 = _mm256_fmadd_pd(x0, z11, z9);
                                    px0 += 4, py += 4;
                                    _mm_prefetch(&px0[64], _MM_HINT_T0);
                                    _mm_prefetch(&py[64], _MM_HINT_T0);
                                }
                                if (alpha != 1.0) {
                                    y0 = _mm256_broadcast_sd(&alpha);
                                    z0 *= y0, z3 *= y0, z6 *= y0, z9 *= y0;
                                }
                                if (kk != 0) {
                                    z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                    z3 += _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                    z6 += _mm256_loadu_pd(&pz[0 + 2 * ldc]);
                                    z9 += _mm256_loadu_pd(&pz[0 + 3 * ldc]);
                                } else if (beta != 0.0) {
                                    y0 = _mm256_broadcast_sd(&beta);
                                    z0 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                    z3 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                    z6 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 2 * ldc]);
                                    z9 +=
                                        y0 * _mm256_loadu_pd(&pz[0 + 3 * ldc]);
                                }
                                _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                                _mm256_storeu_pd(&pz[0 + 1 * ldc], z3);
                                _mm256_storeu_pd(&pz[0 + 2 * ldc], z6);
                                _mm256_storeu_pd(&pz[0 + 3 * ldc], z9);
                            }
                            xnni += 4;
                        }
                        if ((nni - xnni) & 2) {
                            __m128d w0, w1, w2, w3, wx0;
                            for (size_t xj = 0; xj < xnnj; xj += 4) {
                                const double *px0 = &pa[xnni * nnk];
                                const double *py = &pb[xj * nnk];
                                double *pz = &c[(ii + xnni) + (jj + xj) * ldc];
                                w0 = _mm_setzero_pd();
                                w1 = _mm_setzero_pd();
                                w2 = _mm_setzero_pd();
                                w3 = _mm_setzero_pd();
                                for (size_t xk = 0; xk < nnk; xk++) {
                                    wx0 = _mm_loadu_pd(px0);
                                    w0 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[0]),
                                                      w0);
                                    w1 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[1]),
                                                      w1);
                                    w2 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[2]),
                                                      w2);
                                    w3 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[3]),
                                                      w3);
                                    px0 += 2, py += 4;
                                    _mm_prefetch(&px0[64], _MM_HINT_T0);
                                    _mm_prefetch(&py[64], _MM_HINT_T0);
                                }
                                if (alpha != 1.0) {
                                    wx0 = _mm_set1_pd(alpha);
                                    w0 *= wx0, w1 *= wx0, w2 *= wx0, w3 *= wx0;
                                }
                                if (kk != 0) {
                                    w0 += _mm_loadu_pd(&pz[0 + 0 * ldc]);
                                    w1 += _mm_loadu_pd(&pz[0 + 1 * ldc]);
                                    w2 += _mm_loadu_pd(&pz[0 + 2 * ldc]);
                                    w3 += _mm_loadu_pd(&pz[0 + 3 * ldc]);
                                } else if (beta != 0.0) {
                                    wx0 = _mm_set1_pd(beta);
                                    w0 += wx0 * _mm_loadu_pd(&pz[0 + 0 * ldc]);
                                    w1 += wx0 * _mm_loadu_pd(&pz[0 + 1 * ldc]);
                                    w2 += wx0 * _mm_loadu_pd(&pz[0 + 2 * ldc]);
                                    w3 += wx0 * _mm_loadu_pd(&pz[0 + 3 * ldc]);
                                }
                                _mm_storeu_pd(&pz[0 + 0 * ldc], w0);
                                _mm_storeu_pd(&pz[0 + 1 * ldc], w1);
                                _mm_storeu_pd(&pz[0 + 2 * ldc], w2);
                                _mm_storeu_pd(&pz[0 + 3 * ldc], w3);
                            }
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1) {
                            for (size_t xj = 0; xj < xnnj; xj += 4) {
                                const double *px0 = &pa[xnni * nnk];
                                const double *py = &pb[xj * nnk];
                                double *pz = &c[(ii + xnni) + (jj + xj) * ldc];
                                z0 = _mm256_setzero_pd();
                                for (size_t xk = 0; xk < nnk; xk++) {
                                    x0 = _mm256_broadcast_sd(px0);
                                    z0 = _mm256_fmadd_pd(
                                        x0, _mm256_loadu_pd(py), z0);
                                    px0++, py += 4;
                                    _mm_prefetch(&px0[64], _MM_HINT_T0);
                                    _mm_prefetch(&py[64], _MM_HINT_T0);
                                }
                                if (alpha != 1.0)
                                    z0 *= _mm256_broadcast_sd(&alpha);
                                if (kk != 0) {
                                    pz[0 + 0 * ldc] += ((double *)(&z0))[0];
                                    pz[0 + 1 * ldc] += ((double *)(&z0))[1];
                                    pz[0 + 2 * ldc] += ((double *)(&z0))[2];
                                    pz[0 + 3 * ldc] += ((double *)(&z0))[3];
                                } else if (beta != 0.0) {
                                    pz[0 + 0 * ldc] = ((double *)(&z0))[0] +
                                                      beta * pz[0 + 0 * ldc];
                                    pz[0 + 1 * ldc] = ((double *)(&z0))[1] +
                                                      beta * pz[0 + 1 * ldc];
                                    pz[0 + 2 * ldc] = ((double *)(&z0))[2] +
                                                      beta * pz[0 + 2 * ldc];
                                    pz[0 + 3 * ldc] = ((double *)(&z0))[3] +
                                                      beta * pz[0 + 3 * ldc];
                                } else {
                                    pz[0 + 0 * ldc] = ((double *)(&z0))[0];
                                    pz[0 + 1 * ldc] = ((double *)(&z0))[1];
                                    pz[0 + 2 * ldc] = ((double *)(&z0))[2];
                                    pz[0 + 3 * ldc] = ((double *)(&z0))[3];
                                }
                            }
                        }
                    }
                    if ((nnj - xnnj) & 2) {
                        size_t xnni = nni >> 3 << 3;
                        for (size_t xi = 0; xi < xnni; xi += 8) {
                            const double *px0 = &pa[xi * nnk];
                            const double *px1 = &pa[(xi + 4) * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xi) + (jj + xnnj) * ldc];
                            z0 = _mm256_setzero_pd();
                            z1 = _mm256_setzero_pd();
                            z3 = _mm256_setzero_pd();
                            z4 = _mm256_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                x0 = _mm256_loadu_pd(px0);
                                x1 = _mm256_loadu_pd(px1);
                                z2 = _mm256_broadcast_sd(&py[0]);
                                z0 = _mm256_fmadd_pd(x0, z2, z0);
                                z1 = _mm256_fmadd_pd(x1, z2, z1);
                                z5 = _mm256_broadcast_sd(&py[1]);
                                z3 = _mm256_fmadd_pd(x0, z5, z3);
                                z4 = _mm256_fmadd_pd(x1, z5, z4);
                                px0 += 4, px1 += 4, py += 2;
                                _mm_prefetch(&px0[64], _MM_HINT_T0);
                                _mm_prefetch(&py[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0) {
                                y0 = _mm256_broadcast_sd(&alpha);
                                z0 *= y0, z1 *= y0, z3 *= y0, z4 *= y0;
                            }
                            if (kk != 0) {
                                z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                z1 += _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                                z3 += _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                z4 += _mm256_loadu_pd(&pz[4 + 1 * ldc]);
                            } else if (beta != 0.0) {
                                y0 = _mm256_broadcast_sd(&beta);
                                z0 += y0 * _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                z1 += y0 * _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                                z3 += y0 * _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                                z4 += y0 * _mm256_loadu_pd(&pz[4 + 1 * ldc]);
                            }
                            _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                            _mm256_storeu_pd(&pz[4 + 0 * ldc], z1);
                            _mm256_storeu_pd(&pz[0 + 1 * ldc], z3);
                            _mm256_storeu_pd(&pz[4 + 1 * ldc], z4);
                        }
                        if ((nni - xnni) & 4) {
                            const double *px0 = &pa[xnni * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xnni) + (jj + xnnj) * ldc];
                            z0 = _mm256_setzero_pd();
                            z3 = _mm256_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                x0 = _mm256_loadu_pd(px0);
                                z2 = _mm256_broadcast_sd(&py[0]);
                                z0 = _mm256_fmadd_pd(x0, z2, z0);
                                z5 = _mm256_broadcast_sd(&py[1]);
                                z3 = _mm256_fmadd_pd(x0, z5, z3);
                                px0 += 4, py += 2;
                                _mm_prefetch(&px0[64], _MM_HINT_T0);
                                _mm_prefetch(&py[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0) {
                                y0 = _mm256_broadcast_sd(&alpha);
                                z0 *= y0, z3 *= y0;
                            }
                            if (kk != 0) {
                                z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                z3 += _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                            } else if (beta != 0.0) {
                                y0 = _mm256_broadcast_sd(&beta);
                                z0 += y0 * _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                z3 += y0 * _mm256_loadu_pd(&pz[0 + 1 * ldc]);
                            }
                            _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                            _mm256_storeu_pd(&pz[0 + 1 * ldc], z3);
                            xnni += 4;
                        }
                        if ((nni - xnni) & 2) {
                            __m128d w0, w1, w2, w3, wx0;
                            const double *px0 = &pa[xnni * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xnni) + (jj + xnnj) * ldc];
                            w0 = _mm_setzero_pd();
                            w1 = _mm_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                wx0 = _mm_loadu_pd(px0);
                                w0 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[0]), w0);
                                w1 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[1]), w1);
                                px0 += 2, py += 2;
                                _mm_prefetch(&px0[64], _MM_HINT_T0);
                                _mm_prefetch(&py[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0) {
                                wx0 = _mm_set1_pd(alpha);
                                w0 *= wx0, w1 *= wx0;
                            }
                            if (kk != 0) {
                                w0 += _mm_loadu_pd(&pz[0 + 0 * ldc]);
                                w1 += _mm_loadu_pd(&pz[0 + 1 * ldc]);
                            } else if (beta != 0.0) {
                                wx0 = _mm_set1_pd(beta);
                                w0 += wx0 * _mm_loadu_pd(&pz[0 + 0 * ldc]);
                                w1 += wx0 * _mm_loadu_pd(&pz[0 + 1 * ldc]);
                            }
                            _mm_storeu_pd(&pz[0 + 0 * ldc], w0);
                            _mm_storeu_pd(&pz[0 + 1 * ldc], w1);
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1) {
                            __m128d w0, w1;
                            const double *px0 = &pa[xnni * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xnni) + (jj + xnnj) * ldc];
                            w0 = _mm_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                w1 = _mm_set1_pd(px0[xk]);
                                w0 = _mm_fmadd_pd(w1, _mm_loadu_pd(py), w0);
                                py += 2;
                                _mm_prefetch(&py[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0)
                                w0 *= _mm_set1_pd(alpha);
                            if (kk != 0) {
                                pz[0 + 0 * ldc] += ((double *)(&w0))[0];
                                pz[0 + 1 * ldc] += ((double *)(&w0))[1];
                            } else if (beta != 0.0) {
                                pz[0 + 0 * ldc] = beta * pz[0 + 0 * ldc] +
                                                  ((double *)(&w0))[0];
                                pz[0 + 1 * ldc] = beta * pz[0 + 1 * ldc] +
                                                  ((double *)(&w0))[1];
                            } else {
                                pz[0 + 0 * ldc] = ((double *)(&w0))[0];
                                pz[0 + 1 * ldc] = ((double *)(&w0))[1];
                            }
                        }
                        xnnj += 2;
                    }
                    if ((nnj - xnnj) & 1) {
                        size_t xnni = nni >> 3 << 3;
                        for (size_t xi = 0; xi < xnni; xi += 8) {
                            const double *px0 = &pa[xi * nnk];
                            const double *px1 = &pa[(xi + 4) * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xi) + (jj + xnnj) * ldc];
                            z0 = _mm256_setzero_pd();
                            z1 = _mm256_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                x0 = _mm256_loadu_pd(px0);
                                x1 = _mm256_loadu_pd(px1);
                                z2 = _mm256_broadcast_sd(&py[xk]);
                                z0 = _mm256_fmadd_pd(x0, z2, z0);
                                z1 = _mm256_fmadd_pd(x1, z2, z1);
                                px0 += 4, px1 += 4;
                                _mm_prefetch(&px0[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0) {
                                y0 = _mm256_broadcast_sd(&alpha);
                                z0 *= y0, z1 *= y0;
                            }
                            if (kk != 0) {
                                z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                z1 += _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                            } else if (beta != 0.0) {
                                y0 = _mm256_broadcast_sd(&beta);
                                z0 += y0 * _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                                z1 += y0 * _mm256_loadu_pd(&pz[4 + 0 * ldc]);
                            }
                            _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                            _mm256_storeu_pd(&pz[4 + 0 * ldc], z1);
                        }
                        if ((nni - xnni) & 4) {
                            const double *px0 = &pa[xnni * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xnni) + (jj + xnnj) * ldc];
                            z0 = _mm256_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                x0 = _mm256_loadu_pd(px0);
                                z2 = _mm256_broadcast_sd(&py[xk]);
                                z0 = _mm256_fmadd_pd(x0, z2, z0);
                                px0 += 4;
                                _mm_prefetch(&px0[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0)
                                z0 *= _mm256_broadcast_sd(&alpha);
                            if (kk != 0)
                                z0 += _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                            else if (beta != 0.0)
                                z0 += _mm256_broadcast_sd(&beta) *
                                      _mm256_loadu_pd(&pz[0 + 0 * ldc]);
                            _mm256_storeu_pd(&pz[0 + 0 * ldc], z0);
                            xnni += 4;
                        }
                        if ((nni - xnni) & 2) {
                            __m128d w0, w1, w2, w3, wx0;
                            const double *px0 = &pa[xnni * nnk];
                            const double *py = &pb[xnnj * nnk];
                            double *pz = &c[(ii + xnni) + (jj + xnnj) * ldc];
                            w0 = _mm_setzero_pd();
                            for (size_t xk = 0; xk < nnk; xk++) {
                                wx0 = _mm_loadu_pd(px0);
                                w0 = _mm_fmadd_pd(wx0, _mm_set1_pd(py[xk]), w0);
                                px0 += 2;
                                _mm_prefetch(&px0[64], _MM_HINT_T0);
                            }
                            if (alpha != 1.0)
                                w0 *= _mm_set1_pd(alpha);
                            if (kk != 0)
                                w0 += _mm_loadu_pd(&pz[0 + 0 * ldc]);
                            else if (beta != 0.0)
                                w0 += _mm_set1_pd(beta) *
                                      _mm_loadu_pd(&pz[0 + 0 * ldc]);
                            _mm_storeu_pd(&pz[0 + 0 * ldc], w0);
                            xnni += 2;
                        }
                        if ((nni - xnni) & 1) {
                            const double *__restrict__ px0 = &pa[xnni * nnk];
                            const double *__restrict__ py = &pb[xnnj * nnk];
                            double *__restrict__ pz =
                                &c[(ii + xnni) + (jj + xnnj) * ldc];
                            double zz0 = 0.0;
                            for (size_t xk = 0; xk < nnk; xk++)
                                zz0 += px0[xk] * py[xk];
                            zz0 *= alpha;
                            if (kk != 0)
                                pz[0 + 0 * ldc] += zz0;
                            else if (beta != 0.0)
                                pz[0 + 0 * ldc] = zz0 + beta * pz[0 + 0 * ldc];
                            else
                                pz[0 + 0 * ldc] = zz0;
                        }
                    }
                }
            }
        }
        if (full_copy == 1 || full_copy == 0) {
            free(xa);
            free(xb);
        } else if (full_copy == 3) {
            free(xb);
        } else if (full_copy == 4) {
            free(xa);
        }
    }
};

}; // namespace block2

#pragma GCC pop_options
#endif
