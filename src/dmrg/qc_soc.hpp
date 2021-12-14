
/*
 * block2: Efficient MPO implementation of quantum chemistry DMRG
 * Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
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

#include "../core/delayed_tensor_functions.hpp"
#include "../core/operator_tensor.hpp"
#include "../core/symbolic.hpp"
#include "../core/tensor_functions.hpp"
#include "../core/threading.hpp"
#include "mpo.hpp"
#include "qc_hamiltonian.hpp"
#include <cassert>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

using namespace std;

namespace block2 {

template <typename, typename, typename = void> struct SOCMPOQC;

// Quantum chemistry MPO for the spin-orbital coupling term (non-spin-adapted)
// HSOC: ds = 0: aa + bb; 2: ab; 4: ba
template <typename S, typename FL>
struct SOCMPOQC<S, FL, typename S::is_sz_t> : MPO<S, FL> {
    SOCMPOQC(const shared_ptr<HamiltonianQC<S, FL>> &hamil, int ds,
             typename S::pg_t dpg)
        : MPO<S, FL>(hamil->n_sites) {
        assert((int)(uint16_t)dpg == (int)dpg);
        // hop/s = 0: aa + bb; 1: ab; 2: ba
        shared_ptr<OpExpr<S>> h_op = make_shared<OpElement<S, FL>>(
            OpNames::HSOC, SiteIndex({(uint16_t)dpg}, {ds == 2, ds == 4}),
            S(0, ds, dpg));
        shared_ptr<OpExpr<S>> i_op = make_shared<OpElement<S, FL>>(
            OpNames::I, SiteIndex(), hamil->vacuum);
        assert(ds == 0 || ds == 2 || ds == 4);
        uint16_t n_sites = hamil->n_sites;
        if (hamil->opf != nullptr &&
            hamil->opf->get_type() == SparseMatrixTypes::CSR) {
            if (hamil->get_n_orbs_left() > 0)
                MPO<S, FL>::sparse_form[0] = 'S';
            if (hamil->get_n_orbs_right() > 0)
                MPO<S, FL>::sparse_form[n_sites - 1] = 'S';
        }
        int n_orbs_big_left = max(hamil->get_n_orbs_left(), 1);
        int n_orbs_big_right = max(hamil->get_n_orbs_right(), 1);
        uint16_t n_orbs =
            hamil->n_sites + n_orbs_big_left - 1 + n_orbs_big_right - 1;
#ifdef _MSC_VER
        vector<vector<shared_ptr<OpExpr<S>>>> c_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            d_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> mc_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            md_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> rd_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            r_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
        vector<vector<shared_ptr<OpExpr<S>>>> mrd_op(
            n_orbs, vector<shared_ptr<OpExpr<S>>>(2)),
            mr_op(n_orbs, vector<shared_ptr<OpExpr<S>>>(2));
#else
        shared_ptr<OpExpr<S>> c_op[n_orbs][2], d_op[n_orbs][2];
        shared_ptr<OpExpr<S>> mc_op[n_orbs][2], md_op[n_orbs][2];
        shared_ptr<OpExpr<S>> rd_op[n_orbs][2], r_op[n_orbs][2];
        shared_ptr<OpExpr<S>> mrd_op[n_orbs][2], mr_op[n_orbs][2];
#endif
        MPO<S, FL>::op = dynamic_pointer_cast<OpElement<S, FL>>(h_op);
        MPO<S, FL>::const_e = 0.0;
        if (hamil->delayed == DelayedOpNames::None)
            MPO<S, FL>::tf = make_shared<TensorFunctions<S, FL>>(hamil->opf);
        else
            MPO<S, FL>::tf =
                make_shared<DelayedTensorFunctions<S, FL>>(hamil->opf);
        MPO<S, FL>::site_op_infos = hamil->site_op_infos;
        const int sz[2] = {1, -1};
        // RSOC[s sp]  = t[s sp]  D[sp]
        // RDSOC[s sp] = t[s sp]* C[sp] = t[sp s] C[sp]
        for (uint16_t m = 0; m < n_orbs; m++)
            for (uint8_t s = 0; s < 2; s++) {
                c_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::C, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil->orb_sym[m]));
                d_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], S::pg_inv(hamil->orb_sym[m])));
                mc_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::C, SiteIndex({m}, {s}),
                    S(1, sz[s], hamil->orb_sym[m]), -1.0);
                md_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::D, SiteIndex({m}, {s}),
                    S(-1, -sz[s], S::pg_inv(hamil->orb_sym[m])), -1.0);
                rd_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::RDSOC,
                    SiteIndex({m}, {s, ds == 0 ? (s << 1) : ((!s) << 1)}),
                    S(1, sz[s], S::pg_mul(dpg, hamil->orb_sym[m])));
                r_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::RSOC,
                    SiteIndex({m}, {s, ds == 0 ? (s << 1) : ((!s) << 1)}),
                    S(-1, -sz[s], S::pg_mul(dpg, S::pg_inv(hamil->orb_sym[m])));
                mrd_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::RDSOC,
                    SiteIndex({m}, {s, ds == 0 ? (s << 1) : ((!s) << 1)}),
                    S(1, sz[s], S::pg_mul(dpg, hamil->orb_sym[m])), -1.0);
                mr_op[m][s] = make_shared<OpElement<S, FL>>(
                    OpNames::RSOC,
                    SiteIndex({m}, {s, ds == 0 ? (s << 1) : ((!s) << 1)}),
                    S(-1, -sz[s], S::pg_mul(dpg, S::pg_inv(hamil->orb_sym[m]))), -1.0);
            }
        this->left_operator_names.resize(n_sites, nullptr);
        this->right_operator_names.resize(n_sites, nullptr);
        this->tensors.resize(n_sites, nullptr);
        for (uint16_t m = 0; m < n_sites; m++)
            this->tensors[m] = make_shared<OperatorTensor<S, FL>>();
        int ntg = threading->activate_global();
#ifdef _MSC_VER
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (int xxm = 0; xxm < (int)n_sites; xxm++) {
            uint16_t xm = (uint16_t)xxm;
#else
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
        for (uint16_t xm = 0; xm < n_sites; xm++) {
#endif
            uint16_t pm = xm;
            int p;
            uint16_t m = pm + n_orbs_big_left - 1;
            shared_ptr<Symbolic<S>> pmat;
            int lshape = 2 + (ds == 0 ? 4 * m : 2 * m);
            int rshape = 2 + (ds == 0 ? 4 * (m + 1) : 2 * (m + 1));
            if (pm == 0)
                pmat = make_shared<SymbolicRowVector<S>>(rshape);
            else if (pm == n_sites - 1)
                pmat = make_shared<SymbolicColumnVector<S>>(lshape);
            else
                pmat = make_shared<SymbolicMatrix<S>>(lshape, rshape);
            Symbolic<S> &mat = *pmat;
            if (pm == 0) {
                mat[{0, 0}] = h_op;
                mat[{0, 1}] = i_op;
                p = 2;
                if (ds == 0) {
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            mat[{0, p + j}] = c_op[j][s];
                        p += m + 1;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            mat[{0, p + j}] = d_op[j][s];
                        p += m + 1;
                    }
                } else if (ds == 2) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        mat[{0, p + j}] = c_op[j][0];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        mat[{0, p + j}] = d_op[j][1];
                    p += m + 1;
                } else {
                    for (uint16_t j = 0; j < m + 1; j++)
                        mat[{0, p + j}] = c_op[j][1];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        mat[{0, p + j}] = d_op[j][0];
                    p += m + 1;
                }
            } else {
                mat[{0, 0}] = i_op;
                mat[{1, 0}] = h_op;
                p = 2;
                if (ds == 0) {
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = r_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{p + j, 0}] = mrd_op[j][s];
                        p += m;
                    }
                } else if (ds == 2) {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = r_op[j][1];
                    p += m;
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = mrd_op[j][0];
                    p += m;
                } else {
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = r_op[j][0];
                    p += m;
                    for (uint16_t j = 0; j < m; j++)
                        mat[{p + j, 0}] = mrd_op[j][1];
                    p += m;
                }
            }
            assert(p == mat.m);
            if (pm != 0 && pm != n_sites - 1) {
                mat[{1, 1}] = i_op;
                p = 2;
                // pointers
                int pi = 1;
                int pc[2] = {2, 2 + m};
                int pd[2] = {2 + m * 2, 2 + m * 3};
                if (ds != 0)
                    pd[0] = 2 + m;
                if (ds == 0) {
                    // C
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pc[s] + j, p + j}] = i_op;
                        mat[{pi, p + m}] = c_op[m][s];
                        p += m + 1;
                    }
                    // D
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            mat[{pd[s] + j, p + j}] = i_op;
                        mat[{pi, p + m}] = d_op[m][s];
                        p += m + 1;
                    }
                } else if (ds == 2) {
                    // C
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pc[0] + j, p + j}] = i_op;
                    mat[{pi, p + m}] = c_op[m][0];
                    p += m + 1;
                    // D
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pd[0] + j, p + j}] = i_op;
                    mat[{pi, p + m}] = d_op[m][1];
                    p += m + 1;
                } else {
                    // C
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pc[0] + j, p + j}] = i_op;
                    mat[{pi, p + m}] = c_op[m][1];
                    p += m + 1;
                    // D
                    for (uint16_t j = 0; j < m; j++)
                        mat[{pd[0] + j, p + j}] = i_op;
                    mat[{pi, p + m}] = d_op[m][0];
                    p += m + 1;
                }
                assert(p == mat.n);
            }
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[pm];
            opt->lmat = opt->rmat = pmat;
            // operator names
            shared_ptr<SymbolicRowVector<S>> plop;
            if (pm == n_sites - 1)
                plop = make_shared<SymbolicRowVector<S>>(1);
            else
                plop = make_shared<SymbolicRowVector<S>>(rshape);
            SymbolicRowVector<S> &lop = *plop;
            lop[0] = h_op;
            if (pm != n_sites - 1) {
                lop[1] = i_op;
                p = 2;
                if (ds == 0) {
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            lop[p + j] = c_op[j][s];
                        p += m + 1;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m + 1; j++)
                            lop[p + j] = d_op[j][s];
                        p += m + 1;
                    }
                } else if (ds == 2) {
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = c_op[j][0];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = d_op[j][1];
                    p += m + 1;
                } else {
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = c_op[j][1];
                    p += m + 1;
                    for (uint16_t j = 0; j < m + 1; j++)
                        lop[p + j] = d_op[j][0];
                    p += m + 1;
                }
                assert(p == rshape);
            }
            this->left_operator_names[pm] = plop;
            shared_ptr<SymbolicColumnVector<S>> prop;
            if (pm == 0)
                prop = make_shared<SymbolicColumnVector<S>>(1);
            else
                prop = make_shared<SymbolicColumnVector<S>>(lshape);
            SymbolicColumnVector<S> &rop = *prop;
            if (pm == 0)
                rop[0] = h_op;
            else {
                rop[0] = i_op;
                rop[1] = h_op;
                p = 2;
                if (ds == 0) {
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            rop[p + j] = r_op[j][s];
                        p += m;
                    }
                    for (uint8_t s = 0; s < 2; s++) {
                        for (uint16_t j = 0; j < m; j++)
                            rop[p + j] = mrd_op[j][s];
                        p += m;
                    }
                } else if (ds == 2) {
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = r_op[j][0];
                    p += m;
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = mrd_op[j][1];
                    p += m;
                } else {
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = r_op[j][1];
                    p += m;
                    for (uint16_t j = 0; j < m; j++)
                        rop[p + j] = mrd_op[j][0];
                    p += m;
                }
                assert(p == lshape);
            }
            this->right_operator_names[pm] = prop;
        }
        SeqTypes seqt = hamil->opf->seq->mode;
        hamil->opf->seq->mode = SeqTypes::None;
        const uint16_t m_start = hamil->get_n_orbs_left() > 0 ? 1 : 0;
        const uint16_t m_end =
            hamil->get_n_orbs_right() > 0 ? n_sites - 1 : n_sites;
#pragma omp parallel for schedule(dynamic) num_threads(ntg)
#ifdef _MSC_VER
        for (int m = (int)m_start; m < (int)m_end; m++) {
#else
        for (uint16_t m = m_start; m < m_end; m++) {
#endif
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[m];
            hamil->filter_site_ops((uint16_t)m, {opt->lmat, opt->rmat},
                                   opt->ops);
        }
        if (hamil->get_n_orbs_left() > 0 && n_sites > 0) {
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[0];
            hamil->filter_site_ops(0, {opt->lmat, opt->rmat}, opt->ops);
        }
        if (hamil->get_n_orbs_right() > 0 && n_sites > 0) {
            shared_ptr<OperatorTensor<S, FL>> opt = this->tensors[n_sites - 1];
            hamil->filter_site_ops(n_sites - 1, {opt->lmat, opt->rmat},
                                   opt->ops);
        }
        hamil->opf->seq->mode = seqt;
        threading->activate_normal();
    }
    void deallocate() override {
        for (int16_t m = this->n_sites - 1; m >= 0; m--)
            this->tensors[m]->deallocate();
    }
};

} // namespace block2
