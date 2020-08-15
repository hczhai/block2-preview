
#include "block2.hpp"
#include <gtest/gtest.h>

using namespace block2;

class TestAncillaH8STO6G : public ::testing::Test {
  protected:
    size_t isize = 1L << 30;
    size_t dsize = 1L << 34;
    void SetUp() override {
        Random::rand_seed(0);
        frame_() = make_shared<DataFrame>(isize, dsize, "nodex");
    }
    void TearDown() override {
        frame_()->activate(0);
        assert(ialloc_()->used == 0 && dalloc_()->used == 0);
        frame_() = nullptr;
    }
};

TEST_F(TestAncillaH8STO6G, TestSU2) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H8.STO6G.R1.8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    mkl_set_num_threads(8);
    mkl_set_dynamic(0);

    vector<double> energies_fted = {
        0.3124038410492045,  -0.0273905176813768, -0.3265074932156511,
        -0.5914620908396366, -0.8276498731818384, -1.0395171725041257,
        -1.2307228748517529, -1.4042806712721763, -1.5626789845611742,
        -1.7079796842651509, -1.8418982445788070};

    vector<double> energies_m500 = {
        0.312403841049,  -0.027389713306, -0.326500930805, -0.591439984794,
        -0.827598404678, -1.039419259243, -1.230558968502, -1.404029934736,
        -1.562319009406, -1.707487414764, -1.841250686976};

    SU2 vacuum(0);
    SU2 target(fcidump->n_sites() * 2, fcidump->twos(),
               PointGroup::swap_pg(pg)(fcidump->isym()));
    int n_physical_sites = fcidump->n_sites();
    int n_sites = n_physical_sites * 2;

    HamiltonianQC<SU2> hamil(vacuum, n_physical_sites, orbsym, fcidump);
    hamil.mu = -1.0;
    hamil.fcidump->e = 0.0;
    hamil.opf->seq->mode = SeqTypes::Simple;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SU2>> mpo =
        make_shared<MPOQC<SU2>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // Ancilla MPO construction
    cout << "Ancilla MPO start" << endl;
    mpo = make_shared<AncillaMPO<SU2>>(mpo);
    cout << "Ancilla MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo =
        make_shared<SimplifiedMPO<SU2>>(mpo, make_shared<RuleQC<SU2>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    uint16_t bond_dim = 500;
    double beta = 0.05;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> te_energies;

    // Ancilla MPSInfo (thermal)
    Random::rand_seed(0);

    shared_ptr<AncillaMPSInfo<SU2>> mps_info_thermal =
        make_shared<AncillaMPSInfo<SU2>>(n_physical_sites, vacuum, target,
                                         hamil.basis, hamil.orb_sym);
    mps_info_thermal->set_thermal_limit();
    mps_info_thermal->tag = "KET";

    // Ancilla MPS (thermal)
    shared_ptr<MPS<SU2>> mps_thermal =
        make_shared<MPS<SU2>>(n_sites, n_sites - 2, 2);
    mps_thermal->initialize(mps_info_thermal);
    mps_thermal->fill_thermal_limit();

    // MPS/MPSInfo save mutable
    mps_thermal->save_mutable();
    mps_thermal->deallocate();
    mps_info_thermal->save_mutable();
    mps_info_thermal->deallocate_mutable();

    // TE ME
    shared_ptr<MovingEnvironment<SU2>> me = make_shared<MovingEnvironment<SU2>>(
        mpo, mps_thermal, mps_thermal, "TE");
    me->init_environments(false);

    shared_ptr<Expect<SU2>> ex =
        make_shared<Expect<SU2>>(me, bond_dim, bond_dim);
    te_energies.push_back(ex->solve(false));

    // Imaginary TE
    shared_ptr<ImaginaryTE<SU2>> te =
        make_shared<ImaginaryTE<SU2>>(me, bdims, TETypes::RK4);
    te->iprint = 1;
    te->n_sub_sweeps = 6;
    te->solve(1, beta / 2, mps_thermal->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    te->n_sub_sweeps = 2;
    te->solve(9, beta / 2, mps_thermal->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    for (size_t i = 0; i < te_energies.size(); i++) {
        cout << "== SU2 =="
             << " BETA = " << setw(10) << fixed << setprecision(4) << (i * beta)
             << " E = " << fixed << setw(22) << setprecision(12)
             << te_energies[i] << " error-fted = " << scientific
             << setprecision(3) << setw(10)
             << (te_energies[i] - energies_fted[i])
             << " error-m500 = " << scientific << setprecision(3) << setw(10)
             << (te_energies[i] - energies_m500[i]) << endl;

        EXPECT_LT(abs(te_energies[i] - energies_m500[i]), 1E-10);
    }

    mps_info_thermal->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}

TEST_F(TestAncillaH8STO6G, TestSZ) {
    shared_ptr<FCIDUMP> fcidump = make_shared<FCIDUMP>();
    PGTypes pg = PGTypes::D2H;
    string filename = "data/H8.STO6G.R1.8.FCIDUMP";
    fcidump->read(filename);
    vector<uint8_t> orbsym = fcidump->orb_sym();
    transform(orbsym.begin(), orbsym.end(), orbsym.begin(),
              PointGroup::swap_pg(pg));

    mkl_set_num_threads(8);
    mkl_set_dynamic(0);

    vector<double> energies_fted = {
        0.3124038410492045,  -0.0273905176813768, -0.3265074932156511,
        -0.5914620908396366, -0.8276498731818384, -1.0395171725041257,
        -1.2307228748517529, -1.4042806712721763, -1.5626789845611742,
        -1.7079796842651509, -1.8418982445788070};

    vector<double> energies_m500 = {
        0.312403841049,  -0.027388048069, -0.326490457632, -0.591401772825,
        -0.827502872933, -1.039228830737, -1.230231051484, -1.403519072586,
        -1.561579406450, -1.706474487633, -1.839921660072};

    SZ vacuum(0);
    SZ target(fcidump->n_sites() * 2, fcidump->twos(),
              PointGroup::swap_pg(pg)(fcidump->isym()));
    int n_physical_sites = fcidump->n_sites();
    int n_sites = n_physical_sites * 2;

    HamiltonianQC<SZ> hamil(vacuum, n_physical_sites, orbsym, fcidump);
    hamil.mu = -1.0;
    hamil.fcidump->e = 0.0;
    hamil.opf->seq->mode = SeqTypes::Simple;

    Timer t;
    t.get_time();
    // MPO construction
    cout << "MPO start" << endl;
    shared_ptr<MPO<SZ>> mpo =
        make_shared<MPOQC<SZ>>(hamil, QCTypes::Conventional);
    cout << "MPO end .. T = " << t.get_time() << endl;

    // Ancilla MPO construction
    cout << "Ancilla MPO start" << endl;
    mpo = make_shared<AncillaMPO<SZ>>(mpo);
    cout << "Ancilla MPO end .. T = " << t.get_time() << endl;

    // MPO simplification
    cout << "MPO simplification start" << endl;
    mpo = make_shared<SimplifiedMPO<SZ>>(mpo, make_shared<RuleQC<SZ>>(), true);
    cout << "MPO simplification end .. T = " << t.get_time() << endl;

    uint16_t bond_dim = 500;
    double beta = 0.05;
    vector<uint16_t> bdims = {bond_dim};
    vector<double> te_energies;

    // Ancilla MPSInfo (thermal)
    Random::rand_seed(0);

    shared_ptr<AncillaMPSInfo<SZ>> mps_info_thermal =
        make_shared<AncillaMPSInfo<SZ>>(n_physical_sites, vacuum, target,
                                        hamil.basis, hamil.orb_sym);
    mps_info_thermal->set_thermal_limit();
    mps_info_thermal->tag = "KET";

    // Ancilla MPS (thermal)
    shared_ptr<MPS<SZ>> mps_thermal =
        make_shared<MPS<SZ>>(n_sites, n_sites - 2, 2);
    mps_thermal->initialize(mps_info_thermal);
    mps_thermal->fill_thermal_limit();

    // MPS/MPSInfo save mutable
    mps_thermal->save_mutable();
    mps_thermal->deallocate();
    mps_info_thermal->save_mutable();
    mps_info_thermal->deallocate_mutable();

    // TE ME
    shared_ptr<MovingEnvironment<SZ>> me =
        make_shared<MovingEnvironment<SZ>>(mpo, mps_thermal, mps_thermal, "TE");
    me->init_environments(false);

    shared_ptr<Expect<SZ>> ex = make_shared<Expect<SZ>>(me, bond_dim, bond_dim);
    te_energies.push_back(ex->solve(false));

    // Imaginary TE
    shared_ptr<ImaginaryTE<SZ>> te =
        make_shared<ImaginaryTE<SZ>>(me, bdims, TETypes::RK4);
    te->iprint = 1;
    te->n_sub_sweeps = 6;
    te->solve(1, beta / 2, mps_thermal->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    te->n_sub_sweeps = 2;
    te->solve(9, beta / 2, mps_thermal->center == 0);

    te_energies.insert(te_energies.end(), te->energies.begin(),
                       te->energies.end());

    for (size_t i = 0; i < te_energies.size(); i++) {
        cout << "== SZ  =="
             << " BETA = " << setw(10) << fixed << setprecision(4) << (i * beta)
             << " E = " << fixed << setw(22) << setprecision(12)
             << te_energies[i] << " error-fted = " << scientific
             << setprecision(3) << setw(10)
             << (te_energies[i] - energies_fted[i])
             << " error-m500 = " << scientific << setprecision(3) << setw(10)
             << (te_energies[i] - energies_m500[i]) << endl;

        EXPECT_LT(abs(te_energies[i] - energies_m500[i]), 1E-10);
    }

    mps_info_thermal->deallocate();
    mpo->deallocate();
    hamil.deallocate();
    fcidump->deallocate();
}
