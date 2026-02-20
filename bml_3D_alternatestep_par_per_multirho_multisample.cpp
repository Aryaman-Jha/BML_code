#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <unordered_map>

#include <omp.h>

// ============================================================
// BASIC GRID HELPERS
// ============================================================
static inline int idx(int x, int y, int L) { return y * L + x; }

// ============================================================
// INITIALIZATION
// ============================================================
static void init_grid_exact(std::vector<uint8_t>& grid, int L, double rho,
                            uint32_t seed, double frac_right = 0.5) {
    const int N = L * L;
    const int total_cars = (int)std::floor(rho * double(N));
    const int n_right = (int)std::floor(frac_right * double(total_cars));
    const int n_down  = total_cars - n_right;

    std::vector<int> pos(N);
    std::iota(pos.begin(), pos.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(pos.begin(), pos.end(), rng);

    std::fill(grid.begin(), grid.end(), 0);
    for (int k = 0; k < n_right; ++k) grid[pos[k]] = 1;
    for (int k = 0; k < n_down;  ++k) grid[pos[n_right + k]] = 2;
}

// ============================================================
// BML UPDATE (UNCHANGED LOGIC) — but now with explicit num_threads
// ============================================================

static int update_right_and_count_parallel(std::vector<uint8_t>& grid, int L, int nthreads) {
    const int N = L * L;
    std::vector<uint8_t> next(N, 0);
    int movable = 0;

    #pragma omp parallel for num_threads(nthreads) reduction(+:movable) schedule(static)
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int i = idx(x,y,L);
            if (grid[i] == 1) {
                int j = idx((x+1)%L, y, L);
                if (grid[j] == 0) { next[j] = 1; movable++; }
                else next[i] = 1;
            } else if (grid[i] == 2) next[i] = 2;
        }
    }

    grid.swap(next);
    return movable;
}

static int update_down_and_count_parallel(std::vector<uint8_t>& grid, int L, int nthreads) {
    const int N = L * L;
    std::vector<uint8_t> next(N, 0);
    int movable = 0;

    #pragma omp parallel for num_threads(nthreads) reduction(+:movable) schedule(static)
    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            int i = idx(x,y,L);
            if (grid[i] == 2) {
                int j = idx(x,(y+1)%L,L);
                if (grid[j] == 0) { next[j] = 2; movable++; }
                else next[i] = 2;
            } else if (grid[i] == 1) next[i] = 1;
        }
    }

    grid.swap(next);
    return movable;
}

// ============================================================
// DSU
// ============================================================

struct DSU {
    std::vector<int> p, r;
    DSU(int n=0) : p(n), r(n,0) { std::iota(p.begin(), p.end(), 0); }

    int find(int a) {
        while (p[a] != a) {
            p[a] = p[p[a]];
            a = p[a];
        }
        return a;
    }

    int find_nopc(int a) const {
        while (p[a] != a) a = p[a];
        return a;
    }

    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (r[a] < r[b]) std::swap(a,b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
    }
};

// ============================================================
// 2D CCA RESULT
// ============================================================

struct CCA2DResult {
    std::vector<int32_t> labels;     // -1 empty
    std::vector<int32_t> comp_size;  // size per component
    int32_t nComp = 0;
};

// ============================================================
// PARALLEL PERIODIC 2D CCA (LABELS + SIZES) — with explicit num_threads
// ============================================================

static CCA2DResult cca_labels_sizes_parallel_periodic(
    const std::vector<uint8_t>& grid, int L, int Bx, int By, int nthreads
) {
    const int N = L*L;
    const int Tx = (L+Bx-1)/Bx;
    const int Ty = (L+By-1)/By;
    const int Ttiles = Tx*Ty;

    std::vector<int> x0(Ttiles),x1(Ttiles),y0(Ttiles),y1(Ttiles);
    for(int ty=0;ty<Ty;++ty) for(int tx=0;tx<Tx;++tx){
        int t=ty*Tx+tx;
        x0[t]=tx*Bx; y0[t]=ty*By;
        x1[t]=std::min(x0[t]+Bx,L);
        y1[t]=std::min(y0[t]+By,L);
    }

    std::vector<int> local_label(N,0), ncomp_tile(Ttiles,0);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for(int t=0;t<Ttiles;++t){
        int W=x1[t]-x0[t], H=y1[t]-y0[t];
        DSU uf(W*H);
        std::vector<uint8_t> occ(W*H,0);
        auto lid=[&](int x,int y){return (y-y0[t])*W+(x-x0[t]);};

        for(int y=y0[t];y<y1[t];++y)
            for(int x=x0[t];x<x1[t];++x)
                if(grid[idx(x,y,L)]!=0) occ[lid(x,y)]=1;

        for(int y=y0[t];y<y1[t];++y)
            for(int x=x0[t];x<x1[t];++x){
                if(!occ[lid(x,y)]) continue;
                if(x>x0[t] && occ[lid(x-1,y)]) uf.unite(lid(x,y),lid(x-1,y));
                if(y>y0[t] && occ[lid(x,y-1)]) uf.unite(lid(x,y),lid(x,y-1));
            }

        std::vector<int> remap(W*H,-1);
        int next=0;
        for(int y=y0[t];y<y1[t];++y)
            for(int x=x0[t];x<x1[t];++x){
                int i=idx(x,y,L);
                if(!occ[lid(x,y)]){ local_label[i]=0; continue; }
                int r=uf.find(lid(x,y));
                if(remap[r]<0) remap[r]=next++;
                local_label[i]=remap[r]+1;
            }
        ncomp_tile[t]=next;
    }

    std::vector<int> tile_off(Ttiles,0);
    for(int t=1;t<Ttiles;++t) tile_off[t]=tile_off[t-1]+ncomp_tile[t-1];
    int Nnodes=tile_off.back()+ncomp_tile.back();
    DSU guf(Nnodes);

    for(int ty=0;ty<Ty;++ty) for(int tx=0;tx<Tx;++tx){
        int t=ty*Tx+tx;
        int tr=ty*Tx+(tx+1)%Tx;
        int td=((ty+1)%Ty)*Tx+tx;

        for(int y=y0[t];y<y1[t];++y){
            int i=idx(x1[t]-1,y,L), j=idx(x0[tr],y,L);
            if(local_label[i]&&local_label[j])
                guf.unite(tile_off[t]+local_label[i]-1,
                          tile_off[tr]+local_label[j]-1);
        }
        for(int x=x0[t];x<x1[t];++x){
            int i=idx(x,y1[t]-1,L), j=idx(x,y0[td],L);
            if(local_label[i]&&local_label[j])
                guf.unite(tile_off[t]+local_label[i]-1,
                          tile_off[td]+local_label[j]-1);
        }
    }

    std::vector<int> root_sz(Nnodes,0);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for(int i=0;i<N;++i) if(grid[i]!=0){
        int tx=(i%L)/Bx, ty=(i/L)/By;
        int t=ty*Tx+tx;
        int nid=tile_off[t]+local_label[i]-1;
        #pragma omp atomic
        root_sz[guf.find_nopc(nid)]++;
    }

    std::vector<int> root2c(Nnodes,-1);
    int nComp=0;
    for(int i=0;i<Nnodes;++i)
        if(guf.find(i)==i && root_sz[i]>0) root2c[i]=nComp++;

    CCA2DResult out;
    out.nComp=nComp;
    out.labels.assign(N,-1);
    out.comp_size.assign(nComp,0);

    for(int i=0;i<Nnodes;++i)
        if(root2c[i]>=0) out.comp_size[root2c[i]]=root_sz[i];

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for(int i=0;i<N;++i) if(grid[i]!=0){
        int tx=(i%L)/Bx, ty=(i/L)/By;
        int t=ty*Tx+tx;
        int nid=tile_off[t]+local_label[i]-1;
        out.labels[i]=root2c[guf.find_nopc(nid)];
    }

    return out;
}

// ============================================================
// 3D TRACKER
// ============================================================

struct Tracker3D {
    std::vector<int> parent,birth,last_seen,used_stamp;
    std::vector<uint8_t> active;
    std::vector<long long> size3D;
    std::vector<int> free_ids;
    int next_cid=0;

    int FIND(int a){
        while(parent[a]!=a){
            parent[a]=parent[parent[a]];
            a=parent[a];
        }
        return a;
    }

    int FIND_nopc(int a) const {
        while(parent[a]!=a) a=parent[a];
        return a;
    }

    int NEW(int t){
        int cid;
        if(!free_ids.empty()){ cid=free_ids.back(); free_ids.pop_back(); }
        else{
            cid=next_cid++;
            parent.push_back(cid);
            birth.push_back(0);
            last_seen.push_back(0);
            used_stamp.push_back(0);
            active.push_back(0);
            size3D.push_back(0);
        }
        parent[cid]=cid;
        birth[cid]=last_seen[cid]=t;
        size3D[cid]=0;
        active[cid]=1;
        return cid;
    }

    int UNION(int a,int b){
        a=FIND(a); b=FIND(b);
        if(a==b) return a;
        parent[b]=a;
        birth[a]=std::min(birth[a],birth[b]);
        last_seen[a]=std::max(last_seen[a],last_seen[b]);
        size3D[a]+=size3D[b];
        active[b]=0;
        return a;
    }
};

// ============================================================
// Running mean update for normalized histograms
// mean_p[key] = mean over trials of p_trial(key)
// ============================================================

static void update_running_mean_prob(
    std::unordered_map<long long, double>& mean_p,
    const std::unordered_map<long long, long long>& counts,
    long long total_counts,
    int trial_index_1based
) {
    // trial_index_1based = 1..M
    if (total_counts <= 0) return;
    const double inv = 1.0 / double(total_counts);
    const double alpha = 1.0 / double(trial_index_1based);

    // Update only keys present in this trial.
    // Keys absent in this trial are treated as p=0; we do not explicitly decay them here.
    // If you want exact decay-to-zero tracking, we can add a cleanup pass later, but not needed in practice.
    for (const auto& kv : counts) {
        long long key = kv.first;
        double p_trial = double(kv.second) * inv;
        double& m = mean_p[key]; // default 0 if new
        m += alpha * (p_trial - m);
    }
}

// ============================================================
// One simulation run for a given rho + seed
// Returns: velocity time series (length T_max) and fills per-run hists (counts)
// ============================================================

struct RunResult {
    std::vector<double> v; // length T_max (full steps)
    std::unordered_map<long long,long long> hist_size;
    std::unordered_map<long long,long long> hist_life;
};

static RunResult run_one(
    int L, double rho, int T_max, uint32_t seed, double frac_right,
    int Bx, int By, int MIN2D, int threads_per_sim
) {
    const int N = L*L;
    const int total_cars=(int)std::floor(rho*N);

    std::vector<uint8_t> grid(N,0);
    init_grid_exact(grid,L,rho,seed,frac_right);

    Tracker3D tr;
    std::vector<int32_t> labels_prev(N,-1), labels_curr;
    std::vector<uint8_t> occ_prev(N,0);
    std::vector<int32_t> size2D_prev,size2D_curr;
    std::vector<int32_t> comp2cid_prev,comp2cid_curr;

    RunResult rr;
    rr.v.assign(T_max, 0.0);

    // t=0
    auto cca0 = cca_labels_sizes_parallel_periodic(grid,L,Bx,By,threads_per_sim);
    labels_prev = cca0.labels;
    size2D_prev = cca0.comp_size;
    comp2cid_prev.assign(cca0.nComp, -1);

    #pragma omp parallel for num_threads(threads_per_sim) schedule(static)
    for(int i=0;i<N;++i) occ_prev[i] = (grid[i]!=0);

    for(int a=0;a<cca0.nComp;++a){
        if(size2D_prev[a] < MIN2D) { comp2cid_prev[a] = -1; continue; }
        int cid = tr.NEW(0);
        tr.size3D[cid] = size2D_prev[a];
        comp2cid_prev[a] = cid;
    }

    // full steps
    for(int t=1;t<=T_max;++t){
        int moved1 = update_right_and_count_parallel(grid, L, threads_per_sim);
        int moved2 = update_down_and_count_parallel(grid, L, threads_per_sim);
        rr.v[t-1] = double(moved1 + moved2) / double(total_cars);

        auto cca = cca_labels_sizes_parallel_periodic(grid,L,Bx,By,threads_per_sim);
        labels_curr = cca.labels;
        size2D_curr = cca.comp_size;
        comp2cid_curr.assign(cca.nComp, -1);

        // build unique pairs (b, rootCID)
        std::vector<std::pair<int,int>> pairs;
        #pragma omp parallel num_threads(threads_per_sim)
        {
            std::vector<std::pair<int,int>> local;
            #pragma omp for schedule(static)
            for(int i=0;i<N;++i) {
                if(!(occ_prev[i] && grid[i]!=0)) continue;

                int a = labels_prev[i];
                int b = labels_curr[i];
                if(a < 0 || b < 0) continue;

                if(size2D_prev[a] < MIN2D) continue;
                if(size2D_curr[b] < MIN2D) continue;

                int cid0 = comp2cid_prev[a];
                if(cid0 < 0) continue;

                int root = tr.FIND_nopc(cid0);
                local.emplace_back(b, root);
            }

            #pragma omp critical
            pairs.insert(pairs.end(), local.begin(), local.end());
        }

        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

        size_t p=0;
        for(int b=0;b<cca.nComp;++b){
            if(size2D_curr[b] < MIN2D) { comp2cid_curr[b] = -1; continue; }

            if(p>=pairs.size() || pairs[p].first!=b){
                int cid = tr.NEW(t);
                tr.size3D[cid] += size2D_curr[b];
                tr.used_stamp[cid] = t;
                comp2cid_curr[b] = cid;
            } else {
                int cid = pairs[p].second;
                while(p<pairs.size() && pairs[p].first==b)
                    cid = tr.UNION(cid, pairs[p++].second);
                cid = tr.FIND(cid);
                tr.size3D[cid] += size2D_curr[b];
                tr.last_seen[cid] = t;
                tr.used_stamp[cid] = t;
                comp2cid_curr[b] = cid;
            }
        }

        // deaths
        for(int cid=0; cid<tr.next_cid; ++cid) {
            if(tr.parent[cid]==cid && tr.active[cid] && tr.used_stamp[cid]!=t){
                int life = tr.last_seen[cid] - tr.birth[cid] + 1;
                rr.hist_size[tr.size3D[cid]]++;
                rr.hist_life[life]++;
                tr.active[cid]=0;
                tr.free_ids.push_back(cid);
            }
        }

        // roll
        labels_prev.swap(labels_curr);
        size2D_prev.swap(size2D_curr);
        comp2cid_prev.swap(comp2cid_curr);

        #pragma omp parallel for num_threads(threads_per_sim) schedule(static)
        for(int i=0;i<N;++i) occ_prev[i] = (grid[i]!=0);
    }

    // finalize survivors at T_max
    for(int cid=0; cid<tr.next_cid; ++cid) {
        if(tr.parent[cid]==cid && tr.active[cid]) {
            int life = T_max - tr.birth[cid] + 1;
            rr.hist_size[tr.size3D[cid]]++;
            rr.hist_life[life]++;
        }
    }

    return rr;
}

// ============================================================
// write histogram map<double> as CSV
// ============================================================
static void write_prob_csv(const std::string& filename,
                           const std::string& xname,
                           const std::unordered_map<long long,double>& prob) {
    std::vector<std::pair<long long,double>> v(prob.begin(), prob.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.first < b.first; });

    std::ofstream out(filename);
    out << xname << ",p\n";
    for (auto& kv : v) out << kv.first << "," << kv.second << "\n";
}

// ============================================================
// write velocity matrix as binary:
// int32 T, int32 M, then M*T doubles in row-major [m][t]
// ============================================================
static void write_vel_bin(const std::string& filename,
                          const std::vector<double>& vel_mt,
                          int M, int T) {
    std::ofstream out(filename, std::ios::binary);
    int32_t TT = (int32_t)T;
    int32_t MM = (int32_t)M;
    out.write((char*)&TT, sizeof(int32_t));
    out.write((char*)&MM, sizeof(int32_t));
    out.write((char*)vel_mt.data(), sizeof(double) * (size_t)M * (size_t)T);
}

// ============================================================
// MAIN
// ============================================================
int main(){
    const int L = 500;
    const int T_max = 5000;     // full steps
    const uint32_t seed0 = 12345;
    const double frac_right = 0.5;

    const int Bx=64, By=64;
    const int MIN2D = 2;

    const int M = 100;
    const std::vector<double> rhos = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9};

    // --- Parallel strategy ---
    // Total physical cores / threads available:
    const int max_threads = omp_get_max_threads();

    // Choose threads per simulation and how many sims to run concurrently.
    // Rule of thumb: threads_per_sim * sims_in_parallel <= max_threads.
    const int threads_per_sim = std::max(1, max_threads / 4);   // e.g. quarter the machine
    const int sims_in_parallel = std::max(1, max_threads / threads_per_sim);

    std::cout << "max_threads=" << max_threads
              << " threads_per_sim=" << threads_per_sim
              << " sims_in_parallel=" << sims_in_parallel << "\n";

    for (size_t ir=0; ir<rhos.size(); ++ir) {
        double rho = rhos[ir];

        // Store velocities for all runs: vel_all[m*T + (t-1)]
        std::vector<double> vel_all((size_t)M * (size_t)T_max, 0.0);

        // Running mean probability histograms across runs
        std::unordered_map<long long, double> mean_p_size;
        std::unordered_map<long long, double> mean_p_life;

        // We parallelize across trials (outer) up to sims_in_parallel
        #pragma omp parallel for num_threads(sims_in_parallel) schedule(dynamic)
        for (int m = 0; m < M; ++m) {
            uint32_t seed = seed0 + (uint32_t)(1000003u * (uint32_t)ir + (uint32_t)m);

            RunResult rr = run_one(L, rho, T_max, seed, frac_right, Bx, By, MIN2D, threads_per_sim);

            // copy velocity into matrix
            for (int t = 0; t < T_max; ++t) {
                vel_all[(size_t)m * (size_t)T_max + (size_t)t] = rr.v[t];
            }

            // normalize THIS trial's histograms, then update running mean
            long long tot_size = 0;
            for (auto& kv : rr.hist_size) tot_size += kv.second;
            long long tot_life = 0;
            for (auto& kv : rr.hist_life) tot_life += kv.second;

            // Update global means (need synchronization)
            #pragma omp critical
            {
                update_running_mean_prob(mean_p_size, rr.hist_size, tot_size, m+1);
                update_running_mean_prob(mean_p_life, rr.hist_life, tot_life, m+1);
            }
        }

        // Write outputs for this rho
        auto rho_tag = std::string("rho") + std::to_string(int(std::round(rho*100))) ;

        write_vel_bin("vel_" + rho_tag + ".bin", vel_all, M, T_max);
        write_prob_csv("hist_size3D_" + rho_tag + ".csv", "size", mean_p_size);
        write_prob_csv("hist_life3D_" + rho_tag + ".csv", "life", mean_p_life);

        std::cout << "Done rho=" << rho << " -> "
                  << "vel_" << rho_tag << ".bin, "
                  << "hist_size3D_" << rho_tag << ".csv, "
                  << "hist_life3D_" << rho_tag << ".csv\n";
    }

    return 0;
}
