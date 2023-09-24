#include <stdint.h>

#include <DarkMatterParticleContainer.H>


using namespace amrex;

/// These are helper functions used when initializing from a morton-ordered
/// binary particle file.
namespace {

  inline uint64_t split(unsigned int a) {
    uint64_t x = a & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8)  & 0x100f00f00f00f00f;
    x = (x | x << 4)  & 0x10c30c30c30c30c3;
    x = (x | x << 2)  & 0x1249249249249249;
    return x;
  }
  
  inline uint64_t get_morton_index(unsigned int x,
                                   unsigned int y,
                                   unsigned int z) {
    uint64_t morton_index = 0;
    morton_index |= split(x) | ( split(y) << 1) | (split(z) << 2);
    return morton_index;
  }  

  struct BoxMortonKey {
    uint64_t morton_id;
    int box_id;
  };

  struct by_morton_id { 
    bool operator()(const BoxMortonKey &a, const BoxMortonKey &b) { 
      return a.morton_id < b.morton_id;
    }
  };

  std::string get_file_name(const std::string& base, int file_num) {
    std::stringstream ss;
    ss << base << file_num;
    return ss.str();
  }

  struct ParticleMortonFileHeader {
    long NP;
    int  DM;
    int  NX;
    int  SZ;
    int  NF;
  };
  
  void ReadHeader(const std::string& dir,
                  const std::string& file,
                  ParticleMortonFileHeader& hdr) {
    std::string header_filename = dir;
    header_filename += "/";
    header_filename += file;
    
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(header_filename, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream HdrFile(fileCharPtrString, std::istringstream::in);

    HdrFile >> hdr.NP;
    HdrFile >> hdr.DM;
    HdrFile >> hdr.NX;
    HdrFile >> hdr.SZ;
    HdrFile >> hdr.NF;    
  }

}



//ACJ
namespace {
  typedef std::pair<int, int> BidNp;

  struct PairCompare {
    bool inverse = false;

    PairCompare(const bool a_inverse=false) : inverse(a_inverse) {}

    bool operator() (const BidNp& lhs, const BidNp& rhs)
    {
      return inverse ? lhs.second > rhs.second : lhs.second < rhs.second;
    }
  };

  typedef std::priority_queue<BidNp, Vector<BidNp>, PairCompare> BidNpHeap;
}
//ACJ

//ACJ
void DarkMatterParticleContainer::countParticle(int lev,
                                          const BoxList& bl,
                                          const IntVect& binsize,
                                          Vector<int>&   pcounts,
                                          Vector<int>&   poffsets)
{
  const auto* boxes = bl.data().data();
  const int   nbox  = bl.size();

  pcounts.clear();
  poffsets.resize(1, 0);

  // find the total number of bins and offsets for each box's bins
  int total_nbin = 0;
  for (int i=0; i<nbox; ++i) {
    IntVect boxsize = boxes[i].size();
    AMREX_ASSERT_WITH_MESSAGE(AMREX_D_TERM(
         (boxsize[0] < binsize[0] || boxsize[0] % binsize[0] == 0),
      && (boxsize[1] < binsize[1] || boxsize[1] % binsize[1] == 0),
      && (boxsize[2] < binsize[2] || boxsize[2] % binsize[2] == 0)),
      "ERROR: For now, the greedy balance requires the box size to be less than"
      " or divisible over the bin size");

    total_nbin += AMREX_D_TERM(  (boxsize[0] + binsize[0] - 1) / binsize[0],
                              * ((boxsize[1] + binsize[1] - 1) / binsize[1]),
                              * ((boxsize[2] + binsize[2] - 1) / binsize[2]));
    poffsets.push_back(total_nbin);
  }
  pcounts.resize(total_nbin, 0);

  // particle tiles and geometry of this level
  const auto& geom   = Geom(lev);
  const auto  domain = geom.Domain();
  const auto  dx_inv = geom.InvCellSizeArray();
  const auto  prob_lo = geom.ProbLoArray();

  amrex::Gpu::DeviceVector<Box> d_boxes(nbox);
  amrex::Gpu::DeviceVector<int> d_poffsets(poffsets.size());
  amrex::Gpu::DeviceVector<int> d_pcounts(pcounts.size(), 0);

  Gpu::copy(Gpu::hostToDevice, bl.data().begin(), bl.data().end(), d_boxes.begin());
  Gpu::copy(Gpu::hostToDevice, poffsets.begin(), poffsets.end(), d_poffsets.begin());

  Gpu::synchronize();

  const auto* p_d_boxes = d_boxes.data();
  const auto* p_d_poffsets = d_poffsets.data();

  auto* p_d_pcounts  = d_pcounts.dataPtr();

  for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
    const auto& aos     = pti.GetArrayOfStructs();
    const auto* pstruct = aos().dataPtr();
    int         np      = pti.numParticles();

    ParallelFor(np, [pstruct, prob_lo, dx_inv, domain, nbox, p_d_boxes,
                     binsize, p_d_pcounts, p_d_poffsets]
      AMREX_GPU_DEVICE (int i) noexcept
      {
        IntVect cell_ijk = getParticleCell(pstruct[i], prob_lo, dx_inv, domain);
        Box     box_tmp;
        for (int ibox=0; ibox<nbox; ++ibox) {
          if (p_d_boxes[ibox].contains(cell_ijk)) {
            int ibin = getTileIndex(cell_ijk, p_d_boxes[ibox], true, binsize, box_tmp);
            ibin += p_d_poffsets[ibox];
            Gpu::Atomic::AddNoRet(p_d_pcounts + ibin, 1);
          }
        }
      });// end parallel for
  }// end for pariter

  Gpu::synchronize();
  Gpu::copy(Gpu::deviceToHost, d_pcounts.begin(), d_pcounts.end(), pcounts.begin());
} 
//ACJ

//ACJ
void DarkMatterParticleContainer::countParticle(int lev, iMultiFab& np_mf)
{
  //get the count of particles in mf_temp
  //MultiFab mf_temp(np_mf.boxArray(), np_mf.DistributionMap(), 1, 0); 
  //DarkMatterParticleContainer::Increment(mf_temp, lev);    
  //copy the values of mf_temp int np_mf
  //Copy (FabArray<DFAB>& dst, FabArray<SFAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
  //Copy(np_mf, mf_temp, 0, 0, 1, 0);
  
  // particle tiles and geometry of this level
  //const auto& geom    = Geom(lev);
  const auto& geom    = this->m_gdb->Geom(lev);
  const auto  domain  = geom.Domain();
  const auto  dx_inv  = geom.InvCellSizeArray();
  const auto  prob_lo = geom.ProbLoArray();

  for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
    const auto& aos     = pti.GetArrayOfStructs();
    const auto* pstruct = aos().dataPtr();
    int         np      = pti.numParticles();

    Array4<int> const& np_a4 = np_mf.array(pti); 
    ParallelFor(np, [pstruct, np_a4, prob_lo, dx_inv, domain]
      AMREX_GPU_DEVICE (int i) noexcept
      {
        IntVect     ijk = getParticleCell(pstruct[i], prob_lo, dx_inv, domain);
        Gpu::Atomic::AddNoRet(&np_a4(ijk[0], ijk[1], ijk[2], 0), 1);
      });// end parallel for
  }// end for pariter

  Gpu::synchronize();
}
//ACJ

//ACJ
Box DarkMatterParticleContainer::chop_and_distribute_box(int o_box_id, int np_target, const Vector<Box>& fbl_vec0, Vector<Box>& fbl_vec, iMultiFab& np_mf_f, int & new_box_np, int room)
{
    const IArrayBox&         np_fab = np_mf_f[o_box_id];
    Array4<int const> const& np_arr = np_fab.const_array();
    const int*               np_ptr = np_arr.dataPtr();
    
    Box remain0 = fbl_vec0[o_box_id]; //fbox0; this is use to get strides so that you can continue to iterate into np_mf_f
    Box remain  = fbl_vec[o_box_id]; //this is actual Vector of boxes for new gridding distribution
    IntVect lo  = remain.smallEnd(); //fbox0.smallEnd();
    IntVect stride {1, remain0.length(0), remain0.length(0)*remain0.length(1)};
    
    int np_target_lo = static_cast<int>(np_target * 0.95);
    int np_target_hi = static_cast<int>(np_target * 1.05);
    int min_chop_dir = -1;

    Vector<int> np_cutoff_bydir (AMREX_SPACEDIM);
    Vector<int> np_surface_bydir(AMREX_SPACEDIM);
    Vector<int> chop_pos_bydir  (AMREX_SPACEDIM);

    for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
      if (remain.length(chop_dir) < 2*greedy_min_grid_size) {
        np_surface_bydir[chop_dir] = std::numeric_limits<int>::max();
        np_cutoff_bydir [chop_dir] = std::numeric_limits<int>::max();
        chop_pos_bydir  [chop_dir] = -1;
        continue;
      }

      int dir0 = (chop_dir + 1) % AMREX_SPACEDIM;
      int dir1 = (chop_dir + 2) % AMREX_SPACEDIM;

      int chop_lo  = remain.smallEnd(chop_dir) + greedy_min_grid_size;
      int chop_hi  = remain.bigEnd(chop_dir);
      int chop_pos = chop_hi;

      int np_tmp       = 0;
      int np_diff_prev = np_target + 1;
     
      // find the chop position with the closest np to np_target
      while (chop_pos >= chop_lo) {
        int np_slice = 0;
        for (int idir0=remain.smallEnd(dir0); idir0<=remain.bigEnd(dir0); ++idir0) {
          for (int idir1=remain.smallEnd(dir1); idir1<=remain.bigEnd(dir1); ++idir1) {
          // the index is based on the offset to the lower bound of the original box
            int iCell = stride[dir0]     * (idir0    - lo[dir0])
                      + stride[dir1]     * (idir1    - lo[dir1])
                      + stride[chop_dir] * (chop_pos - lo[chop_dir]);
            np_slice += *(np_ptr + iCell);
          }
        }
        int np_diff = std::abs(np_target - np_tmp - np_slice);
        if((np_diff>np_diff_prev || np_tmp+np_slice>room) && chop_hi-chop_pos+1>=greedy_min_grid_size)
        {
            ++chop_pos;
            break;
        }
        else if(np_tmp+np_slice>room && chop_hi-chop_pos+1<greedy_min_grid_size)
        {
            Box dummy;
            new_box_np = -1;
            return dummy;
        }
        else
        {
            np_diff_prev = np_diff;
            np_tmp += np_slice;
            --chop_pos;
        }
      }// end while
      chop_pos = chop_pos == chop_lo-1 ? chop_lo : chop_pos;
      chop_pos = chop_pos > chop_hi ? chop_hi : chop_pos; 
      np_cutoff_bydir[chop_dir] = np_tmp;
      chop_pos_bydir [chop_dir] = chop_pos;
      
      // count surface particles
      Box remain_copy(remain);
      Box cutoff(remain_copy.chop(chop_dir, chop_pos));
          
      //Box cutoff (remain_copy.chop(chop_dir, chop_pos));
      int np_surface = 0;
      // z faces
      for (int j=cutoff.smallEnd(1); j<=cutoff.bigEnd(1); ++j) {
        for (int i=cutoff.smallEnd(0); i<=cutoff.bigEnd(0); ++i) {
          np_surface += np_arr(i, j, cutoff.smallEnd(2), 0);
          np_surface += np_arr(i, j, cutoff.bigEnd  (2), 0);
        }
      }
      // y faces
      for (int k=cutoff.smallEnd(2); k<=cutoff.bigEnd(2); ++k) {
        for (int i=cutoff.smallEnd(0); i<=cutoff.bigEnd(0); ++i) {
          np_surface += np_arr(i, cutoff.smallEnd(1), k, 0);
          np_surface += np_arr(i, cutoff.bigEnd  (1), k, 0);
        }
      }
      // x faces
      for (int k=cutoff.smallEnd(2); k<=cutoff.bigEnd(2); ++k) {
        for (int j=cutoff.smallEnd(1); j<=cutoff.bigEnd(1); ++j) {
          np_surface += np_arr(cutoff.smallEnd(0), j, k, 0);
          np_surface += np_arr(cutoff.bigEnd  (0), j, k, 0);
        }
      }
      np_surface_bydir[chop_dir] = np_surface;
    }// end for chop_dir

    Vector<int> within_toler_dirs;
    for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
      if (  np_cutoff_bydir[chop_dir] >= np_target_lo
         && np_cutoff_bydir[chop_dir] <= np_target_hi) {
        within_toler_dirs.push_back(chop_dir);
      }
    }

    // if none of the dir fits, choose the one closest to fit
    if (within_toler_dirs.empty()) {
      int min_diff = std::numeric_limits<int>::max();
      for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
        int diff = std::abs(np_cutoff_bydir[chop_dir] - np_target);
        if (diff < min_diff) {
          min_diff     = diff;
          min_chop_dir = chop_dir;
        }
      }
    }
    // choose the one with least surface particles
    else {
      int min_np_surface = std::numeric_limits<int>::max();
      for (int dir: within_toler_dirs) {
        if (np_surface_bydir[dir] < min_np_surface) {
          min_np_surface = np_surface_bydir[dir];
          min_chop_dir = dir;
        }
      }
    }
    
    // chop the cutoff chunk
    Box new_box;
    if(chop_pos_bydir[min_chop_dir]>0 && np_cutoff_bydir[min_chop_dir]<=room)
    {
        new_box = remain.chop(min_chop_dir, chop_pos_bydir[min_chop_dir]);
        new_box_np = np_cutoff_bydir[min_chop_dir];
        fbl_vec[o_box_id] = remain;
    }
    else
        new_box_np = -2;

      
    return new_box;
}
//ACJ

//ACJ
void get_over_under_load_ranks(Vector<int>& underload_ranks, Vector<int>& overload_ranks, const Vector<int>& pcount_rank, 
        Vector<BidNp>& o_q, Vector<BidNp>& u_q, const int& o_toler_np, const int& u_toler_np)
{
    //do underload ranks
    if(underload_ranks.size()>0){
        Vector<int> tmp_underload_ranks;
        for(int u_rnk : underload_ranks){
            if(pcount_rank[u_rnk] < u_toler_np)
                tmp_underload_ranks.push_back(u_rnk);
        }
        underload_ranks = tmp_underload_ranks;
    }
    else{
        for(int i=0; i<pcount_rank.size(); ++i){
            if(pcount_rank[i] < u_toler_np)
                underload_ranks.push_back(i);
        }
    }
    //do overload ranks
    overload_ranks.clear();
    for (int i=0; i<pcount_rank.size(); ++i){
        if (pcount_rank[i] > o_toler_np) 
            overload_ranks.push_back(i);
    }
    
    o_q.clear();
    u_q.clear();
    // construct queues for over and under loaded ranks
    for (int rank: overload_ranks)
        o_q.push_back(std::make_pair(rank, pcount_rank[rank]));
    for (int rank: underload_ranks)
        u_q.push_back(std::make_pair(rank, pcount_rank[rank]));
    
    std::sort(o_q.begin(), o_q.end(), PairCompare(true)); //descending order
    std::sort(u_q.begin(), u_q.end(), PairCompare(false)); //ascending order

    return;
}
//ACJ

//Simple function testing whether the calling rank is BOTH an overload rank AND has an available underload rank to 
//unload into. If both aren't true wait to execute.
bool should_execute(const Vector<BidNp> & o_q, const Vector<BidNp>& u_q, int& o_rank, 
        int& o_rank_np, int& u_rank, int& u_rank_np, bool& am_overload_rank)
{
    bool should_exec = false;
    am_overload_rank=false;
    int myproc = ParallelDescriptor::MyProc();
    for(int idx = 0; idx < o_q.size(); ++idx){
        if (myproc == o_q[idx].first){ //each overload rank uses underload rank at same index
            o_rank = o_q[idx].first;
            o_rank_np = o_q[idx].second;
            am_overload_rank = true;
            if(idx<u_q.size()){ //means there is an underload rank for this overload rank
                u_rank = u_q[idx].first;
                u_rank_np = u_q[idx].second;
                should_exec = true;
            }
            break;
        }
    }
    return should_exec;
}

//ACJconst amrex::BoxArray& fba, const amrex::DistributionMapping& fdmap
void 
DarkMatterParticleContainer::load_balance(int lev, const amrex::BoxArray& fba, const amrex::DistributionMapping& fdmap, amrex::Real overload_toler, amrex::Real underload_toler, 
        amrex::BoxArray &new_ba, amrex::DistributionMapping &new_dm)
{
    BL_PROFILE("DarkMatterParticleContainer::load_balance()"); 
    // parent grid info
    //const amrex::BoxArray& fba = ParticleBoxArray(lev);
    //const amrex::DistributionMapping& fdmap = ParticleDistributionMap(lev);
    const Vector<int>& fpmap    = fdmap.ProcessorMap();
    BoxList            fbl      = fba.boxList();
    Vector<Box>&       fbl_vec  = fbl.data();
    Vector<Box>        fbl_vec0 = fbl_vec;
    int MyProc = ParallelDescriptor::MyProc(), NProcs = ParallelDescriptor::NProcs();

    // If the mapping between particle and parent grids hasn't been set,
    // set an 1to1 mapping.
    if (m_pboxid_to_fboxid.size()==0) {
        m_pboxid_to_fboxid.resize(fbl.size());
        std::iota(m_pboxid_to_fboxid.begin(), m_pboxid_to_fboxid.end(), 0);
    }

    // count particles in parent grid
    Vector<int> pcount_fbox(fba.size(), 0);
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
        int fboxid = m_pboxid_to_fboxid[pti.index()];
        pcount_fbox[fboxid] += pti.numParticles();
    }
    ParallelDescriptor::ReduceIntSum(pcount_fbox.dataPtr(), pcount_fbox.size());
    //Vector<long> pcount_fbox = this->NumberOfParticlesInGrid(lev, true, false);

    // count particles by rank
    Vector<int> pcount_rank(NProcs, 0), pcount_rank_diff(NProcs, 0);
    std::unordered_map<int,  std::list<BidNp>> rank_box_map;
    for (auto i=0; i<pcount_fbox.size(); ++i) {
        pcount_rank[fpmap[i]] += pcount_fbox[i];
        rank_box_map[fpmap[i]].push_back(BidNp(i, pcount_fbox[i]));
    }

    // re-initialize the map from particle box to fluid box
    m_pboxid_to_fboxid.resize(fbl.size());
    std::iota(m_pboxid_to_fboxid.begin(), m_pboxid_to_fboxid.end(), 0);

    // count total # particles
    /*if (m_total_numparticle <= 0) {
        m_total_numparticle = 0;
        for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
            m_total_numparticle += static_cast<long>(pti.numRealParticles());
        ParallelDescriptor::ReduceLongSum(m_total_numparticle);
    }*/

    // particle counts based on particle grids
    MultiFab np_mf_p(fba, fdmap, 1, 0); //ParticleBoxArray(lev), ParticleDistributionMap(lev), 1, 0);
    np_mf_p.setVal(0);
    //countParticle(lev, np_mf_p);                
    m_total_numparticle = IncrementWithTotal(np_mf_p, lev, false);
    // particle counts based on fluid grids
    iMultiFab np_mf_f(The_Pinned_Arena());
    np_mf_f.define(fba, fdmap, 1, 0);
    // copy the counts over
    //np_mf_f.ParallelCopy(np_mf_p);
    Copy(np_mf_f, np_mf_p, 0,0,1,0);

    // find the underload and overload tolerances
    if (overload_toler<=1)
        amrex::Abort("DarkMatterParticleContainer::load_balance(): overload_toler must be greater than 1");
    Real avg_np = static_cast<Real>(m_total_numparticle) / NProcs;
    int  o_toler_np = static_cast<int>(avg_np * overload_toler);
    int  u_toler_np = static_cast<int>(avg_np); //make any rank below average a potential underloaded rank

    // debug
    Print() << "avg np: "              << avg_np
          << " overload tolerance "  << overload_toler << "\n";

    //now actually start the redivisioning algorithm
    amrex::Print()<<"starting 3D load_balance algorithm..."<<std::flush;

    //new box->processor map
    Vector<int> new_ppmap(fpmap), ppmap_chngs(fpmap.size(), 0);    
    int u_rank, o_rank, u_rank_np, o_rank_np;
    Vector<Box> addl_bl; //this will store the additional boxes that result from splitting
    Vector<int> addl_ppmap, addl_m_pboxid_to_fboxid, underload_ranks, overload_ranks;
    Vector<BidNp> o_q, u_q;
    //first get the under and over load ranks
    get_over_under_load_ranks(underload_ranks, overload_ranks, pcount_rank, o_q, u_q, o_toler_np, u_toler_np);
    //then find out if this rank is an overload rank (and has an underload rank to unload into) and execute 
    //algorithm if both are true.
    bool am_overload_rank = false;
    bool should_exec = should_execute(o_q, u_q, o_rank, o_rank_np, u_rank, u_rank_np, am_overload_rank); 
    
    //need to create communicator with only executing ranks
    MPI_Comm overload_comm, world_comm_dup;
    MPI_Group world_group, overload_group;
    MPI_Comm_dup(ParallelDescriptor::Communicator(), &world_comm_dup);
    MPI_Comm_group(world_comm_dup, &world_group);
    int overload_tag = 0; 

    while(am_overload_rank)
    { 
        //create appropriate comunicator from group
        MPI_Group_incl(world_group, overload_ranks.size(), overload_ranks.dataPtr(), &overload_group);
        MPI_Comm_create_group(world_comm_dup, overload_group, overload_tag, &overload_comm); 
        ParallelContext::push(overload_comm);

        if(should_exec)
        {
            //get the list of boxes for this overload rank
            std::list<BidNp>& o_box_list = rank_box_map[o_rank];
            o_box_list.sort(PairCompare(true));  //sort in descending order
             
            //TODO: could add more intelligence to matching over and underload ranks
            //get number of particles this rank needs to remove
            int num_2_rmv = o_rank_np - o_toler_np;
            int room = avg_np - u_rank_np;
            while(num_2_rmv>0  && room>0)
            { 
                //get the space available and num of particles we will attempt to remove
                int num_do_rmv = std::min(room, num_2_rmv);
                
                //Reduce load as much as possible by simply sending as many boxes as possible to underload rank
                auto it = o_box_list.begin();
                while(num_do_rmv>0 && it!=o_box_list.end())
                {
                    int box_2_mv_id = it->first;
                    int box_2_mv_np = it->second;
                    if(box_2_mv_np<=room)
                    {
                        ppmap_chngs[box_2_mv_id] = u_rank+1; //add 1 so that rank 0 is non-zero, will subtract 1 later
                        num_2_rmv -= box_2_mv_np;
                        num_do_rmv -= box_2_mv_np;
                        u_rank_np += box_2_mv_np;
                        room = avg_np - u_rank_np;
                        pcount_rank_diff[o_rank] -= box_2_mv_np;
                        pcount_rank_diff[u_rank] += box_2_mv_np;
                        it++;
                        o_box_list.erase(std::prev(it));
                    }
                    else
                        it++;
                }
               
                if(num_do_rmv>0) //need to split
                {
                    //choose the biggest box 
                    int o_box_id = o_box_list.front().first;
              
                    //split the box and send chunk to u_rank 
                    //note: below function does necessary updates to fbl_vec
                    int new_box_np;
                    //TODO: better splitting below
                    Box new_box = chop_and_distribute_box(o_box_id, num_do_rmv, fbl_vec0, fbl_vec, np_mf_f, new_box_np, room); 
                    if(new_box_np>=0)
                    {
                        addl_bl.push_back(new_box);

                        // update new dmap
                        addl_ppmap.push_back(u_rank);
                        addl_m_pboxid_to_fboxid.push_back(o_box_id);

                        //make necessary changes to overload box list
                        int o_box_np = o_box_list.front().second;
                        o_box_list.pop_front(); //remove old entry
                        BidNp o_pair(o_box_id, o_box_np-new_box_np);
                        //insert box into proper sorted position
                        int idx = 0;
                        for (auto it = o_box_list.begin(); it!=o_box_list.end(); ++it){
                            if(o_box_np >= it->second) {
                                o_box_list.insert(it, o_pair); 
                                break;
                            }
                            idx++;                    
                        }
                        if(idx==o_box_list.size()) o_box_list.push_back(o_pair); 
                    
                        //update number of particles left to remove
                        num_2_rmv -= new_box_np;
                        u_rank_np += new_box_np;
                        pcount_rank_diff[o_rank] -= new_box_np;
                        pcount_rank_diff[u_rank] += new_box_np;
                    }
                    else //couldn't create new_box that meets criteria, just remove box from list
                    //TODO: consider this is unnecessary if limiting factor is small "room"s not highly dense cells
                    {
                        //idea here is that if we did not add anything to the underload rank load
                        //previously in this (outermost) loop, then the space available is as good as it's going to get
                        //in the sense that if we recieve a new u_rank after re-assessing it can only have the same
                        //or less space, and the space in this one won't change by running the loop again. So just need to 
                        //drop this box. 
                        if(pcount_rank_diff[u_rank]==0){ 
                            num_2_rmv -= o_box_list.front().second;
                            pcount_rank_diff[o_rank] -= o_box_list.front().second;
                            o_box_list.pop_front();
                        }
                        break; //break to re-asses underload ranks in case this one is close to full
                    }
                }
                
                //update count of room available in this underload rank
                room = avg_np - u_rank_np;
                
            }  //end while(num_2_rmv_>0 && room>0)
        }// end if(should_exec) 
       
        //////re-assess which ranks are the under and overload ranks////
        //
        //need to synchronize changes in pcount_rank across processes
        MPI_Allreduce(MPI_IN_PLACE, pcount_rank_diff.dataPtr(), pcount_rank_diff.size(), 
                MPI_INT, MPI_SUM, ParallelContext::CommunicatorSub()); //overload_comm);
        //now add up differences in pcount_rank
        for(int i=0; i<pcount_rank.size(); ++i){
            pcount_rank[i] += pcount_rank_diff[i];
            pcount_rank_diff[i] = 0; //reset pcount_rank_diff
        }
        //now re-assess what the under and overload ranks are
        //this function also does the necessary updates to input vectors 
        get_over_under_load_ranks(underload_ranks, overload_ranks, pcount_rank, 
                o_q, u_q, o_toler_np, u_toler_np); 

        //then need to see if this rank should execute and keep executing if so. 
        //This function also does necessary updating to: o_rank, o_rank_np, u_rank and u_rank_np
        should_exec = should_execute(o_q, u_q, o_rank, o_rank_np, u_rank, u_rank_np, am_overload_rank); 

        //free the overload comm  and group for next iteration
        ParallelContext::pop();
        MPI_Comm_free(&overload_comm); 
        MPI_Group_free(&overload_group);

    } //end while(am_overload_rank)

    //everyone needs to synchronize their changes to new_ppmap
    ParallelDescriptor::ReduceIntMax(ppmap_chngs.dataPtr(), ppmap_chngs.size());      
    for(int i=0 ; i<ppmap_chngs.size(); ++i){
       if(ppmap_chngs[i]) new_ppmap[i] = ppmap_chngs[i]-1; //-1 is just to convert flag to rank
    }
    //write number of new boxes into one vector so every rank knows how to form global list
    Vector<int> new_bxs_per_rank(NProcs, 0);
    new_bxs_per_rank[MyProc] = addl_bl.size();
    ParallelDescriptor::ReduceIntMax(new_bxs_per_rank.dataPtr(), new_bxs_per_rank.size());
    
    //resize fbl_vec and new_ppmap to be able to contain additional entries
    int num_new = std::accumulate(new_bxs_per_rank.begin(), new_bxs_per_rank.end(), 0);
    if(num_new>0) //only do this if a rank added a box
    {
        int write_pos = fbl_vec.size();
        for(int i=0; i<num_new; ++i){
            fbl_vec.push_back(Box(IntVect{std::numeric_limits<int>::min()},
                                IntVect{std::numeric_limits<int>::max()}));
            new_ppmap.push_back(-1);
            m_pboxid_to_fboxid.push_back(-1);
        } 

        //now add new entries (for this rank), in appropriate position.
        //(Will sync with other ranks later)
        //
        write_pos += std::accumulate(new_bxs_per_rank.begin(), new_bxs_per_rank.begin()+MyProc, 0);
        for(int i=0; i<addl_bl.size(); ++i){
            fbl_vec[write_pos+i] = addl_bl[i];
            new_ppmap[write_pos+i] = addl_ppmap[i];
            m_pboxid_to_fboxid[write_pos+i] = addl_m_pboxid_to_fboxid[i];
        }

        //sync the dmaps and pboxid_to_fboxid maps
        ParallelDescriptor::ReduceIntMax(&new_ppmap[new_ppmap.size()-1]-num_new+1, num_new);
        ParallelDescriptor::ReduceIntMax(&m_pboxid_to_fboxid[m_pboxid_to_fboxid.size()-1]-num_new+1, num_new);
    }

    //sync the boxes
    size_t nbox = fbl_vec.size();
    Vector<int> ubound_buf(3*nbox);
    Vector<int> lbound_buf(3*nbox);
    // pack the buffer
    for (size_t i=0; i<nbox; ++i) {
      ubound_buf[3*i]   = fbl_vec[i].bigEnd(0);
      ubound_buf[3*i+1] = fbl_vec[i].bigEnd(1);
      ubound_buf[3*i+2] = fbl_vec[i].bigEnd(2);
      lbound_buf[3*i]   = fbl_vec[i].smallEnd(0);
      lbound_buf[3*i+1] = fbl_vec[i].smallEnd(1);
      lbound_buf[3*i+2] = fbl_vec[i].smallEnd(2);
    }

    ParallelDescriptor::ReduceIntMin(ubound_buf.dataPtr(), 3*nbox);
    ParallelDescriptor::ReduceIntMax(lbound_buf.dataPtr(), 3*nbox);
   
    // unpack the buffer
    for (size_t i=0; i<nbox; ++i) {
      fbl_vec[i] =
        Box(IntVect{lbound_buf[3*i], lbound_buf[3*i+1], lbound_buf[3*i+2]},
            IntVect{ubound_buf[3*i], ubound_buf[3*i+1], ubound_buf[3*i+2]});
    }
    
    // ba and dmap to particle container
    new_ba = BoxArray(fbl); //SetParticleBoxArray(lev, BoxArray(fbl));
    new_dm = DistributionMapping(new_ppmap); //SetParticleDistributionMap(lev, DistributionMapping(new_ppmap)); 
   
    amrex::Print()<<"done."<<std::endl;
}
//ACJ

//ACJ
void
DarkMatterParticleContainer::partitionParticleGrids(int lev, const amrex::BoxArray& fba, const amrex::DistributionMapping& fdmap, amrex::Real overload_toler, 
                                                     amrex::Real underload_toler)
{
   
  // fluid grid info
  const Vector<int>& fpmap    = fdmap.ProcessorMap();
  BoxList            fbl      = fba.boxList();
  Vector<Box>&       fbl_vec  = fbl.data();
  Vector<Box>        fbl_vec0 = fbl_vec;

  // If the mapping between particle and fluid grids hasn't been set,
  // set an 1to1 mapping.
  if (m_pboxid_to_fboxid.size() == 0) {
    m_pboxid_to_fboxid.resize(fbl.size());
    std::iota(m_pboxid_to_fboxid.begin(), m_pboxid_to_fboxid.end(), 0);
  }

  // count particles in fluid grid
  Vector<int> pcount_fbox(fba.size(), 0);
  for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
    int fboxid = m_pboxid_to_fboxid[pti.index()];
    pcount_fbox[fboxid] += pti.numParticles();
  }
  ParallelDescriptor::ReduceIntSum(pcount_fbox.dataPtr(), pcount_fbox.size());

  // count particles by rank
  Vector<int> pcount_rank(ParallelDescriptor::NProcs(), 0);
  for (auto i=0; i<pcount_fbox.size(); ++i) {
    pcount_rank[fpmap[i]] += pcount_fbox[i];
  }

  // count total # particles
  if (m_total_numparticle <= 0) {
    m_total_numparticle = 0;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti)
      m_total_numparticle += static_cast<long>(pti.numRealParticles());
    ParallelDescriptor::ReduceLongSum(m_total_numparticle);
  }

  // find the indices of the overload and underload fluid boxes
  Real avg_np     = static_cast<Real>(m_total_numparticle)
                  / ParallelDescriptor::NProcs();
  int  o_toler_np = static_cast<int>(avg_np * overload_toler);
  int  u_toler_np = static_cast<int>(avg_np * underload_toler);
  Vector<int> overload_fboxid, underload_ranks;
  BoxList     overload_fbl;
  // find overload fluid boxes
  for (auto i=0; i < pcount_fbox.size(); ++i) {
    if (pcount_fbox[i] > o_toler_np) {
      overload_fboxid.push_back(i);
      overload_fbl.push_back(fbl_vec[i]);
    }
  }
  // find underload ranks
  for (auto i=0; i<pcount_rank.size(); ++i)
    if (pcount_rank[i] < u_toler_np)  underload_ranks.push_back(i);
  // debug
  Print() << "avg np: "              << avg_np
          << " overload tolerance "  << overload_toler
          << " underload tolerance " << underload_toler << "\n";

  // re-initialize the map from particle box to fluid box
  m_pboxid_to_fboxid.resize(fbl.size());
  std::iota(m_pboxid_to_fboxid.begin(), m_pboxid_to_fboxid.end(), 0);

  // construct queues for the greedy algorithm
  BidNpHeap o_q;
  for (int i=0; i<overload_fboxid.size(); ++i)
    o_q.push(std::make_pair(i, pcount_fbox[overload_fboxid[i]]));
  BidNpHeap u_q(PairCompare(true));
  for (int rank: underload_ranks)
    u_q.push(std::make_pair(rank, pcount_rank[rank]));

  Vector<int> new_ppmap(fpmap);

  // 1D greedy algorithm
  if (!greedy_3d) {
    amrex::Print()<<"starting 1D greedy algorithm..."<<std::flush;
    // count the particles in bins of the overload boxes
    //currently, each bin is a thin layer of the box
    Vector<int> pcount_bin;           // particle count of bins of overload fbox
    Vector<int> poffset_bin;          // offset of bins of overload fbox
    IntVect     binsize(1024);
    binsize.setVal(greedy_dir, 1);    // thickness of the layer
                                      // layer covers the other 2 direction
    countParticle(lev, overload_fbl, binsize, pcount_bin, poffset_bin);
    ParallelDescriptor::ReduceIntSum(pcount_bin.dataPtr(), pcount_bin.size());

    Vector<int> left_nbin;            // bins left for an overload box
    for (int i=0; i<poffset_bin.size()-1; ++i)
      left_nbin.push_back(poffset_bin[i+1] - poffset_bin[i]);

    // use greedy algorithm to setup the mapping between overload boxes
    // and their cutoff chunks
    while (o_q.size() > 0 && u_q.size() > 0) {
      const BidNp& o_pair = o_q.top();
      const BidNp& u_pair = u_q.top();
      if (o_pair.second < o_toler_np || u_pair.second > avg_np)  break;

      int   room    = amrex::max(static_cast<int>(o_toler_np - u_pair.second), 0);
      int   o_boxid = overload_fboxid[o_pair.first];
      //int   u_boxid = underload_fboxid[u_pair.first];

      // find # bins to chop off
      int   chop_np = 0, chop_nbin = 0;
      for (int ibin = poffset_bin[o_pair.first] + left_nbin[o_pair.first] - 1;
          ibin >= poffset_bin[o_pair.first]; --ibin) {
        int chop_np_tmp = chop_np + pcount_bin[ibin];
        if (  (chop_np_tmp < room && o_pair.second - chop_np_tmp > u_toler_np)
            || chop_nbin < greedy_min_grid_size) {
          chop_np = chop_np_tmp;
          ++chop_nbin;
        }
        else {
          break;
        }
      }// end for ibin

      if (chop_np > room || (o_pair.second - chop_np) < u_toler_np)
        if (chop_np > room)
          break;

      // chop the most overload box
      int chop_pos = fbl_vec[o_boxid].length(greedy_dir)
                   - chop_nbin * binsize[greedy_dir]
                   + fbl_vec[o_boxid].smallEnd(greedy_dir);
      fbl_vec.push_back(fbl_vec[o_boxid].chop(greedy_dir, chop_pos));

      // update mapping for the new particle grid
      m_pboxid_to_fboxid.push_back(o_boxid);
      new_ppmap.push_back(u_pair.first);
      left_nbin[o_pair.first] -= chop_nbin;

      // update np and heap
      BidNp new_o_pair(o_pair.first, o_pair.second - chop_np);
      o_q.pop();
      if (new_o_pair.second > o_toler_np)  o_q.push(new_o_pair);

      BidNp new_u_pair(u_pair.first, u_pair.second + chop_np);
      u_q.pop();
      if (new_u_pair.second < u_toler_np)  u_q.push(new_u_pair); 
    }// end while
    amrex::Print()<<"done."<<std::endl;
  }
  // 3D greedy algorithms
  else {
    
    amrex::Print()<<"starting 3D greedy algorithm..."<<std::flush;
    // particle counts based on particle grids
    iMultiFab np_mf_p(ParticleBoxArray(lev), ParticleDistributionMap(lev), 1, 0);
    np_mf_p.setVal(0);
    // particle counts based on fluid grids
    iMultiFab np_mf_f(The_Pinned_Arena());
    np_mf_f.define(fba, fdmap, 1, 0);
    // count the particles and copy across grids
    countParticle(lev, np_mf_p);                
    np_mf_f.ParallelCopy(np_mf_p);

    std::unordered_map<int, Vector<int>> np_cutoff;
    std::unordered_map<int, Vector<int>> id_cutoff;

    // greedy pass to see which box send how many particles to which rank
    while (o_q.size() > 0 && u_q.size() > 0) {
      const BidNp& o_pair = o_q.top();
      const BidNp& u_pair = u_q.top();
      if (o_pair.second < o_toler_np || u_pair.second > avg_np)  break;

      int o_boxid = overload_fboxid[o_pair.first];
      int chop_np = static_cast<int>(min(o_pair.second - avg_np, avg_np - u_pair.second));

      // Append a placeholder for the cutoff, shape will be set later
      fbl_vec.push_back(Box(IntVect{std::numeric_limits<int>::min()},
                            IntVect{std::numeric_limits<int>::max()}));
      new_ppmap.push_back(u_pair.first);
      m_pboxid_to_fboxid.push_back(o_boxid);


      // the rank that owns the overload box stores the info
      if (fpmap[o_boxid] == ParallelDescriptor::MyProc()) {
        id_cutoff[o_boxid].push_back(fbl_vec.size()-1);
        np_cutoff[o_boxid].push_back(chop_np);
      }

      BidNp new_o_pair(o_pair.first, o_pair.second - chop_np);
      o_q.pop();
      if (new_o_pair.second > o_toler_np)  o_q.push(new_o_pair);

      BidNp new_u_pair(u_pair.first, u_pair.second + chop_np);
      u_q.pop();
      if (new_u_pair.second < u_toler_np)  u_q.push(new_u_pair);
    }// end while

    // setup the shape of cutoffs
    for (auto& kv: id_cutoff) {
      // get # particles in current overload fluid box
      const IArrayBox&         np_fab = np_mf_f[kv.first];
      Array4<int const> const& np_arr = np_fab.const_array();
      const int*               np_ptr = np_arr.dataPtr();

      Box&    fbox0  = fbl_vec0[kv.first];
      Box     remain = fbox0;
      IntVect lo     = fbox0.smallEnd();
      IntVect stride   {1, fbox0.length(0), fbox0.length(0) * fbox0.length(1)};

      // chop the cutoff chunk to fit in the corresponding underload ranks
      // the chop-off workload has be computed by the greedy pass before
      for (int icutoff=0; icutoff<kv.second.size(); ++icutoff) {
        //AllPrintToFile("debug") << "current remain " << remain << "\n";
        int np_target    = np_cutoff[kv.first][icutoff];
        int np_target_lo = static_cast<int>(np_target * underload_toler);
        int np_target_hi = static_cast<int>(np_target * overload_toler);
        int min_chop_dir = -1;

        Vector<int> np_cutoff_bydir (AMREX_SPACEDIM);
        Vector<int> np_surface_bydir(AMREX_SPACEDIM);
        Vector<int> chop_pos_bydir  (AMREX_SPACEDIM);

        for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
          if (remain.length(chop_dir) < 2*greedy_min_grid_size) {
            np_surface_bydir[chop_dir] = std::numeric_limits<int>::max();
            np_cutoff_bydir [chop_dir] = std::numeric_limits<int>::max();
            chop_pos_bydir  [chop_dir] = -1;
            continue;
          }

          int dir0 = (chop_dir + 1) % AMREX_SPACEDIM;
          int dir1 = (chop_dir + 2) % AMREX_SPACEDIM;

          int chop_lo  = remain.smallEnd(chop_dir) + greedy_min_grid_size;
          int chop_hi  = remain.bigEnd(chop_dir);
          int chop_pos = chop_hi;

          int np_tmp       = 0;
          int np_diff_prev = np_target + 1;

          // find the chop position with the closest np to np_target
          while (chop_pos >= chop_lo) {
            int np_slice = 0;
            for (int idir0=remain.smallEnd(dir0); idir0<=remain.bigEnd(dir0); ++idir0) {
              for (int idir1=remain.smallEnd(dir1); idir1<=remain.bigEnd(dir1); ++idir1) {
              // the index is based on the offset to the lower bound of the original box
                int iCell = stride[dir0]     * (idir0    - lo[dir0])
                          + stride[dir1]     * (idir1    - lo[dir1])
                          + stride[chop_dir] * (chop_pos - lo[chop_dir]);
                np_slice += *(np_ptr + iCell);
              }
            }

            int np_diff = std::abs(np_target - np_tmp - np_slice);
            if (np_diff > np_diff_prev && chop_hi - chop_pos + 1 >= greedy_min_grid_size) {
              ++chop_pos;
              break;
            }
            else {
              np_diff_prev = np_diff;
              np_tmp      += np_slice;
              --chop_pos;
            }
          }// end while
          chop_pos = chop_pos == chop_lo-1 ? chop_lo : chop_pos;

          np_cutoff_bydir[chop_dir] = np_tmp;
          chop_pos_bydir [chop_dir] = chop_pos;

          // count surface particles
          Box remain_copy(remain);
          Box cutoff     (remain_copy.chop(chop_dir, chop_pos));
          int np_surface = 0;
          // z faces
          for (int j=cutoff.smallEnd(1); j<=cutoff.bigEnd(1); ++j) {
            for (int i=cutoff.smallEnd(0); i<=cutoff.bigEnd(0); ++i) {
              np_surface += np_arr(i, j, cutoff.smallEnd(2), 0);
              np_surface += np_arr(i, j, cutoff.bigEnd  (2), 0);
            }
          }
          // y faces
          for (int k=cutoff.smallEnd(2); k<=cutoff.bigEnd(2); ++k) {
            for (int i=cutoff.smallEnd(0); i<=cutoff.bigEnd(0); ++i) {
              np_surface += np_arr(i, cutoff.smallEnd(1), k, 0);
              np_surface += np_arr(i, cutoff.bigEnd  (1), k, 0);
            }
          }
          // x faces
          for (int k=cutoff.smallEnd(2); k<=cutoff.bigEnd(2); ++k) {
            for (int j=cutoff.smallEnd(1); j<=cutoff.bigEnd(1); ++j) {
              np_surface += np_arr(cutoff.smallEnd(0), j, k, 0);
              np_surface += np_arr(cutoff.bigEnd  (0), j, k, 0);
            }
          }
          np_surface_bydir[chop_dir] = np_surface;
        }// end for chop_dir

        Vector<int> within_toler_dirs;
        for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
          if (  np_cutoff_bydir[chop_dir] >= np_target_lo
             && np_cutoff_bydir[chop_dir] <= np_target_hi) {
            within_toler_dirs.push_back(chop_dir);
          }
        }

        // if none of the dir fits, choose the one closest to fit
        if (within_toler_dirs.empty()) {
          int min_diff = std::numeric_limits<int>::max();
          for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
            int diff = std::abs(np_cutoff_bydir[chop_dir] - np_target);
            if (diff < min_diff) {
              min_diff     = diff;
              min_chop_dir = chop_dir;
            }
          }
        }
        // choose the one with least surface particles
        else {
          int min_np_surface = std::numeric_limits<int>::max();
          for (int dir: within_toler_dirs) {
            if (np_surface_bydir[dir] < min_np_surface) {
              min_np_surface = np_surface_bydir[dir];
              min_chop_dir   = dir;
            }
          }
        }

        // chop the cutoff chunk
        fbl_vec[id_cutoff[kv.first][icutoff]] = \
          remain.chop(min_chop_dir, chop_pos_bydir[min_chop_dir]);
      }// end for icutoff

      // the last underload rank gets whatever left
      fbl_vec[kv.first] = remain;
    }// end for kv

    // Only the overloaded ranks have set up some of the cutoff boxes
    // use an allreduce to inform all the ranks
    size_t nbox = fbl_vec.size();
    Vector<int> ubound_buf(3*nbox);
    Vector<int> lbound_buf(3*nbox);
    // pack the buffer
    for (size_t i=0; i<nbox; ++i) {
      ubound_buf[3*i]   = fbl_vec[i].bigEnd(0);
      ubound_buf[3*i+1] = fbl_vec[i].bigEnd(1);
      ubound_buf[3*i+2] = fbl_vec[i].bigEnd(2);
      lbound_buf[3*i]   = fbl_vec[i].smallEnd(0);
      lbound_buf[3*i+1] = fbl_vec[i].smallEnd(1);
      lbound_buf[3*i+2] = fbl_vec[i].smallEnd(2);
    }

    ParallelDescriptor::ReduceIntMin(ubound_buf.dataPtr(), 3*nbox);
    ParallelDescriptor::ReduceIntMax(lbound_buf.dataPtr(), 3*nbox);

    for (size_t i=0; i<3*nbox; ++i) {
      if (  ubound_buf[i] == std::numeric_limits<int>::max()
         || lbound_buf[i] == std::numeric_limits<int>::min()) {
        Print() << "Greedy 3D cannot balance the current workload, keep "
                << "current particle box array and distribution map.\n";
        return;
      }
    }

    // unpack the buffer
    for (size_t i=0; i<nbox; ++i) {
      fbl_vec[i] =
        Box(IntVect{lbound_buf[3*i], lbound_buf[3*i+1], lbound_buf[3*i+2]},
            IntVect{ubound_buf[3*i], ubound_buf[3*i+1], ubound_buf[3*i+2]});
    }
   amrex::Print()<<"done."<<std::endl;
  }// end else

  // ba and dmap to particle container
  SetParticleBoxArray(lev, BoxArray(fbl));
  SetParticleDistributionMap(lev, DistributionMapping(new_ppmap)); 
                                                                                      
   return;
}
//ACJ



void
DarkMatterParticleContainer::moveKickDrift (amrex::MultiFab&       acceleration,
                                            int                    lev,
                                            amrex::Real            dt,
                                            amrex::Real            a_old,
                                            amrex::Real            a_half,
                                            int                    where_width)
{
    BL_PROFILE("DarkMatterParticleContainer::moveKickDrift()");

    //If there are no particles at this level
    if (lev >= this->GetParticles().size())
        return;
    const auto dxi = Geom(lev).InvCellSizeArray();

    amrex::MultiFab* ac_ptr;
    if (this->OnSameGrids(lev, acceleration))
    {
        ac_ptr = &acceleration;
    }
    else
    {
        const IntVect& ng = acceleration.nGrowVect();
        ac_ptr = new amrex::MultiFab(this->ParticleBoxArray(lev),
                                     this->ParticleDistributionMap(lev),
                                     acceleration.nComp(),acceleration.nGrow());
        ac_ptr->setVal(0.);
        if(acceleration.boxArray() == ac_ptr->boxArray())//this->finestLevel() == 0)
        {
            ac_ptr->Redistribute(acceleration,0,0,acceleration.nComp(),ng);
            ac_ptr->FillBoundary();
        }
        else
        {
            //ACJ :don't copy ghosts in ParallelCopy (they will overwrite valid cells.)
            //Instead copy cells and then fill ghosts afterwards.
            //ac_ptr->ParallelCopy(acceleration,0,0,acceleration.nComp(), ng,ng);
            ac_ptr->ParallelCopy(acceleration,0,0,acceleration.nComp(),0,0);
            ac_ptr->FillBoundary(); 
        }
    }

    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();

    int do_move = 1;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {

        AoS& particles = pti.GetArrayOfStructs();
        ParticleType* pstruct = particles().data();
        const long np = pti.numParticles();
        int grid    = pti.index();

        const FArrayBox& accel_fab= ((*ac_ptr)[grid]);
        Array4<amrex::Real const> accel= accel_fab.array();

        int nc=AMREX_SPACEDIM;
        amrex::ParallelFor(np,
                           [=] AMREX_GPU_HOST_DEVICE ( long i)
                           {
                             update_dm_particle_single(pstruct[i],nc,
                                                       accel,
                                                       plo,dxi,dt,a_old, a_half,do_move);
                           });
    }

    if (ac_ptr != &acceleration) delete ac_ptr;
    
    ParticleLevel&    pmap          = this->GetParticles(lev);
    if (lev > 0 && sub_cycle)
    {
        amrex::ParticleLocData pld; 
        for (auto& kv : pmap) {
            AoS&  pbox       = kv.second.GetArrayOfStructs();
            const int   n    = pbox.size();

#ifdef _OPENMP
#pragma omp parallel for private(pld) if (Gpu::notInLaunchRegion())
#endif
            for (int i = 0; i < n; i++)
            {
                ParticleType& p = pbox[i];
                if (p.id() <= 0) continue;

                // Move the particle to the proper ghost cell. 
                //      and remove any *ghost* particles that have gone too far
                // Note that this should only negate ghost particles, not real particles.
                if (!this->Where(p, pld, lev, lev, where_width))
                {
                    // Assert that the particle being removed is a ghost particle;
                    // the ghost particle is no longer in relevant ghost cells for this grid.
                    if (p.id() == amrex::GhostParticleID)
                    {
                        p.id() = -1;
                    }
                    else
                    {       
                        int grid = kv.first.first;
                        
                        
                        std::cout << "Oops -- removing particle " << p << " " << this->Index(p, lev) << " " << lev << " " << (this->m_gdb->ParticleBoxArray(lev))[grid] << " " << where_width << std::endl;
                        amrex::Error("Trying to get rid of a non-ghost particle in moveKickDrift");
                    }
                }
            }
        }
    }
}

void
DarkMatterParticleContainer::moveKick (MultiFab&       acceleration,
                                       int             lev,
                                       Real            dt,
                                       Real            a_new,
                                       Real            a_half) 
{
    BL_PROFILE("DarkMatterParticleContainer::moveKick()");

    const auto dxi              = Geom(lev).InvCellSizeArray();

    MultiFab* ac_ptr;
    if (OnSameGrids(lev,acceleration))
    {
        ac_ptr = &acceleration;
    }
    else 
    {
        const IntVect& ng = acceleration.nGrowVect();
        ac_ptr = new amrex::MultiFab(this->ParticleBoxArray(lev),
                                     this->ParticleDistributionMap(lev),
                                     acceleration.nComp(),acceleration.nGrow());
        ac_ptr->setVal(0.);
        if(acceleration.boxArray() == ac_ptr->boxArray())//this->finestLevel() == 0)
        {
            ac_ptr->Redistribute(acceleration,0,0,acceleration.nComp(),ng);
            ac_ptr->FillBoundary();
        }
        else
        {   
            //ACJ: don't copy ghosts using ParallelCopy
            //instead just copy all valid cells and then fill ghosts 
            //ac_ptr->ParallelCopy(acceleration,0,0,acceleration.nComp(),ng,ng);
            ac_ptr->ParallelCopy(acceleration,0,0,acceleration.nComp(),0,0);
            ac_ptr->FillBoundary(); 
        }
    }

    const GpuArray<Real,AMREX_SPACEDIM> plo = Geom(lev).ProbLoArray();

    int do_move = 0;

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {

        AoS& particles = pti.GetArrayOfStructs();
        ParticleType* pstruct = particles().data();
        const long np = pti.numParticles();
        int grid    = pti.index();

        const FArrayBox& accel_fab= ((*ac_ptr)[grid]);
        Array4<amrex::Real const> accel= accel_fab.array();

        int nc=AMREX_SPACEDIM;
        amrex::ParallelFor(np,
                           [=] AMREX_GPU_HOST_DEVICE ( long i)
                           {
                             update_dm_particle_single(pstruct[i],nc,
                                                       accel,
                                                       plo,dxi,dt,a_half,a_new,do_move);
                           });
    }

    
    if (ac_ptr != &acceleration) delete ac_ptr;
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE
void update_dm_particle_single (amrex::ParticleContainer<4, 0>::SuperParticleType&  p,
                                const int nc,
                                amrex::Array4<amrex::Real const> const& acc,
                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                                amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                                const amrex::Real& dt, const amrex::Real& a_prev,
                                const amrex::Real& a_cur, const int& do_move)
{
    amrex::Real half_dt       = 0.5 * dt;
    amrex::Real a_cur_inv    = 1.0 / a_cur;
    amrex::Real dt_a_cur_inv = dt * a_cur_inv;

    amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] + 0.5;
    amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] + 0.5;
    amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] + 0.5;
    
    int i = static_cast<int>(amrex::Math::floor(lx));
    int j = static_cast<int>(amrex::Math::floor(ly));
    int k = static_cast<int>(amrex::Math::floor(lz));
    
    amrex::Real xint = lx - i;
    amrex::Real yint = ly - j;
    amrex::Real zint = lz - k;
    
    amrex::Real sx[] = {amrex::Real(1.)-xint, xint};
    amrex::Real sy[] = {amrex::Real(1.)-yint, yint};
    amrex::Real sz[] = {amrex::Real(1.)-zint, zint};

    for (int d=0; d < AMREX_SPACEDIM; ++d)
    {
      amrex::Real val = 0.0;
        for (int kk = 0; kk<=1; ++kk)
        {
            for (int jj = 0; jj <= 1; ++jj)
            {
                for (int ii = 0; ii <= 1; ++ii)
                {
                    val += sx[amrex::Math::abs(ii-1)]*
                           sy[amrex::Math::abs(jj-1)]*
                           sz[amrex::Math::abs(kk-1)]*acc(i-ii,j-jj,k-kk,d);
                }
            }
        }


        p.rdata(d+1)=a_prev*p.rdata(d+1)+half_dt * val;
        p.rdata(d+1)*=a_cur_inv;
    }        

       if (do_move == 1) 
         {
           for (int comp=0; comp < nc; ++comp) {
             p.pos(comp) = p.pos(comp) + dt_a_cur_inv * p.rdata(comp+1);
           }
         }

}

void
DarkMatterParticleContainer::InitCosmo1ppcMultiLevel(
                        MultiFab& mf, const Real disp_fac[], const Real vel_fac[], 
                        const Real particleMass, int disp_idx, int vel_idx, 
                        BoxArray &baWhereNot, int lev, int nlevs)
{
    BL_PROFILE("DarkMatterParticleContainer::InitCosmo1ppcMultiLevel()");
    const int       MyProc   = ParallelDescriptor::MyProc();
    const Geometry& geom     = m_gdb->Geom(lev);
    const Real*     dx       = geom.CellSize();

    static Vector<int> calls;

    calls.resize(nlevs);

    calls[lev]++;

    if (calls[lev] > 1) return;

    Vector<ParticleLevel>& particles = this->GetParticles();

    particles.reserve(15);  // So we don't ever have to do any copying on a resize.

    particles.resize(nlevs);

    ParticleType p;
    Real         disp[AMREX_SPACEDIM];
    Real         vel[AMREX_SPACEDIM];
    
    Real        mean_disp[AMREX_SPACEDIM]={D_DECL(0,0,0)};


    //
    // The mf should be initialized according to the ics...
    //
    int outside_counter=0;
    long outcount[3]={0,0,0};
    long outcountminus[3]={0,0,0};
    long totalcount=0;
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        FArrayBox&  myFab  = mf[mfi];
        const Box&  vbx    = mfi.validbox();
        const int  *fab_lo = vbx.loVect();
        const int  *fab_hi = vbx.hiVect();
        ParticleLocData pld;
        for (int kx = fab_lo[2]; kx <= fab_hi[2]; kx++)
        {
            for (int jx = fab_lo[1]; jx <= fab_hi[1]; jx++)
            {
                for (int ix = fab_lo[0]; ix <= fab_hi[0]; ix++)
                {
                    IntVect indices(D_DECL(ix, jx, kx));
                    totalcount++;
                    if (baWhereNot.contains(indices)) 
                    {
                       continue;
                    }

                    for (int n = 0; n < AMREX_SPACEDIM; n++)
                    {
                        disp[n] = myFab(indices,disp_idx+n);
                        //
                        // Start with homogeneous distribution (for 1 p per cell in the center of the cell),
                        //
                        p.pos(n) = geom.ProbLo(n) + 
                            (indices[n]+Real(0.5))*dx[n];
                        if(disp[n]*disp_fac[n]>dx[n]/2.0)
                          outcount[n]++;
                        if(disp[n]*disp_fac[n]<-dx[n]/2.0)
                          outcountminus[n]++;
                        mean_disp[n]+=fabs(disp[n]);
                        //
                        // then add the displacement (input values weighted by domain length).
                        //
                        p.pos(n) += disp[n] * disp_fac[n];

                        //
                        // Set the velocities.
                        //
                        vel[n] = myFab(indices,vel_idx+n);
                        p.rdata(n+1) = vel[n] * vel_fac[n];
                    }
                    //
                    // Set the mass of the particle from the input value.
                    //
                    p.rdata(0)  = particleMass;
                    p.id()      = ParticleType::NextID();
                    p.cpu()     = MyProc;
        
                    if (!this->Where(p, pld))
                    {
                        this->PeriodicShift(p);

                        if (!this->Where(p, pld))
                            amrex::Abort("DarkMatterParticleContainer::InitCosmo1ppcMultiLevel():invalid particle");
                    }

                    BL_ASSERT(pld.m_lev >= 0 && pld.m_lev <= m_gdb->finestLevel());
                    //handle particles that ran out of this level into a finer one. 
                    if (baWhereNot.contains(pld.m_cell))
                    {
                      outside_counter++;
                      ParticleType newp[8];
                      ParticleLocData new_pld;
                      for (int i=0;i<8;i++)
                      {
                          newp[i].rdata(0)   = particleMass/8.0;
                          newp[i].id()       = ParticleType::NextID();
                          newp[i].cpu()      = MyProc;
                          for (int dim=0;dim<AMREX_SPACEDIM;dim++)
                          {
                              newp[i].pos(dim)=p.pos(dim)+(2*((i/(1 << dim)) % 2)-1)*dx[dim]/4.0;
                              newp[i].rdata(dim+1)=p.rdata(dim+1);
                          }
                          
                          if (!this->Where(newp[i], new_pld))
                          {
                              this->PeriodicShift(newp[i]);
                              
                              if (!this->Where(newp[i], new_pld))
                                  amrex::Abort("DarkMatterParticleContainer::InitCosmo1ppcMultiLevel():invalid particle");
                          }
                          particles[new_pld.m_lev][std::make_pair(new_pld.m_grid, 
                                                                  new_pld.m_tile)].push_back(newp[i]);
                      }
                      
                    }
                    
                    //
                    // Add it to the appropriate PBox at the appropriate level.
                    //
                    else
                        particles[pld.m_lev][std::make_pair(pld.m_grid, pld.m_tile)].push_back(p);
                }
            }
        }
    }
    Redistribute();
}

/*
  Particle deposition
*/

void
DarkMatterParticleContainer::AssignDensityAndVels (Vector<std::unique_ptr<MultiFab> >& mf, int lev_min) const
{
     AssignDensity(mf, lev_min, AMREX_SPACEDIM+1);
}

void 
DarkMatterParticleContainer::InitFromBinaryMortonFile(const std::string& particle_directory,
                                                      int /*nextra*/, int skip_factor) {
  BL_PROFILE("DarkMatterParticleContainer::InitFromBinaryMortonFile");
  
  ParticleMortonFileHeader hdr;
  ReadHeader(particle_directory, "Header", hdr);    
  
  uint64_t num_parts = hdr.NP;
  int DM             = hdr.DM;
  int NX             = hdr.NX;
  int float_size     = hdr.SZ;
  int num_files      = hdr.NF;
  size_t psize       = (DM + NX) * float_size;
  
  std::string particle_file_base = particle_directory + "/particles.";
  std::vector<std::string> file_names;
  for (int i = 0; i < num_files; ++i)
    file_names.push_back(get_file_name(particle_file_base, i));
  
  const int lev = 0;
  const BoxArray& ba = ParticleBoxArray(lev);
  int num_boxes = ba.size();
  uint64_t num_parts_per_box  = num_parts / num_boxes;
  uint64_t num_parts_per_file = num_parts / num_files;
  uint64_t num_bytes_per_file = num_parts_per_file * psize;
  
  std::vector<BoxMortonKey> box_morton_keys(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    const Box& box = ba[i];
    unsigned int x = box.smallEnd(0);
    unsigned int y = box.smallEnd(1);
    unsigned int z = box.smallEnd(2);
    box_morton_keys[i].morton_id = get_morton_index(x, y, z);
    box_morton_keys[i].box_id = i;
  }
  
  std::sort(box_morton_keys.begin(), box_morton_keys.end(), by_morton_id());
  
  std::vector<int> file_indices(num_boxes);
  for (int i = 0; i < num_boxes; ++i)
    file_indices[box_morton_keys[i].box_id] = i;
  
  ParticleType p;
  for (MFIter mfi = MakeMFIter(lev, false); mfi.isValid(); ++mfi) {  // no tiling
    const int grid = mfi.index();
    const int tile = mfi.LocalTileIndex();      
    auto& particles = GetParticles(lev);
    
    uint64_t start    = file_indices[grid]*num_parts_per_box;
    uint64_t stop     = start + num_parts_per_box;

    int file_num      = start / num_parts_per_file;
    uint64_t seek_pos = (start * psize ) % num_bytes_per_file;
    std::string file_name = file_names[file_num];
    
    std::ifstream ifs;
    ifs.open(file_name.c_str(), std::ios::in|std::ios::binary);
    if (!ifs ) {
      amrex::Print() << "Failed to open file " << file_name << " for reading. \n";
      amrex::Abort();
    } 

    ifs.seekg(seek_pos, std::ios::beg);
    
    for (uint64_t i = start; i < stop; ++i) {
      int next_file = i / num_parts_per_file;
      if (next_file != file_num) {
        file_num = next_file;
        file_name = file_names[file_num];
        ifs.close();
        ifs.open(file_name.c_str(), std::ios::in|std::ios::binary);
        if (!ifs ) {
          amrex::Print() << "Failed to open file " << file_name << " for reading. \n";
          amrex::Abort();
        }
      }

      Vector<float> fpos(DM);
      Vector<float> fextra(NX);
      ifs.read((char*)&fpos[0],   DM*sizeof(float));
      ifs.read((char*)&fextra[0], NX*sizeof(float));
      
      if ( (i - start) % skip_factor == 0 ) {
        AMREX_D_TERM(p.pos(0) = fpos[0];,
                     p.pos(1) = fpos[1];,
                     p.pos(2) = fpos[2];);
        
        for (int comp = 0; comp < NX; comp++)
          p.rdata(AMREX_SPACEDIM+comp) = fextra[comp];
        
        p.rdata(AMREX_SPACEDIM) *= skip_factor;
        
        p.id()  = ParticleType::NextID();
        p.cpu() = ParallelDescriptor::MyProc();
        particles[std::make_pair(grid, tile)].push_back(p);
      }
    }    
  }
  
  Redistribute();
}

