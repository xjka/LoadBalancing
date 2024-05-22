#include <stdint.h>

#include <DarkMatterParticleContainer.H>


void gdb_attach_point(int myrank)
{
    volatile int gdb_i = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout<<"PID "<<getpid()<<" on "<< hostname<<" ready for attach (rank:"<<myrank<<")\n";
    if (myrank==0)
        while (0 == gdb_i)
            sleep(5);
   MPI_Barrier(MPI_COMM_WORLD);
}

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
  
 typedef std::pair<int,int> BidNp;

  class BoxEntry {
      public:
        int fbl_vec0_id, proc_box_id, np;
        BoxEntry(int fid, int bid, int n): fbl_vec0_id(fid), proc_box_id(bid), np(n){} 
  };

  struct PairCompare {
    bool inverse = false;

    PairCompare(const bool a_inverse=false) : inverse(a_inverse) {}

    bool operator() (const BoxEntry& lhs, const BoxEntry& rhs)
    {
        return inverse ? lhs.np > rhs.np : lhs.np < rhs.np;
    }

    bool operator() (const BidNp &lhs, const BidNp &rhs)
    {
        return inverse ? lhs.second > rhs.second : lhs.second < rhs.second;
    }
  };
}
//ACJ


//ACJ
long DarkMatterParticleContainer::countParticle(int lev, iMultiFab& np_mf)
{
  //initialize a multifab based on the boxArray and DistributionMap of np_mf
  MultiFab mf_temp(np_mf.boxArray(), np_mf.DistributionMap(), 1, 0); 
  mf_temp.setVal(0.);
  //put the particle counts in mf_temp and return total number of particles
  long numparticle = IncrementWithTotal(mf_temp, lev, false);    
  //copy the values of mf_temp int np_mf
  //Copy (FabArray<DFAB>& dst, FabArray<SFAB> const& src, int srccomp, int dstcomp, int numcomp, int nghost)
  Copy(np_mf, mf_temp, 0, 0, 1, 0);
  return numparticle;
}
//ACJ

//ACJ
Box DarkMatterParticleContainer::chop_and_distribute_box(Box &remain, const Box &remain0, const IArrayBox& np_fab, int np_target, 
        int & new_box_np, int room, int np_box, int min_grid_size, bool &hi_vs_lo)
{
    //const IArrayBox&         np_fab = np_mf_f[o_fbox_id];
    //Array4<int const> const& np_arr = np_fab.const_array();
    //const int*               np_ptr = np_arr.dataPtr();

    const int *np_ptr = np_fab.const_array().dataPtr();
    IntVect lo  = remain.smallEnd(); 
    IntVect lo0 = remain0.smallEnd();
    //this is used to get strides so that you can continue to iterate into np_mf_f
    IntVect stride {1, remain0.length(0), remain0.length(0)*remain0.length(1)};  
    int min_chop_dir = -1;

    int np_cutoff_bydir = std::numeric_limits<int>::max();
    int chop_pos_bydir = -1;
    int chop_bydir = -1;
    int MyProc = ParallelDescriptor::MyProc();
    
    for (int chop_dir=0; chop_dir<AMREX_SPACEDIM; ++chop_dir) {
        if (remain.length(chop_dir) < 2*min_grid_size) {
            continue;
        }

        int dir0 = (chop_dir + 1) % AMREX_SPACEDIM;
        int dir1 = (chop_dir + 2) % AMREX_SPACEDIM;

        int chop_lo  = remain.smallEnd(chop_dir);
        int chop_hi  = remain.bigEnd(chop_dir);
        int chop_pos = chop_hi;

        int np_tmp       = 0;
        int np_diff_prev[2] = {-np_target, np_box-np_target};
         
        // find the chop position with the closest np to np_target
        int chop_idx[2] = {-1,-1};
        while (chop_pos >= chop_lo and (chop_idx[0]==-1 or chop_idx[1]==-1)) {
            int np_slice = 0; 
            for (int idir0=remain.smallEnd(dir0); idir0<=remain.bigEnd(dir0); ++idir0) {
                for (int idir1=remain.smallEnd(dir1); idir1<=remain.bigEnd(dir1); ++idir1) {
                    // the index is based on the offset to the lower bound of the original box
                    int iCell = stride[dir0]     * (idir0    - lo0[dir0])
                              + stride[dir1]     * (idir1    - lo0[dir1])
                              + stride[chop_dir] * (chop_pos - lo0[chop_dir]);
                    np_slice += *(np_ptr + iCell);
                }
            }
            int np_diff[2] = {np_tmp+np_slice-np_target,  np_box-np_tmp-np_slice-np_target};
            bool valid_chop = chop_hi-chop_pos>=min_grid_size and chop_pos+1-chop_lo>=min_grid_size;
            if(chop_idx[0]==-1 and (np_diff_prev[0]>=0 or np_tmp+np_slice>room) and valid_chop)
            {
                if(np_tmp<=room){ //necessary because valid_chop makes it possible for this not to be true here
                    int chop_diff = np_tmp-np_target;
                    chop_idx[0] = chop_pos+1; 
                    if(chop_pos_bydir!=-1){
                        int other_diff = np_cutoff_bydir - np_target;
                        if((chop_diff>=0) != (other_diff>=0)){
                            if(chop_diff>=0){
                                hi_vs_lo = 0;
                                chop_pos_bydir = chop_idx[0];
                                np_cutoff_bydir = np_tmp;
                                chop_bydir = chop_dir;
                            }
                        }
                        else if(std::abs(chop_diff) < std::abs(other_diff)){
                            hi_vs_lo = 0;
                            chop_pos_bydir = chop_idx[0];
                            np_cutoff_bydir = np_tmp;
                            chop_bydir = chop_dir;                        }
                    }
                    else{
                        hi_vs_lo = 0;
                        np_cutoff_bydir = np_tmp;
                        chop_pos_bydir = chop_idx[0];
                        chop_bydir = chop_dir;
                    }
                }
            }
            if(chop_idx[1]==-1 and np_diff[1]<0 and np_box-np_tmp<=room and valid_chop)
            { 
                int chop_diff = np_box-np_tmp-np_target;
                chop_idx[1] = chop_pos+1;
                if(chop_pos_bydir!=-1){
                    int other_diff = np_cutoff_bydir - np_target;
                    if((chop_diff>=0) != (other_diff>=0)){
                        if(chop_diff>=0){
                            hi_vs_lo = 1;
                            chop_pos_bydir = chop_idx[1];
                            np_cutoff_bydir = np_box-np_tmp;
                            chop_bydir = chop_dir;
                        }
                    }
                    else if(std::abs(chop_diff) < std::abs(other_diff)){
                        hi_vs_lo = 1;
                        chop_pos_bydir = chop_idx[1];
                        np_cutoff_bydir = np_box-np_tmp;
                        chop_bydir = chop_dir;
                    }
                }
                else{
                    hi_vs_lo = 1;
                    np_cutoff_bydir = np_box-np_tmp;
                    chop_pos_bydir = chop_idx[1]; 
                    chop_bydir = chop_dir;
                }
            }
            np_diff_prev[0] = np_diff[0];
            np_diff_prev[1] = np_diff[1];
            np_tmp += np_slice;
            --chop_pos;
        }// end while
    }// end for chop_dir
    
    if(chop_pos_bydir==-1){ //couldn't find a split
        new_box_np = -2;
        Box dummy;
        return dummy;
    } 
    else //split the box and send to appropriate ranks
    {
        Box new_box = remain.chop(chop_bydir, chop_pos_bydir);
        new_box_np  = hi_vs_lo==0 ? np_cutoff_bydir : np_box - np_cutoff_bydir;
        return new_box;
    }
}
//ACJ

//ACJ
// This function gets the list of "overloaded" and "underloaded" ranks and returns them (through editing inputs.)
void get_over_under_load_ranks(Vector<int>& underload_ranks, Vector<int>& overload_ranks, const Vector<int>& pcount_rank, 
        Vector<BidNp>& o_q, Vector<BidNp>& u_q, const int& o_toler_np, const int& u_toler_np)
{
    //do underload ranks
    //If there are underload ranks this is not the first function call during
    //this execution of the algorithm in which case we do not want to reconsider
    //all ranks, as this could cause the algorithm to not terminate due to flip-flopping. 
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
//This function also does necessary updating to: o_rank, o_rank_np, u_rank and u_rank_np.
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

int countBox(Box &box, Box &remain0, const IArrayBox& np_fab){
    //The below is the stride pattern into the original multifab of box counts that
    //essentially saves us from having to recount boxes throughout execution
    IntVect stride {1, remain0.length(0), remain0.length(0)*remain0.length(1)};
    //The below is the pointer into the original multifab of counts
    const int* np_ptr = np_fab.const_array().dataPtr();
    IntVect hi = box.bigEnd();
    IntVect lo = box.smallEnd();
    IntVect lo0 = remain0.smallEnd();
    int np_count=0;
    for(int i0=lo[0]; i0<=hi[0]; ++i0){
        for(int i1=lo[1]; i1<=hi[1]; ++i1){
            for(int i2=lo[2]; i2<=hi[2]; ++i2){
                int iCell = stride[0]*(i0-lo0[0]) + 
                            stride[1]*(i1-lo0[1]) + 
                            stride[2]*(i2-lo0[2]);
                np_count += *(np_ptr+iCell);
            }
        }
    }
    return np_count;
}

void sorted_insert(std::list<BoxEntry>& o_box_list, BoxEntry& be)
{
    //this will insert into a list in descending order
    auto it = std::upper_bound(o_box_list.begin(), o_box_list.end(), be, PairCompare(true));
    o_box_list.insert(it, be);
}

//ACJ 
void 
DarkMatterParticleContainer::load_balance(int lev, const amrex::BoxArray& fba, const amrex::DistributionMapping& fdmap,
        amrex::Real overload_toler, int min_grid_size, amrex::BoxArray &new_ba, 
        amrex::DistributionMapping &new_dm)
{

    //gdb_attach_point(ParallelDescriptor::MyProc());

    BL_PROFILE("DarkMatterParticleContainer::load_balance()"); 
    // parent grid info
    //const amrex::BoxArray&            fba      = ParticleBoxArray(lev);
    //const amrex::DistributionMapping& fdmap    = ParticleDistributionMap(lev);
    const Vector<int>&                fpmap    = fdmap.ProcessorMap();
    BoxList                           fbl      = fba.boxList();
    Vector<Box>&                      fbl_vec  = fbl.data();
    Vector<Box>                       fbl_vec0 = fbl_vec;
    int MyProc = ParallelDescriptor::MyProc(), NProcs = ParallelDescriptor::NProcs();
    int FBL_VEC_SIZE0 = fbl_vec.size();
    
    // If the mapping between particle and parent grids hasn't been set,
    // set an 1to1 mapping.
    if (m_pboxid_to_fboxid.size()==0) {
        m_pboxid_to_fboxid.resize(fbl.size());
        std::iota(m_pboxid_to_fboxid.begin(), m_pboxid_to_fboxid.end(), 0);
    }     

    // count particles in grid
    Vector<int> pcount_fbox(fba.size(),0);//this is fluid grid in this case
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
        int fboxid = m_pboxid_to_fboxid[pti.index()];
        pcount_fbox[fboxid] += pti.numParticles();
    }
    ParallelDescriptor::ReduceIntSum(pcount_fbox.dataPtr(), pcount_fbox.size());    

    // re-initialize the map from particle box to fluid box
    m_pboxid_to_fboxid.resize(fbl.size());
    std::iota(m_pboxid_to_fboxid.begin(), m_pboxid_to_fboxid.end(), 0);    
    
    //Vector<long> pcount_fbox = this->NumberOfParticlesInGrid(lev, true, false); //Only can use if refining DMPC grid
    Vector<Box> proc_box_list  = fbl_vec;

    // count particles by rank
    Vector<int> pcount_rank(NProcs, 0), pcount_rank_diff(NProcs, 0);
    std::unordered_map<int,  std::list<BoxEntry>> rank_box_map;
    for (auto i=0; i<pcount_fbox.size(); ++i) {
        pcount_rank[fpmap[i]] += pcount_fbox[i];
        rank_box_map[fpmap[i]].push_back(BoxEntry(i, i, pcount_fbox[i]));
    }

    // particle counts iMultiFab
    iMultiFab np_mf_f(The_Pinned_Arena());
    np_mf_f.define(fba, fdmap, 1, 0);
    long m_total_numparticle = countParticle(lev, np_mf_f);

    // find the underload and overload tolerances
    if (overload_toler<=1)
        amrex::Abort("DarkMatterParticleContainer::load_balance(): overload_toler must be greater than 1");
    Real avg_np = static_cast<Real>(m_total_numparticle) / NProcs;
    int  o_toler_np = static_cast<int>(avg_np * overload_toler);
    int  u_toler_np = static_cast<int>(avg_np); //make any rank below average a potential underloaded rank

    // debug
    Print() <<"total num particles: "<< m_total_numparticle << "\n"
           << "avg np:              "<< avg_np << "\n"
           << "overload tolerance:  "<< overload_toler << "\n";

    //now actually start the redivisioning algorithm
    Print()<<"starting 3D load_balance algorithm (N_box="<<FBL_VEC_SIZE0<<")--->"<<std::flush;

    //new box->processor map
    Vector<int> m_pboxid_to_fboxid_chngs(m_pboxid_to_fboxid);
    Vector<int> ppmap_chngs(fpmap.size(), -1);     
    int u_rank, o_rank, u_rank_np, o_rank_np;
    Vector<int> underload_ranks, overload_ranks; 
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
            std::list<BoxEntry>& o_box_list = rank_box_map[o_rank];
            o_box_list.sort(PairCompare(true));  //sort in descending order 
             
            //TODO: could add more intelligence to matching over and underload ranks
            //get the number of particles to remove, and the space available
            int num_2_rmv = o_rank_np - o_toler_np;
            int room = avg_np - u_rank_np;
            while(num_2_rmv>0  and room>0)
            { 
                //get the number of particles we will (can) remove
                int num_do_rmv = std::min(room, num_2_rmv);
                
                ///// Reduce load as much as possible by simply sending as many boxes as possible to underload rank /////
                auto it = o_box_list.begin();
                while(num_do_rmv>0 and it!=o_box_list.end())
                {
                    int box_2_mv_id = it->proc_box_id; //needs to be index into the dynamic box list
                    int box_2_mv_np = it->np;
                    if(box_2_mv_np<=room)
                    {
                        ppmap_chngs[box_2_mv_id] = u_rank; //add 1 so that rank 0 is non-zero, will subtract 1 later
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
                ///// if there are still particles to remove, need to split a box/////
                if(num_do_rmv>0) 
                {
                    int o_fbox_id = o_box_list.front().fbl_vec0_id;
                    int o_box_id = o_box_list.front().proc_box_id;
                    int o_box_np = o_box_list.front().np;
                    int new_box_np;
                    bool hi_vs_lo;
                    Box new_box = chop_and_distribute_box(proc_box_list[o_box_id], 
                            fbl_vec0[o_fbox_id], np_mf_f[o_fbox_id], num_do_rmv, new_box_np, room, o_box_np, min_grid_size, hi_vs_lo); 
                    if(new_box_np>=0)
                    {
                        proc_box_list.push_back(new_box);
                        m_pboxid_to_fboxid_chngs.push_back(o_fbox_id);
                        o_box_list.pop_front();
                        int num_removed, num_remaining;
                        if(hi_vs_lo==0){//o_rank is keeping the lo part of chop (normal behavior)
                            num_removed = new_box_np;
                            num_remaining = o_box_np - new_box_np;
                            ppmap_chngs.push_back(u_rank);
                            BoxEntry tmp(o_fbox_id, o_box_id, num_remaining);
                            sorted_insert(o_box_list, tmp);
                        }
                        else{//need to change the low part of split to be owned by urank
                            num_removed = o_box_np - new_box_np;
                            num_remaining = new_box_np;
                            ppmap_chngs.push_back(o_rank);
                            ppmap_chngs[o_box_id] = u_rank;
                            BoxEntry tmp(o_fbox_id, proc_box_list.size()-1, num_remaining);
                            sorted_insert(o_box_list, tmp);
                        }
                        num_2_rmv -= num_removed;
                        num_do_rmv -= num_removed;
                        u_rank_np += num_removed;
                        pcount_rank_diff[o_rank] -= num_removed;
                        pcount_rank_diff[u_rank] += num_removed;
                    }
                    else  
                    //TODO: consider this is unnecessary if limiting factor is small "room"s not highly dense cells
                    {
                        //Split the box into 8 pieces (if we can) only stopping when we reach the grid resolution size. 
                        Vector<Box>new_boxes {proc_box_list[o_box_id]};
                        Vector<int> new_box_counts;
                        IntVect hi = proc_box_list[o_box_id].bigEnd();
                        IntVect lo = proc_box_list[o_box_id].smallEnd();
                        for (int dir=0; dir<AMREX_SPACEDIM; dir++)
                        {
                            Vector<Box> tmp_new_boxes;
                            for(Box &b : new_boxes){
                                if (hi[dir]-lo[dir] >= 2*min_grid_size)
                                    tmp_new_boxes.push_back(b.chop(dir, lo[dir]+std::ceil((hi[dir]-lo[dir])/2.0)));
                            }
                            new_boxes.insert(new_boxes.end(), tmp_new_boxes.begin(), tmp_new_boxes.end());
                        }
                        if(new_boxes.size()>1){
                            //TODO: can do half the iterating by using total counts (which you know initially [o_box_np])
                            for(Box &b : new_boxes){
                                new_box_counts.push_back(countBox(b, fbl_vec0[o_fbox_id], np_mf_f[o_fbox_id]));
                                m_pboxid_to_fboxid_chngs.push_back(o_fbox_id);
                            }
                            
                            //add to synchronization buffers and box list
                            int size0 = proc_box_list.size();
                            proc_box_list[o_box_id] = new_boxes[0]; //this is the original box 
                            for(int i=1;i<new_boxes.size();++i){ //remainging boxes are new
                                proc_box_list.push_back(new_boxes[i]);
                                ppmap_chngs.push_back(o_rank);
                            }
                            //insert new boxes onto box list
                            o_box_list.pop_front();
                            BoxEntry tmp(o_fbox_id, o_box_id, new_box_counts[0]); 
                            sorted_insert(o_box_list, tmp);
                            for(int i=1; i<new_boxes.size();++i){
                                BoxEntry tmp(o_fbox_id, size0+i-1, new_box_counts[i]);
                                sorted_insert(o_box_list, tmp);
                            }
                        }
                        else{
                            //Idea here is that if we did not add anything to the underload rank load
                            //previously in this (outermost) loop, then the space available is as good as it's going to get
                            //in the sense that if we recieve a new u_rank after re-assessing it can only have the same
                            //or less space, and the space in this one won't change by running the loop again. So just need to 
                            //drop this box. 
                            if(pcount_rank_diff[u_rank]==0){ 
                                num_2_rmv -= o_box_np; 
                                pcount_rank_diff[o_rank] -= o_box_np;
                                o_box_list.pop_front();
                            }
                            break; //break to re-asses underload ranks in case this one is close to full
                        }
                    } 
                } //end if(num_do_rmv>0)
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
    Vector<int> new_ppmap(fpmap);
    ParallelDescriptor::ReduceIntMax(ppmap_chngs.dataPtr(), FBL_VEC_SIZE0); //only synchronize original entries      
    for(int i=0 ; i<FBL_VEC_SIZE0; ++i){
       if(ppmap_chngs[i]>=0){
           new_ppmap[i] = ppmap_chngs[i];
           fbl_vec[i]   = proc_box_list[i]; 
       }
    }
    //write number of new boxes into one vector so every rank knows how to form global list
    Vector<int> new_bxs_per_rank(NProcs, 0);
    new_bxs_per_rank[MyProc] = proc_box_list.size() - FBL_VEC_SIZE0;
    ParallelDescriptor::ReduceIntMax(new_bxs_per_rank.dataPtr(), new_bxs_per_rank.size());
    
    //resize fbl_vec and new_ppmap to be able to contain additional entries
    int num_new = std::accumulate(new_bxs_per_rank.begin(), new_bxs_per_rank.end(), 0);
    if(num_new>0) //only do this if a rank added a box
    {
        for(int i=0; i<num_new; ++i){
            fbl_vec.push_back(Box(IntVect{std::numeric_limits<int>::min()},
                                IntVect{std::numeric_limits<int>::max()}));
            new_ppmap.push_back(-1);
            m_pboxid_to_fboxid.push_back(-1);
        } 

        //now add new entries (for this rank), in appropriate position.
        //(Will sync with other ranks later)
        int write_pos = FBL_VEC_SIZE0 + std::accumulate(new_bxs_per_rank.begin(), new_bxs_per_rank.begin()+MyProc, 0);
        for(int i=0; i<new_bxs_per_rank[MyProc]; ++i){
            fbl_vec[write_pos+i] = proc_box_list[FBL_VEC_SIZE0+i];
            new_ppmap[write_pos+i] = ppmap_chngs[FBL_VEC_SIZE0+i];
            m_pboxid_to_fboxid[write_pos+i] = m_pboxid_to_fboxid_chngs[FBL_VEC_SIZE0+i];
        }

        //sync the dmaps and pboxid_to_fboxid maps
        ParallelDescriptor::ReduceIntMax(&new_ppmap.back()-num_new+1, num_new);
        ParallelDescriptor::ReduceIntMax(&m_pboxid_to_fboxid.back()-num_new+1, num_new); 
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
   
    amrex::Print()<<"(N_box="<<nbox<<") done."<<std::endl;
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
    const auto dxi              = Geom(lev).InvCellSizeArray();

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
	if (! m_particle_locator.isValid(GetParGDB())) m_particle_locator.build(GetParGDB());
        m_particle_locator.setGeometry(GetParGDB());
        AmrAssignGrid<DenseBinIteratorFactory<Box>> assign_grid = m_particle_locator.getGridAssignor();

        amrex::ParticleLocData pld; 
        for (auto& kv : pmap) {
            AoS&  particles = kv.second.GetArrayOfStructs();
            ParticleType* pstruct = particles().data();
            const long np = particles.size();
            amrex::ParallelFor(np,
                           [=] AMREX_GPU_HOST_DEVICE ( long i)
                           {
                               //                              amrex::ParticleContainer<4, 0>::SuperParticleType&  p=pstruct[i];
                               auto&  p=pstruct[i];
			       if(p.id()>0) {
                               const auto tup = assign_grid(p, lev, lev, where_width);
                               auto p_boxes = amrex::get<0>(tup);
                               auto p_levs  = amrex::get<1>(tup);
                               if(p_boxes<0||p_levs<0) {
                                   //printf("p:       %d\t%d\t%g %g %g %g %g %g %g id\n",p.id(),p.cpu(),p.pos(0),p.pos(1),p.pos(2),p.rdata(0),p.rdata(1),p.rdata(2),p.rdata(3));
                                   //printf("tup:     %d\t%d\n",amrex::get<0>(tup),amrex::get<1>(tup));
                                   if (p.id() == amrex::GhostParticleID)
                                   {
                                       p.id() = -1;
                                   }
                                   else
                                   {
                                       amrex::Error("Trying to get rid of a non-ghost particle in moveKickDrift");
                                   }
                               }
			       }
                           });
            Gpu::streamSynchronize();
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
    
    Real        mean_disp[AMREX_SPACEDIM]={AMREX_D_DECL(0,0,0)};


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
                    IntVect indices(AMREX_D_DECL(ix, jx, kx));
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

