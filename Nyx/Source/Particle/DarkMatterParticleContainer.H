
#ifndef _DarkMatterParticleContainer_H_
#define _DarkMatterParticleContainer_H_

#include <NyxParticleContainer.H>
#include <queue> //ACJ

//ACJ
/*
struct SoArealData
{
  enum {
    radius = 0,
    mass,
    velx,
    vely,
    velz,
    omegax,
    omegay,
    omegaz,
    dragcoeff,
    dragx,
    dragy,
    dragz,
    count
  };

  static
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  amrex::Real volume (const amrex::Real radius)
  { return (4.0/3.0)*M_PI*(radius*radius*radius); }

  static
  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
  amrex::Real oneOverI (const amrex::Real radius,
                        const amrex::Real mass)
  { return 2.5/(mass*(radius*radius)); }
};

struct SoAintData
{
  enum {
    phase = 0,
    state,
//#if MFIX_POLYDISPERSE
//    ptype,
//#endif
    count
  };
};
//ACJ
*/


class DarkMatterParticleContainer
    : public NyxParticleContainer<1+AMREX_SPACEDIM>
{
public:
    DarkMatterParticleContainer (amrex::Amr* amr)
        : NyxParticleContainer<1+AMREX_SPACEDIM>(amr)
    {
      real_comp_names.clear();
      real_comp_names.push_back("mass");
      real_comp_names.push_back("xvel");
      real_comp_names.push_back("yvel");
      real_comp_names.push_back("zvel");
    }

    //ACJ 
    DarkMatterParticleContainer(const amrex::Geometry& geom, const amrex::DistributionMapping& dmap, const amrex::BoxArray& ba) 
    : NyxParticleContainer<1+AMREX_SPACEDIM>(geom, dmap, ba)
    {
      real_comp_names.clear();
      real_comp_names.push_back("mass");
      real_comp_names.push_back("xvel");
      real_comp_names.push_back("yvel");
      real_comp_names.push_back("zvel");
    }
    //ACJ

    using MyParIter = amrex::ParIter<1+AMREX_SPACEDIM>;
    using MyConstParIter = amrex::ParConstIter<1+AMREX_SPACEDIM>;
    
    //using DMParIter = amrex::ParIter<0,0,SoArealData::count,SoAintData::count>; //ACJ
     
    virtual ~DarkMatterParticleContainer () {}

    void InitCosmo1ppcMultiLevel(amrex::MultiFab& mf, const amrex::Real disp_fac[], const amrex::Real vel_fac[],
                                 const amrex::Real particleMass, int disp_idx, int vel_idx,
                                 amrex::BoxArray &baWhereNot, int lev, int nlevs);

    void AssignDensityAndVels (amrex::Vector<std::unique_ptr<amrex::MultiFab> >& mf, int lev_min = 0) const;

    virtual void moveKickDrift (amrex::MultiFab& acceleration, int level, amrex::Real timestep,
                                amrex::Real a_old = 1.0, amrex::Real a_half = 1.0, int where_width = 0);
    virtual void moveKick      (amrex::MultiFab& acceleration, int level, amrex::Real timestep,
                                amrex::Real a_new = 1.0, amrex::Real a_half = 1.0);

    void InitFromBinaryMortonFile(const std::string& particle_directory, int nextra, int skip_factor);

    long m_total_numparticle = 0; //ACJ
    amrex::Vector<int> m_pboxid_to_fboxid; //ACJ
    void partitionParticleGrids(int lev, const amrex::BoxArray& fba, const amrex::DistributionMapping& fdmap, amrex::Real overload_toler, 
                                amrex::Real underload_toler); //ACJ
    void load_balance(int lev, const amrex::BoxArray& fba, const amrex::DistributionMapping& fdmap, amrex::Real overload_toler, amrex::Real underload_toler); 
    int  greedy_dir           = 0; //ACJ
    bool greedy_3d            = false; //ACJ
    int  greedy_min_grid_size = 1; //ACJ
    
    void setGreedyRegrid(void) //int dir, bool partition_3d, int min_size=2) //ACJ
    { 
        //greedy_dir = dir; 
        //greedy_3d = partition_3d; 
        //greedy_min_grid_size = min_size;
        
        amrex::ParmParse pp("nyx");
        pp.query("greedy_dir", greedy_dir); //ACJ
        pp.query("greedy_3d", greedy_3d); //ACJ
        pp.query("greedy_min_grid_size", greedy_min_grid_size); //ACJ
    } //ACJ
   
    //ACJ 
    void setParticleFluidGridMap(const amrex::Vector<int>& boxmap) 
    {
        m_pboxid_to_fboxid.clear();
        m_pboxid_to_fboxid.insert(m_pboxid_to_fboxid.end(), boxmap.begin(), boxmap.end());
    } 
    //ACJ

    void countParticle(int lev,
                       const amrex::BoxList& bl,
                       const amrex::IntVect& binsize,
                       amrex::Vector<int>& pcounts,
                       amrex::Vector<int>& poffsets); //ACJ

    void countParticle(int lev, amrex::iMultiFab& np_mf); //ACJ
    
    amrex::Box chop_and_distribute_box(int o_box_id, int np_target, const amrex::Vector<amrex::Box>& fbl_vec0,
            amrex::Vector<amrex::Box>& fbl_vec, amrex::iMultiFab& np_mf_f, int & new_box_np, int lev);

};

AMREX_GPU_HOST_DEVICE AMREX_INLINE void update_dm_particle_single (amrex::ParticleContainer<4, 0>::SuperParticleType&  p,
                                                                   const int nc,
                                                                   amrex::Array4<amrex::Real const> const& acc,
                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
                                                                   amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
                                                                   const amrex::Real& dt, const amrex::Real& a_prev,
                                                                   const amrex::Real& a_cur, const int& do_move);

#endif /* _DarkMatterParticleContainer_H_ */
