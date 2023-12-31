
#include <iostream>
#include <iomanip>
#include <sstream>

#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
#include <AMReX_Box.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#ifdef BL_USE_MPI
#include <MemInfo.H>
#endif
#include <Nyx.H>

#ifdef REEBER
#ifdef REEBER_HIST
#include <ReeberAnalysis.H> // This actually works both in situ and in-transit.
#endif
#endif

#include <Nyx_output.H>

std::string inputs_name = "";

#ifdef GIMLET
#include <DoGimletAnalysis.H>
#include <postprocess_tau_fields.H>
#include <fftw3-mpi.h>
#include <MakeFFTWBoxes.H>
#endif

#ifdef HENSON
#include <henson/context.h>
#include <henson/data.h>
#endif

using namespace amrex;

const int NyxHaloFinderSignal(42);
const int resizeSignal(43);
const int GimletSignal(55);
const int quitSignal(-44);

amrex::LevelBld* getLevelBld ();

//ACJ
void write_rank_loads(const bool & dual_grid_load_balance, const std::vector<amrex::Vector<long>> & rank_loads, std::vector<int> steps)
{
    //write the rank loads over times  
    std::string rankloadfile;
    if(dual_grid_load_balance)
        rankloadfile = "rank_loads_dual_grid";
    else
        rankloadfile = "rank_loads";
    std::ofstream rl_outfile(rankloadfile.c_str(), std::ios::out|std::ios::binary); 
    long numsteps = rank_loads.size();        
    rl_outfile.write((char*)&numsteps, sizeof(long));
    for(int & step : steps)
        rl_outfile.write((char*)&step, sizeof(int)); 
    for(auto & ranks : rank_loads){ 
        long numranks = ranks.size(); 
        rl_outfile.write((char*)&numranks, sizeof(long));
        for(auto & load : ranks){    
            rl_outfile.write((char*)&load, sizeof(long)); 
        } 
    } 
    rl_outfile.close(); 
}
//ACJ

void
nyx_main (int argc, char* argv[])
{
    // check to see if it contains --describe
    if (argc >= 2) {
        for (auto i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--describe") {
                Nyx::writeBuildInfo();
                return;
            }
        }
    }
    amrex::Initialize(argc, argv);
    {

    // save the inputs file name for later
    if (argc > 1) {
      if (!strchr(argv[1], '=')) {
        inputs_name = argv[1];
      }
    }
    BL_PROFILE_REGION_START("main()");
    BL_PROFILE_VAR("main()", pmain);

    //
    // Don't start timing until all CPUs are ready to go.
    //
    ParallelDescriptor::Barrier("Starting main.");

    BL_COMM_PROFILE_NAMETAG("main TOP");

    Real dRunTime1 = ParallelDescriptor::second();

    std::cout << std::setprecision(10);

    int max_step;
    Real stop_time;
    ParmParse pp;

    max_step  = -1;
    stop_time = -1.0;

    pp.query("max_step",  max_step);
    pp.query("stop_time", stop_time);

    if (max_step < 0 && stop_time < 0.0)
    {
        amrex::Abort("**** Error: either max_step or stop_time has to be positive!");
    }

    // Reeber has to do some initialization.
#ifdef REEBER
#ifdef REEBER_HIST
    reeber_int = initReeberAnalysis();
#endif
#endif

    // We hard-wire the initial time to 0
    Real strt_time =  0.0;

    Amr *amrptr = new Amr(getLevelBld());
    amrptr->init(strt_time,stop_time);

#ifdef BL_USE_MPI
    // ---- initialize nyx memory monitoring
    MemInfo *mInfo = MemInfo::GetInstance();
    mInfo->LogSummary("MemInit  ");
#endif

    const Real time_before_main_loop = ParallelDescriptor::second();

    BL_PROFILE_VAR_NS("nyx_main::LoadBalanceProfiling()", loadBalanceProfiling);
    std::vector<amrex::Vector<long>> rank_loads; //ACJ
    std::vector<int> rl_steps;
    amrex::ParmParse ppn("nyx"); //ACJ
    int loadBalanceInt = 100, checkInt = 100, dualGridProfileFreq=1; //ACJ
    amrex::Real loadBalanceStartZ = 199; //ACJ
    ppn.query("load_balance_int", loadBalanceInt);  //ACJ
    ppn.query("check_int", checkInt); //ACJ
    ppn.query("load_balance_start_z", loadBalanceStartZ); //ACJ
    ppn.query("dual_grid_profile_freq", dualGridProfileFreq); //ACJ
    
    bool finished(false);
    {
        BL_PROFILE_REGION("R::Nyx::coarseTimeStep");
                       
        
        while ( ! finished)
        {
            // If we set the regrid_on_restart flag and if we are *not* going to take
            // a time step then we want to go ahead and regrid here.
            //
            if (amrptr->RegridOnRestart()) {
                if ( (amrptr->levelSteps(0) >= max_step ) ||
                   ( (stop_time >= 0.0) &&
                     (amrptr->cumTime() >= stop_time)  )    )
                {
               // Regrid only!
               amrptr->RegridOnly(amrptr->cumTime());
                }
            }  

            if (amrptr->okToContinue()
              && (amrptr->levelSteps(0) < max_step || max_step < 0)
              && (amrptr->cumTime() < stop_time || stop_time < 0.0))

            {
           amrptr->coarseTimeStep(stop_time);          // ---- Do a timestep.
#ifdef HENSON
           henson_save_pointer("amr",  amrptr);        // redundant to do every timesetp, but negligible overhead
           henson_save_pointer("dmpc", Nyx::theDMPC());
           henson_yield();
#endif
            } else {
                finished = true;
            }   
         
            //ACJ
            int current_step  = amrptr->levelSteps(0);
            if(current_step%dualGridProfileFreq==0 or current_step%loadBalanceInt==0 or current_step%loadBalanceInt==loadBalanceInt-1)
            {
                BL_PROFILE_VAR_START(loadBalanceProfiling);
                int level_acj = 0;
                amrex::Vector<long> gridloads = Nyx::theDMPC()->NumberOfParticlesInGrid(level_acj, true, false);   
                amrex::Vector<long> rankload(ParallelDescriptor::NProcs(), 0); //ACJ should be vector of processors for each boxid
                
                if(ParallelDescriptor::IOProcessor())
                {
                    //get current rank load (do this every time step)
                    int numboxes = Nyx::theDMPC()->ParticleDistributionMap(level_acj).size(); 
                    for (int boxid=0; boxid < numboxes; boxid++){ 
                        int proc = Nyx::theDMPC()->ParticleDistributionMap(level_acj)[boxid];
                        rankload[proc] += gridloads[boxid];
                    }
                    rank_loads.push_back(rankload); 
                    rl_steps.push_back(current_step);
                    //if output time-step, output current rank_loads
                    //if(amrptr->levelSteps(0)%loadBalanceInt == 0) and (Nyx::new_a >= 1.0/(loadBalanceStartZ + 1.0)))
                    if(current_step%checkInt==0 or current_step%loadBalanceInt==0){ //write output whenever we write a checkpoint
                        write_rank_loads(Nyx::dual_grid_load_balance, rank_loads, rl_steps);
                    }
                }
                BL_PROFILE_VAR_STOP(loadBalanceProfiling);
                //if(current_step%checkInt==0){
                //    amrex::Print()<<"STEP: "<<current_step<<", Tiny Profiling Output:"<<std::endl;
                    //BL_PROFILE_TINY_FLUSH();
                //}
            }
            //ACJ

        }  // ---- end while( ! finished)
    }

    //ACJ
    if (ParallelDescriptor::IOProcessor())
        write_rank_loads(Nyx::dual_grid_load_balance, rank_loads, rl_steps); 
    //ACJ

    const Real time_without_init = ParallelDescriptor::second() - time_before_main_loop;
    if (ParallelDescriptor::IOProcessor()) std::cout << "Time w/o init: " << time_without_init << std::endl;

    // Write final checkpoint and plotfile
    if (amrptr->stepOfLastCheckPoint() < amrptr->levelSteps(0)) {
        amrptr->checkPoint();
    }
    if (amrptr->stepOfLastPlotFile() < amrptr->levelSteps(0)) {
        amrptr->writePlotFile();
    }

    delete amrptr;

    //
    // This MUST follow the above delete as ~Amr() may dump files to disk.
    //
    const int IOProc = ParallelDescriptor::IOProcessorNumber();

    Real dRunTime2 = ParallelDescriptor::second() - dRunTime1;

    ParallelDescriptor::ReduceRealMax(dRunTime2, IOProc);

    if (ParallelDescriptor::IOProcessor())
    {
        std::cout << "Run time = " << dRunTime2 << std::endl;
    }

    BL_PROFILE_VAR_STOP(pmain);
    BL_PROFILE_REGION_STOP("main()");
    BL_PROFILE_SET_RUN_TIME(dRunTime2);

    }
    amrex::Finalize();
}
