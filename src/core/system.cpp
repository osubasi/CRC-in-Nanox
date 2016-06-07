/*************************************************************************************/
/*      Copyright 2015 Barcelona Supercomputing Center                               */
/*                                                                                   */
/*      This file is part of the NANOS++ library.                                    */
/*                                                                                   */
/*      NANOS++ is free software: you can redistribute it and/or modify              */
/*      it under the terms of the GNU Lesser General Public License as published by  */
/*      the Free Software Foundation, either version 3 of the License, or            */
/*      (at your option) any later version.                                          */
/*                                                                                   */
/*      NANOS++ is distributed in the hope that it will be useful,                   */
/*      but WITHOUT ANY WARRANTY; without even the implied warranty of               */
/*      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                */
/*      GNU Lesser General Public License for more details.                          */
/*                                                                                   */
/*      You should have received a copy of the GNU Lesser General Public License     */
/*      along with NANOS++.  If not, see <http://www.gnu.org/licenses/>.             */
/*************************************************************************************/


#ifdef NANOS_RESILIENCY_ENABLED
#include "backupmanager.hpp"
#include "exception/signaltranslator.hpp"
#include "exception/operationfailure.hpp"
#include <cpuid.h>
#include "hashmap.hpp"
#include <nmmintrin.h>
#endif

#include "system.hpp"
#include "config.hpp"
#include "plugin.hpp"
#include "schedule.hpp"
#include "barrier.hpp"
#include "nanos-int.h"
#include "copydata.hpp"
#include "os.hpp"
#include "basethread.hpp"
#include "malign.hpp"
#include "processingelement.hpp"
#include "basethread.hpp"
#include "allocator.hpp"
#include "debug.hpp"

#include "smpthread.hpp"
#include "regiondict.hpp"
#include "smpprocessor.hpp"
#include "location.hpp"
#include "router.hpp"
#include "addressspace.hpp"
#include "globalregt.hpp"

#ifdef SPU_DEV
#include "spuprocessor.hpp"
#endif

#ifdef CLUSTER_DEV
#include "clusternode_decl.hpp"
#include "clusterthread_decl.hpp"
#endif
#include "clustermpiplugin_decl.hpp"

#include "addressspace.hpp"

#ifdef NANOS_INSTRUMENTATION_ENABLED
#   include "mainfunction.hpp"
#endif

#include <mutex>
#include <set>

#include <climits>
#include <cstring>

// extern "C" {
// 
//    // MPI INTERCEPTION
// 
//    int MPI_Init(int *argc, char ***argv);
//    int MPI_Init(int *argc, char ***argv) 
//    {
//       //fprintf(stderr ,"Calling MPI_Init\n");
//       //int result = MPI_Init_C_Wrapper(argc, argv);
//       //int result = PMPI_Init( argc, argv );
//       printf("Just called MPI_Init\n");
//       sys.initClusterMPI(argc, argv);
//       return 0;
//    }
// 
//    int MPI_Init_thread( int *argc, char ***argv, int required, int *provided );
//    int MPI_Init_thread( int *argc, char ***argv, int required, int *provided )
//    {
//       //fprintf(stderr,"Calling MPI_Init_thread\n");
//       //int result = MPI_Init_thread_C_Wrapper(argc, argv, required, provided);
//       //int result = PMPI_Init_thread( argc, argv, required, provided );
//       printf("Just called MPI_Init_thread\n");
//       sys.initClusterMPI(argc, argv);
//       return 0;
//    }
// 
//    // int MPI_Finalize() 
//    // {
//    //    //printf("Calling MPI_Finalize\n");
//    //    FTI_Finalize();
//    //    int result = PMPI_Finalize();
//    //    //printf("Just called MPI_Finalize\n");
//    //    return result;
//    // }
// }

using namespace nanos;

System nanos::sys;

namespace nanos {
namespace PMInterfaceType
{
   extern int * ssCompatibility;
   extern void (*set_interface)( void * );
}
}

// default system values go here
System::System () :
      _atomicWDSeed( 1 ), _threadIdSeed( 0 ), _peIdSeed( 0 ), _SMP("SMP"),
      /*jb _numPEs( INT_MAX ), _numThreads( 0 ),*/ _deviceStackSize( 0 ), _profile( false ),
      _instrument( false ), _verboseMode( false ), _summary( false ), _executionMode( DEDICATED ), _initialMode( POOL ),
      _untieMaster( true ), _delayedStart( false ), _synchronizedStart( true ), _alreadyFinished( false ),
      _predecessorLists( false ), _throttlePolicy ( NULL ),
      _schedStats(), _schedConf(), _defSchedule( "bf" ), _defThrottlePolicy( "hysteresis" ), 
      _defBarr( "centralized" ), _defInstr ( "empty_trace" ), _defDepsManager( "plain" ), _defArch( "smp" ),
      _initializedThreads ( 0 ), /*_targetThreads ( 0 ),*/ _pausedThreads( 0 ),
      _pausedThreadsCond(), _unpausedThreadsCond(),
      _net(), _usingCluster( false ), _usingClusterMPI( false ), _clusterMPIPlugin( NULL ), _usingNode2Node( true ), _usingPacking( true ), _conduit( "udp" ),
      _instrumentation ( NULL ), _defSchedulePolicy( NULL ), _dependenciesManager( NULL ),
      _pmInterface( NULL ), _masterGpuThd( NULL ), _separateMemorySpacesCount(1), _separateAddressSpaces(1024), _hostMemory( ext::getSMPDevice() ),
      _regionCachePolicy( RegionCache::WRITE_BACK ), _regionCachePolicyStr(""), _regionCacheSlabSize(0), _clusterNodes(), _numaNodes(),
      _activeMemorySpaces(), _acceleratorCount(0), _numaNodeMap(), _threadManagerConf(), _threadManager( NULL )
#ifdef GPU_DEV
      , _pinnedMemoryCUDA( NEW CUDAPinnedMemoryManager() )
#endif
#ifdef NANOS_INSTRUMENTATION_ENABLED
      , _mainFunctionEvent( NULL )
      , _enableEvents(), _disableEvents(), _instrumentDefault("default"), _enableCpuidEvent( false )
#endif
      , _lockPoolSize(37), _lockPool( NULL ), _mainTeam (NULL), _simulator(false)
#ifdef NANOS_RESILIENCY_ENABLED
      , _injectionPolicy( "none" )
      , _resiliency_disabled(false)
      , _task_max_trials(1)
      , _backup_pool_size(sysconf(_SC_PAGESIZE ) * sysconf(_SC_PHYS_PAGES) / 20)
      , _crcParam()
      , _hashmap()
#endif
      , _affinityFailureCount( 0 )
      , _createLocalTasks( false )
      , _verboseDevOps( false )
      , _verboseCopies( false )
      , _splitOutputForThreads( false )
      , _userDefinedNUMANode( -1 )
      , _router()
      , _hwloc()
      , _immediateSuccessorDisabled( false )
      , _predecessorCopyInfoDisabled( false )
      , _invalControl( false )
      , _cgAlloc( false )
      , _inIdle( false )
	  , _lazyPrivatizationEnabled (false)
	  , _watchAddr(nullptr)
{
   verbose( "NANOS++ initializing... start" );

   // OS::init must be called here and not in System::start() as it can be too late
   // to locate the program arguments at that point
   OS::init();
   config();

   _lockPool = NEW Lock[_lockPoolSize];

   if ( !_delayedStart ) {
      //std::cerr << "NX_ARGS is:" << (char *)(OS::getEnvironmentVariable( "NX_ARGS" ) != NULL ? OS::getEnvironmentVariable( "NX_ARGS" ) : "NO NX_ARGS: GG!") << std::endl;
      start();
   }
   verbose( "NANOS++ initializing... end" );
}

struct LoadModule
{
   void operator() ( const char *module )
   {
      if ( module ) {
        verbose( "loading ", module, " module" );
        sys.loadPlugin(module);
      }
   }
};

void System::loadArchitectures()
{
   verbose ( "Configuring module manager" );
   _pluginManager.init();
   verbose ( "Loading architectures" );

   
   // load host processor module
   if ( _hostFactory == NULL ) {
     verbose( "loading Host support" );

     if ( !loadPlugin( "pe-"+getDefaultArch() ) )
       fatal ( "Couldn't load host support" );
   }
   ensure( _hostFactory,"No default host factory" );

#ifdef GPU_DEV
   verbose( "loading GPU support" );

   if ( !loadPlugin( "pe-gpu" ) )
      fatal ( "Couldn't load GPU support" );
#endif
   
#ifdef OpenCL_DEV
   verbose( "loading OpenCL support" );
   if ( !loadPlugin( "pe-opencl" ) )
     fatal ( "Couldn't load OpenCL support" );
#endif

#ifdef FPGA_DEV
   verbose( "loading FPGA support" );

   if ( !loadPlugin( "pe-fpga" ) )
       fatal ( "couldn't load FPGA support" );
#endif

#ifdef CLUSTER_DEV
   if ( usingCluster() && usingClusterMPI() ) {
      fatal("Can't use --cluster and --cluster-mpi at the same time,");
   } else if ( usingCluster() ) {
      verbose( "Loading Cluster plugin (" + getNetworkConduit() + ")" ) ;
      if ( !loadPlugin( "pe-cluster-"+getNetworkConduit() ) )
         fatal ( "Couldn't load Cluster support" );
   } else if ( usingClusterMPI() ) {
      verbose( "Loading ClusterMPI plugin (" + getNetworkConduit() + ")" ) ;
      _clusterMPIPlugin = (ext::ClusterMPIPlugin *) loadAndGetPlugin( "pe-clustermpi-"+getNetworkConduit() );
      if ( _clusterMPIPlugin == NULL ) {
         fatal ( "Couldn't load ClusterMPI support" );
      } else {
         _clusterMPIPlugin->init();
      }
   }
#endif

   verbose( "Architectures loaded");

#ifdef MPI_DEV
   char* isOffloadSlave = getenv(const_cast<char*> ("OMPSS_OFFLOAD_SLAVE")); 
   //Plugin->init of MPI will initialize MPI when we are slaves so MPI spawn returns ASAP in the master
   //This plugin does not reserve any PE at initialization time, just perform MPI Init and other actions
   if ( isOffloadSlave ) sys.loadPlugin("arch-mpi");
#endif
}

void System::loadModules ()
{
   verbose ( "Loading modules" );

   const OS::ModuleList & modules = OS::getRequestedModules();
   std::for_each(modules.begin(),modules.end(), LoadModule());
   
   if ( !loadPlugin( "instrumentation-"+getDefaultInstrumentation() ) )
      fatal( "Could not load " + getDefaultInstrumentation() + " instrumentation" );

   // load default dependencies plugin
   verbose( "loading ", getDefaultDependenciesManager(), " dependencies manager support" );

   if ( !loadPlugin( "deps-"+getDefaultDependenciesManager() ) )
      fatal ( "Couldn't load main dependencies manager" );

   ensure( _dependenciesManager,"No default dependencies manager" );

   // load default schedule plugin
   verbose( "loading ", getDefaultSchedule(), " scheduling policy support" );

   if ( !loadPlugin( "sched-"+getDefaultSchedule() ) )
      fatal ( "Couldn't load main scheduling policy" );

   ensure( _defSchedulePolicy,"No default system scheduling factory" );

   verbose( "loading ", getDefaultThrottlePolicy(), " throttle policy" );

   if ( !loadPlugin( "throttle-"+getDefaultThrottlePolicy() ) )
      fatal( "Could not load main cutoff policy" );

   ensure( _throttlePolicy, "No default throttle policy" );

   verbose( "loading ", getDefaultBarrier(), " barrier algorithm" );

   if ( !loadPlugin( "barrier-"+getDefaultBarrier() ) )
      fatal( "Could not load main barrier algorithm" );

   ensure( _defBarrFactory,"No default system barrier factory" );
   
#ifdef NANOS_RESILIENCY_ENABLED
   // load default error injection plugin
   verbose( "loading ", getInjectionPolicy(), " injection policy" );

   if ( !loadPlugin( std::string("injection-")+getInjectionPolicy() ) )
      fatal( "Could not load main error injection policy" );

   ensure( !_injectionPolicy.empty(),"No error injection policy defined" );
#endif

   verbose( "Starting Thread Manager" );

   _threadManager = _threadManagerConf.create();
}

void System::unloadModules ()
{   
   delete _throttlePolicy;
   
   delete _defSchedulePolicy;
   
   //! \todo (#613): delete GPU plugin?
}

// Config Functor
struct ExecInit
{
   std::set<void *> _initialized;

   ExecInit() : _initialized() {}

   void operator() ( const nanos_init_desc_t & init )
   {
      if ( _initialized.find( (void *)init.func ) == _initialized.end() ) {
         init.func( init.data );
         _initialized.insert( ( void * ) init.func );
      }
   }
};

void System::config ()
{
   Config cfg;

   const OS::InitList & externalInits = OS::getInitializationFunctions();
   std::for_each(externalInits.begin(),externalInits.end(), ExecInit());
   
   //! Declare all configuration core's flags
   verbose( "Preparing library configuration" );

   cfg.setOptionsSection( "Core", "Core options of the core of Nanos++ runtime" );

   cfg.registerConfigOption( "stack-size", NEW Config::SizeVar( _deviceStackSize ), "Default stack size (all devices)" );
   cfg.registerArgOption( "stack-size", "stack-size" );
   cfg.registerEnvOption( "stack-size", "NX_STACK_SIZE" );

   cfg.registerConfigOption( "verbose", NEW Config::FlagOption( _verboseMode ),
                             "Activates verbose mode" );
   cfg.registerArgOption( "verbose", "verbose" );

   cfg.registerConfigOption( "summary", NEW Config::FlagOption( _summary ),
                             "Activates summary mode" );
   cfg.registerArgOption( "summary", "summary" );

//! \bug implement execution modes (#146) */
#if 0
   cfg::MapVar<ExecutionMode> map( _executionMode );
   map.addOption( "dedicated", DEDICATED).addOption( "shared", SHARED );
   cfg.registerConfigOption ( "exec_mode", &map, "Execution mode" );
   cfg.registerArgOption ( "exec_mode", "mode" );
#endif

   registerPluginOption( "schedule", "sched", _defSchedule,
                         "Defines the scheduling policy", cfg );
   cfg.registerArgOption( "schedule", "schedule" );
   cfg.registerEnvOption( "schedule", "NX_SCHEDULE" );

   registerPluginOption( "throttle", "throttle", _defThrottlePolicy,
                         "Defines the throttle policy", cfg );
   cfg.registerArgOption( "throttle", "throttle" );
   cfg.registerEnvOption( "throttle", "NX_THROTTLE" );

   cfg.registerConfigOption( "barrier", NEW Config::StringVar ( _defBarr ),
                             "Defines barrier algorithm" );
   cfg.registerArgOption( "barrier", "barrier" );
   cfg.registerEnvOption( "barrier", "NX_BARRIER" );

   registerPluginOption( "instrumentation", "instrumentation", _defInstr,
                         "Defines instrumentation format", cfg );
   cfg.registerArgOption( "instrumentation", "instrumentation" );
   cfg.registerEnvOption( "instrumentation", "NX_INSTRUMENTATION" );

   cfg.registerConfigOption( "no-sync-start", NEW Config::FlagOption( _synchronizedStart, false),
                             "Disables synchronized start" );
   cfg.registerArgOption( "no-sync-start", "disable-synchronized-start" );

   cfg.registerConfigOption( "architecture", NEW Config::StringVar ( _defArch ),
                             "Defines the architecture to use (smp by default)" );
   cfg.registerArgOption( "architecture", "architecture" );
   cfg.registerEnvOption( "architecture", "NX_ARCHITECTURE" );

   registerPluginOption( "deps", "deps", _defDepsManager,
                         "Defines the dependencies plugin", cfg );
   cfg.registerArgOption( "deps", "deps" );
   cfg.registerEnvOption( "deps", "NX_DEPS" );
   

#ifdef NANOS_INSTRUMENTATION_ENABLED
   cfg.registerConfigOption( "instrument-default", NEW Config::StringVar ( _instrumentDefault ),
                             "Set instrumentation event list default (none, all)" );
   cfg.registerArgOption( "instrument-default", "instrument-default" );

   cfg.registerConfigOption( "instrument-enable", NEW Config::StringVarList ( _enableEvents ),
                             "Add events to instrumentation event list" );
   cfg.registerArgOption( "instrument-enable", "instrument-enable" );

   cfg.registerConfigOption( "instrument-disable", NEW Config::StringVarList ( _disableEvents ),
                             "Remove events to instrumentation event list" );
   cfg.registerArgOption( "instrument-disable", "instrument-disable" );

   cfg.registerConfigOption( "instrument-cpuid", NEW Config::FlagOption ( _enableCpuidEvent ),
                             "Add cpuid event when binding is disabled (expensive)" );
   cfg.registerArgOption( "instrument-cpuid", "instrument-cpuid" );
#endif

   /* Cluster: load the cluster support */
   cfg.registerConfigOption ( "enable-cluster", NEW Config::FlagOption ( _usingCluster, true ), "Enables the usage of Nanos++ Cluster" );
   cfg.registerArgOption ( "enable-cluster", "cluster" );
   //cfg.registerEnvOption ( "enable-cluster", "NX_ENABLE_CLUSTER" );
   cfg.registerConfigOption ( "enable-cluster-mpi", NEW Config::FlagOption ( _usingClusterMPI, true ), "Enables the usage of Nanos++ Cluster with MPI applications" );
   cfg.registerArgOption ( "enable-cluster-mpi", "cluster-mpi" );

   cfg.registerConfigOption ( "no-node2node", NEW Config::FlagOption ( _usingNode2Node, false ), "Disables the usage of Slave-to-Slave transfers" );
   cfg.registerArgOption ( "no-node2node", "disable-node2node" );
   cfg.registerConfigOption ( "no-pack", NEW Config::FlagOption ( _usingPacking, false ), "Disables the usage of packing and unpacking of strided transfers" );
   cfg.registerArgOption ( "no-pack", "disable-packed-copies" );

   /* Cluster: select wich module to load mpi or udp */
   cfg.registerConfigOption ( "conduit", NEW Config::StringVar ( _conduit ), "Selects which GasNet conduit will be used" );
   cfg.registerArgOption ( "conduit", "cluster-network" );
   cfg.registerEnvOption ( "conduit", "NX_CLUSTER_NETWORK" );

#if 0 /* _defDeviceName and _defDevice seem unused */
   cfg.registerConfigOption ( "device-priority", NEW Config::StringVar ( _defDeviceName ), "Defines the default device to use");
   cfg.registerArgOption ( "device-priority", "--use-device");
   cfg.registerEnvOption ( "device-priority", "NX_USE_DEVICE");
#endif
   cfg.registerConfigOption( "simulator", NEW Config::FlagOption ( _simulator ),
                             "Nanos++ will be executed by a simulator (disabled as default)" );
   cfg.registerArgOption( "simulator", "simulator" );

#ifdef NANOS_RESILIENCY_ENABLED
   cfg.registerConfigOption("disable_resiliency",
         NEW Config::FlagOption(_resiliency_disabled, true),
         "Disables all resiliency mechanisms. ");
   cfg.registerArgOption("disable_resiliency", "disable-resiliency");
   cfg.registerEnvOption("disable_resiliency", "NX_DISABLE_RESILIENCY");

   cfg.registerConfigOption("task_retrials",
         NEW Config::UintVar(_task_max_trials),
         "Defines the number of times a restartable task can be re-executed (default: 1). ");
   cfg.registerArgOption("task_retrials", "task-retrials");
   cfg.registerEnvOption("task_retrials", "NX_TASK_RETRIALS");

   cfg.registerConfigOption("backup_pool_size",
         NEW Config::SizeVar(_backup_pool_size),
         "Sets the memory pool maximum size (dedicated to store task backups) in bytes. ");
   cfg.registerArgOption("backup_pool_size", "backup-pool-size");
   cfg.registerEnvOption("backup_pool_size", "NX_BACKUP_POOL_SIZE");

   registerPluginOption("error_injection", "error-injection", _injectionPolicy,
         "Selects error injection policy. Used for resiliency evaluation.", cfg);
   cfg.registerArgOption("error_injection", "error-injection");
   cfg.registerEnvOption("error_injection", "NX_ERROR_INJECTION");

   cfg.registerConfigOption("enable_crc", NEW Config::FlagOption(_crc_enabled, true), "Enables CRC protection. ");
   cfg.registerArgOption("enable_crc", "enable-crc");
   cfg.registerEnvOption("enable_crc", "NX_ENABLE_CRC");
#endif

   cfg.registerConfigOption ( "verbose-devops", NEW Config::FlagOption ( _verboseDevOps, true ), "Verbose cache ops" );
   cfg.registerArgOption ( "verbose-devops", "verbose-devops" );
   cfg.registerConfigOption ( "verbose-copies", NEW Config::FlagOption ( _verboseCopies, true ), "Verbose data copies" );
   cfg.registerArgOption ( "verbose-copies", "verbose-copies" );

   cfg.registerConfigOption ( "thd-output", NEW Config::FlagOption ( _splitOutputForThreads, true ), "Create separate files for each thread" );
   cfg.registerArgOption ( "thd-output", "thd-output" );

   cfg.registerConfigOption ( "regioncache-policy", NEW Config::StringVar ( _regionCachePolicyStr ), "Region cache policy, accepted values are : nocache, writethrough, writeback, fpga. Default is writeback." );
   cfg.registerArgOption ( "regioncache-policy", "cache-policy" );
   cfg.registerEnvOption ( "regioncache-policy", "NX_CACHE_POLICY" );

   cfg.registerConfigOption ( "regioncache-slab-size", NEW Config::SizeVar ( _regionCacheSlabSize ), "Region slab size." );
   cfg.registerArgOption ( "regioncache-slab-size", "cache-slab-size" );
   cfg.registerEnvOption ( "regioncache-slab-size", "NX_CACHE_SLAB_SIZE" );

   cfg.registerConfigOption( "disable-immediate-succ", NEW Config::FlagOption( _immediateSuccessorDisabled ), "Disables the usage of getImmediateSuccessor" );
   cfg.registerArgOption( "disable-immediate-succ", "disable-immediate-successor" );

   cfg.registerConfigOption( "disable-predecessor-info", NEW Config::FlagOption( _predecessorCopyInfoDisabled ),
                             "Disables sending the copy_data info to successor WDs." );
   cfg.registerArgOption( "disable-predecessor-info", "disable-predecessor-info" );
   cfg.registerConfigOption( "inval-control", NEW Config::FlagOption( _invalControl ),
                             "Inval control." );
   cfg.registerArgOption( "inval-control", "inval-control" );

   cfg.registerConfigOption( "cg-alloc", NEW Config::FlagOption( _cgAlloc ),
                             "CG alloc." );
   cfg.registerArgOption( "cg-alloc", "cg-alloc" );

   cfg.registerConfigOption ( "enable-lazy-privatization", NEW Config::BoolVar ( _lazyPrivatizationEnabled ),
		   "Enable lazy reduction privatization" );
   cfg.registerArgOption ( "enable-lazy-privatization", "enable-lazy-privatization" );

   _schedConf.config( cfg );

   _hwloc.config( cfg );
   _threadManagerConf.config( cfg );

   verbose ( "Reading Configuration" );

   cfg.init();
   
   // Now read compiler-supplied flags
   // Open the own executable
   void * myself = dlopen(NULL, RTLD_LAZY | RTLD_GLOBAL);

   // Check if the compiler marked myself as requiring priorities (#1041)
   _compilerSuppliedFlags.prioritiesNeeded = dlsym(myself, "nanos_need_priorities_") != NULL;
   
   // Close handle to myself
   dlclose( myself );
}

void System::start ()
{
   _hwloc.loadHwloc();
   
   // Modules can be loaded now
   loadArchitectures();
   loadModules();

   verbose( "Stating PM interface.");
   Config cfg;
   void (*f)(void *) = nanos::PMInterfaceType::set_interface;
   f(NULL);
   _pmInterface->config( cfg );
   cfg.init();
   _pmInterface->start();

   // Instrumentation startup
   NANOS_INSTRUMENT ( sys.getInstrumentation()->filterEvents( _instrumentDefault, _enableEvents, _disableEvents ) );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->initialize() );

   verbose ( "Starting runtime" );

   if ( _regionCachePolicyStr.compare("") != 0 ) {
      //value is set
      if ( _regionCachePolicyStr.compare("nocache") == 0 ) {
         _regionCachePolicy = RegionCache::NO_CACHE;
      } else if ( _regionCachePolicyStr.compare("writethrough") == 0 ) {
         _regionCachePolicy = RegionCache::WRITE_THROUGH;
      } else if ( _regionCachePolicyStr.compare("writeback") == 0 ) {
         _regionCachePolicy = RegionCache::WRITE_BACK;
      } else if ( _regionCachePolicyStr.compare("fpga") == 0 ) {
         _regionCachePolicy = RegionCache::FPGA;
      } else {
         warning( "Invalid option for region cache policy '", _regionCachePolicyStr,
                   "', using default value."
                 );
      }
   }

   //don't allow untiedMaster in cluster, otherwise Nanos finalization crashes
   _smpPlugin->associateThisThread( usingCluster() ? false : getUntieMaster() );

   //Setup MainWD
   WD &mainWD = *myThread->getCurrentWD();
   mainWD._mcontrol.setMainWD();

   if ( _pmInterface->getInternalDataSize() > 0 ) {
      char *data = NEW char[_pmInterface->getInternalDataSize()];
      _pmInterface->initInternalData( data );
      mainWD.setInternalData( data );
   }
   _pmInterface->setupWD( mainWD );

   if ( _defSchedulePolicy->getWDDataSize() > 0 ) {
      char *data = NEW char[ _defSchedulePolicy->getWDDataSize() ];
      _defSchedulePolicy->initWDData( data );
      mainWD.setSchedulerData( reinterpret_cast<ScheduleWDData*>( data ), /* ownedByWD */ true );
   }

   /* Renaming currend thread as Master */
   myThread->rename("Master");
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent (NANOS_STARTUP) );

   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      verbose("addPEs for arch: ", (*it)->getName());
      (*it)->addPEs( _pes );
      (*it)->addDevices( _devices );
   }
   
   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      (*it)->startSupportThreads();
   }   
   
   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it )
   {
      (*it)->startWorkerThreads( _workers );
   }   

   for ( PEList::iterator it = _pes.begin(); it != _pes.end(); it++ ) {
      if ( it->second->isActive() ) {
         _clusterNodes.insert( it->second->getClusterNode() );
         // If this PE is in a NUMA node and has workers
         if ( it->second->isInNumaNode() && ( it->second->getNumThreads() > 0  ) ) {
            // Add the node of this PE to the set of used NUMA nodes
            unsigned node = it->second->getNumaNode() ;
            _numaNodes.insert( node );
         }
         _activeMemorySpaces.insert( it->second->getMemorySpaceId() );
      }
   }
   
   // gmiranda: was completeNUMAInfo() We must do this after the
   // previous loop since we need the size of _numaNodes
   
   unsigned availNUMANodes = 0;
   // #994: this should be the number of NUMA objects in hwloc, but if we don't
   // want to query, this max should be enough
   unsigned maxNUMANode = _numaNodes.empty() ? 1 : *std::max_element( _numaNodes.begin(), _numaNodes.end() );
   // Create the NUMA node translation table. Do this before creating the team,
   // as the schedulers might need the information.
   _numaNodeMap.resize( maxNUMANode + 1, INT_MIN );
   
   for ( std::set<unsigned int>::const_iterator it = _numaNodes.begin();
        it != _numaNodes.end(); ++it )
   {
      unsigned node = *it;
      // If that node has not been translated, yet
      if ( _numaNodeMap[ node ] == INT_MIN )
      {
         verbose( "[NUMA] Mapping from physical node ", node,
                   " to user node ", availNUMANodes
                 );
         _numaNodeMap[ node ] = availNUMANodes;
         // Increase the number of available NUMA nodes
         ++availNUMANodes;
      }
      // Otherwise, do nothing
   }
   verbose( "[NUMA] ", availNUMANodes, " NUMA node(s) available for the user." );

   // For each plugin, notify it's the way to reserve PEs if they are required
   //for ( ArchitecturePlugins::const_iterator it = _archs.begin();
   //     it != _archs.end(); ++it )
   //{
   //   (*it)->createBindingList();
   //}

#ifdef NANOS_RESILIENCY_ENABLED   // compile time disable
   if(sys.isResiliencyEnabled()){// runtime disable
      // Insert a new separate memory address space to store input backups
      BackupManager* mgr = new BackupManager("BackupMgr", _backup_pool_size);

      memory_space_id_t backup_id = addSeparateMemoryAddressSpace( *mgr, true /*allocWide*/, 0 /* slabSize*/ );
      _backupMemory = &getSeparateMemory( backup_id );
   }
#endif   

   _targetThreads = _smpPlugin->getNumThreads();

   // Set up internal data for each worker
   for ( ThreadList::const_iterator it = _workers.begin(); it != _workers.end(); it++ ) {

      WD & threadWD = it->second->getThreadWD();
      if ( _pmInterface->getInternalDataSize() > 0 ) {
         char *data = NEW char[_pmInterface->getInternalDataSize()];
         _pmInterface->initInternalData( data );
         threadWD.setInternalData( data );
      }
      _pmInterface->setupWD( threadWD );

      int schedDataSize = _defSchedulePolicy->getWDDataSize();
      if ( schedDataSize  > 0 ) {
         ScheduleWDData *schedData = reinterpret_cast<ScheduleWDData*>( NEW char[schedDataSize] );
         _defSchedulePolicy->initWDData( schedData );
         threadWD.setSchedulerData( schedData, true );
      }

   }

#if 0 /* _defDeviceName and _defDevice seem unused */
   if ( !_defDeviceName.empty() ) 
   {
       PEList::iterator it;
       for ( it = _pes.begin() ; it != _pes.end(); it++ )
       {
           PE *pe = it->second;
           if ( pe->getDeviceType()->getName() != NULL)
              if ( _defDeviceName == pe->getDeviceType()->getName()  )
                 _defDevice = pe->getDeviceType();
       }
   }
#endif

#ifdef NANOS_RESILIENCY_ENABLED
   error::SignalTranslator<error::OperationFailure> signalToExceptionTranslator;
#endif

   if ( getSynchronizedStart() ) threadReady();

   switch ( getInitialMode() )
   {
      case POOL:
         verbose("Pool model enabled (OmpSs)");
         _mainTeam = createTeam( _workers.size(), /*constraints*/ NULL, /*reuse*/ true, /*enter*/ true, /*parallel*/ false );
         break;
      case ONE_THREAD:
         verbose("One-thread model enabled (OpenMP)");
         _mainTeam = createTeam( 1, /*constraints*/ NULL, /*reuse*/ true, /*enter*/ true, /*parallel*/ true );
         break;
      default:
         fatal("Unknown initial mode!");
         break;
   }

   _router.initialize();
   _net.setParentWD( &mainWD );
   if ( usingCluster() )
   {
      _net.nodeBarrier();
   }

   NANOS_INSTRUMENT ( static InstrumentationDictionary *ID = sys.getInstrumentation()->getInstrumentationDictionary(); )
   NANOS_INSTRUMENT ( static nanos_event_key_t num_threads_key = ID->getEventKey("set-num-threads"); )
   NANOS_INSTRUMENT ( nanos_event_value_t team_size =  (nanos_event_value_t) myThread->getTeam()->size(); )
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raisePointEvents(1, &num_threads_key, &team_size); )
   
   // Paused threads: set the condition checker 
   _pausedThreadsCond.setConditionChecker( EqualConditionChecker<unsigned int >( &_pausedThreads.override(), _workers.size() ) );
   _unpausedThreadsCond.setConditionChecker( EqualConditionChecker<unsigned int >( &_pausedThreads.override(), 0 ) );

   // All initialization is ready, call postInit hooks
   const OS::InitList & externalInits = OS::getPostInitializationFunctions();
   std::for_each(externalInits.begin(),externalInits.end(), ExecInit());

   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent (NANOS_RUNNING) );

   // List unrecognised arguments
   std::string unrecog = Config::getOrphanOptions();
   if ( !unrecog.empty() )
      warning( "Unrecognised arguments: ", unrecog );
   Config::deleteOrphanOptions();
      
   if ( _summary ) {
      environmentSummary();

#ifdef HAVE_CXX11
		// TODO: move this to the new exception design directory
      // If the summary is enabled, print the final execution summary
      // even if the application is terminated by an error.
      std::set_terminate( [](){
         static std::once_flag called_once;
         call_once( called_once, []() {
            sys.executionSummary();
#ifdef __GNUG__
            // Call verbose terminate handler: prints exception information
            // This is a gnu extension
            __gnu_cxx::__verbose_terminate_handler();
#else
            message( "An error was reported and this program will finish." );
            std::abort();
#endif
         });
      });
#endif
   }

   // Thread Manager initialization is delayed until a safe point
   _threadManager->init();

#ifdef NANOS_RESILIENCY_ENABLED //OMER::ENCODING
   if(sys._crc_enabled){
	   unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

	   __get_cpuid(1, &eax, &ebx, &ecx, &edx);

	   if (ecx & bit_SSE4_2){
		   // i have the instruction
		   _crcParam = true;
		//#define CRC_SUPPORTED 
		   //std::cerr << "Intel CRC Exists "<<std::endl;
		   unsigned int temp[256] = {
				0x00000000,0x8f158014,0x1bc776d9,0x94d2f6cd,
				0x378eedb2,0xb89b6da6,0x2c499b6b,0xa35c1b7f,
				0x6f1ddb64,0xe0085b70,0x74daadbd,0xfbcf2da9,
				0x589336d6,0xd786b6c2,0x4354400f,0xcc41c01b,
				0xde3bb6c8,0x512e36dc,0xc5fcc011,0x4ae94005,
				0xe9b55b7a,0x66a0db6e,0xf2722da3,0x7d67adb7,
				0xb1266dac,0x3e33edb8,0xaae11b75,0x25f49b61,
				0x86a8801e,0x09bd000a,0x9d6ff6c7,0x127a76d3,
				0xb99b1b61,0x368e9b75,0xa25c6db8,0x2d49edac,
				0x8e15f6d3,0x010076c7,0x95d2800a,0x1ac7001e,
				0xd686c005,0x59934011,0xcd41b6dc,0x425436c8,
				0xe1082db7,0x6e1dada3,0xfacf5b6e,0x75dadb7a,
				0x67a0ada9,0xe8b52dbd,0x7c67db70,0xf3725b64,
				0x502e401b,0xdf3bc00f,0x4be936c2,0xc4fcb6d6,
				0x08bd76cd,0x87a8f6d9,0x137a0014,0x9c6f8000,
				0x3f339b7f,0xb0261b6b,0x24f4eda6,0xabe16db2,
				0x76da4033,0xf9cfc027,0x6d1d36ea,0xe208b6fe,
				0x4154ad81,0xce412d95,0x5a93db58,0xd5865b4c,
				0x19c79b57,0x96d21b43,0x0200ed8e,0x8d156d9a,
				0x2e4976e5,0xa15cf6f1,0x358e003c,0xba9b8028,
				0xa8e1f6fb,0x27f476ef,0xb3268022,0x3c330036,
				0x9f6f1b49,0x107a9b5d,0x84a86d90,0x0bbded84,
				0xc7fc2d9f,0x48e9ad8b,0xdc3b5b46,0x532edb52,
				0xf072c02d,0x7f674039,0xebb5b6f4,0x64a036e0,
				0xcf415b52,0x4054db46,0xd4862d8b,0x5b93ad9f,
				0xf8cfb6e0,0x77da36f4,0xe308c039,0x6c1d402d,
				0xa05c8036,0x2f490022,0xbb9bf6ef,0x348e76fb,
				0x97d26d84,0x18c7ed90,0x8c151b5d,0x03009b49,
				0x117aed9a,0x9e6f6d8e,0x0abd9b43,0x85a81b57,
				0x26f40028,0xa9e1803c,0x3d3376f1,0xb226f6e5,
				0x7e6736fe,0xf172b6ea,0x65a04027,0xeab5c033,
				0x49e9db4c,0xc6fc5b58,0x522ead95,0xdd3b2d81,
				0xedb48066,0x62a10072,0xf673f6bf,0x796676ab,
				0xda3a6dd4,0x552fedc0,0xc1fd1b0d,0x4ee89b19,
				0x82a95b02,0x0dbcdb16,0x996e2ddb,0x167badcf,
				0xb527b6b0,0x3a3236a4,0xaee0c069,0x21f5407d,
				0x338f36ae,0xbc9ab6ba,0x28484077,0xa75dc063,
				0x0401db1c,0x8b145b08,0x1fc6adc5,0x90d32dd1,
				0x5c92edca,0xd3876dde,0x47559b13,0xc8401b07,
				0x6b1c0078,0xe409806c,0x70db76a1,0xffcef6b5,
				0x542f9b07,0xdb3a1b13,0x4fe8edde,0xc0fd6dca,
				0x63a176b5,0xecb4f6a1,0x7866006c,0xf7738078,
				0x3b324063,0xb427c077,0x20f536ba,0xafe0b6ae,
				0x0cbcadd1,0x83a92dc5,0x177bdb08,0x986e5b1c,
				0x8a142dcf,0x0501addb,0x91d35b16,0x1ec6db02,
				0xbd9ac07d,0x328f4069,0xa65db6a4,0x294836b0,
				0xe509f6ab,0x6a1c76bf,0xfece8072,0x71db0066,
				0xd2871b19,0x5d929b0d,0xc9406dc0,0x4655edd4,
				0x9b6ec055,0x147b4041,0x80a9b68c,0x0fbc3698,
				0xace02de7,0x23f5adf3,0xb7275b3e,0x3832db2a,
				0xf4731b31,0x7b669b25,0xefb46de8,0x60a1edfc,
				0xc3fdf683,0x4ce87697,0xd83a805a,0x572f004e,
				0x4555769d,0xca40f689,0x5e920044,0xd1878050,
				0x72db9b2f,0xfdce1b3b,0x691cedf6,0xe6096de2,
				0x2a48adf9,0xa55d2ded,0x318fdb20,0xbe9a5b34,
				0x1dc6404b,0x92d3c05f,0x06013692,0x8914b686,
				0x22f5db34,0xade05b20,0x3932aded,0xb6272df9,
				0x157b3686,0x9a6eb692,0x0ebc405f,0x81a9c04b,
				0x4de80050,0xc2fd8044,0x562f7689,0xd93af69d,
				0x7a66ede2,0xf5736df6,0x61a19b3b,0xeeb41b2f,
				0xfcce6dfc,0x73dbede8,0xe7091b25,0x681c9b31,
				0xcb40804e,0x4455005a,0xd087f697,0x5f927683,
				0x93d3b698,0x1cc6368c,0x8814c041,0x07014055,
				0xa45d5b2a,0x2b48db3e,0xbf9a2df3,0x308fade7

		   };

		   unsigned int temp2[256] = {
				0x00000000,0xe417f38a,0xcdc391e5,0x29d4626f,
				0x9e6b553b,0x7a7ca6b1,0x53a8c4de,0xb7bf3754,
				0x393adc87,0xdd2d2f0d,0xf4f94d62,0x10eebee8,
				0xa75189bc,0x43467a36,0x6a921859,0x8e85ebd3,
				0x7275b90e,0x96624a84,0xbfb628eb,0x5ba1db61,
				0xec1eec35,0x08091fbf,0x21dd7dd0,0xc5ca8e5a,
				0x4b4f6589,0xaf589603,0x868cf46c,0x629b07e6,
				0xd52430b2,0x3133c338,0x18e7a157,0xfcf052dd,
				0xe4eb721c,0x00fc8196,0x2928e3f9,0xcd3f1073,
				0x7a802727,0x9e97d4ad,0xb743b6c2,0x53544548,
				0xddd1ae9b,0x39c65d11,0x10123f7e,0xf405ccf4,
				0x43bafba0,0xa7ad082a,0x8e796a45,0x6a6e99cf,
				0x969ecb12,0x72893898,0x5b5d5af7,0xbf4aa97d,
				0x08f59e29,0xece26da3,0xc5360fcc,0x2121fc46,
				0xafa41795,0x4bb3e41f,0x62678670,0x867075fa,
				0x31cf42ae,0xd5d8b124,0xfc0cd34b,0x181b20c1,
				0xcc3a92c9,0x282d6143,0x01f9032c,0xe5eef0a6,
				0x5251c7f2,0xb6463478,0x9f925617,0x7b85a59d,
				0xf5004e4e,0x1117bdc4,0x38c3dfab,0xdcd42c21,
				0x6b6b1b75,0x8f7ce8ff,0xa6a88a90,0x42bf791a,
				0xbe4f2bc7,0x5a58d84d,0x738cba22,0x979b49a8,
				0x20247efc,0xc4338d76,0xede7ef19,0x09f01c93,
				0x8775f740,0x636204ca,0x4ab666a5,0xaea1952f,
				0x191ea27b,0xfd0951f1,0xd4dd339e,0x30cac014,
				0x28d1e0d5,0xccc6135f,0xe5127130,0x010582ba,
				0xb6bab5ee,0x52ad4664,0x7b79240b,0x9f6ed781,
				0x11eb3c52,0xf5fccfd8,0xdc28adb7,0x383f5e3d,
				0x8f806969,0x6b979ae3,0x4243f88c,0xa6540b06,
				0x5aa459db,0xbeb3aa51,0x9767c83e,0x73703bb4,
				0xc4cf0ce0,0x20d8ff6a,0x090c9d05,0xed1b6e8f,
				0x639e855c,0x878976d6,0xae5d14b9,0x4a4ae733,
				0xfdf5d067,0x19e223ed,0x30364182,0xd421b208,
				0x9d995363,0x798ea0e9,0x505ac286,0xb44d310c,
				0x03f20658,0xe7e5f5d2,0xce3197bd,0x2a266437,
				0xa4a38fe4,0x40b47c6e,0x69601e01,0x8d77ed8b,
				0x3ac8dadf,0xdedf2955,0xf70b4b3a,0x131cb8b0,
				0xefecea6d,0x0bfb19e7,0x222f7b88,0xc6388802,
				0x7187bf56,0x95904cdc,0xbc442eb3,0x5853dd39,
				0xd6d636ea,0x32c1c560,0x1b15a70f,0xff025485,
				0x48bd63d1,0xacaa905b,0x857ef234,0x616901be,
				0x7972217f,0x9d65d2f5,0xb4b1b09a,0x50a64310,
				0xe7197444,0x030e87ce,0x2adae5a1,0xcecd162b,
				0x4048fdf8,0xa45f0e72,0x8d8b6c1d,0x699c9f97,
				0xde23a8c3,0x3a345b49,0x13e03926,0xf7f7caac,
				0x0b079871,0xef106bfb,0xc6c40994,0x22d3fa1e,
				0x956ccd4a,0x717b3ec0,0x58af5caf,0xbcb8af25,
				0x323d44f6,0xd62ab77c,0xfffed513,0x1be92699,
				0xac5611cd,0x4841e247,0x61958028,0x858273a2,
				0x51a3c1aa,0xb5b43220,0x9c60504f,0x7877a3c5,
				0xcfc89491,0x2bdf671b,0x020b0574,0xe61cf6fe,
				0x68991d2d,0x8c8eeea7,0xa55a8cc8,0x414d7f42,
				0xf6f24816,0x12e5bb9c,0x3b31d9f3,0xdf262a79,
				0x23d678a4,0xc7c18b2e,0xee15e941,0x0a021acb,
				0xbdbd2d9f,0x59aade15,0x707ebc7a,0x94694ff0,
				0x1aeca423,0xfefb57a9,0xd72f35c6,0x3338c64c,
				0x8487f118,0x60900292,0x494460fd,0xad539377,
				0xb548b3b6,0x515f403c,0x788b2253,0x9c9cd1d9,
				0x2b23e68d,0xcf341507,0xe6e07768,0x02f784e2,
				0x8c726f31,0x68659cbb,0x41b1fed4,0xa5a60d5e,
				0x12193a0a,0xf60ec980,0xdfdaabef,0x3bcd5865,
				0xc73d0ab8,0x232af932,0x0afe9b5d,0xeee968d7,
				0x59565f83,0xbd41ac09,0x9495ce66,0x70823dec,
				0xfe07d63f,0x1a1025b5,0x33c447da,0xd7d3b450,
				0x606c8304,0x847b708e,0xadaf12e1,0x49b8e16b


		   };
		   for (int n = 0; n < 256; n++) {
			   _mul_table1_336[n] = temp[n];
			   _mul_table1_672[n] = temp2[n];
		   }

	   }
	   else{
		   // i dont have crc instruction
		   //std::cerr << "Intel CRC Does NOT Exist, Computing CRC in Software"<<std::endl;
			_crcParam = false;
			unsigned long c;
			int n, k;

			for (n = 0; n < 256; n++) {
				c = (unsigned long) n;
				for (k = 0; k < 8; k++) {
					if (c & 1)
						c = 0x04C11DB7 ^ (c >> 1); //0xBA0DC66B //0x04C11DB7
					else
						c = c >> 1;
				}
				_crcTable[n] = c;
			}
	   }
   }
#endif
}

System::~System ()
{
   if ( !_delayedStart ) finish();
}

void System::finish ()
{
   if ( _alreadyFinished ) return;

   if ( usingClusterMPI() ) return;

   _alreadyFinished = true;

   //! \note Instrumentation: first removing RUNNING state from top of the state stack
   //! and then pushing SHUTDOWN state in order to instrument this latest phase
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseOpenStateEvent(NANOS_SHUTDOWN) );

   verbose ( "NANOS++ shutting down.... init" );

   //! \note waiting for remaining tasks
   myThread->getCurrentWD()->waitCompletion( true );

   //! \note switching main work descriptor (current) to the main thread to shutdown the runtime 
   if ( _workers[0]->isSleeping() ) {
      if ( !_workers[0]->hasTeam() ) {
         acquireWorker( myThread->getTeam(), _workers[0], true, false, false );
      }
      _workers[0]->wakeup();
   }
   getMyThreadSafe()->getCurrentWD()->tied().tieTo(*_workers[0]);
   Scheduler::switchToThread(_workers[0]);
   BaseThread *mythread = getMyThreadSafe();
   mythread->getTeam()->getSchedulePolicy().atShutdown();

   ensure( mythread->isMainThread(), "Main thread is not finishing the application!");

   ThreadTeam* team = mythread->getTeam();
   while ( !(team->isStable()) ) memoryFence();

   //! \note stopping all threads
   verbose ( "Joining threads..." );
   for ( PEList::iterator it = _pes.begin(); it != _pes.end(); it++ ) {
      it->second->stopAllThreads();
   }
   verbose ( "...thread has been joined" );


   ensure( _schedStats._readyTasks == 0, "Ready task counter has an invalid value!");

   verbose ( "NANOS++ statistics");
   verbose ( std::dec, (unsigned int) getCreatedTasks(),         " tasks have been executed" );

#ifdef NANOS_RESILIENCY_ENABLED
   verbose ( std::dec, error::FailureStats<error::ErrorInjection>::get(),    " errors injected" );
   verbose ( std::dec, error::FailureStats<error::CheckpointFailure>::get(), " tasks could not be initialized (backup failed)" );
   verbose ( std::dec, error::FailureStats<error::ExecutionFailure>::get(),  " task executions failed" );
   verbose ( std::dec, error::FailureStats<error::TaskRecovery>::get(),      " tasks have been reexecuted" );
   verbose ( std::dec, error::FailureStats<error::DiscardedTask>::get(),     " tasks have been discarded (initialization, parent or sibling(s) failed" );
#endif // NANOS_RESILIENCY_ENABLED
   sys.getNetwork()->nodeBarrier();

   if ( usingCluster() ) {
      _net.nodeBarrier();
   }

   for ( unsigned int nodeCount = 0; nodeCount < _net.getNumNodes(); nodeCount += 1 ) {
      if ( _net.getNodeNum() == nodeCount ) {
         for ( ArchitecturePlugins::const_iterator it = _archs.begin(); it != _archs.end(); ++it )
         {
            (*it)->finalize();
         }
#ifdef CLUSTER_DEV
         if ( _net.getNodeNum() == 0 && usingCluster() ) {
            //message("Master: Created " << createdWds << " WDs.");
            //message("Master: Failed to correctly schedule " << sys.getAffinityFailureCount() << " WDs.");
            //int soft_inv = 0;
            //int hard_inv = 0;

            //#ifdef OpenCL_DEV
            //      if ( _opencls ) {
            //         soft_inv = 0;
            //         hard_inv = 0;
            //         for ( unsigned int idx = 1; idx < _opencls->size(); idx += 1 ) {
            //            soft_inv += _separateAddressSpaces[(*_opencls)[idx]->getMemorySpaceId()]->getSoftInvalidationCount();
            //            hard_inv += _separateAddressSpaces[(*_opencls)[idx]->getMemorySpaceId()]->getHardInvalidationCount();
            //            //max_execd_wds = max_execd_wds >= (*_nodes)[idx]->getExecutedWDs() ? max_execd_wds : (*_nodes)[idx]->getExecutedWDs();
            //            //message("Memory space " << idx <<  " has performed " << _separateAddressSpaces[idx]->getSoftInvalidationCount() << " soft invalidations." );
            //            //message("Memory space " << idx <<  " has performed " << _separateAddressSpaces[idx]->getHardInvalidationCount() << " hard invalidations." );
            //         }
            //      }
            //      message("OpenCLs Soft invalidations: " << soft_inv);
            //      message("OpenCLs Hard invalidations: " << hard_inv);
            //#endif
         }
#endif
      }
      if ( usingCluster() ) {
         _net.nodeBarrier();
      }
   }

   //! \note Master leaves team and finalizes thread structures (before insrumentation ends)
   _workers[0]->finish();

   //! \note finalizing instrumentation (if active)
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->finalize() );

   //! \note stopping and deleting the thread manager
   delete _threadManager;

   //! \note stopping and deleting the programming model interface
   _pmInterface->finish();
   delete _pmInterface;

   //! \note deleting pool of locks
   delete[] _lockPool;

   //! \note deleting main work descriptor
   delete ( WorkDescriptor * ) ( mythread->getCurrentWD() );
   delete ( WorkDescriptor * ) &( mythread->getThreadWD() );

   //! \note deleting loaded slicers
   for ( Slicers::const_iterator it = _slicers.begin(); it !=   _slicers.end(); it++ ) {
      delete ( Slicer * )  it->second;
   }

   //! \note deleting loaded worksharings
   for ( WorkSharings::const_iterator it = _worksharings.begin(); it !=   _worksharings.end(); it++ ) {
      delete ( WorkSharing * )  it->second;
   }
   
   //! \note  printing thread team statistics and deleting it
   if ( team->getScheduleData() != NULL ) team->getScheduleData()->printStats();

   ensure(team->size() == 0, "Trying to finish execution, but team is still not empty");
   delete team;

   //! \note deleting processing elements (but main pe)
   for ( PEList::iterator it = _pes.begin(); it != _pes.end(); it++ ) {
      if ( it->first != (unsigned int)mythread->runningOn()->getId() ) {
         delete it->second;
      }
   }
   
   for ( unsigned int idx = 1; idx < _separateMemorySpacesCount; idx += 1 ) {
      delete _separateAddressSpaces[ idx ];
   }
   
   //! \note unload modules
   unloadModules();

   //! \note deleting dependency manager
   delete _dependenciesManager;

   //! \note deleting last processing element
   delete _pes[mythread->runningOn()->getId() ];

   //! \note deleting allocator (if any)
   if ( allocator != NULL ) free (allocator);

   verbose ( "NANOS++ shutting down.... end" );
   //! \note printing execution summary
   if ( _summary ) executionSummary();

   _net.finalize(); //this can call exit (because of GASNet)
}

#ifdef NANOS_RESILIENCY_ENABLED
unsigned int System::getCRC32(uint64_t address)
{
	crcStruct *str = this->_hashmap.find(address);
	if(str==NULL)
	{
		str = (crcStruct*)malloc(sizeof(crcStruct *));
		str->crc1 = 0;
		str->crc2 = 0;
		str->crc3 = 0;
		bool inserted = false;
		_hashmap.insert(address, *str, inserted);
		return 0;
	}
	if(str->crc1==str->crc2){
		return str->crc1;
	}
	else if(str->crc1==str->crc3){
		return str->crc1;
	}
	else {
		return str->crc2;
	}

}

void System::setCRC32(uint64_t address, unsigned int inCrc32 )
{
	crcStruct *str = this->_hashmap.find(address);
	if(str==NULL)
	{
		str = (crcStruct*)malloc(sizeof(crcStruct *));
		str->crc1 = inCrc32;
		str->crc2 = inCrc32;
		str->crc3 = inCrc32;
		bool inserted = false;
		_hashmap.insert(address, *str, inserted);
		return;
	}
	str->crc1 = inCrc32;
	str->crc2 = inCrc32;
	str->crc3 = inCrc32;
}

unsigned int System::compute_Crc32(unsigned int initval, void *buf, size_t bufLen)
{
	   unsigned int crc32 = initval;
	   unsigned char *byteBuf, *byteBuf2;
		size_t i;
		size_t res =  bufLen/1024;
		size_t remainder = bufLen%1024;

		byteBuf = (unsigned char*) buf;
		for (i=0; i < (res*1024); i+=1024) {
			crc32 = subcompute_Crc32(crc32, &byteBuf[i]);
		}
		byteBuf2 = (unsigned char*) &byteBuf[res*1024];
		for (i=0; i < remainder; i++) {
			crc32 = _mm_crc32_u8(crc32, byteBuf2[i]);
		}
		return crc32;
}

unsigned int System::subcompute_Crc32(long long unsigned int initval, void *buffer){
    uint64_t crc0, crc1, crc2, tmp; 
    uint64_t *p_buffer;
    uint64_t BLOCKSIZE = 336;
    uint64_t BLOCKSIZE8 = (BLOCKSIZE / 8);

    p_buffer = (uint64_t *)buffer;
    crc1 = crc2 = 0;

    // Do first 8 bytes here for better pipelining
    crc0 = _mm_crc32_u64((uint64_t)initval, p_buffer[0]);

    for(int i = 0; i < 42; i++){
      crc1 = _mm_crc32_u64(crc1, (uint64_t)(p_buffer[1 + 1*BLOCKSIZE8 + i]));
      crc2 = _mm_crc32_u64(crc2, (uint64_t)(p_buffer[1 + 2*BLOCKSIZE8 + i]));
      crc0 = _mm_crc32_u64(crc0, (uint64_t)(p_buffer[1 + 0*BLOCKSIZE8 + i]));
    }

    // merge in crc1
    tmp  = p_buffer[127];
    tmp ^= _mul_table1_336[crc1 & 0xFF];
    tmp ^= (_mul_table1_336[(crc1 >>  8) & 0xFF]) <<  8;
    tmp ^= (_mul_table1_336[(crc1 >> 16) & 0xFF]) << 16;
    tmp ^= (_mul_table1_336[(crc1 >> 24) & 0xFF]) << 24;

    // merge in crc0
    tmp ^= _mul_table1_672[crc0 & 0xFF];
    tmp ^= (_mul_table1_672[(crc0 >>  8) & 0xFF]) <<  8;
    tmp ^= (_mul_table1_672[(crc0 >> 16) & 0xFF]) << 16;
    tmp ^= (_mul_table1_672[(crc0 >> 24) & 0xFF]) << 24;

    return (unsigned int) _mm_crc32_u64(crc2, (uint64_t)tmp);
}

unsigned int System::compute_Crc32_Software(unsigned int initval, void *buf, size_t bufLen)
{

    unsigned int crc32 = initval;
    unsigned char *byteBuf;
    size_t i;

    // accumulate crc32 for buffer
    //crc32 = inCrc32 ^ 0xFFFFFFFF;
    byteBuf = (unsigned char*) buf;
    for (i=0; i < bufLen; i++) {
    		crc32 = (crc32 >> 8) ^ _crcTable[ (crc32 ^ (byteBuf[i])) & 0xFF ];

    }
    return( crc32 ^ 0xFFFFFFFF );
}

void System::startComputeCRC(WD &wd){
	if(_crcParam){
		for (unsigned int index = 0; index < wd.getNumCopies(); index++) {
			if (wd.getCopies()[index].isOutput()) {
				CopyData cd = wd.getCopies()[index];
				unsigned int inCrc32 = 0xFFFFFFFF;
				size_t len = cd.getSize();
				void *buffer = (void*) cd.getAddress();////
				uint64_t key = cd.getAddress().value();
				if(cd.getNumDimensions()>1){
					for (unsigned int i = 0; i < cd.getDimensions()[1].accessed_length; i += 1){
							uint64_t address = cd.getAddress().value() + cd.getDimensions()[0].accessed_length * i;
							buffer = (void *) address;
							len =  cd.getDimensions()[0].accessed_length;
							inCrc32 = sys.compute_Crc32(inCrc32, buffer, len);
					}
				}
				else{
					inCrc32 = sys.compute_Crc32(inCrc32, buffer, len);
				}
				setCRC32(key, inCrc32);
				//std::cerr << "Computed CRC "<<inCrc32<<std::endl;
			}
		}
	}
	else{
		for (unsigned int index = 0; index < wd.getNumCopies(); index++) {
			if (wd.getCopies()[index].isOutput()) {
				CopyData cd = wd.getCopies()[index];
				unsigned int inCrc32 = 0xFFFFFFFF;
				size_t len = cd.getSize();
				void *buffer = (void*) cd.getAddress();////
				uint64_t key = cd.getAddress().value();
				if(cd.getNumDimensions()>1){
					for (unsigned int i = 0; i < cd.getDimensions()[1].accessed_length; i += 1){
							uint64_t address = cd.getAddress().value() + cd.getDimensions()[0].accessed_length * i;
							buffer = (void *) address;
							len =  cd.getDimensions()[0].accessed_length;
							inCrc32 = sys.compute_Crc32_Software(inCrc32, buffer, len);
					}
				}
				else{
					inCrc32 = sys.compute_Crc32_Software(inCrc32, buffer, len);
				}
				setCRC32(key, inCrc32);
				//std::cerr << "Computed CRC "<<inCrc32<<std::endl;
			}
		}
	}

}

bool System::checkSDCviaCRC32(WD &wd){
	bool result = false;
	if(_crcParam){
		for (unsigned int index = 0; index < wd.getNumCopies(); index++) {
			if (wd.getCopies()[index].isInput()) {
				CopyData cd = wd.getCopies()[index];
				unsigned int computedCRC = 0xFFFFFFFF;
				size_t len = cd.getSize();
				void *buffer = (void*) cd.getAddress();////
				uint64_t key = cd.getAddress().value();
				if(cd.getNumDimensions()>1){
					for (unsigned int i = 0; i < cd.getDimensions()[1].accessed_length; i += 1){
							uint64_t address = cd.getAddress().value() + cd.getDimensions()[0].accessed_length * i;
							buffer = (void *) address;
							len =  cd.getDimensions()[0].accessed_length;
							computedCRC = sys.compute_Crc32(computedCRC, buffer, len);
					}
				}else{
					computedCRC = sys.compute_Crc32(computedCRC, buffer, len);
				}
				unsigned int storedCRC = getCRC32(key);
				//std::cerr << "Computed CRC "<<computedCRC << " vs. Stored CRC "<<storedCRC<<std::endl;
				if(storedCRC == 0){
					setCRC32(key,computedCRC);
					continue;
				}
				if(storedCRC != computedCRC){
					setCRC32(key, computedCRC);
					result = true;
				}
			}
		}
	}
	else{
			for (unsigned int index = 0; index < wd.getNumCopies(); index++) {
					if (wd.getCopies()[index].isInput()) {
						CopyData cd = wd.getCopies()[index];
						unsigned int computedCRC = 0xFFFFFFFF;
						size_t len = cd.getSize();
						void *buffer = (void*) cd.getAddress();
						uint64_t key = cd.getAddress().value();
						if(cd.getNumDimensions()>1){
							for (unsigned int i = 0; i < cd.getDimensions()[1].accessed_length; i += 1){
									uint64_t address = cd.getAddress().value() + cd.getDimensions()[0].accessed_length * i;
									buffer = (void *) address;
									len =  cd.getDimensions()[0].accessed_length;
									computedCRC = sys.compute_Crc32_Software(computedCRC, buffer, len);
							}
						}else{
							computedCRC = sys.compute_Crc32_Software(computedCRC, buffer, len);
						}
						unsigned int storedCRC = getCRC32(key);
						//std::cerr << "Computed CRC "<<computedCRC << " vs. Stored CRC "<<storedCRC<<std::endl;
						if(storedCRC == 0){
							setCRC32(key,computedCRC);
							continue;
						}
						if(storedCRC != computedCRC){
							setCRC32(key, computedCRC);
							result = true;
						}
					}
				}
	}
	//std::cerr <<"Does SDC Check Pass "<<result<<std::endl;
	return result;
}

void System::restore(WD &wd) {
	//std::cerr << "Restoring Task "<<wd.getId()<<std::endl;
   debug ( "Resiliency CRC: Task ", wd.getId(), " is being recovered to be re-executed further on.");
   // Wait for successors to finish.
   //wd.waitCompletion();

   // Restore the data
   wd._mcontrol.restoreBackupData();

   while ( !wd._mcontrol.isDataRestored( wd ) ) {
      myThread->idle();
   }
   debug ( "Resiliency: Task ", wd.getId(), " recovery complete.");
   // Reset invalid state
   //wd.setInvalid(false);
}

#endif

/*! \brief Creates a new WD
 *
 *  This function creates a new WD, allocating memory space for device ptrs and
 *  data when necessary. 
 *
 *  \param [in,out] uwd is the related addr for WD if this parameter is null the
 *                  system will allocate space in memory for the new WD
 *  \param [in] num_devices is the number of related devices
 *  \param [in] devices is a vector of device descriptors 
 *  \param [in] data_size is the size of the related data
 *  \param [in,out] data is the related data (allocated if needed)
 *  \param [in] uwg work group to relate with
 *  \param [in] props new WD properties
 *  \param [in] num_copies is the number of copy objects of the WD
 *  \param [in] copies is vector of copy objects of the WD
 *  \param [in] num_dimensions is the number of dimension objects associated to the copies
 *  \param [in] dimensions is vector of dimension objects
 *
 *  When it does a full allocation the layout is the following:
 *  <pre>
 *  +---------------+
 *  |     WD        |
 *  +---------------+
 *  |    data       |
 *  +---------------+
 *  |  dev_ptr[0]   |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |  dev_ptr[N]   |
 *  +---------------+
 *  |     DD0       |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     DDN       |
 *  +---------------+
 *  |    copy0      |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |    copyM      |
 *  +---------------+
 *  |     dim0      |
 *  +---------------+
 *  |     ....      |
 *  +---------------+
 *  |     dimM      |
 *  +---------------+
 *  |   PM Data     |
 *  +---------------+
 *  </pre>
 */
void System::createWD ( WD **uwd, size_t num_devices, nanos_device_t *devices, size_t data_size, size_t data_align,
                        void **data, WD *uwg, nanos_wd_props_t *props, nanos_wd_dyn_props_t *dyn_props,
                        size_t num_copies, nanos_copy_data_t **copies, size_t num_dimensions,
                        nanos_region_dimension_internal_t **dimensions, nanos_translate_args_t translate_args,
                        const char *description, Slicer *slicer )
{
   ensure( num_devices > 0, "WorkDescriptor has no devices" );

   unsigned int i;
   char *chunk = 0;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs, size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t offset_Sched;
   size_t total_size;

   // WD doesn't need to compute offset, it will always be the chunk allocated address

   // Computing Data info
   size_Data = (data != NULL && *data == NULL)? data_size:0;
   if ( *uwd == NULL ) offset_Data = NANOS_ALIGNED_MEMORY_OFFSET(0, sizeof(WD), data_align );
   else offset_Data = 0; // if there are no wd allocated, it will always be the chunk allocated address

   // Computing Data Device pointers and Data Devicesinfo
   size_DPtrs    = sizeof(DD *) * num_devices;
   offset_DPtrs  = NANOS_ALIGNED_MEMORY_OFFSET(offset_Data, size_Data, __alignof__( DD*) );

   // Computing Copies info
   if ( num_copies != 0 ) {
      size_CopyData = sizeof(CopyData);
      size_Copies   = size_CopyData * num_copies;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, __alignof__(nanos_copy_data_t) );
      // There must be at least 1 dimension entry
      size_Dimensions = num_dimensions * sizeof(nanos_region_dimension_internal_t);
      offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, __alignof__(nanos_region_dimension_internal_t) );
   } else {
      size_Copies = 0;
      // No dimensions
      size_Dimensions = 0;
      offset_Copies = offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, 1);
   }

   // Computing Internal Data info and total size
   static size_t size_PMD   = _pmInterface->getInternalDataSize();
   if ( size_PMD != 0 ) {
      static size_t align_PMD = _pmInterface->getInternalDataAlignment();
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_Dimensions, size_Dimensions, align_PMD );
   } else {
      offset_PMD = offset_Dimensions;
      size_PMD = size_Dimensions;
   }
   
   // Compute Scheduling Data size
   static size_t size_Sched = _defSchedulePolicy->getWDDataSize();
   if ( size_Sched != 0 )
   {
      static size_t align_Sched =  _defSchedulePolicy->getWDDataAlignment();
      offset_Sched = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD, size_PMD, align_Sched );
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_Sched,size_Sched,1);
   }
   else
   {
      offset_Sched = offset_PMD; // Needed by compiler unused variable error
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD,size_PMD,1);
   }

   chunk = NEW char[total_size];
   if ( props != NULL ) {
      if (props->clear_chunk)
          memset(chunk, 0, sizeof(char) * total_size);
   }

   // allocating WD and DATA
   if ( *uwd == NULL ) *uwd = (WD *) chunk;
   if ( data != NULL && *data == NULL ) *data = (chunk + offset_Data);

   // allocating Device Data
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_devices ; i ++ ) dev_ptrs[i] = ( DD* ) devices[i].factory( devices[i].arg );

   //std::cerr << "num_copies=" << num_copies <<" copies=" <<copies << " num_dimensions=" <<num_dimensions << " dimensions=" << dimensions<< std::endl;
   //ensure ((num_copies==0 && copies==NULL && num_dimensions==0 && dimensions==NULL) || (num_copies!=0 && copies!=NULL && num_dimensions!=0 && dimensions!=NULL ), "Number of copies and copy data conflict" );
   ensure ((num_copies==0 && copies==NULL && num_dimensions==0 /*&& dimensions==NULL*/ ) || (num_copies!=0 && copies!=NULL && num_dimensions!=0 && dimensions!=NULL ), "Number of copies and copy data conflict" );
   

   // allocating copy-ins/copy-outs
   if ( copies != NULL && *copies == NULL ) {
      *copies = ( CopyData * ) (chunk + offset_Copies);
      ::bzero(*copies, size_Copies);
      *dimensions = ( nanos_region_dimension_internal_t * ) ( chunk + offset_Dimensions );
   }

   WD * wd;
   wd =  new (*uwd) WD( num_devices, dev_ptrs, data_size, data_align, data != NULL ? *data : NULL,
                        num_copies, (copies != NULL)? *copies : NULL, translate_args, description );

   if ( slicer ) wd->setSlicer(slicer);

   // Set WD's socket
   wd->setNUMANode( sys.getUserDefinedNUMANode() );
   
   // Set total size
   wd->setTotalSize(total_size );
   
   if ( wd->getNUMANode() >= (int)sys.getNumNumaNodes() )
      throw NANOS_INVALID_PARAM;

   // All the implementations for a given task will have the same ID
   wd->setVersionGroupId( ( unsigned long ) devices );

   // initializing internal data
   if ( size_PMD > 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      wd->setInternalData( chunk + offset_PMD );
   }
   
   // Create Scheduling data
   if ( size_Sched > 0 ){
      _defSchedulePolicy->initWDData( chunk + offset_Sched );
      ScheduleWDData * sched_Data = reinterpret_cast<ScheduleWDData*>( chunk + offset_Sched );
      wd->setSchedulerData( sched_Data, /*ownedByWD*/ false );
   }

   // add to workdescriptor
   if ( uwg != NULL ) {
      WD * wg = ( WD * )uwg;
      wg->addWork( *wd );
   }

   // set properties
   if ( props != NULL ) {
      if ( props->tied ) wd->tied();
   }

   // Set dynamic properties
   if ( dyn_props != NULL ) {
      wd->setPriority( dyn_props->priority );
      wd->setFinal ( dyn_props->flags.is_final );
      wd->setRecoverable ( dyn_props->flags.is_recover);
      if ( dyn_props->flags.is_implicit ) wd->setImplicit();
      wd->setCallback(dyn_props->callback);
      wd->setArguments(dyn_props->arguments);

   }

   if ( dyn_props && dyn_props->tie_to ) wd->tieTo( *( BaseThread * )dyn_props->tie_to );
   
   /* DLB */
   // In case the master have been busy crating tasks 
   // every 10 tasks created I'll check if I must return claimed cpus
   // or there are available cpus idle
   if(_atomicWDSeed.value()%10==0){
      _threadManager->returnClaimedCpus();
      _threadManager->acquireResourcesIfNeeded();
   }

   if (_createLocalTasks) {
      wd->tieToLocation( 0 );
   }

   //Copy reduction data from parent
   if (uwg) wd->copyReductions((WorkDescriptor *)uwg);
}

/*! \brief Duplicates the whole structure for a given WD
 *
 *  \param [out] uwd is the target addr for the new WD
 *  \param [in] wd is the former WD
 *
 *  \return void
 *
 *  \par Description:
 *
 *  This function duplicates the given WD passed as a parameter copying all the
 *  related data included in the layout (devices ptr, data and DD). First it computes
 *  the size for the layout, then it duplicates each one of the chunks (Data,
 *  Device's pointers, internal data, etc). Finally calls WorkDescriptor constructor
 *  using new and placement.
 *
 *  \sa WorkDescriptor, createWD 
 */
void System::duplicateWD ( WD **uwd, WD *wd)
{
   unsigned int i, num_Devices, num_Copies, num_Dimensions;
   DeviceData **dev_data;
   void *data = NULL;
   char *chunk = 0, *chunk_iter;

   size_t size_CopyData;
   size_t size_Data, offset_Data, size_DPtrs, offset_DPtrs, size_Copies, offset_Copies, size_Dimensions, offset_Dimensions, offset_PMD;
   size_t offset_Sched;
   size_t total_size;

   // WD doesn't need to compute offset, it will always be the chunk allocated address

   // Computing Data info
   size_Data = wd->getDataSize();
   if ( *uwd == NULL ) offset_Data = NANOS_ALIGNED_MEMORY_OFFSET(0, sizeof(WD), wd->getDataAlignment() );
   else offset_Data = 0; // if there are no wd allocated, it will always be the chunk allocated address

   // Computing Data Device pointers and Data Devicesinfo
   num_Devices = wd->getNumDevices();
   dev_data = wd->getDevices();
   size_DPtrs    = sizeof(DD *) * num_Devices;
   offset_DPtrs  = NANOS_ALIGNED_MEMORY_OFFSET(offset_Data, size_Data, __alignof__( DD*) );

   // Computing Copies info
   num_Copies = wd->getNumCopies();
   num_Dimensions = 0;
   for ( i = 0; i < num_Copies; i += 1 ) {
      num_Dimensions += wd->getCopies()[i].getNumDimensions();
   }
   if ( num_Copies != 0 ) {
      size_CopyData = sizeof(CopyData);
      size_Copies   = size_CopyData * num_Copies;
      offset_Copies = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, __alignof__(nanos_copy_data_t) );
      // There must be at least 1 dimension entry
      size_Dimensions = num_Dimensions * sizeof(nanos_region_dimension_internal_t);
      offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_Copies, size_Copies, __alignof__(nanos_region_dimension_internal_t) );
   } else {
      size_Copies = 0;
      // No dimensions
      size_Dimensions = 0;
      offset_Copies = offset_Dimensions = NANOS_ALIGNED_MEMORY_OFFSET(offset_DPtrs, size_DPtrs, 1);
   }

   // Computing Internal Data info and total size
   static size_t size_PMD   = _pmInterface->getInternalDataSize();
   if ( size_PMD != 0 ) {
      static size_t align_PMD = _pmInterface->getInternalDataAlignment();
      offset_PMD = NANOS_ALIGNED_MEMORY_OFFSET(offset_Dimensions, size_Dimensions, align_PMD);
   } else {
      offset_PMD = offset_Copies;
      size_PMD = size_Copies;
   }

   // Compute Scheduling Data size
   static size_t size_Sched = _defSchedulePolicy->getWDDataSize();
   if ( size_Sched != 0 )
   {
      static size_t align_Sched =  _defSchedulePolicy->getWDDataAlignment();
      offset_Sched = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD, size_PMD, align_Sched );
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_Sched,size_Sched,1);
   }
   else
   {
      offset_Sched = offset_PMD; // Needed by compiler unused variable error
      total_size = NANOS_ALIGNED_MEMORY_OFFSET(offset_PMD,size_PMD,1);
   }

   chunk = NEW char[total_size];

   // allocating WD and DATA; if size_Data == 0 data keep the NULL value
   if ( *uwd == NULL ) *uwd = (WD *) chunk;
   if ( size_Data != 0 ) {
      data = chunk + offset_Data;
      memcpy ( data, wd->getData(), size_Data );
   }

   // allocating Device Data
   DD **dev_ptrs = ( DD ** ) (chunk + offset_DPtrs);
   for ( i = 0 ; i < num_Devices; i ++ ) {
      dev_ptrs[i] = dev_data[i]->clone();
   }

   // allocate copy-in/copy-outs
   CopyData *wdCopies = ( CopyData * ) (chunk + offset_Copies);
   chunk_iter = chunk + offset_Copies;
   nanos_region_dimension_internal_t *dimensions = ( nanos_region_dimension_internal_t * ) ( chunk + offset_Dimensions );
   for ( i = 0; i < num_Copies; i++ ) {
      CopyData *wdCopiesCurr = ( CopyData * ) chunk_iter;
      *wdCopiesCurr = wd->getCopies()[i];
      memcpy( dimensions, wd->getCopies()[i].getDimensions(), sizeof( nanos_region_dimension_internal_t ) * wd->getCopies()[i].getNumDimensions() );
      wdCopiesCurr->setDimensions( dimensions );
      dimensions += wd->getCopies()[i].getNumDimensions();
      chunk_iter += size_CopyData;
   }

   // creating new WD 
   //FIXME jbueno (#758) should we have to take into account dimensions?
   new (*uwd) WD( *wd, dev_ptrs, wdCopies, data );

   // Set total size
   (*uwd)->setTotalSize(total_size );
   
   // initializing internal data
   if ( size_PMD != 0) {
      _pmInterface->initInternalData( chunk + offset_PMD );
      (*uwd)->setInternalData( chunk + offset_PMD );
      memcpy ( chunk + offset_PMD, wd->getInternalData(), size_PMD );
   }
   
   // Create Scheduling data
   if ( size_Sched > 0 ){
      _defSchedulePolicy->initWDData( chunk + offset_Sched );
      ScheduleWDData * sched_Data = reinterpret_cast<ScheduleWDData*>( chunk + offset_Sched );
      (*uwd)->setSchedulerData( sched_Data, /*ownedByWD*/ false );
   }
}

void System::setupWD ( WD &work, WD *parent )
{
   work.setDepth( parent->getDepth() +1 );
   
   // Inherit priority
   if ( parent != NULL ){
      // Add the specified priority to its parent's
      work.setPriority( work.getPriority() + parent->getPriority() );
   }

   /**************************************************/
   /*********** selective node executuion ************/
   /**************************************************/
   //if (_net.getNodeNum() == 0) work.tieTo(*_workers[ 1 + nanos::ext::GPUConfig::getGPUCount() + ( work.getId() % ( _net.getNumNodes() - 1 ) ) ]);
   /**************************************************/
   /**************************************************/

   //  ext::SMPDD * workDD = dynamic_cast<ext::SMPDD *>( &work.getActiveDevice());
   //if (_net.getNodeNum() == 0)
   //         std::cerr << "wd " << work.getId() << " depth is: " << work.getDepth() << " @func: " << (void *) workDD->getWorkFct() << std::endl;
#if 0
#ifdef CLUSTER_DEV
   if (_net.getNodeNum() == 0)
   {
      //std::cerr << "tie wd " << work.getId() << " to my thread" << std::endl;
      //ext::SMPDD * workDD = dynamic_cast<ext::SMPDD *>( &work.getActiveDevice());
      switch ( work.getDepth() )
      {
         //case 1:
         //   //std::cerr << "tie wd " << work.getId() << " to my thread, @func: " << (void *) workDD->getWorkFct() << std::endl;
         //   work.tieTo( *myThread );
         //   break;
         //case 1:
            //if (work.canRunIn( ext::GPU) )
            //{
            //   work.tieTo( *_masterGpuThd );
            //}
         //   break;
         default:
            break;
            std::cerr << "wd " << work.getId() << " depth is: " << work.getDepth() << " @func: " << (void *) workDD->getWorkFct() << std::endl;
      }
   }
#endif
#endif
   // Prepare private copy structures to use relative addresses
   work.prepareCopies();

   // Invoke pmInterface
   
   _pmInterface->setupWD(work);
   Scheduler::updateCreateStats(work);
}

void System::submit ( WD &work )
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_SUBMIT );

/*
   if (_net.getNodeNum() > 0 ) setupWD( work, getSlaveParentWD() );
   else setupWD( work, myThread->getCurrentWD() );
*/

   work.submit();
}

/*! \brief Submit WorkDescriptor to its parent's  dependencies domain
 */
void System::submitWithDependencies (WD& work, size_t numDataAccesses, DataAccess* dataAccesses)
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_SUBMIT_WITH_DEPENDENCIES );
/*
   setupWD( work, myThread->getCurrentWD() );
*/
   WD *current = myThread->getCurrentWD(); 
   current->submitWithDependencies( work, numDataAccesses , dataAccesses);
}

/*! \brief Wait on the current WorkDescriptor's domain for some dependenices to be satisfied
 */
void System::waitOn( size_t numDataAccesses, DataAccess* dataAccesses )
{
   WD* current = myThread->getCurrentWD();
   current->waitOn( numDataAccesses, dataAccesses );
}

void System::inlineWork ( WD &work )
{
   SchedulePolicy* policy = getDefaultSchedulePolicy();
   policy->onSystemSubmit( work, SchedulePolicy::SYS_INLINE_WORK );
   //! \todo choose actual (active) device...
   if ( Scheduler::checkBasicConstraints( work, *myThread ) ) {
      work._mcontrol.preInit();
      work._mcontrol.initialize( *( myThread->runningOn() ) );
      bool result;
      do {
         result = work._mcontrol.allocateTaskMemory();
         if ( !result ) {
            myThread->idle();
         }
      } while( result == false );
      Scheduler::inlineWork( &work, /*schedule*/ false );
   }
   else fatal ("System: Trying to execute inline a task violating basic constraints");
}

/* \brief Returns an unocupied worker
 *
 * This function is called when creating a team. We must look for teamless workers and
 * meet the coditions:
 *    - If binding is enabled, the thread must be running on an Active PE
 *    - The thread must not have team, nor nextTeam
 *    - The thread must be either running and idling, or blocked.
 *
 */
BaseThread * System::getUnassignedWorker ( void )
{
   BaseThread *thread;

   for ( ThreadList::iterator it = _workers.begin(); it != _workers.end(); it++ ) {
      thread = it->second;

      // skip iteration if binding is enabled and the thread is running on a deactivated CPU
      bool cpu_active = thread->runningOn()->isActive();
      if ( _smpPlugin->getBinding() && !cpu_active ) {
         continue;
      }

      thread->lock();
      if ( !thread->hasTeam() && !thread->getNextTeam() ) {

         // Thread may be idle and running or blocked but its CPU is active
         if ( !thread->isSleeping() || thread->runningOn()->isActive() ) {
            thread->reserve(); // set team flag only
            thread->unlock();
            return thread;
         }
      }
      thread->unlock();
   }

   //! \note If no thread has found, return NULL.
   return NULL;
}

BaseThread * System::getWorker ( unsigned int n )
{
   BaseThread *worker = NULL;
   ThreadList::iterator elem = _workers.find( n );
   if ( elem != _workers.end() ) {
      worker = elem->second;
   }
   return worker;
}

void System::acquireWorker ( ThreadTeam * team, BaseThread * thread, bool enter, bool star, bool creator )
{
   int thId = team->addThread( thread, star, creator );
   TeamData *data = NEW TeamData();
   if ( creator ) data->setCreator( true );

   data->setStar(star);

   SchedulePolicy &sched = team->getSchedulePolicy();
   ScheduleThreadData *sthdata = 0;
   if ( sched.getThreadDataSize() > 0 )
      sthdata = sched.createThreadData();

   data->setId(thId);
   data->setTeam(team);
   data->setScheduleData(sthdata);
   if ( creator )
      data->setParentTeamData(thread->getTeamData());

   if ( enter ) thread->enterTeam( data );
   else thread->setNextTeamData( data );

   debug( "added thread ", thread,
          " with id ", thId,
          " to ", team
        );
}

int System::getNumWorkers( DeviceData *arch )
{
   int n = 0;

   for ( ThreadList::iterator it = _workers.begin(); it != _workers.end(); it++ ) {
      if ( it->second->runningOn()->supports( *arch->getDevice() ) ) {
         n++;
      }
   }
   return n;
}

int System::getNumThreads( void ) const
{
   int n = 0;
   n = _smpPlugin->getNumThreads();
   return n;
}

ThreadTeam * System::createTeam ( unsigned nthreads, void *constraints, bool reuse, bool enter, bool parallel )
{
   //! \note Getting default scheduler
   SchedulePolicy *sched = sys.getDefaultSchedulePolicy();

   //! \note Getting scheduler team data (if any)
   ScheduleTeamData *std = ( sched->getTeamDataSize() > 0 )? sched->createTeamData() : NULL;

   //! \note create team object
   ThreadTeam * team = NEW ThreadTeam( nthreads, *sched, std, *_defBarrFactory(), *(_pmInterface->getThreadTeamData()),
                                       reuse? myThread->getTeam() : NULL );

   debug( "Creating team ", team,
          " of ", nthreads, " threads"
        );

   unsigned int remaining_threads = nthreads;

   //! \note Reusing current thread
   if ( reuse ) {
      acquireWorker( team, myThread, /* enter */ enter, /* staring */ true, /* creator */ true );
      remaining_threads--;
   }

   //! \note Getting rest of the members 
   while ( remaining_threads > 0 ) {

      BaseThread *thread = getUnassignedWorker();
      // Check if we don't have a worker because it needs to be created
      if ( !thread && _workers.size() < nthreads ) {
         _smpPlugin->createWorker( _workers );
         continue;
      }
      ensure( thread != NULL, "I could not get the required threads to create the team");

      thread->lock();
      acquireWorker( team, thread, /*enter*/ enter, /* staring */ parallel, /* creator */ false );
      thread->setNextTeam( NULL );
      thread->wakeup();
      thread->unlock();

      remaining_threads--;
   }

   team->init();

   return team;
}

void System::endTeam ( ThreadTeam *team )
{
   debug("Destroying thread team ", team,
         " with size ", team->size()
        );

   /* For OpenMP applications
      At the end of the parallel return the claimed cpus
   */
   _threadManager->returnClaimedCpus();
   while ( team->size ( ) > 0 ) {
      // FIXME: Is it really necessary?
      memoryFence();
   }
   while ( team->getFinalSize ( ) > 0 ) {
      // FIXME: Is it really necessary?
      memoryFence();
   }
   
   fatal_cond( team->size() > 0, "Trying to end a team with running threads");
   
   delete team;
}

void System::waitUntilThreadsPaused ()
{
   // Wait until all threads are paused
   _pausedThreadsCond.wait();
}

void System::waitUntilThreadsUnpaused ()
{
   // Wait until all threads are paused
   _unpausedThreadsCond.wait();
}
 
void System::addPEsAndThreadsToTeam(PE **pes, int num_pes, BaseThread** threads, int num_threads) {  
    //Insert PEs to the team
    for (int i=0; i<num_pes; i++){
        _pes.insert( std::make_pair( pes[i]->getId(), pes[i] ) );
    }
    //Insert the workers to the team
    for (int i=0; i<num_threads; i++){
        _workers.insert( std::make_pair( threads[i]->getId(), threads[i] ) );
        acquireWorker( _mainTeam , threads[i] );
    }
}

void System::environmentSummary( void )
{
   /* Get Prog. Model string */
   std::string prog_model;
   switch ( getInitialMode() )
   {
      case POOL:
         prog_model = "OmpSs";
         break;
      case ONE_THREAD:
         prog_model = "OpenMP";
         break;
      default:
         prog_model = "Unknown";
         break;
   }

   message( "========== Nanos++ Initial Environment Summary ==========" );
   message( "=== PID:                 ", getpid() );
   message( "=== Num. worker threads: ", _workers.size() );
   message( "=== System CPUs:         ", _smpPlugin->getBindingMaskString() );
   message( "=== Binding:             ", std::boolalpha, _smpPlugin->getBinding() );
   message( "=== Prog. Model:         ", prog_model );
   message( "=== Priorities:          ", (getPrioritiesNeeded() ? "Needed" : "Not needed"), " / ", ( _defSchedulePolicy->usingPriorities() ? "enabled" : "disabled" ) );

   for ( ArchitecturePlugins::const_iterator it = _archs.begin();
        it != _archs.end(); ++it ) {
      message( "=== Plugin:              ", (*it)->getName() );
      message( "===  | PEs:              ", (*it)->getNumPEs() );
      message( "===  | Threads:          ", (*it)->getNumThreads() );
      message( "===  | Worker Threads:   ", (*it)->getNumWorkers() );
   }
#ifdef NANOS_RESILIENCY_ENABLED
   message( "=== Runtime resiliency:  ", ( sys.isResiliencyEnabled()? "Enabled": "Disabled") );
#else
   message( "=== Runtime resiliency:  Disabled" );
#endif
   NANOS_INSTRUMENT ( sys.getInstrumentation()->getInstrumentationDictionary()->printEventVerbosity(); )

   message( "=========================================================" );

   // Get start time
   _summaryStartTime = time(NULL);
}

void System::executionSummary( void )
{
   time_t seconds = time(NULL) -_summaryStartTime;
   message( "============ Nanos++ Final Execution Summary ==================" );
   message( "=== Application ended in ", seconds, " seconds" );
   message( "=== ", std::dec, getCreatedTasks(),         " tasks have been executed" );
#ifdef NANOS_RESILIENCY_ENABLED
   message( "=== ", std::dec, error::FailureStats<error::ErrorInjection>::get(),    " errors injected" );
   message( "=== ", std::dec, error::FailureStats<error::CheckpointFailure>::get(), " tasks could not be initialized (backup failed)" );
   message( "=== ", std::dec, error::FailureStats<error::ExecutionFailure>::get(),  " task executions failed" );
   message( "=== ", std::dec, error::FailureStats<error::TaskRecovery>::get(),      " tasks have been reexecuted" );
   message( "=== ", std::dec, error::FailureStats<error::DiscardedTask>::get(),     " tasks have been discarded (initialization, parent or sibling(s) failed" );
#endif // NANOS_RESILIENCY_ENABLED
   message( "===============================================================" );
}

//If someone needs argc and argv, it may be possible, but then a fortran 
//main should be done too
void System::ompss_nanox_main(void *addr, const char* file, int line){
    #ifdef MPI_DEV
    if (getenv("OMPSS_OFFLOAD_SLAVE")){
        //Plugin->init of MPI will do everything and then exit(0)
        sys.loadPlugin("arch-mpi");
    }
    #endif
    #ifdef CLUSTER_DEV
    nanos::ext::ClusterNode::clusterWorker();
    #endif
    
    #ifdef NANOS_RESILIENCY_ENABLED
    if(sys.isResiliencyEnabled()) {
       // Register signal handlers again.
       // May be necessary if some other library overloads
       // our handler on initialization (e.g. Fortran)
       using namespace nanos::error;
       SignalTranslator<OperationFailure> operationFailureTranslator;
    }
    #endif

#ifdef NANOS_INSTRUMENTATION_ENABLED
   _mainFunctionEvent = new instrumentation::MainFunctionEvent( addr, file, line );
#endif
}

void System::ompss_nanox_main_end()
{
#ifdef NANOS_INSTRUMENTATION_ENABLED
   ensure( _mainFunctionEvent != NULL, "Calling ompss_nanox_main_end() before ompss_nanox_main()" );
   delete _mainFunctionEvent;
#endif
}

global_reg_t System::_registerMemoryChunk(void *addr, std::size_t len) {
   CopyData cd;
   nanos_region_dimension_internal_t dim;
   dim.lower_bound = 0;
   dim.size = len;
   dim.accessed_length = len;
   cd.setBaseAddress( addr );
   cd.setDimensions( &dim );
   cd.setNumDimensions( 1 );
   global_reg_t reg;
   getHostMemory().getRegionId( cd, reg, *((WD *) 0), 0 );
   return reg;
}

global_reg_t System::_registerMemoryChunk_2dim(void *addr, std::size_t rows, std::size_t cols, std::size_t elem_size) {
   CopyData cd;
   nanos_region_dimension_internal_t dim[2];
   dim[0].lower_bound = 0;
   dim[0].size = cols * elem_size;
   dim[0].accessed_length = cols * elem_size;
   dim[1].lower_bound = 0;
   dim[1].size = rows;
   dim[1].accessed_length = rows;
   cd.setBaseAddress( addr );
   cd.setDimensions( &dim[0] );
   cd.setNumDimensions( 2 );
   global_reg_t reg;
   getHostMemory().getRegionId( cd, reg, *((WD *) 0), 0 );
   return reg;
}

void System::_distributeObject( global_reg_t &reg, unsigned int start_node, std::size_t num_nodes ) {
   CopyData cd;
   std::size_t num_dims = reg.getNumDimensions();
   nanos_region_dimension_internal_t dims[num_dims];
   cd.setBaseAddress( (void *) reg.getRealFirstAddress() );
   cd.setDimensions( &dims[0] );
   cd.setNumDimensions( 2 );
   reg.fillDimensionData( dims );
   std::size_t size_per_node = dims[ num_dims-1 ].size / num_nodes;
   std::size_t rest_size = dims[ num_dims -1 ].size % num_nodes;

   std::size_t assigned_size = 0;
   for ( std::size_t node_idx = 0; node_idx < num_nodes; node_idx += 1 ) {
      dims[ num_dims-1 ].lower_bound = assigned_size;
      dims[ num_dims-1 ].accessed_length = size_per_node + (node_idx < rest_size);
      assigned_size += size_per_node + (node_idx < rest_size);
      global_reg_t fragmented_reg;
      getHostMemory().getRegionId( cd, fragmented_reg, *((WD *) 0), 0 );
      std::cerr << "fragment " << node_idx << " is "; fragmented_reg.key->printRegion(std::cerr, fragmented_reg.id); std::cerr << std::endl;
      fragmented_reg.key->addFixedRegion( fragmented_reg.id );
      unsigned int version = 0;
      NewLocationInfoList missing_parts;
      NewNewRegionDirectory::__getLocation( fragmented_reg.key, fragmented_reg.id, missing_parts, version, *((WD *) 0) );
      memory_space_id_t loc = 0;
      for ( std::vector<SeparateMemoryAddressSpace *>::iterator it = _separateAddressSpaces.begin(); it != _separateAddressSpaces.end(); it++ ) {
         if ( *it != NULL ) {
            if ((*it)->getNodeNumber() == (start_node + node_idx) ) {
               fragmented_reg.setOwnedMemory(loc);
            }
         }
         loc++;
      }
   }
}

void System::registerNodeOwnedMemory(unsigned int node, void *addr, std::size_t len) {
   memory_space_id_t loc = 0;
   if ( node == 0 ) {
      global_reg_t reg = _registerMemoryChunk( addr, len );
      reg.setOwnedMemory(loc);
   } else {
      //_separateAddressSpaces[0] is always NULL (because loc = 0 is the local node memory)
      for ( std::vector<SeparateMemoryAddressSpace *>::iterator it = _separateAddressSpaces.begin(); it != _separateAddressSpaces.end(); it++ ) {
         if ( *it != NULL ) {
            if ((*it)->getNodeNumber() == node) {
               global_reg_t reg = _registerMemoryChunk( addr, len );
               reg.setOwnedMemory(loc);
            }
         }
         loc++;
      }
   }
}

void System::stickToProducer(void *addr, std::size_t len) {
   if ( _net.getNodeNum() == Network::MASTER_NODE_NUM ) {
      CopyData cd;
      nanos_region_dimension_internal_t dim;
      dim.lower_bound = 0;
      dim.size = len;
      dim.accessed_length = len;
      cd.setBaseAddress( addr );
      cd.setDimensions( &dim );
      cd.setNumDimensions( 1 );
      global_reg_t reg;
      getHostMemory().getRegionId( cd, reg, *((WD *) 0), 0 );
      reg.key->setKeepAtOrigin( true );
   }
}

void System::setCreateLocalTasks( bool value ) {
   _createLocalTasks = value;
}

memory_space_id_t System::addSeparateMemoryAddressSpace( Device &arch, bool allocWide, std::size_t slabSize ) {
   memory_space_id_t id = getNewSeparateMemoryAddressSpaceId();
   SeparateMemoryAddressSpace *mem = NEW SeparateMemoryAddressSpace( id, arch, allocWide, slabSize );
   _separateAddressSpaces[ id ] = mem;
   return id;
}

void System::registerObject( int numObjects, nanos_copy_data_internal_t *obj ) {
   for ( int i = 0; i < numObjects; i += 1 ) {
      _hostMemory.registerObject( &obj[i] );
   }
}

void System::unregisterObject( int numObjects, void *base_addresses ) {
   memory::Address* addrs = reinterpret_cast<memory::Address*>(base_addresses);
   for ( int i = 0; i < numObjects; i += 1 ) {
      _hostMemory.unregisterObject((void*)(addrs[i]));
   }
}

void System::switchToThread( unsigned int thid )
{
   if ( thid > _workers.size() ) return;

   Scheduler::switchToThread(_workers[thid]);
}

int System::initClusterMPI(int *argc, char ***argv) {
   return _clusterMPIPlugin->initNetwork(argc, argv);
}

void System::finalizeClusterMPI() {
   _clusterMPIPlugin->getClusterThread()->stop();
   _clusterMPIPlugin->getClusterThread()->join();
   //! \note finalizing instrumentation (if active)
   NANOS_INSTRUMENT ( sys.getInstrumentation()->raiseCloseStateEvent() );
   NANOS_INSTRUMENT ( sys.getInstrumentation()->finalize() );
   //_net.finalizeNoBarrier();
   //std::cerr << "AFTER _net.finalizeNoBarrier()" << std::endl;
}

void System::stopFirstThread( void ) {
   //FIXME: this assumes that mainWD is tied to thread 0
   _workers[0]->stop();
}

void System::notifyIntoBlockingMPICall() {
   NANOS_INSTRUMENT(static nanos_event_key_t ikey = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
   static int created = 0;
   if ( _schedStats._createdTasks.value() > created ) {
      _inIdle = true;
      NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 4444 );)
      *myThread->_file << "created " << ( _schedStats._createdTasks.value() - created ) << " tasks. Send msg." << std::endl;
      created = _schedStats._createdTasks.value();
      //*myThread->_file << "Into blocking mpi call: " << "[Created: " << _schedStats._createdTasks.value() << " Ready: " << _schedStats._readyTasks.value() << " Total: " << _schedStats._totalTasks.value() << "]" << std::endl;
      _net.broadcastIdle();
   }
}

void System::notifyOutOfBlockingMPICall() {
   NANOS_INSTRUMENT(static nanos_event_key_t ikey = sys.getInstrumentation()->getInstrumentationDictionary()->getEventKey("debug");)
   if ( _inIdle ) {
      NANOS_INSTRUMENT(sys.getInstrumentation()->raiseOpenBurstEvent( ikey, 0 );)
      _inIdle = false;
   }
   //*myThread->_file << "Out of blocking mpi call: " << "[Created: " << _schedStats._createdTasks.value() << " Ready: " << _schedStats._readyTasks.value() << " Total: " << _schedStats._totalTasks.value() << "]" << std::endl;
}

void System::notifyIdle( unsigned int node ) {
#ifdef CLUSTER_DEV
   if ( !_inIdle ) {
      unsigned int friend_node = (_net.getNodeNum() + 1) % _net.getNumNodes();
      if ( node == friend_node ) {
         ext::SMPMultiThread *cluster_thd = (ext::SMPMultiThread *) _clusterMPIPlugin->getClusterThread();
         unsigned int thd_idx = node > _net.getNodeNum() ? node - 1 : node;
         ext::ClusterThread *node_thd = (ext::ClusterThread *) cluster_thd->getThreadVector()[ thd_idx ];
         *myThread->_file << "Node " << node << ", my friend, is idle!! " << node_thd->runningOn()->getClusterNode() << std::endl;
         acquireWorker( _mainTeam,  node_thd, true, false, false );
         *myThread->_file << "Added to team " << _mainTeam << std::endl;
         _mainTeam->addExpectedThread(node_thd);
      }
   }
#endif
}

void System::disableHelperNodes() {
}
