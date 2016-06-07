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

#include "plugin.hpp"
#include "openclconfig.hpp"
#include "openclplugin.hpp"
#include <dlfcn.h>

#ifdef HAVE_OPENCL_OPENCL_H
#include <OpenCL/opencl.h>
#endif

#ifdef HAVE_CL_OPENCL_H
#include <CL/opencl.h>
#endif

namespace nanos {
namespace ext {
   
   std::string OpenCLPlugin::_devTy = "ALL";
   // All found devices
   std::map<cl_device_id, cl_context> OpenCLPlugin::_devices;

   void OpenCLPlugin::config( Config &cfg )
   {
      cfg.setOptionsSection( "OpenCL Arch", "OpenCL specific options" );
      // Select the device to use.
      cfg.registerConfigOption( "opencl-device-type",
                                NEW Config::StringVar( _devTy ),
                                "Defines the OpenCL device type to use "
                                "(ALL, CPU, GPU, ACCELERATOR)" );
      cfg.registerEnvOption( "opencl-device-type", "NX_OPENCL_DEVICE_TYPE" );
      cfg.registerArgOption( "opencl-device-type", "opencl-device-type" );
   
      OpenCLConfig::prepare( cfg );
   }

   void OpenCLPlugin::init()
   {
      OpenCLConfig::apply( _devTy, &_devices );
      _opencls = NEW std::vector<nanos::ext::OpenCLProcessor *>(nanos::ext::OpenCLConfig::getOpenCLDevicesCount(), (nanos::ext::OpenCLProcessor *) NULL); 
      _openclThreads = NEW std::vector<nanos::ext::OpenCLThread *>(nanos::ext::OpenCLConfig::getOpenCLDevicesCount(), (nanos::ext::OpenCLThread *) NULL); 
      for ( unsigned int openclC = 0; openclC < nanos::ext::OpenCLConfig::getOpenCLDevicesCount() ; openclC++ ) {
         memory_space_id_t id = sys.addSeparateMemoryAddressSpace( ext::OpenCLDev, nanos::ext::OpenCLConfig::getAllocWide(), 0 );
         SeparateMemoryAddressSpace &oclmemory = sys.getSeparateMemory( id );
         oclmemory.setAcceleratorNumber( sys.getNewAcceleratorId() );

         ext::SMPProcessor *core = sys.getSMPPlugin()->getLastFreeSMPProcessorAndReserve();
         if ( core == NULL ) {
            fatal("Unable to get a core to run the GPU thread.");
         }
         core->setNumFutureThreads( 1 );

         OpenCLProcessor *ocl = NEW nanos::ext::OpenCLProcessor( openclC, id, core, oclmemory );
         (*_opencls)[openclC] = ocl;
      }
   }
   
   /*unsigned OpenCLPlugin::getPEsInNode( unsigned node ) const
   {
      // TODO: make it work correctly
      // If it is the last node, assign
      //if ( node == ( sys.getNumSockets() - 1 ) )
   }*/
   
   unsigned OpenCLPlugin::getNumHelperPEs() const
   {
      return OpenCLConfig::getOpenCLDevicesCount();
   }

//   unsigned OpenCLPlugin::getNumPEs() const
//   {
//      return OpenCLConfig::getOpenCLDevicesCount();
//   }
   
   unsigned OpenCLPlugin::getNumThreads() const
   {
      return OpenCLConfig::getOpenCLDevicesCount();
   }
   
   void OpenCLPlugin::createBindingList()
   {
////      /* As we now how many devices we have and how many helper threads we
////       * need, reserve a PE for them */
//      for ( unsigned i = 0; i < OpenCLConfig::getOpenCLDevicesCount(); ++i )
//      {
//         // As we don't have NUMA info, don't request an specific node
//         bool numa = false;
//         // TODO: if HWLOC is available, use it.
//         int node = sys.getNumSockets() - 1;
//         bool reserved;
//         unsigned pe = sys.reservePE( numa, node, reserved );
//         
//         // Now add this node to the binding list
//         addBinding( pe );
//      }
   }

   PE* OpenCLPlugin::createPE( unsigned id, unsigned uid )
   {
      //pe->setNUMANode( sys.getNodeOfPE( pe->getId() ) );
//      memory_space_id_t mid = sys.getNewSeparateMemoryAddressSpaceId();
//      SeparateMemoryAddressSpace *oclmemory = NEW SeparateMemoryAddressSpace( mid, ext::OpenCLDev, nanos::ext::OpenCLConfig::getAllocWide());
//      oclmemory->setNodeNumber( 0 );
//      //ext::OpenCLMemorySpace *oclmemspace = NEW ext::OpenCLMemorySpace();
//      //oclmemory->setSpecificData( oclmemspace );
//      sys.addSeparateMemory(mid,oclmemory);
//      nanos::ext::OpenCLProcessor *openclPE = NEW nanos::ext::OpenCLProcessor( getBinding(id), id, uid, mid, *oclmemory );
//      
//      openclPE->setNUMANode( sys.getNodeOfPE( openclPE->getId() ) ); 
//      return openclPE;
      return NULL;
   }

   void OpenCLPlugin::addPEs( std::map<unsigned int, ProcessingElement *> &pes ) const {
      for ( std::vector<OpenCLProcessor *>::const_iterator it = _opencls->begin(); it != _opencls->end(); it++ ) {
         pes.insert( std::make_pair( (*it)->getId(), *it ) );
      }
   }
   
   void OpenCLPlugin::addDevices( DeviceList &devices ) const
   {
      if ( !_opencls->empty() ) {
         std::vector<const Device *> const &pe_archs = ( *_opencls->begin() )->getDeviceTypes();
         for ( std::vector<const Device *>::const_iterator it = pe_archs.begin();
               it != pe_archs.end(); it++ ) {
            devices.insert( *it );
         }
      }
   }
   
   
   void OpenCLPlugin::startSupportThreads() {
      for ( unsigned int openclC = 0; openclC < _opencls->size(); openclC += 1 ) {
         OpenCLProcessor *ocl = (*_opencls)[openclC];
         (*_openclThreads)[openclC] = (ext::OpenCLThread *) &ocl->startOpenCLThread();
      }
   }
   
   void OpenCLPlugin::startWorkerThreads( std::map<unsigned int, BaseThread *> &workers ) {
      for ( std::vector<OpenCLThread *>::iterator it = _openclThreads->begin(); it != _openclThreads->end(); it++ ) {
         workers.insert( std::make_pair( (*it)->getId(), *it ) );
      }
   }
   
   void OpenCLPlugin::finalize() {
   }

} // End namespace ext.
} // End namespace nanos.


DECLARE_PLUGIN("arch-opencl",nanos::ext::OpenCLPlugin);

