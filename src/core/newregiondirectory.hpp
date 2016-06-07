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

#ifndef NANOS_NEWNEWDIRECTORY_H
#define NANOS_NEWNEWDIRECTORY_H

#include "deviceops.hpp"
#include "version.hpp"

namespace nanos {

inline NewNewDirectoryEntryData::NewNewDirectoryEntryData() : Version( 1 )
   //, _writeLocation( -1 )
   , _ops()
   , _location()
   , _pes()
   , _rooted( -1 )
   , _home( -1 )
   , _setLock() 
   , _firstWriterPE( NULL )
   , _baseAddress( nullptr )
{
   _location.insert(0);
}

inline NewNewDirectoryEntryData::NewNewDirectoryEntryData( memory_space_id_t home ) : Version( 1 )
   //, _writeLocation( -1 )
   , _ops()
   , _location()
   , _pes()
   , _rooted( -1 )
   , _home( home )
   , _setLock() 
   , _firstWriterPE( NULL )
   , _baseAddress( nullptr )
{
   _location.insert( home );
}

inline NewNewDirectoryEntryData::NewNewDirectoryEntryData( const NewNewDirectoryEntryData &de ) : Version( de )
   //, _writeLocation( de._writeLocation )
   , _ops()
   , _location( de._location )
   , _pes( de._pes )
   , _rooted( de._rooted )
   , _home( de._home )
   , _setLock()
   , _firstWriterPE( de._firstWriterPE )
   , _baseAddress( de._baseAddress )
{
}

inline NewNewDirectoryEntryData::~NewNewDirectoryEntryData() {
}

inline NewNewDirectoryEntryData & NewNewDirectoryEntryData::operator= ( NewNewDirectoryEntryData &de ) {
   Version::operator=( de );
   //_writeLocation = de._writeLocation;
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   while ( !de._setLock.tryAcquire() ) {
      myThread->idle();
   }
   _location.clear();
   _pes.clear();
   _location.insert( de._location.begin(), de._location.end() );
   _pes.insert( de._pes.begin(), de._pes.end() );
   _rooted = de._rooted;
   _home = de._home;
   _firstWriterPE = de._firstWriterPE;
   _baseAddress = de._baseAddress;
   de._setLock.release();
   _setLock.release();
   return *this;
}

//inline bool NewNewDirectoryEntryData::hasWriteLocation() const {
//   return ( _writeLocation != -1 );
//}

// inline int NewNewDirectoryEntryData::getWriteLocation() const {
//    return _writeLocation;
// }
// 
// inline void NewNewDirectoryEntryData::setWriteLocation( int id ) {
//    _writeLocation = id;
// }

inline void NewNewDirectoryEntryData::addAccess( ProcessingElement *pe, memory_space_id_t loc, unsigned int version ) {
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   //*myThread->_file << "+++++++++++++++++v entry " << (void *) this << " v++++++++++++++++++++++" << std::endl;
   if ( version > this->getVersion() ) {
      //*myThread->_file << "Upgrading version to " << version << " @location " << id << std::endl;
      _location.clear();
      //_writeLocation = id;
      this->setVersion( version );
      _location.insert( loc );
      if ( pe != NULL && loc == pe->getMemorySpaceId() ) {
         _pes.insert( pe );
      }
      if ( version == 2 ) {
         _firstWriterPE = pe;
      }
   } else if ( version == this->getVersion() ) {
      //*myThread->_file << "Equal version (" << version << ") @location " << id << std::endl;
      // entry is going to be replicated, so it must be that multiple copies are used as inputs only
      _location.insert( loc );
      if ( pe != NULL && loc == pe->getMemorySpaceId() ) {
         _pes.insert( pe );
      }
      // if ( _location.size() > 1 )
      // {
      //    _writeLocation = -1;
      // }
   } else {
     //*myThread->_file << "FIXME: wrong case, current version is " << this->getVersion() << " and requested is " << version << " @location " << id <<std::endl;
   }
   //*myThread->_file << "+++++++++++++++++^ entry " << (void *) this << " ^++++++++++++++++++++++" << std::endl;
   _setLock.release();
}

inline void NewNewDirectoryEntryData::addRootedAccess( memory_space_id_t loc, unsigned int version ) {
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   ensure(version == this->getVersion(), "addRootedAccess of already accessed entry." );
   _location.clear();
   //_writeLocation = id;
   this->setVersion( version );
   _location.insert( loc );
   _rooted = loc;
   _setLock.release();
}

inline bool NewNewDirectoryEntryData::delAccess( memory_space_id_t from ) {
   bool result;
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   _location.erase( from );
   std::set< ProcessingElement * >::iterator it = _pes.begin();
   while ( it != _pes.end() ) {
      if ( (*it)->getMemorySpaceId() == from ) {
         std::set< ProcessingElement * >::iterator toBeErased = it;
         ++it;
         _pes.erase( toBeErased );
      } else {
         ++it;
      }
   }
   result = _location.empty();
   _setLock.release();
   return result;
}

inline bool NewNewDirectoryEntryData::isLocatedIn( ProcessingElement *pe, unsigned int version ) {
   bool result;
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   if ( _location.empty() ) {
      *myThread->_file << " Warning: empty _location set, it is likely that an invalidation is ongoing for this region. " << std::endl;
   }
   result = ( version <= this->getVersion() && _location.count( pe->getMemorySpaceId() ) > 0 );
   _setLock.release();
   return result;
}

inline bool NewNewDirectoryEntryData::isLocatedIn( ProcessingElement *pe ) {
   return this->isLocatedIn( pe->getMemorySpaceId() );
}

inline bool NewNewDirectoryEntryData::isLocatedIn( memory_space_id_t loc ) {
   bool result;
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   result = ( _location.count( loc ) > 0 );
   if ( !result && _location.size() == 0 ) { //locations.size = 0 means we are invalidating
      result = (loc == 0);
   }
   _setLock.release();
   return result;
}

inline void NewNewDirectoryEntryData::print(std::ostream &o) const {
   o << " V: " << this->getVersion() << " Locs: ";
   for ( std::set< memory_space_id_t >::iterator it = _location.begin(); it != _location.end(); it++ ) {
      o << *it << " ";
   }
   o << std::endl;
}

inline int NewNewDirectoryEntryData::getFirstLocation() {
   int result;
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   result = *(_location.begin());
   _setLock.release();
   return result;
}

inline int NewNewDirectoryEntryData::getNumLocations() {
   int result;
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
   result = _location.size();
   _setLock.release();
   return result;
}

inline DeviceOps *NewNewDirectoryEntryData::getOps() {
   return &_ops;
}

inline std::set< memory_space_id_t > const &NewNewDirectoryEntryData::getLocations() const {
   return _location;
}

inline memory_space_id_t NewNewDirectoryEntryData::getRootedLocation() const{
   return _rooted;
}


inline bool NewNewDirectoryEntryData::isRooted() const{
   return _rooted != (memory_space_id_t) -1;
}

inline ProcessingElement *NewNewDirectoryEntryData::getFirstWriterPE() const {
   return _firstWriterPE;
}

inline void NewNewDirectoryEntryData::setBaseAddress(memory::Address addr) {
   _baseAddress = addr;
}

inline memory::Address NewNewDirectoryEntryData::getBaseAddress() const {
   return _baseAddress;
}

inline memory_space_id_t NewNewDirectoryEntryData::getHome() const {
   return _home;
}

inline void NewNewDirectoryEntryData::lock() {
   while ( !_setLock.tryAcquire() ) {
      myThread->idle();
   }
}

inline void NewNewDirectoryEntryData::unlock() {
   _setLock.release();
}

inline NewNewRegionDirectory::RegionDirectoryKey NewNewRegionDirectory::getRegionDirectoryKeyRegisterIfNeeded( CopyData const &cd, WD const *wd ) {
   return getRegionDictionaryRegisterIfNeeded( cd, wd );
}

inline NewNewRegionDirectory::RegionDirectoryKey NewNewRegionDirectory::getRegionDirectoryKey( CopyData const &cd ) {
   return getRegionDictionary( cd );
}

inline NewNewRegionDirectory::RegionDirectoryKey NewNewRegionDirectory::getRegionDirectoryKey( memory::Address addr ) {
   return getRegionDictionary( addr );
}

inline void NewNewRegionDirectory::__getLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &missingParts, unsigned int &version, WD const &wd )
{
   dict->lockObject();
   dict->registerRegion( reg, missingParts, version );
   dict->unlockObject();
}

} // namespace nanos

#endif
