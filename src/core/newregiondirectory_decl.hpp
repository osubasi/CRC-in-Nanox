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

#ifndef NEW_NANOS_NEW2DIRECTORY_DECL_H
#define NEW_NANOS_NEW2DIRECTORY_DECL_H

#include "regiondict_decl.hpp"
#include "globalregt_decl.hpp"
#include "deviceops_decl.hpp"
#include "workdescriptor_fwd.hpp"
#include "processingelement_fwd.hpp"

namespace nanos {

   class NewNewDirectoryEntryData : public Version {
      private:
         //int _writeLocation;
         //int _invalidated;
         DeviceOps _ops;
         std::set< memory_space_id_t > _location;
         std::set< ProcessingElement *> _pes;
         memory_space_id_t _rooted;
         memory_space_id_t _home;
         Lock _setLock;
         ProcessingElement * _firstWriterPE;
         memory::Address _baseAddress;
      public:
         NewNewDirectoryEntryData();
         NewNewDirectoryEntryData( memory_space_id_t home );
         NewNewDirectoryEntryData( const NewNewDirectoryEntryData &de );
         ~NewNewDirectoryEntryData();
         NewNewDirectoryEntryData & operator= ( NewNewDirectoryEntryData &de );
         // bool hasWriteLocation() const ;
         // int getWriteLocation() const ;
         // void setWriteLocation( int id ) ;
         void addAccess( ProcessingElement *pe, memory_space_id_t loc, unsigned int version );
         void addRootedAccess( memory_space_id_t loc, unsigned int version );
         bool delAccess( memory_space_id_t id );
         bool isLocatedIn( ProcessingElement *pe, unsigned int version );
         bool isLocatedIn( ProcessingElement *pe );
         bool isLocatedIn( memory_space_id_t loc );
         bool isRooted() const;
         memory_space_id_t getRootedLocation() const;
         void print(std::ostream &o) const ;
         int getFirstLocation();
         ProcessingElement *getFirstWriterPE() const;
         int getNumLocations();
         void setOps( DeviceOps *ops );
         std::set< memory_space_id_t > const &getLocations() const;
         DeviceOps *getOps() ;
         void setBaseAddress(memory::Address addr);
         memory::Address getBaseAddress() const;
         memory_space_id_t getHome() const;
         void lock();
         void unlock();
         friend std::ostream & operator<< (std::ostream &o, NewNewDirectoryEntryData const &entry);
   };

  /*! \class NewDirectory
   *  \brief Stores copy accesses controls their versions and if they are dirty in any cache
   */
   class NewNewRegionDirectory
   {
      private:
         class Object {
            GlobalRegionDictionary *_object;
            CopyData *_registeredObject;

            public:
            Object() : _object( NULL ), _registeredObject( NULL ) {}
            Object( GlobalRegionDictionary *dict, CopyData *cd = NULL ) :
               _object( dict ), _registeredObject( cd ) {}
            Object( Object const& o ) : _object( o._object ),
               _registeredObject( o._registeredObject ) {}
            Object &operator=( Object const& o ) {
               this->_object = o._object;
               this->_registeredObject = o._registeredObject;
               return *this;
            }
            ~Object() {
               destroyDictionary();
               delete _registeredObject;
            }
            GlobalRegionDictionary *getGlobalRegionDictionary() const {
               return _object;
            }
            CopyData *getRegisteredObject() const {
               return _registeredObject;
            }
            void destroyDictionary() {
               for ( unsigned int reg_id = 1; reg_id < _object->getRegionNodeCount()+1; reg_id += 1 ) {
                  NewNewDirectoryEntryData *entry = ( NewNewDirectoryEntryData * ) _object->getRegionData( reg_id );
                  delete entry;
               }
               delete _object;
               _object = NULL;
            }
            void resetGlobalRegionDictionary() {
               destroyDictionary();
               if ( _registeredObject != NULL ) {
                  _object = NEW GlobalRegionDictionary( *_registeredObject );
                  _object->setRegisteredObject( _registeredObject );
                  NewNewDirectoryEntryData *entry = getDirectoryEntry( *_object, 1 );
                  if ( entry == NULL ) {
                     entry = NEW NewNewDirectoryEntryData();
                     _object->setRegionData( 1, entry ); //resetGlobalRegionDictionary
                  }
               }
            }
            void setGlobalRegionDictionary( GlobalRegionDictionary *object ) {
               _object = object;
            }
         };
         struct HashBucket {
            Lock _lock;
            MemoryMap< Object > *_bobjects;
            HashBucket();
            HashBucket( HashBucket const & hb );
            HashBucket &operator=( HashBucket const &hb );
            ~HashBucket();
         };

         MemoryMap<memory::Address> _keys;
         size_t                     _keysSeed;
         Lock                       _keysLock;
         std::vector< HashBucket >  _objects;

      private:

         /*! \brief NewDirectory copy constructor (private) 
          */
         NewNewRegionDirectory( const NewNewRegionDirectory &dir );

         /*! \brief NewDirectory copy assignment operator (private) 
          */
         const NewNewRegionDirectory & operator= ( const NewNewRegionDirectory &dir );

         GlobalRegionDictionary *getRegionDictionaryRegisterIfNeeded( CopyData const &cd, WD const *wd );
         GlobalRegionDictionary *getRegionDictionary( CopyData const &cd );
         GlobalRegionDictionary *getRegionDictionary( memory::Address addr, bool canFail );
         static void addSubRegion( GlobalRegionDictionary &dict, std::list< std::pair< reg_t, reg_t > > &partsList, reg_t regionToInsert );
         memory::Address _getKey( memory::Address addr, std::size_t len, WD const *wd );
         memory::Address _getKey( memory::Address addr ) const;
         void _unregisterObjects( std::map< memory::Address, MemoryMap< Object > * > &objects );
         void _invalidateObjectsFromDevices( std::map< memory::Address, MemoryMap< Object > * > &objects );

      public:
         typedef GlobalRegionDictionary *RegionDirectoryKey;
         //typedef std::pair< Region, NewNewDirectoryEntryData const *> LocationInfo;
         RegionDirectoryKey getRegionDirectoryKey( CopyData const &cd );
         RegionDirectoryKey getRegionDirectoryKey( memory::Address addr );
         RegionDirectoryKey getRegionDirectoryKeyRegisterIfNeeded( CopyData const &cd, WD const *wd );
         void synchronize( WD &wd );
         void synchronize( WD &wd, memory::Address addr );
         void synchronize( WD &wd, std::size_t numDataAccesses, DataAccess *data );

         /*! \brief NewDirectory default constructor
          */
         NewNewRegionDirectory();

         /*! \brief NewDirectory destructor
          */
         ~NewNewRegionDirectory();

         void invalidate( CacheRegionDictionary *regions, unsigned int from );

         void print() const;
         void lock();
         void tryLock();
         GlobalRegionDictionary &getDictionary( CopyData const &cd );

         static NewNewDirectoryEntryData *getDirectoryEntry( GlobalRegionDictionary &dict, reg_t id );

         static reg_t _getLocation( RegionDirectoryKey dict, CopyData const &cd, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe, unsigned int version );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe );
         static bool isLocatedIn( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc );
         static unsigned int getVersion( RegionDirectoryKey dict, reg_t id, bool increaseVersion );
         static void addAccess( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe, memory_space_id_t loc, unsigned int version );
         static void addRootedAccess( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc, unsigned int version );
         static bool delAccess( RegionDirectoryKey dict, reg_t id, memory_space_id_t memorySpaceId );
         static bool isOnlyLocated( RegionDirectoryKey dict, reg_t id, ProcessingElement *pe );
         static bool isOnlyLocated( RegionDirectoryKey dict, reg_t id, memory_space_id_t loc );
         static void invalidate( RegionDirectoryKey dict, reg_t id );
         static bool hasBeenInvalidated( RegionDirectoryKey dict, reg_t id );
         static void updateFromInvalidated( RegionDirectoryKey dict, reg_t id, reg_t from );
         static unsigned int getFirstLocation( RegionDirectoryKey dict, reg_t id );
         static DeviceOps *getOps( RegionDirectoryKey dict, reg_t id );
         static void setOps( RegionDirectoryKey dict, reg_t id, DeviceOps *ops );


         static void __getLocation( RegionDirectoryKey dict, reg_t reg, NewLocationInfoList &loc, unsigned int &version, WD const &wd );
         static void addRegionId( RegionDirectoryKey dict, reg_t masterId, reg_t localId );
         static reg_t getLocalRegionIdFromMasterRegionId( RegionDirectoryKey dict, reg_t localId );
         static void addMasterRegionId( RegionDirectoryKey dict, reg_t masterId, reg_t localId );
         reg_t getLocalRegionId( void *hostObject, reg_t hostRegionId );

         void registerObject(nanos_copy_data_internal_t *obj);
         void unregisterObject( memory::Address baseAddr );
   };

} // namespace nanos

#endif
