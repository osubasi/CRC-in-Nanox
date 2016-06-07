CRC Implementation
--------------------
--------------------
AUTHOR: Omer Subasi, Polytechnic University of Catalonia and Barcelona Supercomputing Center

CRC Implementation is integrated with Jorge Bellon’s resiliency version. 
Main CRC functionality is in system.cpp and system_decl.hpp under src/core/ directory. 
Two types of CRC computation are implemented:
	- Hardware-accelerated
	- Software-only

Runtime checks whether the underlying architecture supports the Intel CRC instructions and automatically chooses between the hardware-accelerated and software-only implementations in the beginning of the execution.

The environment flag “–enable-crc=yes” enables CRC mechanism.

The Nanox configuration call has to include “CXXFLAG=-march=native” for CRC instructions.

The implementation is tested with
	- Sample program written for CRC mechanism features such as initialization, recovery.
	- OmpSs Benchmarks (Both SMP and OmpSs+MPI).

The source code is available at OmpSs downloads website (https://pm.bsc.es/ompss-downloads) and the direct link is
	- https://pm.bsc.es/sites/default/files/ftp/nanox/ad-hoc/nanox-mem_reliability-0.9a-2016-02-06.tar.gz

Implementation Details
----------------------
The implementation relies heavily on previously implemented resiliency mechanism in Nanox. 
For instance, error recovery is completely done through the resiliency mechanism, no new snapshots are taken.
It introduces very few data structures in system_decl.hpp and a number of functions residing in system.cpp.
Rest of the implementation extends to smpdd.cpp under src/arch/ directory and workdescriptor.cpp under src/core directory.
Runtime flag is implemented for enabling of the CRC mechanism.
The main functionality resides in system.cpp.
