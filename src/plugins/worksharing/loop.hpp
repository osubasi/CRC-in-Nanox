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

#include "nanos-int.h"
#include "atomic_decl.hpp"

namespace nanos {
namespace ext {

typedef struct {
   int                   lowerBound;   // loop lower bound
   int                   upperBound;   // loop upper bound
   int                   loopStep;     // loop step
   int                   chunkSize;    // loop chunk size
   int                   numOfChunks;  // number of chunks for the loop
   Atomic<int>           currentChunk; // current chunk ready to execute
} WorkSharingLoopInfo;

#if 0
   int                   neths;    // additional data to expand team
   nanos_thread_t       *eths;     // additional data to expand team
#endif

} // namespace ext
} // namespace nanos