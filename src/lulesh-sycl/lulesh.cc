/*
   Copyright (c) 2010-2013.
   Lawrence Livermore National Security, LLC.
   Produced at the Lawrence Livermore National Laboratory.
   LLNL-CODE-461231
   All rights reserved.

   This file is part of LULESH, Version 2.0.
   Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
 * Addition of regions to make work more representative of multi-material codes
 * Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
 more representative of our actual working set sizes
 * Single source distribution supports pure serial, pure OpenMP, MPI-only, 
 and MPI+OpenMP
 * Addition of ability to visualize the mesh using VisIt 
https://wci.llnl.gov/codes/visit/download.html
 * Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
 printf(" where [opts] is one or more of:\n");
 printf(" -q              : quiet mode - suppress all stdout\n");
 printf(" -i <iterations> : number of cycles to run\n");
 printf(" -s <size>       : length of cube mesh along side\n");
 printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
 printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
 printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
 printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
 printf(" -p              : Print out progress\n");
 printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
 printf(" -h              : This message\n");
 printf("\n\n");

 *Notable changes in LULESH 2.0

 * Split functionality into different files
 lulesh.cc - where most (all?) of the timed functionality lies
 lulesh-comm.cc - MPI functionality
 lulesh-init.cc - Setup code
 lulesh-viz.cc  - Support for visualization option
 lulesh-util.cc - Non-timed functions
 *
 * The concept of "regions" was added, although every region is the same ideal
 *    gas material, and the same sedov blast wave problem is still the only
 *    problem its hardcoded to solve.
 * Regions allow two things important to making this proxy app more representative:
 *   Four of the LULESH routines are now performed on a region-by-region basis,
 *     making the memory access patterns non-unit stride
 *   Artificial load imbalances can be easily introduced that could impact
 *     parallelization strategies.  
 * The load balance flag changes region assignment.  Region number is raised to
 *   the power entered for assignment probability.  Most likely regions changes
 *   with MPI process id.
 * The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
 *   entered multiple. The cost of 5% is 10x the entered multiple.
 * MPI and OpenMP were added, and coalesced into a single version of the source
 *   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
  *
* Possible Differences in Final Release (other changes possible)
  *
  * High Level mesh structure to allow data structure transformations
  * Different default parameters
  * Minor code performance changes and cleanup

  TODO in future versions
  * Add reader for (truly) unstructured meshes, probably serial only
  * CMake based build system

  //////////////

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  * Redistributions of source code must retain the above copyright
  notice, this list of conditions and the disclaimer below.

  * Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the disclaimer (as noted below)
  in the documentation and/or other materials provided with the
  distribution.

  * Neither the name of the LLNS/LLNL nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
  THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
      BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
      DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
      NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


  Additional BSD Notice

  1. This notice is required to be provided under our contract with the U.S.
  Department of Energy (DOE). This work was produced at Lawrence Livermore
  National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

  2. Neither the United States Government nor Lawrence Livermore National
  Security, LLC nor any of their employees, makes any warranty, express
  or implied, or assumes any liability or responsibility for the accuracy,
  completeness, or usefulness of any information, apparatus, product, or
  process disclosed, or represents that its use would not infringe
  privately-owned rights.

  3. Also, reference herein to any specific commercial products, process, or
  services by trade name, trademark, manufacturer or otherwise does not
  necessarily constitute or imply its endorsement, recommendation, or
  favoring by the United States Government or Lawrence Livermore National
  Security, LLC. The views and opinions of authors expressed herein do not
  necessarily state or reflect those of the United States Government or
  Lawrence Livermore National Security, LLC, and shall not be used for
  advertising or product endorsement purposes.

  */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <climits>
#include <iostream>
#include <sstream>
#include <limits>
#include <fstream>
#include <string>
#ifdef VERIFY
#include <cassert>
#include <random>
#endif
#include "lulesh.h"

#define EPSILON 1e-7

#define THREADS 256

#if !defined(USE_GPU)
#define USE_GPU 1
#endif

#define ZERO  Real_t(0)
#define HALF  Real_t(0.5)
#define ONE   Real_t(1.0)
#define THREE Real_t(3.0)
#define FOUR  Real_t(4.0)
#define C1    Real_t(.1111111e-36)
#define C2    Real_t(.3333333e-18)
#define SEVEN Real_t(7.0)
#define EIGHT Real_t(8.0)
#define C1S   Real_t(2.0/3.0)
#define SIXTH Real_t(1.0/6.0)

#define PTINY Real_t(1e-36)

  /*********************************/
  /* Data structure implementation */
  /*********************************/

  /* might want to add access methods so that memory can be */
  /* better managed, as in luleshFT */

template <typename T>
T *Allocate(size_t size)
{
  return static_cast<T *>(malloc(sizeof(T)*size)) ;
}

template <typename T>
void Release(T **ptr)
{
  if (*ptr != NULL) {
    free(*ptr) ;
    *ptr = NULL ;
  }
}

/******************************************/

/* Work Routines */

static inline
void TimeIncrement(Domain& domain)
{
  Real_t targetdt = domain.stoptime() - domain.time() ;

  if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
    Real_t ratio ;
    Real_t olddt = domain.deltatime() ;

    /* This will require a reduction in parallel */
    Real_t gnewdt = Real_t(1.0e+20) ;
    Real_t newdt ;
    if (domain.dtcourant() < gnewdt) {
      gnewdt = domain.dtcourant() / Real_t(2.0) ;
    }
    if (domain.dthydro() < gnewdt) {
      gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
    }

    newdt = gnewdt;

    ratio = newdt / olddt ;
    if (ratio >= Real_t(1.0)) {
      if (ratio < domain.deltatimemultlb()) {
        newdt = olddt ;
      }
      else if (ratio > domain.deltatimemultub()) {
        newdt = olddt*domain.deltatimemultub() ;
      }
    }

    if (newdt > domain.dtmax()) {
      newdt = domain.dtmax() ;
    }
    domain.deltatime() = newdt ;
  }

  /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
  if ((targetdt > domain.deltatime()) &&
      (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
    targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
  }

  if (targetdt < domain.deltatime()) {
    domain.deltatime() = targetdt ;
  }

  domain.time() += domain.deltatime() ;

  ++domain.cycle() ;
}


/******************************************/
static inline
void CalcElemShapeFunctionDerivatives(
    Real_t const x[],
    Real_t const y[],
    Real_t const z[],
    Real_t b[][8],
    Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
     */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}
/******************************************/
static inline
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
    Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
    Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
    Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
    const Real_t x0, const Real_t y0, const Real_t z0,
    const Real_t x1, const Real_t y1, const Real_t z1,
    const Real_t x2, const Real_t y2, const Real_t z2,
    const Real_t x3, const Real_t y3, const Real_t z3)
{
  Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
  Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
  Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
  Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
  Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
  Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
  Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
  Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
  Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

  *normalX0 += areaX;
  *normalX1 += areaX;
  *normalX2 += areaX;
  *normalX3 += areaX;

  *normalY0 += areaY;
  *normalY1 += areaY;
  *normalY2 += areaY;
  *normalY3 += areaY;

  *normalZ0 += areaZ;
  *normalZ1 += areaZ;
  *normalZ2 += areaZ;
  *normalZ3 += areaZ;
}

/******************************************/
static inline
void CalcElemNodeNormals(
    Real_t pfx[8],
    Real_t pfy[8],
    Real_t pfz[8],
    const Real_t x[8],
    const Real_t y[8],
    const Real_t z[8])
{
  for (Index_t i = 0 ; i < 8 ; ++i) {
    pfx[i] = Real_t(0.0);
    pfy[i] = Real_t(0.0);
    pfz[i] = Real_t(0.0);
  }
  /* evaluate face one: nodes 0, 1, 2, 3 */
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
      &pfx[1], &pfy[1], &pfz[1],
      &pfx[2], &pfy[2], &pfz[2],
      &pfx[3], &pfy[3], &pfz[3],
      x[0], y[0], z[0], x[1], y[1], z[1],
      x[2], y[2], z[2], x[3], y[3], z[3]);
  /* evaluate face two: nodes 0, 4, 5, 1 */
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
      &pfx[4], &pfy[4], &pfz[4],
      &pfx[5], &pfy[5], &pfz[5],
      &pfx[1], &pfy[1], &pfz[1],
      x[0], y[0], z[0], x[4], y[4], z[4],
      x[5], y[5], z[5], x[1], y[1], z[1]);
  /* evaluate face three: nodes 1, 5, 6, 2 */
  SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
      &pfx[5], &pfy[5], &pfz[5],
      &pfx[6], &pfy[6], &pfz[6],
      &pfx[2], &pfy[2], &pfz[2],
      x[1], y[1], z[1], x[5], y[5], z[5],
      x[6], y[6], z[6], x[2], y[2], z[2]);
  /* evaluate face four: nodes 2, 6, 7, 3 */
  SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
      &pfx[6], &pfy[6], &pfz[6],
      &pfx[7], &pfy[7], &pfz[7],
      &pfx[3], &pfy[3], &pfz[3],
      x[2], y[2], z[2], x[6], y[6], z[6],
      x[7], y[7], z[7], x[3], y[3], z[3]);
  /* evaluate face five: nodes 3, 7, 4, 0 */
  SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
      &pfx[7], &pfy[7], &pfz[7],
      &pfx[4], &pfy[4], &pfz[4],
      &pfx[0], &pfy[0], &pfz[0],
      x[3], y[3], z[3], x[7], y[7], z[7],
      x[4], y[4], z[4], x[0], y[0], z[0]);
  /* evaluate face six: nodes 4, 7, 6, 5 */
  SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
      &pfx[7], &pfy[7], &pfz[7],
      &pfx[6], &pfy[6], &pfz[6],
      &pfx[5], &pfy[5], &pfz[5],
      x[4], y[4], z[4], x[7], y[7], z[7],
      x[6], y[6], z[6], x[5], y[5], z[5]);
}
/******************************************/
static inline
void SumElemStressesToNodeForces( const Real_t B[][8],
    const Real_t stress_xx,
    const Real_t stress_yy,
    const Real_t stress_zz,
    Real_t fx[], Real_t fy[], Real_t fz[] )
{
  for(Index_t i = 0; i < 8; i++) {
    fx[i] = -( stress_xx * B[0][i] );
    fy[i] = -( stress_yy * B[1][i] );
    fz[i] = -( stress_zz * B[2][i] );
  }
}
/******************************************/


/******************************************/
static inline
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
    const Real_t x3, const Real_t x4, const Real_t x5,
    const Real_t y0, const Real_t y1, const Real_t y2,
    const Real_t y3, const Real_t y4, const Real_t y5,
    const Real_t z0, const Real_t z1, const Real_t z2,
    const Real_t z3, const Real_t z4, const Real_t z5,
    Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

  *dvdx =
    (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
    (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
    (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
  *dvdy =
    - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
    (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
    (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

  *dvdz =
    - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
    (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
    (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

  *dvdx *= twelfth;
  *dvdy *= twelfth;
  *dvdz *= twelfth;
}

/******************************************/
static inline
void CalcElemVolumeDerivative(
    Real_t dvdx[8],
    Real_t dvdy[8],
    Real_t dvdz[8],
    const Real_t x[8],
    const Real_t y[8],
    const Real_t z[8])
{
  VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
      y[1], y[2], y[3], y[4], y[5], y[7],
      z[1], z[2], z[3], z[4], z[5], z[7],
      &dvdx[0], &dvdy[0], &dvdz[0]);
  VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
      y[0], y[1], y[2], y[7], y[4], y[6],
      z[0], z[1], z[2], z[7], z[4], z[6],
      &dvdx[3], &dvdy[3], &dvdz[3]);
  VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
      y[3], y[0], y[1], y[6], y[7], y[5],
      z[3], z[0], z[1], z[6], z[7], z[5],
      &dvdx[2], &dvdy[2], &dvdz[2]);
  VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
      y[2], y[3], y[0], y[5], y[6], y[4],
      z[2], z[3], z[0], z[5], z[6], z[4],
      &dvdx[1], &dvdy[1], &dvdz[1]);
  VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
      y[7], y[6], y[5], y[0], y[3], y[1],
      z[7], z[6], z[5], z[0], z[3], z[1],
      &dvdx[4], &dvdy[4], &dvdz[4]);
  VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
      y[4], y[7], y[6], y[1], y[0], y[2],
      z[4], z[7], z[6], z[1], z[0], z[2],
      &dvdx[5], &dvdy[5], &dvdz[5]);
  VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
      y[5], y[4], y[7], y[2], y[1], y[3],
      z[5], z[4], z[7], z[2], z[1], z[3],
      &dvdx[6], &dvdy[6], &dvdz[6]);
  VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
      y[6], y[5], y[4], y[3], y[2], y[0],
      z[6], z[5], z[4], z[3], z[2], z[0],
      &dvdx[7], &dvdy[7], &dvdz[7]);
}


/******************************************/
static inline
Real_t CalcElemVolume(
    const Real_t x0, const Real_t x1,
    const Real_t x2, const Real_t x3,
    const Real_t x4, const Real_t x5,
    const Real_t x6, const Real_t x7,
    const Real_t y0, const Real_t y1,
    const Real_t y2, const Real_t y3,
    const Real_t y4, const Real_t y5,
    const Real_t y6, const Real_t y7,
    const Real_t z0, const Real_t z1,
    const Real_t z2, const Real_t z3,
    const Real_t z4, const Real_t z5,
    const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
  ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
        dy31 + dy72, dy63, dy20,
        dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
        dy43 + dy57, dy64, dy70,
        dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
        dy14 + dy25, dy61, dy50,
        dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

/******************************************/
//inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
  return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
      y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
      z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}
/******************************************/
static inline
Real_t AreaFace(
    const Real_t x0, const Real_t x1,
    const Real_t x2, const Real_t x3,
    const Real_t y0, const Real_t y1,
    const Real_t y2, const Real_t y3,
    const Real_t z0, const Real_t z1,
    const Real_t z2, const Real_t z3)
{
  Real_t fx = (x2 - x0) - (x3 - x1);
  Real_t fy = (y2 - y0) - (y3 - y1);
  Real_t fz = (z2 - z0) - (z3 - z1);
  Real_t gx = (x2 - x0) + (x3 - x1);
  Real_t gy = (y2 - y0) + (y3 - y1);
  Real_t gz = (z2 - z0) + (z3 - z1);
  Real_t area =
    (fx * fx + fy * fy + fz * fz) *
    (gx * gx + gy * gy + gz * gz) -
    (fx * gx + fy * gy + fz * gz) *
    (fx * gx + fy * gy + fz * gz);
  return area ;
}
/******************************************/
#define max(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })

static inline
Real_t CalcElemCharacteristicLength(
    const Real_t x[8],
    const Real_t y[8],
    const Real_t z[8],
    const Real_t volume)
{
  Real_t a, charLength = Real_t(0.0);

  a = AreaFace(x[0],x[1],x[2],x[3],
      y[0],y[1],y[2],y[3],
      z[0],z[1],z[2],z[3]) ;
  charLength = max(a,charLength) ;

  a = AreaFace(x[4],x[5],x[6],x[7],
      y[4],y[5],y[6],y[7],
      z[4],z[5],z[6],z[7]) ;
  charLength = max(a,charLength) ;

  a = AreaFace(x[0],x[1],x[5],x[4],
      y[0],y[1],y[5],y[4],
      z[0],z[1],z[5],z[4]) ;
  charLength = max(a,charLength) ;

  a = AreaFace(x[1],x[2],x[6],x[5],
      y[1],y[2],y[6],y[5],
      z[1],z[2],z[6],z[5]) ;
  charLength = max(a,charLength) ;

  a = AreaFace(x[2],x[3],x[7],x[6],
      y[2],y[3],y[7],y[6],
      z[2],z[3],z[7],z[6]) ;
  charLength = max(a,charLength) ;

  a = AreaFace(x[3],x[0],x[4],x[7],
      y[3],y[0],y[4],y[7],
      z[3],z[0],z[4],z[7]) ;
  charLength = max(a,charLength) ;

  charLength = Real_t(4.0) * volume / sycl::sqrt(charLength);

  return charLength;
}

/******************************************/
static inline
void CalcElemVelocityGradient(
    const Real_t* const xvel,
    const Real_t* const yvel,
    const Real_t* const zvel,
    const Real_t b[][8],
    const Real_t detJ,
    Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];

  d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
      + pfx[1] * (xvel[1]-xvel[7])
      + pfx[2] * (xvel[2]-xvel[4])
      + pfx[3] * (xvel[3]-xvel[5]) );

  d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
      + pfy[1] * (yvel[1]-yvel[7])
      + pfy[2] * (yvel[2]-yvel[4])
      + pfy[3] * (yvel[3]-yvel[5]) );

  d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
      + pfz[1] * (zvel[1]-zvel[7])
      + pfz[2] * (zvel[2]-zvel[4])
      + pfz[3] * (zvel[3]-zvel[5]) );

  dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
      + pfx[1] * (yvel[1]-yvel[7])
      + pfx[2] * (yvel[2]-yvel[4])
      + pfx[3] * (yvel[3]-yvel[5]) );

  dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
      + pfy[1] * (xvel[1]-xvel[7])
      + pfy[2] * (xvel[2]-xvel[4])
      + pfy[3] * (xvel[3]-xvel[5]) );

  dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
      + pfx[1] * (zvel[1]-zvel[7])
      + pfx[2] * (zvel[2]-zvel[4])
      + pfx[3] * (zvel[3]-zvel[5]) );

  dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
      + pfz[1] * (xvel[1]-xvel[7])
      + pfz[2] * (xvel[2]-xvel[4])
      + pfz[3] * (xvel[3]-xvel[5]) );

  dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
      + pfy[1] * (zvel[1]-zvel[7])
      + pfy[2] * (zvel[2]-zvel[4])
      + pfy[3] * (zvel[3]-zvel[5]) );

  dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
      + pfz[1] * (yvel[1]-yvel[7])
      + pfz[2] * (yvel[2]-yvel[4])
      + pfz[3] * (yvel[3]-yvel[5]) );
  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}
/******************************************/


/******************************************/

static inline
void CalcCourantConstraintForElems(Domain &domain, Index_t length,
    Index_t *regElemlist,
    Real_t qqc, Real_t& dtcourant)
{
  Index_t threads = NT;
  static Index_t *courant_elem_per_thread;
  static Real_t *dtcourant_per_thread;
  static bool first = true;
  if (first) {
    courant_elem_per_thread = new Index_t[threads];
    dtcourant_per_thread = new Real_t[threads];
    first = false;
  }

  {
    Real_t   qqc2 = Real_t(64.0) * qqc * qqc ;
    Real_t   dtcourant_tmp = dtcourant;
    Index_t  courant_elem  = -1 ;

    for (Index_t thread_num = 0; thread_num < NT; thread_num++) {

      //#pragma omp for 
      for (Index_t i = 0 ; i < length ; ++i) {
        Index_t indx = regElemlist[i] ;
        Real_t dtf = domain.ss(indx) * domain.ss(indx) ;

        if ( domain.vdov(indx) < Real_t(0.) ) {
          dtf = dtf
            + qqc2 * domain.arealg(indx) * domain.arealg(indx)
            * domain.vdov(indx) * domain.vdov(indx) ;
        }

        dtf = SQRT(dtf) ;
        dtf = domain.arealg(indx) / dtf ;

        if (domain.vdov(indx) != Real_t(0.)) {
          if ( dtf < dtcourant_tmp ) {
            dtcourant_tmp = dtf ;
            courant_elem  = indx ;
          }
        }
      }

      dtcourant_per_thread[thread_num]    = dtcourant_tmp ;
      courant_elem_per_thread[thread_num] = courant_elem ;
    }
  }

  for (Index_t i = 1; i < threads; ++i) {
    if (dtcourant_per_thread[i] < dtcourant_per_thread[0] ) {
      dtcourant_per_thread[0]    = dtcourant_per_thread[i];
      courant_elem_per_thread[0] = courant_elem_per_thread[i];
    }
  }

  if (courant_elem_per_thread[0] != -1) {
    dtcourant = dtcourant_per_thread[0] ;
  }

  return ;

}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain &domain, Index_t length,
    Index_t *regElemlist, Real_t dvovmax, Real_t& dthydro)
{
  Index_t threads = NT;
  static Index_t *hydro_elem_per_thread;
  static Real_t *dthydro_per_thread;
  static bool first = true;
  if (first) {
    hydro_elem_per_thread = new Index_t[threads];
    dthydro_per_thread = new Real_t[threads];
    first = false;
  }

  //#pragma omp parallel firstprivate(length, dvovmax)
  {
    Real_t dthydro_tmp = dthydro ;
    Index_t hydro_elem = -1 ;

    for (Index_t thread_num = 0; thread_num < NT; thread_num++) {

      //#pragma omp for
      for (Index_t i = 0 ; i < length ; ++i) {
        Index_t indx = regElemlist[i] ;

        if (domain.vdov(indx) != Real_t(0.)) {
          Real_t dtdvov = dvovmax / (FABS(domain.vdov(indx))+Real_t(1.e-20)) ;

          if ( dthydro_tmp > dtdvov ) {
            dthydro_tmp = dtdvov ;
            hydro_elem = indx ;
          }
        }
      }

      dthydro_per_thread[thread_num]    = dthydro_tmp ;
      hydro_elem_per_thread[thread_num] = hydro_elem ;
    }
  }

  for (Index_t i = 1; i < threads; ++i) {
    if(dthydro_per_thread[i] < dthydro_per_thread[0]) {
      dthydro_per_thread[0]    = dthydro_per_thread[i];
      hydro_elem_per_thread[0] =  hydro_elem_per_thread[i];
    }
  }

  if (hydro_elem_per_thread[0] != -1) {
    dthydro =  dthydro_per_thread[0] ;
  }

  return ;
}

/******************************************/

static inline
void CalcTimeConstraintsForElems(Domain& domain) {

  // Initialize conditions to a very large value
  domain.dtcourant() = 1.0e+20;
  domain.dthydro() = 1.0e+20;

  for (Index_t r=0 ; r < domain.numReg() ; ++r) {
    /* evaluate time constraint */
    CalcCourantConstraintForElems(domain, domain.regElemSize(r),
        domain.regElemlist(r),
        domain.qqc(),
        domain.dtcourant()) ;

    /* check hydro constraint */
    CalcHydroConstraintForElems(domain, domain.regElemSize(r),
        domain.regElemlist(r),
        domain.dvovmax(),
        domain.dthydro()) ;
  }
}
/******************************************/


/******************************************/

int main(int argc, char *argv[])
{
  Domain *locDom ;
  Int_t numRanks ;
  Int_t myRank ;
  struct cmdLineOpts opts;

  numRanks = 1;
  myRank = 0;

  /* Set defaults that can be overridden by command line opts */
  opts.its = 9999999;
  opts.nx  = 30;
  opts.numReg = 11;
  opts.numFiles = (int)(numRanks+10)/9;
  opts.showProg = 0;
  opts.quiet = 0;
  opts.viz = 0;
  opts.balance = 1;
  opts.cost = 1;
  opts.iteration_cap = 0;

  ParseCommandLineOptions(argc, argv, myRank, &opts);

  if ((myRank == 0) && (opts.quiet == 0)) {
    printf("Running problem size %d^3 per domain until completion\n", opts.nx);
    printf("Num processors: %d\n", numRanks);
    printf("Num threads (hardcoded): %d\n", NT);
    printf("Total number of elements: %lld\n\n", (long long int)(numRanks*opts.nx*opts.nx*opts.nx));
    printf("To run other sizes, use -s <integer>.\n");
    printf("To run a fixed number of iterations, use -i <integer>.\n");
    printf("To run a more or less balanced region set, use -b <integer>.\n");
    printf("To change the relative costs of regions, use -c <integer>.\n");
    printf("To print out progress, use -p\n");
    printf("To write an output file for VisIt, use -v\n");
    printf("To only execute the first iteration, use -z (used when profiling: nvprof --metrics all)\n");
    printf("See help (-h) for more options\n\n");
  }

  // Set up the mesh and decompose. Assumes regular cubes for now
  Int_t col, row, plane, side;
  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

  // Build the main data structure and initialize it
  locDom = new Domain(numRanks, col, row, plane, opts.nx,
      side, opts.numReg, opts.balance, opts.cost) ;


  // BEGIN timestep to solution */
  timeval start;
  gettimeofday(&start, NULL) ;
  // Compute elem to reglist correspondence
  Index_t k = 0;
  for (Int_t r=0 ; r<locDom->numReg() ; r++) {
    Index_t numElemReg = locDom->regElemSize(r);
    Index_t *regElemList = locDom->regElemlist(r);
    Index_t rep;
    //Determine load imbalance for this region
    //round down the number with lowest cost
    if(r < locDom->numReg()/2)
      rep = 1;
    //you don't get an expensive region unless you at least have 5 regions
    else if(r < (locDom->numReg() - (locDom->numReg()+15)/20))
      rep = 1 + locDom->cost();
    //very expensive regions
    else
      rep = 10 * (1+ locDom->cost());
    // std::cout << "Elems: " << numElemReg << " Reps: " << rep << "\n";
    for (Index_t e=0 ; e<numElemReg ; e++){
      locDom->m_elemRep[regElemList[e]] = rep;
      locDom->m_elemElem[k] = regElemList[e];
      k++;
    }
  }

  //export persistent data to GPU
  Index_t numNode = locDom->numNode();
  Index_t numElem = locDom->numElem() ;
  Index_t numElem8 = numElem * 8;

  printf("numNode=%d numElem=%d\n", numNode, numElem);

#ifdef VERIFY
  assert(numElem > 0);
#endif

  Real_t *x = &locDom->m_x[0];
  Real_t *y = &locDom->m_y[0];
  Real_t *z = &locDom->m_z[0];
  Real_t *fx = &locDom->m_fx[0];
  Real_t *fy = &locDom->m_fy[0];
  Real_t *fz = &locDom->m_fz[0];
  Real_t *xd = &locDom->m_xd[0];
  Real_t *yd = &locDom->m_yd[0];
  Real_t *zd = &locDom->m_zd[0];
  Real_t *xdd = &locDom->m_xdd[0];
  Real_t *ydd = &locDom->m_ydd[0];
  Real_t *zdd = &locDom->m_zdd[0];

  Real_t *p = &locDom->m_p[0];
  Real_t *q = &locDom->m_q[0];
  Real_t *volo = &locDom->m_volo[0];
  Real_t *nodalMass = &locDom->m_nodalMass[0];
  Index_t *symmX = &locDom->m_symmX[0];
  Index_t *symmY = &locDom->m_symmY[0];
  Index_t *symmZ = &locDom->m_symmZ[0];
  Real_t *delv = &locDom->m_delv[0];
  Real_t *arealg = &locDom->m_arealg[0];
  Real_t *dxx = &locDom->m_dxx[0];
  Real_t *dyy = &locDom->m_dyy[0];
  Real_t *dzz = &locDom->m_dzz[0];
  Index_t *elemBC = &locDom->m_elemBC[0];
  Index_t *lxim = &locDom->m_lxim[0];
  Index_t *lxip = &locDom->m_lxip[0];
  Index_t *letam = &locDom->m_letam[0];
  Index_t *letap = &locDom->m_letap[0];
  Index_t *lzetam = &locDom->m_lzetam[0];
  Index_t *lzetap = &locDom->m_lzetap[0];
  Real_t *elemMass = &locDom->m_elemMass[0];
  Index_t *elemRep = &locDom->m_elemRep[0];
  Index_t *elemElem = &locDom->m_elemElem[0];
  Real_t *v = &locDom->m_v[0];
  Real_t *vdov = &locDom->m_vdov[0];
  Real_t *e = &locDom->m_e[0];
  Real_t *ss = &locDom->m_ss[0];

  Index_t *nodelist = &locDom->m_nodelist[0];

#ifdef VERIFY
  std::mt19937 gen(19937);
  std::uniform_real_distribution<> dis(0.1, 1);
  for (int i = 0; i < numNode; i++) {
    xd[i] = dis(gen);
    yd[i] = dis(gen);
    zd[i] = dis(gen);
  }
#endif

  buffer<Real_t, 1> d_x(x, numNode);
  buffer<Real_t, 1> d_y(y, numNode);
  buffer<Real_t, 1> d_z(z, numNode);
  buffer<Real_t, 1> d_fx(fx, numNode);
  buffer<Real_t, 1> d_fy(fy, numNode);
  buffer<Real_t, 1> d_fz(fz, numNode);
  buffer<Real_t, 1> d_xd(xd, numNode);
  buffer<Real_t, 1> d_yd(yd, numNode);
  buffer<Real_t, 1> d_zd(zd, numNode);
  buffer<Real_t, 1> d_xdd(xdd, numNode);
  buffer<Real_t, 1> d_ydd(ydd, numNode);
  buffer<Real_t, 1> d_zdd(zdd, numNode);
  buffer<Index_t, 1> d_nodelist(nodelist, numElem8);
  d_x.set_final_data(nullptr);
  d_y.set_final_data(nullptr);
  d_z.set_final_data(nullptr);
  d_fx.set_final_data(nullptr);
  d_fy.set_final_data(nullptr);
  d_fz.set_final_data(nullptr);
  d_xd.set_final_data(nullptr);
  d_yd.set_final_data(nullptr);
  d_zd.set_final_data(nullptr);
  d_xdd.set_final_data(nullptr);
  d_ydd.set_final_data(nullptr);
  d_zdd.set_final_data(nullptr);

  // factor out the buffers from the loop
  buffer<Real_t, 1> d_determ(numElem);
  buffer<Real_t, 1> d_fx_elem(numElem8);
  buffer<Real_t, 1> d_fy_elem(numElem8);
  buffer<Real_t, 1> d_fz_elem(numElem8);
  buffer<Real_t, 1> d_dvdx(numElem8);
  buffer<Real_t, 1> d_dvdy(numElem8);
  buffer<Real_t, 1> d_dvdz(numElem8);
  buffer<Real_t, 1> d_x8n(numElem8);
  buffer<Real_t, 1> d_y8n(numElem8);
  buffer<Real_t, 1> d_z8n(numElem8);
  buffer<Real_t, 1> d_sigxx (numElem);
  buffer<Real_t, 1> d_sigyy (numElem);
  buffer<Real_t, 1> d_sigzz (numElem);
  buffer<Real_t, 1> d_delv_xi (numElem);
  buffer<Real_t, 1> d_delx_xi (numElem);
  buffer<Real_t, 1> d_delv_eta (numElem);
  buffer<Real_t, 1> d_delx_eta (numElem);
  buffer<Real_t, 1> d_delv_zeta (numElem);
  buffer<Real_t, 1> d_delx_zeta (numElem);
  buffer<Real_t, 1> d_p (p, numElem);
  buffer<Real_t, 1> d_q (q, numElem);
  buffer<Real_t, 1> d_volo(volo, numElem);
  buffer<Real_t, 1> d_v(v, numElem);
  buffer<int, 1> d_vol_error(1);

  Index_t *nodeElemStart = &locDom->m_nodeElemStart[0];
  Index_t len1 = numNode + 1;
  Index_t *nodeElemCornerList = &locDom->m_nodeElemCornerList[0];
  Index_t len2 = nodeElemStart[numNode];

  buffer<Index_t, 1> d_nodeElemStart(nodeElemStart, len1); 
  buffer<Index_t, 1> d_nodeElemCornerList(nodeElemCornerList, len2); 

  Real_t  gamma[32] __attribute__((__aligned__(64)));
  gamma[0] = Real_t( 1.);
  gamma[1] = Real_t( 1.);
  gamma[2] = Real_t(-1.);
  gamma[3] = Real_t(-1.);
  gamma[4] = Real_t(-1.);
  gamma[5] = Real_t(-1.);
  gamma[6] = Real_t( 1.);
  gamma[7] = Real_t( 1.);
  gamma[8] = Real_t( 1.);
  gamma[9] = Real_t(-1.);
  gamma[10] = Real_t(-1.);
  gamma[11] = Real_t( 1.);
  gamma[12] = Real_t(-1.);
  gamma[13] = Real_t( 1.);
  gamma[14] = Real_t( 1.);
  gamma[15] = Real_t(-1.);
  gamma[16] = Real_t( 1.);
  gamma[17] = Real_t(-1.);
  gamma[18] = Real_t( 1.);
  gamma[19] = Real_t(-1.);
  gamma[20] = Real_t( 1.);
  gamma[21] = Real_t(-1.);
  gamma[22] = Real_t( 1.);
  gamma[23] = Real_t(-1.);
  gamma[24] = Real_t(-1.);
  gamma[25] = Real_t( 1.);
  gamma[26] = Real_t(-1.);
  gamma[27] = Real_t( 1.);
  gamma[28] = Real_t( 1.);
  gamma[29] = Real_t(-1.);
  gamma[30] = Real_t( 1.);
  gamma[31] = Real_t(-1.);

  buffer<Real_t, 1> d_gamma(gamma, 32);
  buffer<Real_t, 1> d_ss(ss, numElem);
  buffer<Real_t, 1> d_elemMass(elemMass, numElem);
  buffer<Real_t, 1> d_nodalMass(nodalMass, numNode);

  Index_t size = locDom->sizeX();
  Index_t numNodeBC = (size+1)*(size+1) ;
  buffer<Index_t, 1> d_symmX(symmX, numNodeBC);
  buffer<Index_t, 1> d_symmY(symmY, numNodeBC);
  buffer<Index_t, 1> d_symmZ(symmZ, numNodeBC);

  buffer<Real_t, 1> d_vdov (vdov, numElem);
  buffer<Real_t, 1> d_delv (delv, numElem);
  buffer<Real_t, 1> d_arealg (arealg, numElem);
  buffer<Real_t, 1> d_dxx (dxx, numElem);
  buffer<Real_t, 1> d_dyy (dyy, numElem);
  buffer<Real_t, 1> d_dzz (dzz, numElem);
  buffer<Real_t, 1> d_vnew (numElem);

  buffer<Index_t, 1> d_lzetam(lzetam, numElem);
  buffer<Index_t, 1> d_lzetap(lzetap, numElem);
  buffer<Index_t, 1> d_letap(letap, numElem);
  buffer<Index_t, 1> d_letam(letam, numElem);
  buffer<Index_t, 1> d_lxip(lxip, numElem);
  buffer<Index_t, 1> d_lxim(lxim, numElem);
  buffer<Index_t, 1> d_elemBC(elemBC, numElem);
  buffer<Real_t, 1> d_ql(numElem);
  buffer<Real_t, 1> d_qq(numElem);
  buffer<Real_t, 1> d_e (e, numElem);
  buffer<Index_t, 1> d_elemRep (elemRep, numElem);
  buffer<Index_t, 1> d_elemElem (elemElem, numElem);

  d_symmX.set_final_data(nullptr);
  d_symmY.set_final_data(nullptr);
  d_symmZ.set_final_data(nullptr);
  d_dxx.set_final_data(nullptr);
  d_dyy.set_final_data(nullptr);
  d_dzz.set_final_data(nullptr);
  d_ss.set_final_data(nullptr);
  d_vdov.set_final_data(nullptr);
  d_delv.set_final_data(nullptr);
  d_arealg.set_final_data(nullptr);
  d_p.set_final_data(nullptr);
  d_q.set_final_data(nullptr);
  d_e.set_final_data(nullptr);
  d_v.set_final_data(nullptr);
  d_nodalMass.set_final_data(nullptr);
  d_elemMass.set_final_data(nullptr);
  d_elemRep.set_final_data(nullptr);
  d_elemElem.set_final_data(nullptr);
  d_elemBC.set_final_data(nullptr);
  d_lzetam.set_final_data(nullptr);
  d_lzetap.set_final_data(nullptr);
  d_letap.set_final_data(nullptr);
  d_letam.set_final_data(nullptr);
  d_lxip.set_final_data(nullptr);
  d_lxim.set_final_data(nullptr);

  sycl::queue &device_queue = locDom->device_queue;

  // error checking on the host
  Real_t *determ = Allocate<Real_t>(numElem) ;

  // resize m_dxx, m_dyy, and m_dzz
  locDom->AllocateStrains(numElem);

  // resize position and velocity gradients
  Int_t allElem = numElem +  /* local elem */
      2*locDom->sizeX()*locDom->sizeY() + /* plane ghosts */
      2*locDom->sizeX()*locDom->sizeZ() + /* row ghosts */
      2*locDom->sizeY()*locDom->sizeZ() ; /* col ghosts */

  locDom->AllocateGradients(numElem, allElem);

  while((locDom->time() < locDom->stoptime()) && (locDom->cycle() < opts.its)) {

    TimeIncrement(*locDom) ;
    //==============================================================================
    // LagrangeLeapFrog(*locDom) ;
    //==============================================================================

    //=============================================================
    // calculate nodal forces, accelerations, velocities, positions, with
    // applied boundary conditions and slide surface considerations 
    //LagrangeNodal(domain);
    //=============================================================

    Domain &domain = *locDom;

    const Real_t deltaTime = domain.deltatime() ;
    Real_t u_cut = domain.u_cut() ;

    //=============================================================================
    // time of boundary condition evaluation is beginning of step for force and
    // acceleration boundary conditions. 
    //CalcForceForNodes(domain);  
    //=============================================================================

    //=====================================================================
    // CalcVolumeForceForElems(domain) 
    //=====================================================================

    Real_t  hgcoef = domain.hgcoef() ;

    range<1> gws_elem ((numElem+THREADS-1)/THREADS*THREADS);
    range<1> gws_node ((numNode+THREADS-1)/THREADS*THREADS);
    range<1> lws (THREADS);

    device_queue.submit([&] (handler &cgh) {
      auto sigxx = d_sigxx.get_access<sycl_discard_write>(cgh);
      auto sigyy = d_sigyy.get_access<sycl_discard_write>(cgh);
      auto sigzz = d_sigzz.get_access<sycl_discard_write>(cgh);
      auto p = d_p.get_access<sycl_read>(cgh);
      auto q = d_q.get_access<sycl_read>(cgh);
      cgh.parallel_for<class hgc>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        Index_t i = item.get_global_id(0);
        if (i >= numElem) return;
        sigxx[i] = sigyy[i] = sigzz[i] = - p[i] - q[i] ;
      });
    });

    //==============================================================================================
    // IntegrateStressForElems( domain, sigxx, sigyy, sigzz, determ, numElem, domain.numNode())
    //==============================================================================================

    device_queue.submit([&] (handler &cgh) {
      auto fx_elem = d_fx_elem.get_access<sycl_discard_write>(cgh);
      auto fy_elem = d_fy_elem.get_access<sycl_discard_write>(cgh);
      auto fz_elem = d_fz_elem.get_access<sycl_discard_write>(cgh);
      auto x = d_x.get_access<sycl_read>(cgh);
      auto y = d_y.get_access<sycl_read>(cgh);
      auto z = d_z.get_access<sycl_read>(cgh);
      auto nodelist = d_nodelist.get_access<sycl_read>(cgh);
      auto sigxx = d_sigxx.get_access<sycl_read>(cgh);
      auto sigyy = d_sigyy.get_access<sycl_read>(cgh);
      auto sigzz = d_sigzz.get_access<sycl_read>(cgh);
      auto determ = d_determ.get_access<sycl_discard_write>(cgh);

      cgh.parallel_for<class integrateStress>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        Index_t k = item.get_global_id(0);
        if (k >= numElem) return;

        const Index_t* const elemToNode = nodelist.get_pointer() + Index_t(8)*k;
        Real_t B[3][8] ;// shape function derivatives
        Real_t x_local[8] ;
        Real_t y_local[8] ;
        Real_t z_local[8] ;
        determ[k] = Real_t(10.0);

        // get nodal coordinates from global arrays and copy into local arrays.
        Index_t nd0i = elemToNode[0] ;
        Index_t nd1i = elemToNode[1] ;
        Index_t nd2i = elemToNode[2] ;
        Index_t nd3i = elemToNode[3] ;
        Index_t nd4i = elemToNode[4] ;
        Index_t nd5i = elemToNode[5] ;
        Index_t nd6i = elemToNode[6] ;
        Index_t nd7i = elemToNode[7] ;

        x_local[0] = x[nd0i];
        x_local[1] = x[nd1i];
        x_local[2] = x[nd2i];
        x_local[3] = x[nd3i];
        x_local[4] = x[nd4i];
        x_local[5] = x[nd5i];
        x_local[6] = x[nd6i];
        x_local[7] = x[nd7i];

        y_local[0] = y[nd0i];
        y_local[1] = y[nd1i];
        y_local[2] = y[nd2i];
        y_local[3] = y[nd3i];
        y_local[4] = y[nd4i];
        y_local[5] = y[nd5i];
        y_local[6] = y[nd6i];
        y_local[7] = y[nd7i];

        z_local[0] = z[nd0i];
        z_local[1] = z[nd1i];
        z_local[2] = z[nd2i];
        z_local[3] = z[nd3i];
        z_local[4] = z[nd4i];
        z_local[5] = z[nd5i];
        z_local[6] = z[nd6i];
        z_local[7] = z[nd7i];

        // Volume calculation involves extra work for numerical consistency
        CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &determ[k]);

        CalcElemNodeNormals( B[0], B[1], B[2], x_local, y_local, z_local );

        // Eliminate thread writing conflicts at the nodes by giving
        // each element its own copy to write to
        SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
            &fx_elem[k*8],
            &fy_elem[k*8],
            &fz_elem[k*8] ) ;
      });
    });

    device_queue.submit([&] (handler &cgh) {
      auto fx_elem = d_fx_elem.get_access<sycl_read>(cgh);
      auto fy_elem = d_fy_elem.get_access<sycl_read>(cgh);
      auto fz_elem = d_fz_elem.get_access<sycl_read>(cgh);
      auto fx = d_fx.get_access<sycl_discard_write>(cgh);
      auto fy = d_fy.get_access<sycl_discard_write>(cgh);
      auto fz = d_fz.get_access<sycl_discard_write>(cgh);
      auto nodeElemStart = d_nodeElemStart.get_access<sycl_read>(cgh);
      auto nodeElemCornerList = d_nodeElemCornerList.get_access<sycl_read>(cgh);

      cgh.parallel_for<class acc_final_force>(nd_range<1>(gws_node, lws), [=] (nd_item<1> item) {
        Index_t gnode = item.get_global_id(0);
        if (gnode >= numNode) return;
        // element count
        const Index_t count = nodeElemStart[gnode+1] - nodeElemStart[gnode];//domain.nodeElemCount(gnode) ;
        // list of all corners
        const Index_t *cornerList = nodeElemCornerList.get_pointer() + nodeElemStart[gnode];//domain.nodeElemCornerList(gnode) ;
        Real_t fx_tmp = Real_t(0.0) ;
        Real_t fy_tmp = Real_t(0.0) ;
        Real_t fz_tmp = Real_t(0.0) ;
        for (Index_t i=0 ; i < count ; ++i) {
          Index_t elem = cornerList[i] ;
          fx_tmp += fx_elem[elem] ;
          fy_tmp += fy_elem[elem] ;
          fz_tmp += fz_elem[elem] ;
        }
        fx[gnode] = fx_tmp ;
        fy[gnode] = fy_tmp ;
        fz[gnode] = fz_tmp ;
      });
    });

    // check for negative element volume on the host

    device_queue.submit([&] (handler &cgh) {
      auto d_acc = d_determ.get_access<sycl_read>(cgh);
      cgh.copy(d_acc, determ);
    }).wait();

#ifdef _OPENMP
#pragma omp parallel for firstprivate(numElem)
#endif
    for ( Index_t k=0 ; k<numElem ; ++k ) {
      if (determ[k] <= Real_t(0.0)) {
        exit(VolumeError);
      }
    }

    //=================================================================================
    // CalcHourglassControlForElems(device_queue, domain, determ, hgcoef) ;  
    //=================================================================================

    int vol_error = -1;

    device_queue.submit([&] (handler &cgh) {
      auto acc = d_vol_error.get_access<sycl_write>(cgh);
      cgh.copy(&vol_error, acc);
    });

    device_queue.submit([&] (handler &cgh) {
      auto dvdx = d_dvdx.get_access<sycl_discard_write>(cgh);
      auto dvdy = d_dvdy.get_access<sycl_discard_write>(cgh);
      auto dvdz = d_dvdz.get_access<sycl_discard_write>(cgh);
      auto x8n = d_x8n.get_access<sycl_discard_write>(cgh);
      auto y8n = d_y8n.get_access<sycl_discard_write>(cgh);
      auto z8n = d_z8n.get_access<sycl_discard_write>(cgh);
      auto determ = d_determ.get_access<sycl_write>(cgh);
      auto x = d_x.get_access<sycl_read>(cgh);
      auto y = d_y.get_access<sycl_read>(cgh);
      auto z = d_z.get_access<sycl_read>(cgh);
      auto nodelist = d_nodelist.get_access<sycl_read>(cgh);
      auto volo = d_volo.get_access<sycl_read>(cgh);
      auto v = d_v.get_access<sycl_read>(cgh);
      auto vol_error = d_vol_error.get_access<sycl_write>(cgh);

      cgh.parallel_for<class hgcfe>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i >= numElem) return;

        Real_t  x1[8],  y1[8],  z1[8] ;
        Real_t pfx[8], pfy[8], pfz[8] ;

        const Index_t* elemToNode = nodelist.get_pointer() + Index_t(8)*i;

        // inline the function manually
        Index_t nd0i = elemToNode[0] ;
        Index_t nd1i = elemToNode[1] ;
        Index_t nd2i = elemToNode[2] ;
        Index_t nd3i = elemToNode[3] ;
        Index_t nd4i = elemToNode[4] ;
        Index_t nd5i = elemToNode[5] ;
        Index_t nd6i = elemToNode[6] ;
        Index_t nd7i = elemToNode[7] ;

        x1[0] = x[nd0i];
        x1[1] = x[nd1i];
        x1[2] = x[nd2i];
        x1[3] = x[nd3i];
        x1[4] = x[nd4i];
        x1[5] = x[nd5i];
        x1[6] = x[nd6i];
        x1[7] = x[nd7i];

        y1[0] = y[nd0i];
        y1[1] = y[nd1i];
        y1[2] = y[nd2i];
        y1[3] = y[nd3i];
        y1[4] = y[nd4i];
        y1[5] = y[nd5i];
        y1[6] = y[nd6i];
        y1[7] = y[nd7i];

        z1[0] = z[nd0i];
        z1[1] = z[nd1i];
        z1[2] = z[nd2i];
        z1[3] = z[nd3i];
        z1[4] = z[nd4i];
        z1[5] = z[nd5i];
        z1[6] = z[nd6i];
        z1[7] = z[nd7i];

        CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

        /* load into temporary storage for FB Hour Glass control */
        for(Index_t ii=0;ii<8;++ii){
          Index_t jj=8*i+ii;

          dvdx[jj] = pfx[ii];
          dvdy[jj] = pfy[ii];
          dvdz[jj] = pfz[ii];
          x8n[jj]  = x1[ii];
          y8n[jj]  = y1[ii];
          z8n[jj]  = z1[ii];
        }

        determ[i] = volo[i] * v[i];

        /* Do a check for negative volumes */
        if ( v[i] <= Real_t(0.0) ) {
          vol_error[0] = i;
        }
      });
    });

#ifdef VERIFY
    auto dvdx = d_dvdx.get_access<sycl_read>();
    auto dvdy = d_dvdy.get_access<sycl_read>();
    auto dvdz = d_dvdz.get_access<sycl_read>();
    auto x8n = d_x8n.get_access<sycl_read>();
    auto y8n = d_y8n.get_access<sycl_read>();
    auto z8n = d_z8n.get_access<sycl_read>();

    device_queue.submit([&] (handler &cgh) {
      auto d_acc = d_determ.get_access<sycl_read>(cgh);
      cgh.copy(d_acc, determ);
    }).wait();

    // volumn derivative
    for (int i = 0; i < numElem8; i++) {
      printf("vd %d %f %f %f %f %f %f %f\n", 
          i, dvdx[i], dvdy[i], dvdz[i], x8n[i], y8n[i], z8n[i], determ[i/8]);
    }
#endif

    device_queue.submit([&] (handler &cgh) {
      auto vol_error_acc = d_vol_error.get_access<sycl_read>(cgh);
      cgh.copy(vol_error_acc, &vol_error);
    }).wait();

    if (vol_error >= 0){
      printf("VolumeError: negative volumn\n");
      exit(VolumeError);
    }

    if ( hgcoef > Real_t(0.) ) {

      //Index_t *nodeElemStart = &domain.m_nodeElemStart[0];
      //Index_t len1 = numNode + 1;
      //Index_t *nodeElemCornerList = &domain.m_nodeElemCornerList[0];
      //Index_t len2 = nodeElemStart[numNode];

      device_queue.submit([&] (handler &cgh) {
        auto dvdx = d_dvdx.get_access<sycl_read>(cgh);
        auto dvdy = d_dvdy.get_access<sycl_read>(cgh);
        auto dvdz = d_dvdz.get_access<sycl_read>(cgh);
        auto x8n = d_x8n.get_access<sycl_read>(cgh);
        auto y8n = d_y8n.get_access<sycl_read>(cgh);
        auto z8n = d_z8n.get_access<sycl_read>(cgh);
        auto determ = d_determ.get_access<sycl_read>(cgh);
        auto xd = d_xd.get_access<sycl_read>(cgh);
        auto yd = d_yd.get_access<sycl_read>(cgh);
        auto zd = d_zd.get_access<sycl_read>(cgh);
        auto ss = d_ss.get_access<sycl_read>(cgh);
        auto elemMass = d_elemMass.get_access<sycl_read>(cgh);
        auto nodelist = d_nodelist.get_access<sycl_read>(cgh);
        auto gamma = d_gamma.get_access<sycl_read>(cgh);
        auto fx_elem = d_fx_elem.get_access<sycl_discard_write>(cgh);
        auto fy_elem = d_fy_elem.get_access<sycl_discard_write>(cgh);
        auto fz_elem = d_fz_elem.get_access<sycl_discard_write>(cgh);

        cgh.parallel_for<class fb>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
          int i2 = item.get_global_id(0);
          if (i2 >= numElem) return;

          Index_t i3 = 8*i2;

          const Index_t* elemToNode = nodelist.get_pointer() + i3;

          Real_t hgfx[8], hgfy[8], hgfz[8] ;

          Real_t coefficient;

          Real_t hourgam[8][4];
          Real_t xd1[8], yd1[8], zd1[8] ;

          Real_t volinv = ONE/determ[i2];
          Real_t ss1, mass1, volume13 ;


          for(Index_t i1=0;i1<4;++i1) {

            Real_t hourmodx =
              x8n[i3]   * gamma[i1*8+0] + x8n[i3+1] * gamma[i1*8+1] +
              x8n[i3+2] * gamma[i1*8+2] + x8n[i3+3] * gamma[i1*8+3] +
              x8n[i3+4] * gamma[i1*8+4] + x8n[i3+5] * gamma[i1*8+5] +
              x8n[i3+6] * gamma[i1*8+6] + x8n[i3+7] * gamma[i1*8+7];

            Real_t hourmody =
              y8n[i3]   * gamma[i1*8+0] + y8n[i3+1] * gamma[i1*8+1] +
              y8n[i3+2] * gamma[i1*8+2] + y8n[i3+3] * gamma[i1*8+3] +
              y8n[i3+4] * gamma[i1*8+4] + y8n[i3+5] * gamma[i1*8+5] +
              y8n[i3+6] * gamma[i1*8+6] + y8n[i3+7] * gamma[i1*8+7];

            Real_t hourmodz =
              z8n[i3]   * gamma[i1*8+0] + z8n[i3+1] * gamma[i1*8+1] +
              z8n[i3+2] * gamma[i1*8+2] + z8n[i3+3] * gamma[i1*8+3] +
              z8n[i3+4] * gamma[i1*8+4] + z8n[i3+5] * gamma[i1*8+5] +
              z8n[i3+6] * gamma[i1*8+6] + z8n[i3+7] * gamma[i1*8+7];

            hourgam[0][i1] = gamma[i1*8+0] - volinv*(dvdx[i3  ] * hourmodx +
                dvdy[i3  ] * hourmody +
                dvdz[i3  ] * hourmodz );

            hourgam[1][i1] = gamma[i1*8+1] - volinv*(dvdx[i3+1] * hourmodx +
                dvdy[i3+1] * hourmody +
                dvdz[i3+1] * hourmodz );

            hourgam[2][i1] = gamma[i1*8+2] - volinv*(dvdx[i3+2] * hourmodx +
                dvdy[i3+2] * hourmody +
                dvdz[i3+2] * hourmodz );

            hourgam[3][i1] = gamma[i1*8+3] - volinv*(dvdx[i3+3] * hourmodx +
                dvdy[i3+3] * hourmody +
                dvdz[i3+3] * hourmodz );

            hourgam[4][i1] = gamma[i1*8+4] - volinv*(dvdx[i3+4] * hourmodx +
                dvdy[i3+4] * hourmody +
                dvdz[i3+4] * hourmodz );

            hourgam[5][i1] = gamma[i1*8+5] - volinv*(dvdx[i3+5] * hourmodx +
                dvdy[i3+5] * hourmody +
                dvdz[i3+5] * hourmodz );

            hourgam[6][i1] = gamma[i1*8+6] - volinv*(dvdx[i3+6] * hourmodx +
                dvdy[i3+6] * hourmody +
                dvdz[i3+6] * hourmodz );

            hourgam[7][i1] = gamma[i1*8+7] - volinv*(dvdx[i3+7] * hourmodx +
                dvdy[i3+7] * hourmody +
                dvdz[i3+7] * hourmodz );

          }

          /* compute forces */
          /* store forces into h arrays (force arrays) */

          ss1 = ss[i2];
          mass1 = elemMass[i2];
          volume13 = sycl::cbrt(determ[i2]);

          Index_t n0si2 = elemToNode[0];
          Index_t n1si2 = elemToNode[1];
          Index_t n2si2 = elemToNode[2];
          Index_t n3si2 = elemToNode[3];
          Index_t n4si2 = elemToNode[4];
          Index_t n5si2 = elemToNode[5];
          Index_t n6si2 = elemToNode[6];
          Index_t n7si2 = elemToNode[7];

          xd1[0] = xd[n0si2];
          xd1[1] = xd[n1si2];
          xd1[2] = xd[n2si2];
          xd1[3] = xd[n3si2];
          xd1[4] = xd[n4si2];
          xd1[5] = xd[n5si2];
          xd1[6] = xd[n6si2];
          xd1[7] = xd[n7si2];

          yd1[0] = yd[n0si2];
          yd1[1] = yd[n1si2];
          yd1[2] = yd[n2si2];
          yd1[3] = yd[n3si2];
          yd1[4] = yd[n4si2];
          yd1[5] = yd[n5si2];
          yd1[6] = yd[n6si2];
          yd1[7] = yd[n7si2];

          zd1[0] = zd[n0si2];
          zd1[1] = zd[n1si2];
          zd1[2] = zd[n2si2];
          zd1[3] = zd[n3si2];
          zd1[4] = zd[n4si2];
          zd1[5] = zd[n5si2];
          zd1[6] = zd[n6si2];
          zd1[7] = zd[n7si2];

          coefficient = hgcoef * Real_t(-0.01) * ss1 * mass1 / volume13;

          Real_t hxx[4], hyy[4], hzz[4];

          for(Index_t i = 0; i < 4; i++) {
            hxx[i] = hourgam[0][i] * xd1[0] + hourgam[1][i] * xd1[1] +
              hourgam[2][i] * xd1[2] + hourgam[3][i] * xd1[3] +
              hourgam[4][i] * xd1[4] + hourgam[5][i] * xd1[5] +
              hourgam[6][i] * xd1[6] + hourgam[7][i] * xd1[7];
          }
          for(Index_t i = 0; i < 8; i++) {
            hgfx[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
          }
          for(Index_t i = 0; i < 4; i++) {
            hyy[i] = hourgam[0][i] * yd1[0] + hourgam[1][i] * yd1[1] +
              hourgam[2][i] * yd1[2] + hourgam[3][i] * yd1[3] +
              hourgam[4][i] * yd1[4] + hourgam[5][i] * yd1[5] +
              hourgam[6][i] * yd1[6] + hourgam[7][i] * yd1[7];
          }
          for(Index_t i = 0; i < 8; i++) {
            hgfy[i] = coefficient *
              (hourgam[i][0] * hyy[0] + hourgam[i][1] * hyy[1] +
               hourgam[i][2] * hyy[2] + hourgam[i][3] * hyy[3]);
          }
          for(Index_t i = 0; i < 4; i++) {
            hzz[i] = hourgam[0][i] * zd1[0] + hourgam[1][i] * zd1[1] +
              hourgam[2][i] * zd1[2] + hourgam[3][i] * zd1[3] +
              hourgam[4][i] * zd1[4] + hourgam[5][i] * zd1[5] +
              hourgam[6][i] * zd1[6] + hourgam[7][i] * zd1[7];
          }
          for(Index_t i = 0; i < 8; i++) {
            hgfz[i] = coefficient *
              (hourgam[i][0] * hzz[0] + hourgam[i][1] * hzz[1] +
               hourgam[i][2] * hzz[2] + hourgam[i][3] * hzz[3]);
          }

          // With the threaded version, we write into local arrays per elem
          // so we don't have to worry about race conditions

          Real_t *fx_local = fx_elem.get_pointer() + i3 ;
          fx_local[0] = hgfx[0];
          fx_local[1] = hgfx[1];
          fx_local[2] = hgfx[2];
          fx_local[3] = hgfx[3];
          fx_local[4] = hgfx[4];
          fx_local[5] = hgfx[5];
          fx_local[6] = hgfx[6];
          fx_local[7] = hgfx[7];

          Real_t *fy_local = fy_elem.get_pointer() + i3 ;
          fy_local[0] = hgfy[0];
          fy_local[1] = hgfy[1];
          fy_local[2] = hgfy[2];
          fy_local[3] = hgfy[3];
          fy_local[4] = hgfy[4];
          fy_local[5] = hgfy[5];
          fy_local[6] = hgfy[6];
          fy_local[7] = hgfy[7];

          Real_t *fz_local = fz_elem.get_pointer() + i3 ;
          fz_local[0] = hgfz[0];
          fz_local[1] = hgfz[1];
          fz_local[2] = hgfz[2];
          fz_local[3] = hgfz[3];
          fz_local[4] = hgfz[4];
          fz_local[5] = hgfz[5];
          fz_local[6] = hgfz[6];
          fz_local[7] = hgfz[7];
        });
      });

      device_queue.submit([&] (handler &cgh) {
        auto fx_elem = d_fx_elem.get_access<sycl_read>(cgh);
        auto fy_elem = d_fy_elem.get_access<sycl_read>(cgh);
        auto fz_elem = d_fz_elem.get_access<sycl_read>(cgh);
        auto fx = d_fx.get_access<sycl_discard_write>(cgh);
        auto fy = d_fy.get_access<sycl_discard_write>(cgh);
        auto fz = d_fz.get_access<sycl_discard_write>(cgh);
        auto nodeElemStart = d_nodeElemStart.get_access<sycl_read>(cgh);
        auto nodeElemCornerList = d_nodeElemCornerList.get_access<sycl_read>(cgh);

        cgh.parallel_for<class collect_final_force>(nd_range<1>(gws_node, lws), [=] (nd_item<1> item) {
          Index_t gnode = item.get_global_id(0);
          if (gnode >= numNode) return;
          // element count
          const Index_t count = nodeElemStart[gnode+1] - nodeElemStart[gnode];//domain.nodeElemCount(gnode) ;
          // list of all corners
          const Index_t *cornerList = nodeElemCornerList.get_pointer() + nodeElemStart[gnode];//domain.nodeElemCornerList(gnode) ;
          Real_t fx_tmp = Real_t(0.0) ;
          Real_t fy_tmp = Real_t(0.0) ;
          Real_t fz_tmp = Real_t(0.0) ;
          for (Index_t i=0 ; i < count ; ++i) {
            Index_t elem = cornerList[i] ;
            fx_tmp += fx_elem[elem] ;
            fy_tmp += fy_elem[elem] ;
            fz_tmp += fz_elem[elem] ;
          }
          fx[gnode] = fx_tmp ;
          fy[gnode] = fy_tmp ;
          fz[gnode] = fz_tmp ;
        });
      });

#ifdef VERIFY
      device_queue.submit([&] (handler &cgh) {
        auto acc = d_fx.get_access<sycl_read>(cgh);
        cgh.copy(acc, fx);
      });
      device_queue.submit([&] (handler &cgh) {
        auto acc = d_fy.get_access<sycl_read>(cgh);
        cgh.copy(acc, fy);
      });
      device_queue.submit([&] (handler &cgh) {
        auto acc = d_fz.get_access<sycl_read>(cgh);
        cgh.copy(acc, fz);
      });

      device_queue.wait();

      for (int i = 0; i < numNode; i++)
        printf("fb: %d %f %f %f\n", i, fx[i], fy[i], fz[i]); 
#endif 
    } // if ( hgcoef > Real_t(0.) ) 


    //===========================================================================
    //CalcAccelerationForNodes(domain, domain.numNode());   // IN: fx  OUT: m_xdd
    //===========================================================================

    device_queue.submit([&] (handler &cgh) {
      auto fx = d_fx.get_access<sycl_read>(cgh);
      auto fy = d_fy.get_access<sycl_read>(cgh);
      auto fz = d_fz.get_access<sycl_read>(cgh);
      auto nodalMass = d_nodalMass.get_access<sycl_read>(cgh);
      auto xdd = d_xdd.get_access<sycl_discard_write>(cgh);
      auto ydd = d_ydd.get_access<sycl_discard_write>(cgh);
      auto zdd = d_zdd.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class AccelerationForNodes>(
        nd_range<1>(gws_node, lws), [=] (nd_item<1> item) {
        Index_t i = item.get_global_id(0);
        if (i >= numNode) return;
        Real_t one_over_nMass = Real_t(1.) / nodalMass[i];
        xdd[i] = fx[i] * one_over_nMass;
        ydd[i] = fy[i] * one_over_nMass;
        zdd[i] = fz[i] * one_over_nMass;
      });
    });

    //======================================================================================
    //ApplyAccelerationBoundaryConditionsForNodes(domain); // uses m_xdd
    //======================================================================================
    //Index_t size = domain.sizeX();
    //Index_t numNodeBC = (size+1)*(size+1) ;


    Index_t s1 = domain.symmXempty();
    Index_t s2 = domain.symmYempty();
    Index_t s3 = domain.symmZempty();

    device_queue.submit([&] (handler &cgh) {
      auto symmX = d_symmX.get_access<sycl_read>(cgh);
      auto symmY = d_symmY.get_access<sycl_read>(cgh);
      auto symmZ = d_symmZ.get_access<sycl_read>(cgh);
      auto xdd = d_xdd.get_access<sycl_write>(cgh);
      auto ydd = d_ydd.get_access<sycl_write>(cgh);
      auto zdd = d_zdd.get_access<sycl_write>(cgh);
      cgh.parallel_for<class ApplyAccelerationBoundaryConditionsForNodes>(
        nd_range<1>(range<1>((numNodeBC+255)/256*256), range<1>(256)), [=] (nd_item<1> item) {
        Index_t i = item.get_global_id(0);
        if (i >= numNodeBC) return;
        if ((!s1) != 0) xdd[symmX[i]] = Real_t(0.0) ;
        if ((!s2) != 0) ydd[symmY[i]] = Real_t(0.0) ;
        if ((!s3) != 0) zdd[symmZ[i]] = Real_t(0.0) ;
      });
    });

    //=================================================================
    // CalcVelocityForNodes( domain, delt, u_cut, domain.numNode()) ; //uses m_xd and m_xdd
    //=================================================================

    device_queue.submit([&] (handler &cgh) {
      auto xd = d_xd.get_access<sycl_read_write>(cgh);
      auto yd = d_yd.get_access<sycl_read_write>(cgh);
      auto zd = d_zd.get_access<sycl_read_write>(cgh);
      auto xdd = d_xdd.get_access<sycl_read>(cgh);
      auto ydd = d_ydd.get_access<sycl_read>(cgh);
      auto zdd = d_zdd.get_access<sycl_read>(cgh);
      cgh.parallel_for<class CalcVelocityForNodes>(
        nd_range<1>(gws_node, lws), [=] (nd_item<1> item) {
        Index_t i = item.get_global_id(0);
        if (i >= numNode) return;

        Real_t xdtmp = xd[i] + xdd[i] * deltaTime;
        // FABS is not compiled with target regions in mind
        // To get around this, compute the absolute value manually:
        // if( xdtmp > Real_t(0.0) && xdtmp < u_cut || Real_t(-1.0) * xdtmp < u_cut)
        if( sycl::fabs(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
        xd[i] = xdtmp ;

        Real_t ydtmp = yd[i] + ydd[i] * deltaTime;
        if( sycl::fabs(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
        yd[i] = ydtmp ;

        Real_t zdtmp = zd[i] + zdd[i] * deltaTime;
        if( sycl::fabs(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
        zd[i] = zdtmp ;
      });
    });

    //=================================================================================
    // CalcPositionForNodes( domain, delt, domain.numNode() );  //uses m_xd and m_x 
    //=================================================================================
    device_queue.submit([&] (handler &cgh) {
      auto x = d_x.get_access<sycl_read_write>(cgh);
      auto y = d_y.get_access<sycl_read_write>(cgh);
      auto z = d_z.get_access<sycl_read_write>(cgh);
      auto xd = d_xd.get_access<sycl_read>(cgh);
      auto yd = d_yd.get_access<sycl_read>(cgh);
      auto zd = d_zd.get_access<sycl_read>(cgh);
      cgh.parallel_for<class CalcPositionForNodes>(
          nd_range<1>(gws_node, lws), [=] (nd_item<1> item) {
          Index_t i = item.get_global_id(0);
          if (i >= numNode) return;
          x[i] += xd[i] * deltaTime;
          y[i] += yd[i] * deltaTime;
          z[i] += zd[i] * deltaTime;
      });
    });

#ifdef VERIFY
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_xd.get_access<sycl_read>(cgh);
      cgh.copy(acc, xd);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_yd.get_access<sycl_read>(cgh);
      cgh.copy(acc, yd);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_zd.get_access<sycl_read>(cgh);
      cgh.copy(acc, zd);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_x.get_access<sycl_read>(cgh);
      cgh.copy(acc, x);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_y.get_access<sycl_read>(cgh);
      cgh.copy(acc, y);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_z.get_access<sycl_read>(cgh);
      cgh.copy(acc, z);
    });

    device_queue.wait();

    for (int i = 0; i < numNode; i++)
      printf("CalcPositionForNodes: %d %f %f %f %f %f %f\n", 
          i, x[i], y[i], z[i], xd[i], yd[i], zd[i]); 
#endif 

    //=========================================================
    // calculate element quantities (i.e. velocity gradient & q), and update material states 
    // LagrangeElements(domain);
    //=========================================================


    //========================================================================
    // void CalcKinematicsForElems( Domain &domain, Real_t *vnew, 
    //========================================================================
    device_queue.submit([&] (handler &cgh) {
      auto xd = d_xd.get_access<sycl_read>(cgh);
      auto yd = d_yd.get_access<sycl_read>(cgh);
      auto zd = d_zd.get_access<sycl_read>(cgh);
      auto x = d_x.get_access<sycl_read>(cgh);
      auto y = d_y.get_access<sycl_read>(cgh);
      auto z = d_z.get_access<sycl_read>(cgh);
      auto nodeList = d_nodelist.get_access<sycl_read>(cgh);
      auto volo = d_volo.get_access<sycl_read>(cgh);
      auto v = d_v.get_access<sycl_read>(cgh);
      auto delv = d_delv.get_access<sycl_discard_write>(cgh);
      auto arealg = d_arealg.get_access<sycl_discard_write>(cgh);
      auto dxx = d_dxx.get_access<sycl_discard_write>(cgh);
      auto dyy = d_dyy.get_access<sycl_discard_write>(cgh);
      auto dzz = d_dzz.get_access<sycl_discard_write>(cgh);
      auto vnew = d_vnew.get_access<sycl_discard_write>(cgh);

      cgh.parallel_for<class kintec>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        int k = item.get_global_id(0);
        if (k >= numElem) return;

        Real_t B[3][8] ; // shape function derivatives 
        Real_t D[6] ;
        Real_t x_local[8] ;
        Real_t y_local[8] ;
        Real_t z_local[8] ;
        Real_t xd_local[8] ;
        Real_t yd_local[8] ;
        Real_t zd_local[8] ;
        Real_t detJ = Real_t(0.0) ;

        Real_t volume ;
        Real_t relativeVolume ;
        const Index_t* elemToNode = nodeList.get_pointer() + Index_t(8)*k;

        // get nodal coordinates from global arrays and copy into local arrays.

        Index_t nd0i = elemToNode[0] ;
        Index_t nd1i = elemToNode[1] ;
        Index_t nd2i = elemToNode[2] ;
        Index_t nd3i = elemToNode[3] ;
        Index_t nd4i = elemToNode[4] ;
        Index_t nd5i = elemToNode[5] ;
        Index_t nd6i = elemToNode[6] ;
        Index_t nd7i = elemToNode[7] ;

        x_local[0] = x[nd0i];
        x_local[1] = x[nd1i];
        x_local[2] = x[nd2i];
        x_local[3] = x[nd3i];
        x_local[4] = x[nd4i];
        x_local[5] = x[nd5i];
        x_local[6] = x[nd6i];
        x_local[7] = x[nd7i];

        y_local[0] = y[nd0i];
        y_local[1] = y[nd1i];
        y_local[2] = y[nd2i];
        y_local[3] = y[nd3i];
        y_local[4] = y[nd4i];
        y_local[5] = y[nd5i];
        y_local[6] = y[nd6i];
        y_local[7] = y[nd7i];

        z_local[0] = z[nd0i];
        z_local[1] = z[nd1i];
        z_local[2] = z[nd2i];
        z_local[3] = z[nd3i];
        z_local[4] = z[nd4i];
        z_local[5] = z[nd5i];
        z_local[6] = z[nd6i];
        z_local[7] = z[nd7i];

        // volume calculations
        volume = CalcElemVolume(x_local, y_local, z_local );
        relativeVolume = volume / volo[k] ;
        vnew[k] = relativeVolume ;
        delv[k] = relativeVolume - v[k] ;

        // set characteristic length
        arealg[k] = CalcElemCharacteristicLength(x_local, y_local, z_local,
            volume);

        // get nodal velocities from global array and copy into local arrays.
        for( Index_t lnode=0 ; lnode<8 ; ++lnode )
        {
          Index_t gnode = elemToNode[lnode];
          xd_local[lnode] = xd[gnode];
          yd_local[lnode] = yd[gnode];
          zd_local[lnode] = zd[gnode];
        }

        Real_t dt2 = Real_t(0.5) * deltaTime;
        for ( Index_t j=0 ; j<8 ; ++j )
        {
          x_local[j] -= dt2 * xd_local[j];
          y_local[j] -= dt2 * yd_local[j];
          z_local[j] -= dt2 * zd_local[j];
        }

        CalcElemShapeFunctionDerivatives( x_local, y_local, z_local,
            B, &detJ );

        CalcElemVelocityGradient( xd_local, yd_local, zd_local,
            B, detJ, D );

        // put velocity gradient quantities into their global arrays.
        dxx[k] = D[0];
        dyy[k] = D[1];
        dzz[k] = D[2];
      });
    });

    vol_error = -1; // reset volumn error
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_vol_error.get_access<sycl_write>(cgh);
      cgh.copy(&vol_error, acc);
    });

    device_queue.submit([&] (handler &cgh) {
      auto dxx = d_dxx.get_access<sycl_read_write>(cgh);
      auto dyy = d_dyy.get_access<sycl_read_write>(cgh);
      auto dzz = d_dzz.get_access<sycl_read_write>(cgh);
      auto vnew = d_vnew.get_access<sycl_read>(cgh);
      auto vdov = d_vdov.get_access<sycl_discard_write>(cgh);
      auto vol_error = d_vol_error.get_access<sycl_write>(cgh);
      cgh.parallel_for<class strainRate>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        int k = item.get_global_id(0);
        if (k >= numElem) return;

        // calc strain rate and apply as constraint (only done in FB element)
        Real_t vvdov = dxx[k] + dyy[k] + dzz[k] ;
        Real_t vdovthird = vvdov/Real_t(3.0) ;

        // make the rate of deformation tensor deviatoric
        vdov[k] = vvdov;
        dxx[k] -= vdovthird ;  //LG:   why to update dxx?  it is deallocated right after
        dyy[k] -= vdovthird ;
        dzz[k] -= vdovthird ;

        // See if any volumes are negative, and take appropriate action.
        if (vnew[k] <= Real_t(0.0))
        {
          vol_error[0] = k;
        }
      });
    });

    device_queue.submit([&] (handler &cgh) {
      auto acc = d_vdov.get_access<sycl_read>(cgh);
      cgh.copy(acc, vdov);
    });

    device_queue.submit([&] (handler &cgh) {
      auto acc = d_vol_error.get_access<sycl_read>(cgh);
      cgh.copy(acc, &vol_error);
    });

    device_queue.wait();

#ifdef VERIFY
    for ( Index_t k=0 ; k<numElem ; ++k )
    {
      printf("kintec: %d %f\n", k, vdov[k]);
    }
#endif

    if (vol_error >= 0){
      printf("VolumeError: negative volumn\n");
      exit(VolumeError);
    }

    //======================================================= 
    //CalcQForElems(domain, vnew) ;
    //======================================================= 

    //================================================================
    // Calculate velocity gradients 
    //CalcMonotonicQGradientsForElems(domain, vnew);
    //================================================================

    device_queue.submit([&] (handler &cgh) {
      auto xd = d_xd.get_access<sycl_read>(cgh);
      auto yd = d_yd.get_access<sycl_read>(cgh);
      auto zd = d_zd.get_access<sycl_read>(cgh);
      auto x = d_x.get_access<sycl_read>(cgh);
      auto y = d_y.get_access<sycl_read>(cgh);
      auto z = d_z.get_access<sycl_read>(cgh);
      auto nodelist = d_nodelist.get_access<sycl_read>(cgh);
      auto volo = d_volo.get_access<sycl_read>(cgh);
      auto delv_eta = d_delv_eta.get_access<sycl_discard_write>(cgh);
      auto delx_eta = d_delx_eta.get_access<sycl_discard_write>(cgh);
      auto delv_zeta = d_delv_zeta.get_access<sycl_discard_write>(cgh);
      auto delx_zeta = d_delx_zeta.get_access<sycl_discard_write>(cgh);
      auto delv_xi = d_delv_xi.get_access<sycl_discard_write>(cgh);
      auto delx_xi = d_delx_xi.get_access<sycl_discard_write>(cgh);
      auto vnew = d_vnew.get_access<sycl_read>(cgh);

      cgh.parallel_for<class monotonicQgradients>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i >= numElem) return;

        Real_t ax,ay,az ;
        Real_t dxv,dyv,dzv ;

        const Index_t *elemToNode = nodelist.get_pointer() + Index_t(8) * i;
        Index_t n0 = elemToNode[0] ;
        Index_t n1 = elemToNode[1] ;
        Index_t n2 = elemToNode[2] ;
        Index_t n3 = elemToNode[3] ;
        Index_t n4 = elemToNode[4] ;
        Index_t n5 = elemToNode[5] ;
        Index_t n6 = elemToNode[6] ;
        Index_t n7 = elemToNode[7] ;

        Real_t x0 = x[n0] ;
        Real_t x1 = x[n1] ;
        Real_t x2 = x[n2] ;
        Real_t x3 = x[n3] ;
        Real_t x4 = x[n4] ;
        Real_t x5 = x[n5] ;
        Real_t x6 = x[n6] ;
        Real_t x7 = x[n7] ;

        Real_t y0 = y[n0] ;
        Real_t y1 = y[n1] ;
        Real_t y2 = y[n2] ;
        Real_t y3 = y[n3] ;
        Real_t y4 = y[n4] ;
        Real_t y5 = y[n5] ;
        Real_t y6 = y[n6] ;
        Real_t y7 = y[n7] ;

        Real_t z0 = z[n0] ;
        Real_t z1 = z[n1] ;
        Real_t z2 = z[n2] ;
        Real_t z3 = z[n3] ;
        Real_t z4 = z[n4] ;
        Real_t z5 = z[n5] ;
        Real_t z6 = z[n6] ;
        Real_t z7 = z[n7] ;

        Real_t xv0 = xd[n0] ;
        Real_t xv1 = xd[n1] ;
        Real_t xv2 = xd[n2] ;
        Real_t xv3 = xd[n3] ;
        Real_t xv4 = xd[n4] ;
        Real_t xv5 = xd[n5] ;
        Real_t xv6 = xd[n6] ;
        Real_t xv7 = xd[n7] ;

        Real_t yv0 = yd[n0] ;
        Real_t yv1 = yd[n1] ;
        Real_t yv2 = yd[n2] ;
        Real_t yv3 = yd[n3] ;
        Real_t yv4 = yd[n4] ;
        Real_t yv5 = yd[n5] ;
        Real_t yv6 = yd[n6] ;
        Real_t yv7 = yd[n7] ;

        Real_t zv0 = zd[n0] ;
        Real_t zv1 = zd[n1] ;
        Real_t zv2 = zd[n2] ;
        Real_t zv3 = zd[n3] ;
        Real_t zv4 = zd[n4] ;
        Real_t zv5 = zd[n5] ;
        Real_t zv6 = zd[n6] ;
        Real_t zv7 = zd[n7] ;

        Real_t vol = volo[i] * vnew[i] ;
        Real_t norm = Real_t(1.0) / ( vol + PTINY ) ;

        Real_t dxj = Real_t(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
        Real_t dyj = Real_t(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
        Real_t dzj = Real_t(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

        Real_t dxi = Real_t( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
        Real_t dyi = Real_t( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
        Real_t dzi = Real_t( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

        Real_t dxk = Real_t( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
        Real_t dyk = Real_t( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
        Real_t dzk = Real_t( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

        /* find delvk and delxk ( i cross j ) */

        ax = dyi*dzj - dzi*dyj ;
        ay = dzi*dxj - dxi*dzj ;
        az = dxi*dyj - dyi*dxj ;

        delx_zeta[i] = vol / sycl::sqrt(ax*ax + ay*ay + az*az + PTINY) ;

        ax *= norm ;
        ay *= norm ;
        az *= norm ;

        dxv = Real_t(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
        dyv = Real_t(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
        dzv = Real_t(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

        delv_zeta[i] = ax*dxv + ay*dyv + az*dzv ;

        /* find delxi and delvi ( j cross k ) */

        ax = dyj*dzk - dzj*dyk ;
        ay = dzj*dxk - dxj*dzk ;
        az = dxj*dyk - dyj*dxk ;

        delx_xi[i] = vol / sycl::sqrt(ax*ax + ay*ay + az*az + PTINY) ;

        ax *= norm ;
        ay *= norm ;
        az *= norm ;

        dxv = Real_t(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
        dyv = Real_t(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
        dzv = Real_t(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

        delv_xi[i] = ax*dxv + ay*dyv + az*dzv ;

        /* find delxj and delvj ( k cross i ) */

        ax = dyk*dzi - dzk*dyi ;
        ay = dzk*dxi - dxk*dzi ;
        az = dxk*dyi - dyk*dxi ;

        delx_eta[i] = vol / sycl::sqrt(ax*ax + ay*ay + az*az + PTINY) ;

        ax *= norm ;
        ay *= norm ;
        az *= norm ;

        dxv = Real_t(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
        dyv = Real_t(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
        dzv = Real_t(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

        delv_eta[i] = ax*dxv + ay*dyv + az*dzv ;
      });
    });

    //=========================================================
    // CalcMonotonicQForElems(domain, vnew) ;
    //=========================================================
    Real_t monoq_limiter_mult = domain.monoq_limiter_mult();
    Real_t monoq_max_slope = domain.monoq_max_slope();
    Real_t qlc_monoq = domain.qlc_monoq();
    Real_t qqc_monoq = domain.qqc_monoq();

    device_queue.submit([&] (handler &cgh) {
      auto elemBC = d_elemBC.get_access<sycl_read>(cgh);
      auto elemMass = d_elemMass.get_access<sycl_read>(cgh);
      auto ql = d_ql.get_access<sycl_discard_write>(cgh);
      auto qq = d_qq.get_access<sycl_discard_write>(cgh);
      auto vdov = d_vdov.get_access<sycl_read>(cgh);
      auto volo = d_volo.get_access<sycl_read>(cgh);
      auto delv_eta = d_delv_eta.get_access<sycl_read>(cgh);
      auto delx_eta = d_delx_eta.get_access<sycl_read>(cgh);
      auto delv_zeta = d_delv_zeta.get_access<sycl_read>(cgh);
      auto delx_zeta = d_delx_zeta.get_access<sycl_read>(cgh);
      auto delv_xi = d_delv_xi.get_access<sycl_read>(cgh);
      auto delx_xi = d_delx_xi.get_access<sycl_read>(cgh);
      auto lxim = d_lxim.get_access<sycl_read>(cgh);
      auto lxip = d_lxip.get_access<sycl_read>(cgh);
      auto lzetam = d_lzetam.get_access<sycl_read>(cgh);
      auto lzetap = d_lzetap.get_access<sycl_read>(cgh);
      auto letap = d_letap.get_access<sycl_read>(cgh);
      auto letam = d_letam.get_access<sycl_read>(cgh);
      auto vnew = d_vnew.get_access<sycl_read>(cgh);

      cgh.parallel_for<class monotonicQ>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        Index_t  i = item.get_global_id(0);
        if (i >= numElem) return;

        Real_t qlin, qquad ;
        Real_t phixi, phieta, phizeta ;
        Int_t bcMask = elemBC[i] ;
        Real_t delvm = 0.0, delvp =0.0;

        /*  phixi     */
        Real_t norm = Real_t(1.) / (delv_xi[i]+ PTINY ) ;

        switch (bcMask & XI_M) {
        case XI_M_COMM: /* needs comm data */
        case 0:         delvm = delv_xi[lxim[i]]; break ;
        case XI_M_SYMM: delvm = delv_xi[i] ;       break ;
        case XI_M_FREE: delvm = Real_t(0.0) ;      break ;
        default: //fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
        delvm = 0; /* ERROR - but quiets the compiler */
        break;
        }
        switch (bcMask & XI_P) {
          case XI_P_COMM: /* needs comm data */
          case 0:         delvp = delv_xi[lxip[i]] ; break ;
          case XI_P_SYMM: delvp = delv_xi[i] ;       break ;
          case XI_P_FREE: delvp = Real_t(0.0) ;      break ;
          default: //fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
              delvp = 0; /* ERROR - but quiets the compiler */
              break;
        }

        delvm = delvm * norm ;
        delvp = delvp * norm ;

        phixi = Real_t(.5) * ( delvm + delvp ) ;

        delvm *= monoq_limiter_mult ;
        delvp *= monoq_limiter_mult ;

        if ( delvm < phixi ) phixi = delvm ;
        if ( delvp < phixi ) phixi = delvp ;
        if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
        if ( phixi > monoq_max_slope) phixi = monoq_max_slope;

        /*  phieta     */
        norm = Real_t(1.) / ( delv_eta[i] + PTINY ) ;

        switch (bcMask & ETA_M) {
          case ETA_M_COMM: /* needs comm data */
          case 0:          delvm = delv_eta[letam[i]] ; break ;
          case ETA_M_SYMM: delvm = delv_eta[i] ;        break ;
          case ETA_M_FREE: delvm = Real_t(0.0) ;        break ;
          default: //fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
               delvm = 0; /* ERROR - but quiets the compiler */
               break;
        }
        switch (bcMask & ETA_P) {
          case ETA_P_COMM: /* needs comm data */
          case 0:          delvp = delv_eta[letap[i]] ; break ;
          case ETA_P_SYMM: delvp = delv_eta[i] ;        break ;
          case ETA_P_FREE: delvp = Real_t(0.0) ;        break ;
          default: 
               delvp = 0; /* ERROR - but quiets the compiler */
               break;
        }

        delvm = delvm * norm ;
        delvp = delvp * norm ;

        phieta = Real_t(.5) * ( delvm + delvp ) ;

        delvm *= monoq_limiter_mult ;
        delvp *= monoq_limiter_mult ;

        if ( delvm  < phieta ) phieta = delvm ;
        if ( delvp  < phieta ) phieta = delvp ;
        if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
        if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

        /*  phizeta     */
        norm = Real_t(1.) / ( delv_zeta[i] + PTINY ) ;

        switch (bcMask & ZETA_M) {
          case ZETA_M_COMM: /* needs comm data */
          case 0:           delvm = delv_zeta[lzetam[i]] ; break ;
          case ZETA_M_SYMM: delvm = delv_zeta[i] ;         break ;
          case ZETA_M_FREE: delvm = Real_t(0.0) ;          break ;
          default: 
                delvm = 0; /* ERROR - but quiets the compiler */
                break;
        }
        switch (bcMask & ZETA_P) {
          case ZETA_P_COMM: /* needs comm data */
          case 0:           delvp = delv_zeta[lzetap[i]] ; break ;
          case ZETA_P_SYMM: delvp = delv_zeta[i] ;         break ;
          case ZETA_P_FREE: delvp = Real_t(0.0) ;          break ;
          default:
                delvp = 0; /* ERROR - but quiets the compiler */
                break;
        }

        delvm = delvm * norm ;
        delvp = delvp * norm ;

        phizeta = Real_t(.5) * ( delvm + delvp ) ;

        delvm *= monoq_limiter_mult ;
        delvp *= monoq_limiter_mult ;

        if ( delvm   < phizeta ) phizeta = delvm ;
        if ( delvp   < phizeta ) phizeta = delvp ;
        if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
        if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

        /* Remove length scale */

        if ( vdov[i] > Real_t(0.) )  {
          qlin  = Real_t(0.) ;
          qquad = Real_t(0.) ;
        }
        else {
          Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
          Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
          Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

          if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
          if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
          if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

          Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

          qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

          qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
        }

        qq[i] = qquad ;
        ql[i] = qlin  ;
      });
    });

#ifdef VERIFY
    auto h_qq = d_qq.get_access<sycl_read>();
    auto h_ql = d_ql.get_access<sycl_read>();
    for (int i = 0; i < numElem; i++) {
      printf("mqr: %d %f %f\n", i, h_qq[i], h_ql[i]);
    }
#endif

    /* Don't allow excessive artificial viscosity */
    Index_t idx = -1; 
    for (Index_t i=0; i<numElem; ++i) {
      if ( domain.q(i) > domain.qstop() ) {
        idx = i ;
        break ;
      }
    }

    if(idx >= 0) {
      printf("QStopError\n");
      exit(QStopError);
    }

    //=================================================
    //ApplyMaterialPropertiesForElems(domain, vnew) ;
    //=================================================
    Real_t  e_cut = domain.e_cut() ;
    Real_t  p_cut = domain.p_cut() ;
    //Real_t  ss4o3 = domain.ss4o3() ;
    Real_t  q_cut = domain.q_cut() ;
    Real_t  v_cut = domain.v_cut() ;

    Real_t eosvmax = domain.eosvmax() ;
    Real_t eosvmin = domain.eosvmin() ;
    Real_t pmin    = domain.pmin() ;
    Real_t emin    = domain.emin() ;
    Real_t rho0    = domain.refdens() ;

    device_queue.submit([&] (handler &cgh) {
      auto ql = d_ql.get_access<sycl_read>(cgh);
      auto qq = d_qq.get_access<sycl_read>(cgh);
      auto delv = d_delv.get_access<sycl_read>(cgh);
      auto elemRep = d_elemRep.get_access<sycl_read>(cgh);
      auto elemElem = d_elemElem.get_access<sycl_read>(cgh);
      auto q = d_q.get_access<sycl_read_write>(cgh);
      auto p = d_p.get_access<sycl_read_write>(cgh);
      auto e = d_e.get_access<sycl_read_write>(cgh);
      auto ss = d_ss.get_access<sycl_read_write>(cgh);
      auto v = d_v.get_access<sycl_read_write>(cgh);
      auto vnewc = d_vnew.get_access<sycl_read_write>(cgh);

      cgh.parallel_for<class eos>(nd_range<1>(gws_elem, lws), [=] (nd_item<1> item) {
        int elem = item.get_global_id(0);
        if (elem >= numElem) return;
        Index_t rep = elemRep[elem];
        Real_t e_old, delvc, p_old, q_old, qq_old, ql_old;
        Real_t p_new, q_new, e_new;
        Real_t work, compression, compHalfStep, bvc, pbvc, pHalfStep;
        Real_t vchalf ;
        Real_t vhalf ;
        Real_t ssc ;
        Real_t q_tilde ;
        Real_t ssTmp ;

        if (eosvmin != ZERO) {
          if (vnewc[elem] < eosvmin)
            vnewc[elem] = eosvmin ;
        }

        if (eosvmax != ZERO) {
          if (vnewc[elem] > eosvmax)
            vnewc[elem] = eosvmax ;
        }

        // This check may not make perfect sense in LULESH, but
        // it's representative of something in the full code -
        // just leave it in, please
        Real_t vc = v[elem] ;
        if (eosvmin != ZERO) {
          if (vc < eosvmin)
            vc = eosvmin ;
        }
        if (eosvmax != ZERO) {
          if (vc > eosvmax)
            vc = eosvmax ;
        }

        Real_t vnewc_t = vnewc[elem];

        Real_t e_temp    =    e[elem];
        Real_t delv_temp = delv[elem];
        Real_t p_temp    =    p[elem];
        Real_t q_temp    =    q[elem];
        Real_t qq_temp   =   qq[elem];
        Real_t ql_temp   =   ql[elem];

        for(Index_t j = 0; j < rep; j++) {

          e_old  =    e_temp ;
          delvc  = delv_temp ;
          p_old  =    p_temp ;
          q_old  =    q_temp ;
          qq_old =   qq_temp ;
          ql_old =   ql_temp ;

          compression = ONE / vnewc_t - ONE;
          vchalf = vnewc_t - delvc * HALF;
          compHalfStep = ONE / vchalf - ONE;
          if (vnewc_t <= eosvmin) { /* impossible due to calling func? */
            compHalfStep = compression ;
          }
          if (vnewc_t >= eosvmax) { /* impossible due to calling func? */
            p_old        = ZERO ;
            compression  = ZERO ;
            compHalfStep = ZERO ;
          }
          work = ZERO ;

          e_new = e_old - HALF * delvc * (p_old + q_old)
            + HALF * work;

          if (e_new  < emin) {
            e_new = emin ;
          }

          bvc = C1S * (compHalfStep + ONE);
          pbvc = C1S;

          pHalfStep = bvc * e_new ;

          if (sycl::fabs(pHalfStep) <  p_cut)
            pHalfStep = ZERO ;

          if (vnewc_t >= eosvmax) /* impossible condition here? */
            pHalfStep = ZERO ;

          if (pHalfStep <  pmin)
            pHalfStep   = pmin ;

          vhalf = ONE / (ONE + compHalfStep) ;

          if (delvc > ZERO) {
            q_new /* = qq_old[elem] = ql_old[elem] */ = ZERO ;
          } else {
            ssc = ( pbvc * e_new + vhalf * vhalf * bvc * pHalfStep ) / rho0 ;

            if ( ssc <= C1 ) {
              ssc = C2 ;
            } else {
              ssc = sycl::sqrt(ssc) ;
            }

            q_new = (ssc*ql_old + qq_old) ;
          }

          e_new = e_new + HALF * delvc * (THREE*(p_old + q_old) - FOUR*(pHalfStep + q_new)) ;

          e_new += HALF * work;

          if (sycl::fabs(e_new) < e_cut) {
            e_new = ZERO  ;
          }
          if (e_new < emin) {
            e_new = emin ;
          }

          bvc = C1S * (compression + ONE);
          pbvc = C1S;

          p_new = bvc * e_new ;

          if (sycl::fabs(p_new) < p_cut)
            p_new = ZERO ;

          if (vnewc_t >= eosvmax) /* impossible condition here? */
            p_new = ZERO ;

          if (p_new  <  pmin)
            p_new = pmin ;


          if (delvc > ZERO) {
            q_tilde = ZERO ;
          }
          else {
            Real_t ssc = ( pbvc * e_new + vnewc_t * vnewc_t * bvc * p_new ) / rho0 ;

            if ( ssc <= C1 ) {
              ssc = C2 ;
            } else {
              ssc = sycl::sqrt(ssc) ;
            }

            q_tilde = (ssc * ql_old + qq_old) ;
          }

          e_new = e_new - (SEVEN*(p_old + q_old) - EIGHT*(pHalfStep + q_new)
              + (p_new + q_tilde)) * delvc*SIXTH ;

          if (sycl::fabs(e_new) < e_cut) {
            e_new = ZERO  ;
          }
          if (e_new < emin) {
            e_new = emin ;
          }

          bvc = C1S * (compression + ONE);
          pbvc = C1S;

          p_new = bvc * e_new ;

          if (sycl::fabs(p_new) <  p_cut)
            p_new = ZERO ;

          if (vnewc_t >= eosvmax) /* impossible condition here? */
            p_new = ZERO ;

          if (p_new < pmin)
            p_new = pmin ;
          if (delvc <= ZERO) {
            ssc = ( pbvc * e_new + vnewc_t * vnewc_t * bvc * p_new ) / rho0 ;

            if (ssc <= C1) {
              ssc = C2 ;
            } else {
              ssc = sycl::sqrt(ssc) ;
            }

            q_new = (ssc*ql_old + qq_old) ;

            if (sycl::fabs(q_new) < q_cut) q_new = ZERO ;
          }
        } //this is the end of the rep loop

        p[elem] = p_new ;
        e[elem] = e_new ;
        q[elem] = q_new ;

        ssTmp = (pbvc * e_new + vnewc_t * vnewc_t * bvc * p_new) / rho0;
        if (ssTmp <= C1) {
          ssTmp = C2;
        } else {
          ssTmp = sycl::sqrt(ssTmp);
        }
        ss[elem] = ssTmp ;

        if ( sycl::fabs(vnewc_t - ONE) < v_cut )
          vnewc_t = ONE ;

        v[elem] = vnewc_t ;
      });
    });

    device_queue.submit([&] (handler &cgh) {
      auto acc = d_ss.get_access<sycl_read>(cgh);
      cgh.copy(acc, ss);
    });

    device_queue.submit([&] (handler &cgh) {
      auto acc = d_arealg.get_access<sycl_read>(cgh);
      cgh.copy(acc, arealg);
    });

    device_queue.wait();

#ifdef VERIFY
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_p.get_access<sycl_read>(cgh);
      cgh.copy(acc, p);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_q.get_access<sycl_read>(cgh);
      cgh.copy(acc, q);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_e.get_access<sycl_read>(cgh);
      cgh.copy(acc, e);
    });
    device_queue.submit([&] (handler &cgh) {
      auto acc = d_v.get_access<sycl_read>(cgh);
      cgh.copy(acc, v);
    });

    device_queue.wait();

    for (int i = 0; i < numElem; i++) {
      printf("eos: %f %f %f %f %f\n", q[i], p[i], e[i], ss[i], v[i]);
    }
#endif

    CalcTimeConstraintsForElems(domain);

    if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0)) {
      printf("cycle = %d, time = %e, dt=%e\n",
          locDom->cycle(), double(locDom->time()), double(locDom->deltatime()) ) ;
    }
    if (opts.iteration_cap == 1){
      break;
    }
    opts.iteration_cap -= 1;
  }

  // Use reduced max elapsed time
  double elapsed_time;
  timeval end;
  gettimeofday(&end, NULL) ;
  elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
  double elapsed_timeG;
  elapsed_timeG = elapsed_time;

  // Write out final viz file */
  if (opts.viz) {
    DumpToVisit(*locDom, opts.numFiles, myRank, numRanks) ;
  }

  if ((myRank == 0) && (opts.quiet == 0)) {
    VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, opts.nx, numRanks);
  }

  // Release resources
  locDom->DeallocateStrains();
  locDom->DeallocateGradients();
  Release(&determ) ;
  delete(locDom);

  return 0 ;
}
