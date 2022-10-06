// Copyright (c) MONAI Consortium
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
//#include <ATen/ATen.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cooperative_groups.h>
//#include <cooperative_groups/reduce.h>
#include <cuda/pipeline>
#include <vector>
//#include <cuda/annotated_ptr>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/memcpy_async.h>

#include <cmath>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <assert.h>
#include "device_launch_parameters.h"

using namespace cooperative_groups;


#pragma once

//constants describing the meaning of main shared memory spaces
constexpr uint32_t localWorkQueLength = 300;
constexpr uint32_t startOfLocalWorkQ = 4160;
constexpr uint32_t lengthOfMainShmem = 4460;//4460;
constexpr uint32_t begResShmem = 1088;
constexpr uint32_t begfirstRegShmem = 2112;
constexpr uint32_t begSecRegShmem = 3136;
constexpr uint32_t begSourceShmem = 32;


//added to linear index meta in order to  mark weather block is of type gold or not
constexpr uint32_t  isGoldOffset = (UINT16_MAX * 10);



/***************************************
 * structs
 * ********************************/


 /**
 In order to be able to use cuda malloc 3d we will implemnt it as a series
 of 3d arrays
 */
#pragma once
extern "C" struct MetaDataCPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;


    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count  9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter
     //12) global FPandFn offset 13)globalIterationNumb
    //array3dWithDimsCPU<unsigned int> minMaxes;
    unsigned int minMaxes[20];
    uint32_t* resultList;

};

#pragma once
extern "C" struct MetaDataGPU {
    int metaXLength;
    int MetaYLength;
    int MetaZLength;
    int totalMetaLength;

    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ - minimal and maximum coordinates of blocks with some entries of intrest
    //7)global FP count; 8)global FN count 9) workQueueCounter 10)resultFP globalCounter 11) resultFn globalCounter
    //12) global FPandFn offset 13)globalIterationNumb

    unsigned int* minMaxes;

    //represents x from description of main Arr
    unsigned int mainArrXLength;
    //have length 4x
    unsigned int mainArrSectionLength;

    unsigned int metaDataSectionLength = 20;

    // now we will store here also calculated by min maxes kernel values of minimum and maximumvalues
        //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    unsigned int maxX;
    unsigned int minX;
    unsigned int maxY;
    unsigned int minY;
    unsigned int maxZ;
    unsigned int minZ;
};








/*
* Basically holding the arguments for master functions controlling full preparation to get all for Housedorff kernel
*/
#pragma once
template <typename TFF>
struct ForFullBoolPrepArgs {

    //metadata struct
    MetaDataCPU metaData;

    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    torch::Tensor goldArr;
    torch::Tensor segmArr;

    int Nx;
    int Ny;
    int Nz;

    TFF numberToLookFor;// what we will look for in arrays
    //number and dimensionality of threads and blocks required to lounch bool kernel
    dim3 threads;
    int blocks;
    //threads and blocks for first metadata pass kernel
    int threadsFirstMetaDataPass;
    int blocksFirstMetaDataPass;
    //threads and blocks for main pass
    dim3 threadsMainPass;
    int blocksMainPass;
    //threads and blocks for padding pass
    dim3 threadsPaddingPass;
    int blocksPaddingPass;
    //threads and blocks for non first metadata passes
    int threadsOtherMetaDataPasses;
    int blocksOtherMetaDataPasses;
    // will establish how many points we want to include in dilatation and how many we can ignore so typically set to 95% - so we will ignore only 5% most distant
    float robustnessPercent = 1.0;  // 0.95;

    int32_t* resultListPointerMeta;
    int32_t* resultListPointerLocalCPU;
    int32_t* resultListPointerIterNumb;

};


/*
* Basically holding the arguments for main kernel in the FullBoolPrep
*/
#pragma once
template <typename TFB>
struct ForBoolKernelArgs {
    //matadata struct
    MetaDataGPU metaData;
    // dimensions of data block
    int dbXLength;
    int dbYLength;
    int dbZLength;
    // gold standard and segmentation output array
    torch::Tensor goldArr;
    torch::Tensor segmArr;
    TFB numberToLookFor;
    int Nx;
    int Ny;
    int Nz;

    int32_t* resultListPointerMeta;
    int32_t* resultListPointerLocal;
    int32_t* resultListPointerIterNumb;

    torch::Tensor resultListPointerIterNumbTensor;
    torch::Tensor resultListPointerLocalTensor;
    torch::Tensor resultListPointerMetaTensor;


    torch::Tensor resGold;
    torch::Tensor resSegm;


    uint32_t* origArrsPointer;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    uint32_t* metaDataArrPointer;

    uint32_t* workQueuePointer;
    unsigned int* minMaxes;


    /*
main array with all required data  organized in sections for each metadata block
x-  is block dimx times block dimy
now what occupies what positions
##mainArrA
(0) - x-1 : reducedGoldRef
(x) - 2x-1 : reducedSegmRef
##mainArrB
() - 3x-1 : reducedGoldPrev
(x) - 4x-1 : reducedSegmPrev

##metaDataArr
0: empty
1 :fpCount
2 :fnCount
3 :fpCounter
4 :fnCounter
5 :fpOffset
6 :fnOffset
7 :isActiveGold
8 :isFullGold
9 :isActiveSegm
10 :isFullSegm
11 :isToBeActivatedGold
12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
13 : top
14 : bottom
15 : left
16 : right
17 : anterior
18 : posterior
19 : empty
20 : empty


###main arrays
0-x : reducedGold
(x+1) - 2x : reducedSegm
*/


    float robustnessPercent = 1.0;// 0.95;

};




/*
data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
*/
struct occupancyCalcData {
    int warpsNumbForMinMax;
    int blockSizeForMinMax;

    int warpsNumbForboolPrepareKernel;
    int blockSizeFoboolPrepareKernel;

    int theadsForFirstMetaPass;
    int blockForFirstMetaPass;

    int warpsNumbForMainPass;
    int blockForMainPass;
};



/***************************************
 * utils
 * ********************************/


 /*
 copy from host to device
 */
#pragma once
inline MetaDataGPU allocateMetaDataOnGPU(MetaDataCPU metaDataCPU, unsigned int*& minMaxes) {
    MetaDataGPU res;

    metaDataCPU.minMaxes[1] = 0;
    metaDataCPU.minMaxes[2] = 1000;
    metaDataCPU.minMaxes[3] = 0;
    metaDataCPU.minMaxes[4] = 1000;
    metaDataCPU.minMaxes[5] = 0;
    metaDataCPU.minMaxes[6] = 1000;
    metaDataCPU.minMaxes[7] = 0;
    metaDataCPU.minMaxes[8] = 0;
    metaDataCPU.minMaxes[9] = 0;
    metaDataCPU.minMaxes[10] = 0;
    metaDataCPU.minMaxes[11] = 0;
    metaDataCPU.minMaxes[12] = 0;
    metaDataCPU.minMaxes[13] = 0;
    metaDataCPU.minMaxes[14] = 0;
    metaDataCPU.minMaxes[15] = 0;
    metaDataCPU.minMaxes[16] = 0;
    metaDataCPU.minMaxes[17] = 0;
    metaDataCPU.minMaxes[18] = 0;
    metaDataCPU.minMaxes[19] = 0;
    metaDataCPU.minMaxes[20] = 0;

    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpy(minMaxes, metaDataCPU.minMaxes, size, cudaMemcpyHostToDevice);

    return res;

}

/***************************************
 * utils
 * ********************************/

 /*
 setting the linear index of metadata blocks that are in given direction if there is no such (out of range) we will save it as UINT32_MAX
 */
#pragma once
template <typename TCC>
__device__ inline void setNeighbourBlocks(ForBoolKernelArgs<TCC> fbArgs
    , uint8_t idX, uint8_t inArrIndex, bool predicate, uint32_t toAdd
    , uint32_t linIdexMeta, MetaDataGPU metaData, uint32_t localBlockMetaData[20]) {

    if ((threadIdx.x == idX) && (threadIdx.y == 0)) {
        if (predicate) {


            localBlockMetaData[inArrIndex] = (linIdexMeta + toAdd);
        }
        else {
            localBlockMetaData[inArrIndex] = isGoldOffset;
        }
    };
}


/*
gettinng  array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPI>
inline __device__ uint32_t* getSourceReduced(const ForBoolKernelArgs<TXPI>& fbArgs, const int(&iterationNumb)[1]) {


    if ((iterationNumb[0] & 1) == 0) {
        return fbArgs.mainArrAPointer;

    }
    else {
        return fbArgs.mainArrBPointer;
    }


}


/*
gettinng target array for dilatations
basically arrays will alternate between iterations once one will be source other target then they will switch - we will decide upon knowing
wheather the iteration number is odd or even
*/
#pragma once
template <typename TXPPI>
inline __device__ uint32_t* getTargetReduced(const ForBoolKernelArgs<TXPPI>& fbArgs, const  int(&iterationNumb)[1]) {

    if ((iterationNumb[0] & 1) == 0) {

        return fbArgs.mainArrBPointer;

    }
    else {
        return fbArgs.mainArrAPointer;

    }

}


/*
dilatation up and down - using bitwise operators
*/
#pragma once
inline __device__ uint32_t bitDilatate(const uint32_t& x) {
    return ((x) >> 1) | (x) | ((x) << 1);
}

/*
return 1 if at given position of given number bit is set otherwise 0
*/
#pragma once
inline __device__ uint32_t isBitAt(const uint32_t& numb, const int pos) {
    return (numb & (1 << (pos)));
}




/***************************************
 * MinMaxes kernel
 * ********************************/
 /*

 iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
 */
#pragma once
template <typename TYO>
__global__ void getMinMaxes(ForBoolKernelArgs<TYO> fbArgs
    , unsigned int* minMaxes
    , TYO* goldArr
    , TYO* segmArr
    , MetaDataGPU metaData
) {

    thread_block cta = this_thread_block();

    //shared memory
    __shared__ bool anyInGold[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ unsigned int minMaxesInShmem[7];

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) { minMaxesInShmem[1] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 0)) { minMaxesInShmem[2] = 1000; };

    if ((threadIdx.x == 3) && (threadIdx.y == 0)) { minMaxesInShmem[3] = 0; };
    if ((threadIdx.x == 4) && (threadIdx.y == 0)) { minMaxesInShmem[4] = 1000; };

    if ((threadIdx.x == 5) && (threadIdx.y == 0)) { minMaxesInShmem[5] = 0; };
    if ((threadIdx.x == 6) && (threadIdx.y == 0)) { minMaxesInShmem[6] = 1000; };

    if ((threadIdx.x == 7) && (threadIdx.y == 0)) { anyInGold[1] = false; };


    if ((threadIdx.x == 1) && (threadIdx.y == 0) && (blockIdx.x == 0)) {

    }

    __syncthreads();

    //main metadata iteration
    for (auto linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int  xMeta = linIdexMeta % metaData.metaXLength;
        int   zMeta = int(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * metaData.MetaYLength))));
        int   yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //iterating over data block
        for (uint8_t xLoc = threadIdx.x; xLoc < 32; xLoc += blockDim.x) {
            uint32_t x = xMeta * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = yMeta * fbArgs.dbYLength + yLoc;//absolute position

                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    // resetting
                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = zMeta * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.Nz) {
                            //first array gold
                            //uint8_t& zLocRef = zLoc; uint8_t& yLocRef = yLoc; uint8_t& xLocRef = xLoc;

                            // setting bits
                            bool goldBool = goldArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;  // (getTensorRow<TYU>(tensorslice, fbArgs.goldArr, fbArgs.goldArr.Ny, y, z)[x] == fbArgs.numberToLookFor);
                            bool segmBool = segmArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            if (goldBool || segmBool) {
                                anyInGold[0] = true;
                            }
                        }
                    }
                }

                //  __syncthreads();
                  //waiting so shared memory will be loaded evrywhere
                  //on single thread we do last sum reduction

                  /////////////////// setting min and maxes
  //    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
                __syncthreads();

                if ((threadIdx.x == 0) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[1] = max(xMeta, minMaxesInShmem[1]); };
                if ((threadIdx.x == 1) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[2] = min(xMeta, minMaxesInShmem[2]); };

                if ((threadIdx.x == 2) && (threadIdx.y == 0) && anyInGold[0]) {

                    minMaxesInShmem[3] = max(yMeta, minMaxesInShmem[3]);

                };
                if ((threadIdx.x == 3) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[4] = min(yMeta, minMaxesInShmem[4]); };

                if ((threadIdx.x == 4) && (threadIdx.y == 0) && anyInGold[0]) { minMaxesInShmem[5] = max(zMeta, minMaxesInShmem[5]); };
                if ((threadIdx.x == 5) && (threadIdx.y == 0) && anyInGold[0]) {
                    minMaxesInShmem[6] = min(zMeta, minMaxesInShmem[6]);
                };
                __syncthreads(); // just to reduce the warp divergence
                anyInGold[0] = false;




            }
        }

    }
    __syncthreads();

    auto active = coalesced_threads();

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicMax(&minMaxes[1], minMaxesInShmem[1]);
    };

    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {

        atomicMin(&minMaxes[2], minMaxesInShmem[2]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicMax(&minMaxes[3], minMaxesInShmem[3]);

    };

    if ((threadIdx.x == 2) && (threadIdx.y == 0)) {
        atomicMin(&minMaxes[4], minMaxesInShmem[4]);
    };



    if (threadIdx.x == 3 && threadIdx.y == 0) {
        atomicMax(&minMaxes[5], minMaxesInShmem[5]);
    };

    if (threadIdx.x == 4 && threadIdx.y == 0) {
        atomicMin(&minMaxes[6], minMaxesInShmem[6]);

    };
}


/***************************************
 * boolPrepareKernel
 * ********************************/


 /*
 iteration over metadata - becouse metadata may be small and to maximize occupancy we use linear index and then clalculate xMeta,ymeta,zMeta from this linear index ...
 */
#pragma once
template <typename TYO>
__global__ void boolPrepareKernel(ForBoolKernelArgs<TYO> fbArgs
    , MetaDataGPU metaData, uint32_t* origArrs, uint32_t* metaDataArr
    , TYO* goldArr
    , TYO* segmArr
    , unsigned int* minMaxes) {

    ////////////some initializations
    bool goldBool = false;
    bool segmBool = false;
    bool isNotEmpty = false;

    thread_block cta = this_thread_block();
    thread_block_tile<32> tile = tiled_partition<32>(cta);
    uint32_t sumFp = 0;
    uint32_t sumFn = 0;

    //shared memory

    //TODO() make it dynamically sized
    __shared__ uint32_t sharedForGold[1024];
    __shared__ uint32_t sharedForSegm[1024];


    //for storing fp and fn sums to later accumulate it to global values
    __shared__ uint32_t fpSFnS[2];
    __shared__ uint32_t localBlockMetaData[20];

    __shared__ bool anyInGold[1];
    __shared__ bool anyInSegm[1];
    //__shared__ uint32_t reduction_s[32];
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    __shared__ int minMaxesInShmem[7];

    if ((threadIdx.x == 1) && (threadIdx.y == 1)) { fpSFnS[0] = 0; };
    if ((threadIdx.x == 2) && (threadIdx.y == 1)) { fpSFnS[1] = 0; };
    if ((threadIdx.x == 3) && (threadIdx.y == 1)) { anyInGold[1] = false; };
    if ((threadIdx.x == 4) && (threadIdx.y == 1)) { anyInSegm[1] = false; };



    __syncthreads();

    /////////////////////////


    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += gridDim.x) {
        //we get from linear index  the coordinates of the metadata block of intrest
        int xMeta = int(linIdexMeta % (metaData.metaXLength));
        int zMeta = int(floor((float)(linIdexMeta / (metaData.metaXLength * metaData.MetaYLength))));
        int yMeta = int(floor((float)((linIdexMeta - ((zMeta * metaData.metaXLength * metaData.MetaYLength) + xMeta)) / metaData.metaXLength)));
        //reset
        isNotEmpty = false;
        sumFp = 0;
        sumFn = 0;
        anyInGold[0] = false;
        anyInSegm[0] = false;
        //iterating over data block
        __syncthreads();
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    // resetting
                    sharedForGold[xLoc + yLoc * fbArgs.dbXLength] = 0;
                    sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] = 0;


                    for (uint8_t zLoc = 0; zLoc < fbArgs.dbZLength; zLoc++) {
                        uint32_t z = (zMeta + metaData.minZ) * fbArgs.dbZLength + zLoc;//absolute position
                        if (z < fbArgs.Nz) {
                            //char* tensorslice;

                            //first array gold
                            bool goldBool = goldArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            bool segmBool = segmArr[x + y * fbArgs.Nx + z * fbArgs.Nx * fbArgs.Ny] == fbArgs.numberToLookFor;
                            //goldBool = true;

                            // setting bits
                            sharedForGold[xLoc + yLoc * fbArgs.dbXLength] |= goldBool << zLoc;
                            sharedForSegm[xLoc + yLoc * fbArgs.dbXLength] |= segmBool << zLoc;
                            // setting value of local boolean marking that any of the entries was evaluated to true in either of arrays
                            isNotEmpty = (isNotEmpty || (goldBool || segmBool));
                            sumFp += (!goldBool && segmBool);
                            sumFn += (goldBool && !segmBool);
                            if (goldBool)  anyInGold[0] = true;
                            if (segmBool)  anyInSegm[0] = true;

                        }
                    }
                }

            }
        }
        //reset local metadata
        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            localBlockMetaData[threadIdx.x] = 0;
        }



        isNotEmpty = __syncthreads_or(isNotEmpty);
        //exporting to global memory
        for (uint8_t xLoc = threadIdx.x; xLoc < fbArgs.dbXLength; xLoc += blockDim.x) {
            uint32_t x = (xMeta + metaData.minX) * fbArgs.dbXLength + xLoc;//absolute position
            for (uint8_t yLoc = threadIdx.y; yLoc < fbArgs.dbYLength; yLoc += blockDim.y) {
                uint32_t  y = (yMeta + metaData.minY) * fbArgs.dbYLength + yLoc;//absolute position
                if (y < fbArgs.Ny && x < fbArgs.Nx) {

                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc] = sharedForGold[yLoc * 32 + xLoc];
                    origArrs[linIdexMeta * metaData.mainArrSectionLength + yLoc * 32 + xLoc + metaData.mainArrXLength] = sharedForSegm[yLoc * 32 + xLoc];


                }
            }
        }

        __syncthreads();

        /////adding the block and total number of the Fp's and Fn's
        sumFp = reduce(tile, sumFp, plus<uint32_t>());
        sumFn = reduce(tile, sumFn, plus<uint32_t>());
        //reusing shared memory and adding accumulated values from tiles
        if (tile.thread_rank() == 0) {
            sharedForGold[tile.meta_group_rank()] = sumFp;
            sharedForSegm[tile.meta_group_rank()] = sumFn;
        }
        __syncthreads();//waiting so shared memory will be loaded evrywhere
        //on single thread we do last sum reduction
        auto active = coalesced_threads();



        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty) {
            sharedForGold[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForGold[33] += sharedForGold[i];


            };
            fpSFnS[0] += sharedForGold[33];// will be needed later for global set
            localBlockMetaData[1] = sharedForGold[33];

        }
        if ((threadIdx.x == 0) && (threadIdx.y == 1) && isNotEmpty) {


            sharedForSegm[33] = 0;//reset
            for (int i = 0; i < tile.meta_group_size(); i += 1) {
                sharedForSegm[33] += sharedForSegm[i];
            };
            fpSFnS[1] += sharedForSegm[33];// will be needed later for global set
            //setting metadata
            localBlockMetaData[2] = sharedForSegm[33];

        }

        //marking as active
//FP pass
        if ((threadIdx.x == 0) && (threadIdx.y == 0) && isNotEmpty && anyInGold[0]) {
            localBlockMetaData[7] = 1;

        };
        //FN pass
        if ((threadIdx.x == 1) && (threadIdx.y == 0) && isNotEmpty && anyInSegm[0]) {
            localBlockMetaData[9] = 1;

        };


        //after we streamed over all block we save also information about indicies of the surrounding blocks - given they are in range if not UINT32_MAX will be saved
        //top



        setNeighbourBlocks(fbArgs, 3, 13, (zMeta > 0), (-(metaData.metaXLength * metaData.MetaYLength)), linIdexMeta, metaData, localBlockMetaData);//top
        setNeighbourBlocks(fbArgs, 4, 14, (zMeta < (metaData.MetaZLength - 1)), (metaData.metaXLength * metaData.MetaYLength), linIdexMeta, metaData, localBlockMetaData);//bottom

        setNeighbourBlocks(fbArgs, 6, 15, (xMeta > 0), (-1), linIdexMeta, metaData, localBlockMetaData);//left
        setNeighbourBlocks(fbArgs, 7, 16, (xMeta < (metaData.metaXLength - 1)), 1, linIdexMeta, metaData, localBlockMetaData);//right

        setNeighbourBlocks(fbArgs, 8, 17, (yMeta < (metaData.MetaYLength - 1)), metaData.metaXLength, linIdexMeta, metaData, localBlockMetaData);//anterior
        setNeighbourBlocks(fbArgs, 9, 18, (yMeta > 0), (-metaData.metaXLength), linIdexMeta, metaData, localBlockMetaData);//posterior

        if ((threadIdx.x < 20) && (threadIdx.y == 0)) {
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + threadIdx.x] = localBlockMetaData[threadIdx.x];
        };

        __syncthreads(); // just to reduce the warp divergence


    }



    __syncthreads();


    //setting global fp and fn
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[7]), fpSFnS[0]);
    };

    if ((threadIdx.x == 1) && (threadIdx.y == 0)) {
        atomicAdd(&(minMaxes[8]), fpSFnS[1]);

    };
}




/***************************************
 * firstMetaPrepareKernel
 * ********************************/

 /*
 we add here to appropriate queue data  about metadata of blocks of intrest
 minMaxesPos- marks in minmaxes the postion of global offset counter -12) global FP offset 13) global FnOffset
 offsetMetadataArr- arrays from metadata holding data about result list offsets it can be either fbArgs.metaData.fpOffset or fbArgs.metaData.fnOffset
 */
#pragma once
__device__ inline void addToQueue(uint32_t linIdexMeta, uint8_t isGold
    , unsigned int fpFnLocCounter[1], uint32_t localWorkQueue[1600], uint32_t localOffsetQueue[1600], unsigned int localWorkQueueCounter[1]
    , uint8_t countIndexNumb, uint8_t isActiveIndexNumb, uint8_t offsetIndexNumb
    , uint32_t* metaDataArr, MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue) {

    unsigned int count = metaDataArr[linIdexMeta * metaData.metaDataSectionLength + countIndexNumb];
    //given fp is non zero we need to  add this to local queue
    if (metaDataArr[linIdexMeta * metaData.metaDataSectionLength + isActiveIndexNumb] == 1) {

        // printf("adding to local in first meta pass linIdexMeta %d isGold %d isActiveIndexNumb %d \n  ", linIdexMeta, isGold, isActiveIndexNumb);

        count = atomicAdd_block(&fpFnLocCounter[0], count);
        unsigned int  old = atomicAdd_block(&localWorkQueueCounter[0], 1);
        //we check weather we still have space in shared memory
        if (old < 1590) {// so we still have space in shared memory
            // will be equal or above isGoldOffset  if it is gold pass
            localWorkQueue[old] = linIdexMeta + (isGoldOffset * isGold);
            localOffsetQueue[old] = uint32_t(count);
        }
        else {// so we do not have any space more in the sared memory  - it is unlikely so we will just in this case save immidiately to global memory
            old = atomicAdd(&(minMaxes[9]), old);
            //workQueue
            workQueue[old] = linIdexMeta + (isGoldOffset * isGold);
            //and offset
            metaDataArr[linIdexMeta * metaData.metaDataSectionLength + offsetIndexNumb] = atomicAdd(&(minMaxes[12]), count);
        };
    }
}


#pragma once
template <typename PYO>
__global__ void firstMetaPrepareKernel(ForBoolKernelArgs<PYO> fbArgs
    , MetaDataGPU metaData, unsigned int* minMaxes, uint32_t* workQueue
    , uint32_t* origArrs, uint32_t* metaDataArr) {

    //////initializations
    thread_block cta = this_thread_block();
    char* tensorslice;// needed for iterations over 3d arrays
   //local offset counters  for fp and fn's
    __shared__ unsigned int fpFnLocCounter[1];
    // used to store the start position in global memory for whole block
    __shared__ unsigned int globalOffsetForBlock[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    //used as local work queue counter
    __shared__ unsigned int localWorkQueueCounter[1];
    //according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes so it will give us 1600 length of shared memory
    //so here we will store locally the calculated offsets and coordinates of meta data block of intrest marking also wheather we are  talking about gold or segmentation pass (fp or fn )
    __shared__ uint32_t localWorkQueue[1600];
    __shared__ uint32_t localOffsetQueue[1600];
    if ((threadIdx.x == 0)) {
        fpFnLocCounter[0] = 0;
    }
    if ((threadIdx.x == 1)) {
        localWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 2)) {
        globalWorkQueueCounter[0] = 0;
    }
    if ((threadIdx.x == 3)) {
        globalOffsetForBlock[0] = 0;
    }
    __syncthreads();


    // classical grid stride loop - in case of unlikely event we will run out of space we will empty it prematurly
    //main metadata iteration
    for (uint32_t linIdexMeta = blockIdx.x * blockDim.x + threadIdx.x; linIdexMeta < metaData.totalMetaLength; linIdexMeta += blockDim.x * gridDim.x) {

        //goldpass
        addToQueue(linIdexMeta, 0
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 1, 9, 6
            , metaDataArr, metaData, minMaxes, workQueue);
        //segmPass
        addToQueue(linIdexMeta, 1
            , fpFnLocCounter, localWorkQueue, localOffsetQueue, localWorkQueueCounter
            , 2, 7, 5
            , metaDataArr, metaData, minMaxes, workQueue);


    }
    __syncthreads();
    if ((threadIdx.x == 0)) {
        globalOffsetForBlock[0] = atomicAdd(&(minMaxes[12]), (fpFnLocCounter[0]));

    };
    if ((threadIdx.x == 1)) {
        if (localWorkQueueCounter[0] > 0) {
            globalWorkQueueCounter[0] = atomicAdd(&(minMaxes[9]), (localWorkQueueCounter[0]));
        }
    }
    __syncthreads();


    //setting offsets
    for (uint32_t i = threadIdx.x; i < localWorkQueueCounter[0]; i += blockDim.x) {
        workQueue[globalWorkQueueCounter[0] + i] = localWorkQueue[i];

        //FP pass
        if (localWorkQueue[i] >= isGoldOffset) {
            metaDataArr[(localWorkQueue[i] - isGoldOffset) * metaData.metaDataSectionLength + 5] = localOffsetQueue[i] + globalOffsetForBlock[0];
        }
        //FN pass
        else {
            metaDataArr[(localWorkQueue[i]) * metaData.metaDataSectionLength + 6] = localOffsetQueue[i] + globalOffsetForBlock[0];

        };


    }



};




/***************************************
 * memory allocations
 * ********************************/


 /*
 Get arguments for kernels
 */
#pragma once
template <typename TCC>
inline ForBoolKernelArgs<TCC> getArgsForKernel(ForFullBoolPrepArgs<TCC>& mainFunArgs
    , int& warpsNumbForMainPass, int& blockForMainPass
    , const int xLen, const int yLen, const int zLen, cudaStream_t stream
) {


    mainFunArgs.Nx = xLen;
    mainFunArgs.Ny = yLen;
    mainFunArgs.Nz = zLen;


    unsigned int* minMaxes;
    size_t sizeminMaxes = sizeof(unsigned int) * 20;
    cudaMallocAsync(&minMaxes, sizeminMaxes, stream);
    ForBoolKernelArgs<TCC> res;

    res.Nx = xLen;
    res.Ny = yLen;
    res.Nz = zLen;

    res.metaData = allocateMetaDataOnGPU(mainFunArgs.metaData, minMaxes);
    res.metaData.minMaxes = minMaxes;
    res.minMaxes = minMaxes;
    res.numberToLookFor = mainFunArgs.numberToLookFor;
    res.dbXLength = 32;
    res.dbYLength = warpsNumbForMainPass;
    res.dbZLength = 32;

    //printf("in setting bool args ylen %d dbYlen %d calculated meta %d  \n ", yLen, res.dbYLength, int(ceil(yLen / res.dbYLength)));
    res.metaData.metaXLength = int(ceil(xLen / res.dbXLength));
    res.metaData.MetaYLength = int(ceil(yLen / res.dbYLength));;
    res.metaData.MetaZLength = int(ceil(zLen / res.dbZLength));;
    res.metaData.minX = 0;
    res.metaData.minY = 0;
    res.metaData.minZ = 0;
    res.metaData.maxX = res.metaData.metaXLength;
    res.metaData.maxY = res.metaData.MetaYLength;
    res.metaData.maxZ = res.metaData.MetaZLength;

    res.metaData.totalMetaLength = res.metaData.metaXLength * res.metaData.MetaYLength * res.metaData.MetaZLength;
    res.goldArr = mainFunArgs.goldArr;
    res.segmArr = mainFunArgs.segmArr;


    return res;
}


#pragma once
/*
allocate memory after first kernel
*/
template <typename ZZR>
inline MetaDataGPU allocateMemoryAfterMinMaxesKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream) {
    ////reduced arrays
    uint32_t* origArr;
    uint32_t* metaDataArr;
    uint32_t* workQueue;
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    //read an modify
    //1)maxX 2)minX 3)maxY 4) minY 5) maxZ 6) minZ
    //7)global FP count; 8)global FN count
    unsigned int xRange = cpuArgs.metaData.minMaxes[1] - cpuArgs.metaData.minMaxes[2] + 1;
    unsigned int yRange = cpuArgs.metaData.minMaxes[3] - cpuArgs.metaData.minMaxes[4] + 1;
    unsigned int zRange = cpuArgs.metaData.minMaxes[5] - cpuArgs.metaData.minMaxes[6] + 1;
    unsigned int totalMetaLength = (xRange) * (yRange) * (zRange);
    //updating size informations
    gpuArgs.metaData.metaXLength = xRange;
    gpuArgs.metaData.MetaYLength = yRange;
    gpuArgs.metaData.MetaZLength = zRange;
    gpuArgs.metaData.totalMetaLength = totalMetaLength;
    //saving min maxes
    gpuArgs.metaData.maxX = cpuArgs.metaData.minMaxes[1];
    gpuArgs.metaData.minX = cpuArgs.metaData.minMaxes[2];
    gpuArgs.metaData.maxY = cpuArgs.metaData.minMaxes[3];
    gpuArgs.metaData.minY = cpuArgs.metaData.minMaxes[4];
    gpuArgs.metaData.maxZ = cpuArgs.metaData.minMaxes[5];
    gpuArgs.metaData.minZ = cpuArgs.metaData.minMaxes[6];

    //allocating needed memory
    // main array
    unsigned int mainArrXLength = gpuArgs.dbXLength * gpuArgs.dbYLength;
    unsigned int mainArrSectionLength = (mainArrXLength * 2);
    gpuArgs.metaData.mainArrXLength = mainArrXLength;
    gpuArgs.metaData.mainArrSectionLength = mainArrSectionLength;

    size_t sizeB = totalMetaLength * mainArrSectionLength * sizeof(uint32_t);
    //cudaMallocAsync(&mainArr, sizeB, 0);
    size_t sizeorigArr = totalMetaLength * (mainArrXLength * 2) * sizeof(uint32_t);
    cudaMallocAsync(&origArr, sizeorigArr, stream);
    size_t sizemetaDataArr = totalMetaLength * (20) * sizeof(uint32_t) + 100;
    cudaMallocAsync(&metaDataArr, sizemetaDataArr, stream);
    size_t sizeC = (totalMetaLength * 2 * sizeof(uint32_t) + 50);
    cudaMallocAsync(&workQueue, sizeC, stream);
    gpuArgs.origArrsPointer = origArr;
    gpuArgs.metaDataArrPointer = metaDataArr;
    gpuArgs.workQueuePointer = workQueue;
    return gpuArgs.metaData;
};




/*
becouse we need a lot of the additional memory spaces to minimize memory consumption allocations will be postponed after first kernel run enabling
*/
#pragma once
template <typename ZZR>
inline int allocateMemoryAfterBoolKernel(ForBoolKernelArgs<ZZR>& gpuArgs, ForFullBoolPrepArgs<ZZR>& cpuArgs, cudaStream_t stream, bool resIterNeeded, bool res3DNeeded) {

    int32_t* resultListPointerMeta;
    int32_t* resultListPointerLocal;
    int32_t* resultListPointerIterNumb;
    uint32_t* mainArrAPointer;
    uint32_t* mainArrBPointer;
    //copy on cpu
    size_t size = sizeof(unsigned int) * 20;
    cudaMemcpyAsync(cpuArgs.metaData.minMaxes, gpuArgs.metaData.minMaxes, size, cudaMemcpyDeviceToHost, stream);

    unsigned int fpPlusFn = cpuArgs.metaData.minMaxes[7] + cpuArgs.metaData.minMaxes[8];
    size = sizeof(int32_t) * (fpPlusFn + 50);


    cudaMallocAsync(&resultListPointerLocal, size, stream);
    cudaMallocAsync(&resultListPointerIterNumb, size, stream);
    cudaMallocAsync(&resultListPointerMeta, size, stream);


    // we will allocate only if we need
    if (resIterNeeded) {
        auto options =
            torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, 0)
            .requires_grad(false);

        auto resultListPointerIterNumbTensor = torch::empty((fpPlusFn + 50), options);
        gpuArgs.resultListPointerIterNumbTensor = resultListPointerIterNumbTensor;




    }
    if (res3DNeeded) {
        auto options =
            torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, 0)
            .requires_grad(false);

        auto resultListPointerLocalTensor = torch::empty((fpPlusFn + 50), options);
        auto resultListPointerMetaTensor = torch::empty((fpPlusFn + 50), options);
        gpuArgs.resultListPointerLocalTensor = resultListPointerLocalTensor;
        gpuArgs.resultListPointerMetaTensor = resultListPointerMetaTensor;

        int totalLength= gpuArgs.Nx*gpuArgs.Ny*gpuArgs.Nz;
        
        // options = torch::TensorOptions()  krowa
        // .dtype(torch::kFloat32)
        // .device(torch::kCUDA,0)
        // .requires_grad(false)
        // ;

        gpuArgs.resGold = torch::zeros(totalLength,options);//.to(torch::kCUDA);
        gpuArgs.resSegm = torch::zeros(totalLength,options);//.to(torch::kCUDA);



    }

    auto xRange = gpuArgs.metaData.metaXLength;
    auto yRange = gpuArgs.metaData.MetaYLength;
    auto zRange = gpuArgs.metaData.MetaZLength;


    size_t sizeB = gpuArgs.metaData.totalMetaLength * gpuArgs.metaData.mainArrSectionLength * sizeof(uint32_t);


    cudaMallocAsync(&mainArrAPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrAPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);


    cudaMallocAsync(&mainArrBPointer, sizeB, 0);
    cudaMemcpyAsync(mainArrBPointer, gpuArgs.origArrsPointer, sizeB, cudaMemcpyDeviceToDevice, stream);

    //just in order set it to 0
    int32_t* resultListPointerMetaCPU = (int32_t*)calloc(fpPlusFn + 50, sizeof(int32_t));
    cudaMemcpyAsync(resultListPointerMeta, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(resultListPointerIterNumb, resultListPointerMetaCPU, size, cudaMemcpyHostToDevice, stream);
    free(resultListPointerMetaCPU);

    gpuArgs.resultListPointerMeta = resultListPointerMeta;
    gpuArgs.resultListPointerLocal = resultListPointerLocal;
    gpuArgs.resultListPointerIterNumb = resultListPointerIterNumb;

    gpuArgs.mainArrAPointer = mainArrAPointer;
    gpuArgs.mainArrBPointer = mainArrBPointer;


    return fpPlusFn;
};







/***************************************
 * main kernel
 * ********************************/
template <typename TKKI>
inline __global__ void mainPassKernel(ForBoolKernelArgs<TKKI> fbArgs) {



    thread_block cta = cooperative_groups::this_thread_block();

    grid_group grid = cooperative_groups::this_grid();

    /*
    * according to https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556 it is good to keep shared memory below 16kb kilo bytes
    main shared memory spaces
    0-1023 : sourceShmem
    1024-2047 : resShmem
    2048-3071 : first register space
    3072-4095 : second register space
    4096-  4127: small 32 length resgister 3 space
    4128-4500 (372 length) : place for local work queue in dilatation kernels
    */
    // __shared__ uint32_t mainShmem[lengthOfMainShmem];
    __shared__ uint32_t mainShmem[lengthOfMainShmem];
    //cuda::associate_access_property(&mainShmem, cuda::access_property::shared{});



    constexpr size_t stages_count = 2; // Pipeline stages number

    // Allocate shared storage for a two-stage cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;

    //cuda::pipeline<cuda::thread_scope_thread>  pipeline = cuda::make_pipeline(cta, &shared_state);
    cuda::pipeline<cuda::thread_scope_block>  pipeline = cuda::make_pipeline(cta, &shared_state);



    //usefull for iterating through local work queue
    __shared__ bool isGoldForLocQueue[localWorkQueLength];
    // holding data about paddings


    // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
    __shared__ bool isAnythingInPadding[6];

    __shared__ bool isBlockFull[2];

    __shared__ uint32_t lastI[1];


    //variables needed for all threads
    __shared__ int iterationNumb[1];
    __shared__ unsigned int globalWorkQueueOffset[1];
    __shared__ unsigned int globalWorkQueueCounter[1];
    __shared__ unsigned int localWorkQueueCounter[1];
    // keeping data wheather gold or segmentation pass should continue - on the basis of global counters

    __shared__ unsigned int localTotalLenthOfWorkQueue[1];
    //counters for per block number of results added in this iteration
    __shared__ unsigned int localFpConter[1];
    __shared__ unsigned int localFnConter[1];

    __shared__ unsigned int blockFpConter[1];
    __shared__ unsigned int blockFnConter[1];

    __shared__ unsigned int fpFnLocCounter[1];

    //result list offset - needed to know where to write a result in a result list
    __shared__ unsigned int resultfpOffset[1];
    __shared__ unsigned int resultfnOffset[1];

    __shared__ unsigned int worQueueStep[1];


    /* will be used to store all of the minMaxes varibles from global memory (from 7 to 11)
    0 : global FP count;
    1 : global FN count;
    2 : workQueueCounter
    3 : resultFP globalCounter
    4 : resultFn globalCounter
    */
    __shared__ unsigned int localMinMaxes[5];

    /* will be used to store all of block metadata
  nothing at  0 index
 1 :fpCount
 2 :fnCount
 3 :fpCounter
 4 :fnCounter
 5 :fpOffset
 6 :fnOffset
 7 :isActiveGold
 8 :isFullGold
 9 :isActiveSegm
 10 :isFullSegm
 11 :isToBeActivatedGold
 12 :isToBeActivatedSegm
 12 :isToBeActivatedSegm
//now linear indexes of the blocks in all sides - if there is no block in given direction it will equal UINT32_MAX
 13 : top
 14 : bottom
 15 : left
 16 : right
 17 : anterior
 18 : posterior
    */

    __shared__ uint32_t localBlockMetaData[40];

    /*
 //now linear indexes of the previous block in all sides - if there is no block in given direction it will equal UINT32_MAX
 0 : top
 1 : bottom
 2 : left
 3 : right
 4 : anterior
 5 : posterior
    */


    /////used mainly in meta passes

//    __shared__ unsigned int fpFnLocCounter[1];
    __shared__ bool isGoldPassToContinue[1];
    __shared__ bool isSegmPassToContinue[1];





    //initializations and loading
    if (threadIdx.x == 9 && threadIdx.y == 0) { iterationNumb[0] = -1; };
    if (threadIdx.x == 11 && threadIdx.y == 0) {
        isGoldPassToContinue[0] = true;
    };
    if (threadIdx.x == 12 && threadIdx.y == 0) {
        isSegmPassToContinue[0] = true;

    };


    //here we caclulate the offset for given block depending on length of the workqueue and number of the  available blocks in a grid
    // - this will give us number of work queue items per block - we will calculate offset on the basis of the block number
    __syncthreads();

    do {

        for (uint8_t isPaddingPass = 0; isPaddingPass < 2; isPaddingPass++) {


            /////////////////////////****************************************************************************************************************
              /// dilataions

    //initial cleaning  and initializations include loading min maxes
            if (threadIdx.x == 7 && threadIdx.y == 0 && !isPaddingPass) {
                iterationNumb[0] += 1;
            };

            if (threadIdx.x == 6 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            };

            if (threadIdx.x == 1 && threadIdx.y == 0) {
                blockFpConter[0] = 0;
            };
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                blockFnConter[0] = 0;
            };
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localFpConter[0] = 0;
            };
            if (threadIdx.x == 4 && threadIdx.y == 0) {
                localFnConter[0] = 0;
            };
            if (threadIdx.x == 9 && threadIdx.y == 0) {
                isBlockFull[0] = true;
            };
            if (threadIdx.x == 9 && threadIdx.y == 1) {
                isBlockFull[1] = true;
            };

            if (threadIdx.x == 10 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            };



            if (threadIdx.x == 0 && threadIdx.y == 0) {
                localTotalLenthOfWorkQueue[0] = fbArgs.minMaxes[9];
                globalWorkQueueOffset[0] = floor((float)(localTotalLenthOfWorkQueue[0] / gridDim.x)) + 1;
                worQueueStep[0] = min(localWorkQueLength, globalWorkQueueOffset[0]);
            };

            if (threadIdx.y == 1) {
                cooperative_groups::memcpy_async(cta, (&localMinMaxes[0]), (&fbArgs.minMaxes[7]), cuda::aligned_size_t<4>(sizeof(unsigned int) * 5));
            }

            __syncthreads();

            /// load work QueueData into shared memory
            for (uint32_t bigloop = blockIdx.x * globalWorkQueueOffset[0]; bigloop < ((blockIdx.x + 1) * globalWorkQueueOffset[0]); bigloop += worQueueStep[0]) {

                //grid stride loop - sadly most of threads will be idle
               ///////// loading to work queue
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    for (uint16_t ii = cta.thread_rank(); ii < worQueueStep[0]; ii += cta.size()) {

                        mainShmem[startOfLocalWorkQ + ii] = fbArgs.workQueuePointer[bigloop + ii];
                        isGoldForLocQueue[ii] = (mainShmem[startOfLocalWorkQ + ii] >= isGoldOffset);
                        mainShmem[startOfLocalWorkQ + ii] = mainShmem[startOfLocalWorkQ + ii] - isGoldOffset * isGoldForLocQueue[ii];


                    }

                }
                //now all of the threads in the block needs to have the same i value so we will increment by 1 we are preloading to the pipeline block metaData
                ////##### pipeline Step 0

                __syncthreads();




                //loading metadata
                pipeline.producer_acquire();
                if (((bigloop) < localTotalLenthOfWorkQueue[0]) && ((bigloop) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {

                    cuda::memcpy_async(cta, (&localBlockMetaData[0]),
                        (&fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ] * fbArgs.metaData.metaDataSectionLength])
                        , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                }
                pipeline.producer_commit();


                __syncthreads();

                for (uint32_t i = 0; i < worQueueStep[0]; i += 1) {




                    if (((bigloop + i) < localTotalLenthOfWorkQueue[0]) && ((bigloop + i) < ((blockIdx.x + 1) * globalWorkQueueOffset[0]))) {



                        pipeline.producer_acquire();
                        cuda::memcpy_async(cta, &mainShmem[begSourceShmem], &getSourceReduced(fbArgs, iterationNumb)[
                            mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                            cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength), pipeline);
                        pipeline.producer_commit();

                        //just so pipeline will work well
                        pipeline.consumer_wait();



                        pipeline.consumer_release();
                        __syncthreads();

                        ///////// step 1 load top and process main data
                                        //load top
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 13]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process main
                        pipeline.consumer_wait();
                        //marking weather block is already full and no more dilatations are possible
                        if (__popc(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]) < 32) {
                            isBlockFull[i & 1] = false;
                        }
                        mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] = bitDilatate(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32]);
                        pipeline.consumer_release();

                        ///////// step 2 load bottom and process top
                                        //load bottom
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 14]
                                * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process top
                        pipeline.consumer_wait();


                        if (localBlockMetaData[(i & 1) * 20 + 13] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 0)) {
                                // printf("setting padding top val %d \n ", isAnythingInPadding[0]);
                                isAnythingInPadding[0] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32] >> 31) & 1) << 0;
                        }

                        pipeline.consumer_release();
                        __syncthreads();

                        /////////// step 3 load right  process bottom
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 16] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process bototm
                        pipeline.consumer_wait();


                        if (localBlockMetaData[(i & 1) * 20 + 14] < isGoldOffset) {
                            if (isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], 31)) {
                                isAnythingInPadding[1] = true;
                            };
                            // if in bit of intrest of neighbour block is set
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] |= ((mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] >> 0) & 1) << 31;
                        }



                        /*  dilatateHelperTopDown(1, mainShmem, isAnythingInPadding, localBlockMetaData, 14
                              , 0, 31
                              , begSecRegShmem, i);*/

                        pipeline.consumer_release();
                        /////////// step 4 load left process right
                                        //load left
                        pipeline.producer_acquire();
                        if (mainShmem[startOfLocalWorkQ + i] > 0) {
                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[(mainShmem[startOfLocalWorkQ + i] - 1) * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process right
                        pipeline.consumer_wait();

                        if (threadIdx.x == (fbArgs.dbXLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 16] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[3] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + (threadIdx.y * 32)];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x + 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        pipeline.consumer_release();
                        __syncthreads();
                        /////// step 5 load anterior process left
                                        //load anterior
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {

                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 17] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();
                        //process left
                        pipeline.consumer_wait();

                        // so we first check for corner cases
                        if (threadIdx.x == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 15] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[2] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + 31 + threadIdx.y * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x - 1) + (threadIdx.y) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        pipeline.consumer_release();
                        __syncthreads();

                        /////// step 6 load posterior process anterior
                                        //load posterior
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {


                            cuda::memcpy_async(cta, (&mainShmem[begSecRegShmem]),
                                &getSourceReduced(fbArgs, iterationNumb)[localBlockMetaData[(i & 1) * 20 + 18] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])],
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);
                        }
                        pipeline.producer_commit();

                        //process anterior
                        pipeline.consumer_wait();

                        // so we first check for corner cases
                        if (threadIdx.y == (fbArgs.dbYLength - 1)) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 17] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[4] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begfirstRegShmem + threadIdx.x];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y + 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }


                        pipeline.consumer_release();
                        __syncthreads();

                        /////// step 7
                                       //load reference if needed or data for next iteration if there is such
                                        //process posterior, save data from res shmem to global memory also we mark weather block is full
                        pipeline.producer_acquire();

                        //if block should be validated we load data for validation
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                        > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            cuda::memcpy_async(cta, (&mainShmem[begfirstRegShmem]),
                                &fbArgs.origArrsPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (isGoldForLocQueue[i])], //we look for
                                cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                                , pipeline);

                        }
                        else {//if we are not validating we immidiately start loading data for next loop
                            if (i + 1 < worQueueStep[0]) {
                                cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);


                            }
                        }


                        pipeline.producer_commit();

                        //processPosteriorAndSaveResShmem

                        pipeline.consumer_wait();
                        //dilatate posterior


                        // so we first check for corner cases
                        if (threadIdx.y == 0) {
                            // now we need to load the data from the neigbouring blocks
                            //first checking is there anything to look to
                            if (localBlockMetaData[(i & 1) * 20 + 18] < isGoldOffset) {
                                //now we load - we already done earlier up and down so now we are considering only anterior, posterior , left , right possibilities
                                if (mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32] > 0) {
                                    isAnythingInPadding[5] = true;

                                };
                                mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32] =
                                    mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                    | mainShmem[begSecRegShmem + threadIdx.x + (fbArgs.dbYLength - 1) * 32];

                            };
                        }
                        else {//given we are not in corner case we need just to do the dilatation using biwise or with the data inside the block
                            mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32]
                                = mainShmem[begSourceShmem + (threadIdx.x) + (threadIdx.y - 1) * 32]
                                | mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];

                        }

                        //now all data should be properly dilatated we save it to global memory
                        //try save target reduced via mempcy async ...


                        //cuda::memcpy_async(cta,
                        //    &getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])]
                        //    , (&mainShmem[begResShmem]),
                        //    cuda::aligned_size_t<128>(sizeof(uint32_t) * fbArgs.metaData.mainArrXLength)
                        //    , pipeline);



                        getTargetReduced(fbArgs, iterationNumb)[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.mainArrSectionLength + fbArgs.metaData.mainArrXLength * (1 - isGoldForLocQueue[i])
                            + threadIdx.x + threadIdx.y * 32]
                            = mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32];





                        pipeline.consumer_release();

                        __syncthreads();

                        //////// step 8 basically in order to complete here anyting the count need to be bigger than counter
                                                      // loading for next block if block is not to be validated it was already done earlier
                        pipeline.producer_acquire();
                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                            if (i + 1 < worQueueStep[0]) {


                                cuda::memcpy_async(cta, (&localBlockMetaData[((i + 1) & 1) * 20]),
                                    (&fbArgs.metaDataArrPointer[(mainShmem[startOfLocalWorkQ + 1 + i])
                                        * fbArgs.metaData.metaDataSectionLength])
                                    , cuda::aligned_size_t<4>(sizeof(uint32_t) * 20), pipeline);

                            }
                        }
                        pipeline.producer_commit();




                        __syncthreads();

                        //validation - so looking for newly covered voxel for opposite array so new fps or new fns
                        pipeline.consumer_wait();

                        if (localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 1)] //fp for gold and fn count for not gold
                            > localBlockMetaData[(i & 1) * 20 + ((1 - isGoldForLocQueue[i]) + 3)]) {// so count is bigger than counter so we should validate
                                        // now we look through bits and when some is set we call it a result
                            #pragma unroll
                            for (uint8_t bitPos = 0; bitPos < 32; bitPos++) {
                                //if any bit here is set it means it should be added to result list
                                if (isBitAt(mainShmem[begResShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && !isBitAt(mainShmem[begSourceShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    && isBitAt(mainShmem[begfirstRegShmem + threadIdx.x + threadIdx.y * 32], bitPos)
                                    ) {

                                    //just re
                                    mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = 0;
                                    ////// IMPORTANT for some reason in order to make it work resultfnOffset and resultfnOffset swith places
                                    if (isGoldForLocQueue[i]) {
                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFpConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 6] + localBlockMetaData[(i & 1) * 20 + 3]);

                                    }
                                    else {

                                        mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32] = uint32_t(atomicAdd_block(&(localFnConter[0]), 1) + localBlockMetaData[(i & 1) * 20 + 5] + localBlockMetaData[(i & 1) * 20 + 4]);


                                    };
                                    //   add results to global memory
                                    //we add one gere jjust to distinguish it from empty result
                                    fbArgs.resultListPointerMeta[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(mainShmem[startOfLocalWorkQ + i] + (isGoldOffset * isGoldForLocQueue[i]) + 1);
                                    fbArgs.resultListPointerLocal[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t((fbArgs.dbYLength * 32 * bitPos) + (threadIdx.y * 32) + (threadIdx.x));
                                    fbArgs.resultListPointerIterNumb[mainShmem[begSecRegShmem + threadIdx.x + threadIdx.y * 32]] = uint32_t(iterationNumb[0] + 1);




                                }

                            };

                        }
                        /////////
                        pipeline.consumer_release();

                        /// /// cleaning

                        __syncthreads();

                        if (threadIdx.x == 9 && threadIdx.y == 2) {// this is how it is encoded wheather it is gold or segm block

         //executed in case of previous block
                            if (isBlockFull[i & 1] && i >= 0) {
                                //setting data in metadata that block is full
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 10 - (isGoldForLocQueue[i] * 2)] = true;
                            }
                            //resetting for some reason  block 0 gets as full even if it should not ...
                            isBlockFull[i & 1] = true;// mainShmem[startOfLocalWorkQ + i]>0;//!isPaddingPass;
                        };




                        //we do it only for non padding pass
                        if (threadIdx.x < 6 && threadIdx.y == 1 && !isPaddingPass) {
                            //executed in case of previous block
                            if (i >= 0) {

                                if (localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] < isGoldOffset) {

                                    if (isAnythingInPadding[threadIdx.x]) {
                                        // holding data weather we have anything in padding 0)top  1)bottom, 2)left 3)right, 4)anterior, 5)posterior,
                                        fbArgs.metaDataArrPointer[localBlockMetaData[(i & 1) * 20 + 13 + threadIdx.x] * fbArgs.metaData.metaDataSectionLength + 12 - isGoldForLocQueue[i]] = 1;
                                    }

                                }
                            }
                            isAnythingInPadding[threadIdx.x] = false;
                        };






                        if (threadIdx.x == 7 && threadIdx.y == 0) {
                            //this will be executed only if fp or fn counters are bigger than 0 so not during first pass
                            if (localFpConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 3] += localFpConter[0];

                                blockFpConter[0] += localFpConter[0];
                                localFpConter[0] = 0;
                            }


                        };
                        if (threadIdx.x == 8 && threadIdx.y == 0) {

                            if (localFnConter[0] > 0) {
                                fbArgs.metaDataArrPointer[mainShmem[startOfLocalWorkQ + i] * fbArgs.metaData.metaDataSectionLength + 4] += localFnConter[0];

                                blockFnConter[0] += localFnConter[0];
                                localFnConter[0] = 0;
                            }
                        };

                        __syncthreads();

                    }
                }

                //here we are after all of the blocks planned to be processed by this block are

                // just for pipeline to work
                pipeline.consumer_wait();



                pipeline.consumer_release();

            }

            __syncthreads();

            //     updating global counters
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                if (blockFpConter[0] > 0) {
                    atomicAdd(&(fbArgs.minMaxes[10]), (blockFpConter[0]));
                }
            };
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                if (blockFnConter[0] > 0) {
                    //if (blockFnConter[0]>10) {
                    //    printf("Fn %d  ", blockFnConter[0]);
                    //}
                    atomicAdd(&(fbArgs.minMaxes[11]), (blockFnConter[0]));
                }
            };

            // in first thread block we zero work queue counter
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                if (blockIdx.x == 0) {

                    fbArgs.minMaxes[9] = 0;
                }
            };

            grid.sync();
            /////////////////////////****************************************************************************************************************

            /// metadata pass


            // preparation loads
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                fpFnLocCounter[0] = 0;
            }
            if (threadIdx.x == 1 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 2 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;
            }
            if (threadIdx.x == 3 && threadIdx.y == 0) {
                localWorkQueueCounter[0] = 0;

            }

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isGoldPassToContinue[0]
                    = ((fbArgs.minMaxes[7] * fbArgs.robustnessPercent) > fbArgs.minMaxes[10]);

            };

            if (threadIdx.x == 0 && threadIdx.y == 1) {

                isSegmPassToContinue[0]
                    = ((fbArgs.minMaxes[8] * fbArgs.robustnessPercent) > fbArgs.minMaxes[11]);
            };


            __syncthreads();

            /////////////////////////////////

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //goldpass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);
                        //setting to be activated to 0
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 11] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7] = 1;


                    };

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //gold pass
                    if (isGoldPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 7]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 8]) {

                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta + (isGoldOffset);

                    };

                }
            }

            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];
                }
                __syncthreads();

            }

            __syncthreads();

            if (threadIdx.x == 0 && threadIdx.y == 0) {

                localWorkQueueCounter[0] = 0;
            }
            __syncthreads();

            for (uint32_t linIdexMeta = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
                ; linIdexMeta <= fbArgs.metaData.totalMetaLength
                ; linIdexMeta += (blockDim.x * blockDim.y * gridDim.x)
                ) {


                if (isPaddingPass == 0) {

                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;

                        //setting to be activated to 0
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 12] = 0;
                        //setting active to 1
                        fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9] = 1;

                    }

                }
                //contrary to number it is when we are not in padding pass
                else {
                    //segm pass
                    if ((isSegmPassToContinue[0] && fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 9]
                        && !fbArgs.metaDataArrPointer[linIdexMeta * fbArgs.metaData.metaDataSectionLength + 10])) {



                        mainShmem[atomicAdd_block(&localWorkQueueCounter[0], 1)] = linIdexMeta;
                    }

                }
            }
            __syncthreads();

            if (localWorkQueueCounter[0] > 0) {
                if (threadIdx.x == 0 && threadIdx.y == 0) {
                    globalWorkQueueCounter[0] = atomicAdd(&(fbArgs.minMaxes[9]), (localWorkQueueCounter[0]));


                }
                __syncthreads();
                for (uint32_t linI = threadIdx.y * blockDim.x + threadIdx.x; linI < localWorkQueueCounter[0]; linI += blockDim.x * blockDim.y) {
                    fbArgs.workQueuePointer[globalWorkQueueCounter[0] + linI] = mainShmem[linI];

                }

            }



            grid.sync();
        }



    } while (isGoldPassToContinue[0] || isSegmPassToContinue[0]);


    //setting global iteration number to local one
    if (blockIdx.x == 0) {
        if (threadIdx.x == 2 && threadIdx.y == 0) {
            fbArgs.metaData.minMaxes[13] = (iterationNumb[0] + 1);
        }
    }
}




/***************************************
 * putting all kernels and memory allocations together
 * ********************************/


 /*
 get data from occupancy calculator API used to get optimal number of thread blocks and threads per thread block
 */
template <typename T>
inline occupancyCalcData getOccupancy() {

    occupancyCalcData res;

    int blockSize; // The launch configurator returned block size
    int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize; // The actual grid size needed, based on input size

    // for min maxes kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)getMinMaxes<T>,
        0);
    res.warpsNumbForMinMax = blockSize / 32;
    res.blockSizeForMinMax = minGridSize;

    // for min maxes kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.warpsNumbForboolPrepareKernel = blockSize / 32;
    res.blockSizeFoboolPrepareKernel = minGridSize;
    // for first meta pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)boolPrepareKernel<T>,
        0);
    res.theadsForFirstMetaPass = blockSize;
    res.blockForFirstMetaPass = minGridSize;
    //for main pass kernel
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)mainPassKernel<T>,
        0);
    res.warpsNumbForMainPass = blockSize / 32;
    res.blockForMainPass = minGridSize;

    //printf("warpsNumbForMainPass %d blockForMainPass %d  ", res.warpsNumbForMainPass, res.blockForMainPass);
    return res;
}



/*
TODO consider representing as a CUDA graph
executing Algorithm as CUDA graph  based on official documentation and
https://codingbyexample.com/2020/09/25/cuda-graph-usage/
*/
#pragma once
template <typename T>
ForBoolKernelArgs<T> executeHausdoff(ForFullBoolPrepArgs<T>& fFArgs, const int WIDTH, const int HEIGHT, const int DEPTH, occupancyCalcData& occData,
    cudaStream_t stream, bool resToSave, float robustnessPercent, bool resIterNeeded, bool res3DNeeded) {
    fFArgs.robustnessPercent = robustnessPercent;
    T* goldArrPointer = (T*)fFArgs.goldArr.data_ptr();
    T* segmArrPointer = (T*)fFArgs.segmArr.data_ptr();

    ForBoolKernelArgs<T> fbArgs = getArgsForKernel<T>(fFArgs, occData.warpsNumbForMainPass, occData.blockForMainPass, WIDTH, HEIGHT, DEPTH, stream);

    getMinMaxes << <occData.blockSizeForMinMax, dim3(32, occData.warpsNumbForMinMax), 0, stream >> > (fbArgs, fbArgs.minMaxes
        , goldArrPointer
        , segmArrPointer
        , fbArgs.metaData);



    fbArgs.metaData = allocateMemoryAfterMinMaxesKernel(fbArgs, fFArgs, stream);
    fbArgs.robustnessPercent = robustnessPercent;
    boolPrepareKernel << <occData.blockSizeFoboolPrepareKernel, dim3(32, occData.warpsNumbForboolPrepareKernel), 0, stream >> > (
        fbArgs, fbArgs.metaData, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer
        , goldArrPointer
        , segmArrPointer
        , fbArgs.minMaxes);


    int fpPlusFn = allocateMemoryAfterBoolKernel(fbArgs, fFArgs, stream, resIterNeeded, res3DNeeded);



    firstMetaPrepareKernel << <occData.blockForFirstMetaPass, occData.theadsForFirstMetaPass, 0, stream >> > (fbArgs, fbArgs.metaData, fbArgs.minMaxes, fbArgs.workQueuePointer, fbArgs.origArrsPointer, fbArgs.metaDataArrPointer);



    void* kernel_args[] = { &fbArgs };
    cudaLaunchCooperativeKernel((void*)(mainPassKernel<int>), occData.blockForMainPass, dim3(32, occData.warpsNumbForMainPass), kernel_args, 0, stream);

    int len = fbArgs.resultListPointerIterNumbTensor.sizes()[0]; //krowa
    auto size = sizeof(int32_t) * len;


    //copy to the output tensor the rsult
    if (resIterNeeded) {
        cudaMemcpyAsync(fbArgs.resultListPointerIterNumbTensor.data_ptr(), fbArgs.resultListPointerIterNumb, size, cudaMemcpyDeviceToDevice, stream);

        //krowa
       // cudaMemcpyAsync(fbArgs.resultListPointerLocalTensor.data_ptr(), fbArgs.resultListPointerLocal, size, cudaMemcpyDeviceToDevice, stream);
        //cudaMemcpyAsync(fbArgs.resultListPointerMetaTensor.data_ptr(), fbArgs.resultListPointerMeta, size, cudaMemcpyDeviceToDevice, stream);

    
    }
    if (res3DNeeded) {
        //krowa
        cudaMemcpyAsync(fbArgs.resultListPointerLocalTensor.data_ptr(), fbArgs.resultListPointerLocal, size, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(fbArgs.resultListPointerMetaTensor.data_ptr(), fbArgs.resultListPointerMeta, size, cudaMemcpyDeviceToDevice, stream);
    }




    cudaFreeAsync(fbArgs.resultListPointerMeta, stream);
    cudaFreeAsync(fbArgs.resultListPointerLocal, stream);
    cudaFreeAsync(fbArgs.resultListPointerIterNumb, stream);
    cudaFreeAsync(fbArgs.workQueuePointer, stream);
    cudaFreeAsync(fbArgs.origArrsPointer, stream);
    cudaFreeAsync(fbArgs.metaDataArrPointer, stream);
    cudaFreeAsync(fbArgs.mainArrAPointer, stream);
    cudaFreeAsync(fbArgs.mainArrBPointer, stream);


    return fbArgs;

}



//    modified  from https://github.com/pytorch/pytorch/blob/61d6c4386459441710fb4cfa2929a3f77e95e5f7/aten/src/ATen/Dispatch.h
#define AT_DISPATCH_ALL_TYPESWithBool(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                 \
    switch (_st) {                                                              \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Byte, uint8_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Char, int8_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int32_t, __VA_ARGS__)     \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Short, int16_t, __VA_ARGS__)   \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Bool, bool, __VA_ARGS__)   \
      default:                                                                  \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");          \
    }                                                                           \
  }()







template <typename T>
int getHausdorffDistance_CUDA_Generic(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, bool resIterNeeded, at::Tensor numberToLookFor, bool res3DNeeded) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    //size_t size = sizeof(unsigned int) * 20;
    //unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    //metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor.item<T>();
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, resIterNeeded, res3DNeeded);

    size_t sizeMinMax = sizeof(unsigned int) * 20;
    //making sure we have all resultsto copy on cpu
    cudaDeviceSynchronize();
    cudaMemcpy(metaData.minMaxes, fbArgs.metaData.minMaxes, sizeMinMax, cudaMemcpyDeviceToHost);

    int result = metaData.minMaxes[13];

    cudaFreeAsync(fbArgs.minMaxes, stream1);
    //free(minMaxesCPU);


    cudaStreamDestroy(stream1);
    cudaDeviceSynchronize();

    return result;
}




template <typename T>
at::Tensor getHausdorffDistance_CUDA_FullResList_local(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, at::Tensor numberToLookFor) {
    //TODO() use https ://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    //size_t size = sizeof(unsigned int) * 20;
    //unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    //metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor.item<T>();
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, true, false);




    cudaFreeAsync(fbArgs.minMaxes, stream1);
    //free(metaData.minMaxesCPU);



    cudaStreamDestroy(stream1);
    int len = fbArgs.resultListPointerIterNumbTensor.sizes()[0]; //krowa

    return fbArgs.resultListPointerIterNumbTensor;
}

/*
Functions for pybind
*/
int getHausdorffDistance_CUDA(at::Tensor goldStandard,
    at::Tensor algoOutput
    , const int WIDTH, const  int HEIGHT, const  int DEPTH
    , const float robustnessPercent, at::Tensor numberToLookFor) {

    int res = 0;

    AT_DISPATCH_ALL_TYPESWithBool(goldStandard.type(), "getHausdorffDistance_CUDA", ([&] {
        res = getHausdorffDistance_CUDA_Generic<scalar_t>(goldStandard, algoOutput, WIDTH, HEIGHT, DEPTH, robustnessPercent, false, numberToLookFor, false);

        }));
    return res;
}


at::Tensor getHausdorffDistance_CUDA_FullResList(at::Tensor goldStandard,
    at::Tensor algoOutput
    , const int WIDTH, const  int HEIGHT, const  int DEPTH
    , const float robustnessPercent, at::Tensor numberToLookFor) {


    at::Tensor res;
    AT_DISPATCH_ALL_TYPESWithBool(goldStandard.type(), "getHausdorffDistance_CUDA_FullResList", ([&] {
        res = getHausdorffDistance_CUDA_FullResList_local<scalar_t>(goldStandard, algoOutput, WIDTH, HEIGHT, DEPTH, robustnessPercent, numberToLookFor);

        }));

    return res;
}




/***************************************
 *enable getting localizations of the voxels that contributed to HD
 * ********************************/



 /*
 on the basis of result lists return the location of each voxel that contributed to Hausdorff distance and how much it contributed
 fbArgs - struct with needed data
 resGold - tensor where output will be stored from gold mask dilatations
 resSegm - tensor where output will be stored from algorithm output dilatations
 len - length of result list we will iterate over
 */
template <typename T>
__global__ void get3Dres_local_kernel(ForBoolKernelArgs<T> fbArgs, int32_t* resGold, int32_t* resSegm, int len, int32_t* iterNumbb) {

    //simple grid stride loop
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        if (fbArgs.resultListPointerLocal[i] > 0 || fbArgs.resultListPointerMeta[i] > 0) {
            uint32_t linIdexMeta = fbArgs.resultListPointerMeta[i] - (isGoldOffset * (fbArgs.resultListPointerMeta[i] >= isGoldOffset)) - 1;
            uint32_t xMeta = linIdexMeta % fbArgs.metaData.metaXLength;
            uint32_t zMeta = uint32_t(floor((float)(linIdexMeta / (fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength))));
            uint32_t yMeta = uint32_t(floor((float)((linIdexMeta - ((zMeta * fbArgs.metaData.metaXLength * fbArgs.metaData.MetaYLength) + xMeta)) / fbArgs.metaData.metaXLength)));

            auto linLocal = fbArgs.resultListPointerLocal[i];
            auto xLoc = linLocal % fbArgs.dbXLength;
            auto zLoc = uint32_t(floor((float)(linLocal / (32 * fbArgs.dbYLength))));
            auto yLoc = uint32_t(floor((float)((linLocal - ((zLoc * 32 * fbArgs.dbYLength) + xLoc)) / 32)));

            // setting appropriate  spot in the result to a given value
            if (fbArgs.resultListPointerMeta[i] >= isGoldOffset) {
                //resGold[1] = 1.0;
                resGold[(xMeta * 32 + xLoc) + (yMeta * fbArgs.dbYLength + yLoc) * fbArgs.Nx + (zMeta * 32 + zLoc) * fbArgs.Nx * fbArgs.Ny] = (iterNumbb[i]);
            }
       
        else {
            resSegm[(xMeta * 32 + xLoc) + (yMeta * fbArgs.dbYLength + yLoc) * fbArgs.Nx + (zMeta * 32 + zLoc) * fbArgs.Nx * fbArgs.Ny] = (iterNumbb[i]);
        }
            uint32_t x = xMeta * 32 + xLoc;
            uint32_t y = yMeta * fbArgs.dbYLength + yLoc;
            uint32_t z = zMeta * 32 + zLoc;
            int32_t iterNumb = iterNumbb[i];
        }
    }
}



/*
takes two 3D tensord and computes the element wise avarage from two entries and save result in resGold
voxelsNumber - number of voxel in resGold = resSegm
*/
template <typename T>
__global__ void elementWiseAverage(ForBoolKernelArgs<T> fbArgs, int32_t* resGold, int32_t* resSegm, int voxelsNumber,int maxEl) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < voxelsNumber; i += blockDim.x * gridDim.x) {
        resGold[i] = (resGold[i] + resSegm[i]);//*maxEl;
    }
}




/*
3D tensor with data how much given voxel contributed to result in gold mask segmentations  other mask dilatations (the mean of those)
*/
template <typename T>
at::Tensor getHausdorffDistance_CUDA_3Dres_local(at::Tensor goldStandard,
    at::Tensor algoOutput
    , int WIDTH, int HEIGHT, int DEPTH, float robustnessPercent, at::Tensor numberToLookFor) {

    
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    MetaDataCPU metaData;
    //size_t size = sizeof(unsigned int) * 20;
    //unsigned int* minMaxesCPU = (unsigned int*)malloc(size);
    //metaData.minMaxes = minMaxesCPU;

    ForFullBoolPrepArgs<T> forFullBoolPrepArgs;
    forFullBoolPrepArgs.metaData = metaData;
    forFullBoolPrepArgs.numberToLookFor = numberToLookFor.item<T>();
    forFullBoolPrepArgs.goldArr = goldStandard;
    forFullBoolPrepArgs.segmArr = algoOutput;

    occupancyCalcData occData = getOccupancy<T>();

    ForBoolKernelArgs<T> fbArgs = executeHausdoff(forFullBoolPrepArgs, WIDTH, HEIGHT, DEPTH, occData, stream1, false, robustnessPercent, true, true);


    cudaDeviceSynchronize();

    // int totalLength= WIDTH*HEIGHT* DEPTH;
    //     auto options = torch::TensorOptions()
    //     .dtype(torch::kFloat32)
    //     //.device(torch::kCUDA,0)
    //     .requires_grad(false)
    //     ;

    // at::Tensor resGold = torch::zeros(totalLength,options);//.to(torch::kCUDA);
    // at::Tensor resSegm = torch::zeros(totalLength,options);//.to(torch::kCUDA);

    //at::Tensor resGold = torch::zeros({ WIDTH, HEIGHT, DEPTH }, options);
    //at::Tensor resSegm = torch::empty({ WIDTH, HEIGHT, DEPTH }, options);


    int len = fbArgs.resultListPointerIterNumbTensor.sizes()[0]; //krowa


    //occupancy calculator
    int minGridSize = 0;
    int blockSize = 0;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)get3Dres_local_kernel<T>,
        0);

    
    //simple one dimensional kernel
    get3Dres_local_kernel << <minGridSize, blockSize, 0, stream1 >> > (fbArgs, (int32_t*)fbArgs.resGold.data_ptr()
    , (int32_t*)fbArgs.resSegm.data_ptr()
    , len
    ,(int32_t*)fbArgs.resultListPointerIterNumbTensor.data_ptr());


    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)elementWiseAverage<T>,
        0);

    int maxEl = forFullBoolPrepArgs.metaData.minMaxes[13];

    //get element wise average
    elementWiseAverage << <minGridSize, blockSize, 0, stream1 >> > (fbArgs, (int32_t*)fbArgs.resGold.data_ptr(), (int32_t*)fbArgs.resSegm.data_ptr(), WIDTH * HEIGHT * DEPTH, maxEl);


    cudaFreeAsync(fbArgs.minMaxes, stream1);

    cudaStreamDestroy(stream1);

    cudaDeviceSynchronize();

    return  fbArgs.resGold;

}



at::Tensor getHausdorffDistance_CUDA_3Dres(at::Tensor goldStandard,
    at::Tensor algoOutput
    , const int WIDTH, const  int HEIGHT, const  int DEPTH
    , const float robustnessPercent, at::Tensor numberToLookFor) {


    at::Tensor res;
    AT_DISPATCH_ALL_TYPESWithBool(goldStandard.type(), "getHausdorffDistance_CUDA_3Dres_local", ([&] {
        res = getHausdorffDistance_CUDA_3Dres_local<scalar_t>(goldStandard, algoOutput, WIDTH, HEIGHT, DEPTH, robustnessPercent, numberToLookFor);

        }));

    return res;

}


/*************************************************************
 *Oliviera Algorithm
 *************************************************************************************/

 typedef unsigned char uchar;
 typedef unsigned int uint;
 #pragma once
 class Volume {
  private:
   bool* volume;
   int width, height, depth;
   int getLinearIndex(int x, int y, int z);
 
  public:
   bool getVoxelValue(int x, int y, int z);
   bool getPixelValue(int x, int y);
   uint getWidth();
   uint getHeight();
   uint getDepth();
   bool* getVolume();
   void setVoxelValue(bool value, int x, int y, int z);
   void setPixelValue(bool value, int x, int y);
   Volume(int width, int height, int depth);
   Volume(int width, int height);
   void dispose();
 };
 
 #define CUDA_DEVICE_INDEX 0 // setting the index of your CUDA device
 
 #define IS_3D \
   1 // setting this to 0 would grant a very slightly improvement on the performance if working with images only
 #define CHEBYSHEV \
   0 // if not set to 1, then this algorithm would use an Euclidean-like metric, it is just an approximation.
 // It can be changed according to the structuring element
 #pragma once
 class HausdorffDistance {
  private:
   void print(cudaError_t error, char* msg);
 
  public:
   int computeDistance(Volume* img1, Volume* img2, bool* d_img1, bool* d_img2);
 };
 
 inline Volume::Volume(const int width, const int height, const int depth) {
   this->width = width;
   this->height = height;
   this->depth = depth;
   volume = (bool*)calloc(width * height * depth, sizeof(bool));
 }
 
 #pragma once
 inline Volume::Volume(const int width, const int height) {
   this->width = width;
   this->height = height;
   this->depth = 1;
   volume = (bool*)calloc(width * height * depth, sizeof(bool));
 }
 #pragma once
 inline int Volume::getLinearIndex(const int x, const int y, const int z) {
   const int a = 1, b = width, c = (width) * (height);
   return a * x + b * y + c * z;
 }
 
 inline uint Volume::getWidth() {
   return this->width;
 }
 inline uint Volume::getHeight() {
   return this->height;
 }
 inline uint Volume::getDepth() {
   return this->depth;
 }
 inline bool* Volume::getVolume() {
   return this->volume;
 }
 inline bool Volume::getPixelValue(int x, int y) {
   return this->volume[getLinearIndex(x, y, 0)];
 }
 #pragma once
 inline bool Volume::getVoxelValue(int x, int y, int z) {
   return volume[getLinearIndex(x, y, z)];
 }
 #pragma once
 inline void Volume::setPixelValue(bool value, const int x, const int y) {
   volume[getLinearIndex(x, y, 0)] = value;
 }
 #pragma once
 inline void Volume::setVoxelValue(bool value, const int x, const int y, const int z) {
   volume[getLinearIndex(x, y, z)] = value;
 }
 #pragma once
 inline void Volume::dispose() {
   free(volume);
 }
 
 typedef unsigned char uchar;
 typedef unsigned int uint;
 
 #pragma once
 __device__ int finished; // global variable that contains a boolean which indicates when to stop the kernel processing
 #pragma once
 __constant__ __device__ int WIDTH, HEIGHT, DEPTH; // constant variables that contain the size of the volume
 
 #pragma once
 __global__ void dilate(
     const bool* IMG1,
     const bool* IMG2,
     const bool* img1Read,
     const bool* img2Read,
     bool* img1Write,
     bool* img2Write) {
   const int id = blockDim.x * blockIdx.x + threadIdx.x;
 #if !IS_3D
   const int x = id % WIDTH, y = id / WIDTH;
 #else
   const int x = id % WIDTH, y = (id / WIDTH) % HEIGHT, z = (id / WIDTH) / HEIGHT;
 #endif
 
   if (id < WIDTH * HEIGHT * DEPTH) {
     if (img1Read[id]) {
       if (x + 1 < WIDTH)
         img1Write[id + 1] = true;
       if (x - 1 >= 0)
         img1Write[id - 1] = true;
       if (y + 1 < HEIGHT)
         img1Write[id + WIDTH] = true;
       if (y - 1 >= 0)
         img1Write[id - WIDTH] = true;
 #if IS_3D // if working with 3d volumes, then the 3D part
       if (z + 1 < DEPTH)
         img1Write[id + WIDTH * HEIGHT] = true;
       if (z - 1 >= 0)
         img1Write[id - WIDTH * HEIGHT] = true;
 #endif
 
 #if CHEBYSHEV
       // diagonals
       if (x + 1 < WIDTH && y - 1 >= 0)
         img1Write[id - WIDTH + 1] = true;
       if (x - 1 >= 0 && y - 1 >= 0)
         img1Write[id - WIDTH - 1] = true;
       if (x + 1 < WIDTH && y + 1 < HEIGHT)
         img1Write[id + WIDTH + 1] = true;
       if (x - 1 >= 0 && y + 1 < HEIGHT)
         img1Write[id + WIDTH - 1] = true;
 #if IS_3D // if working with 3d volumes, then the 3D part
       if (z + 1 < DEPTH && x + 1 < WIDTH && y - 1 >= 0)
         img1Write[id - WIDTH + 1 + WIDTH * HEIGHT] = true;
       if (z + 1 < DEPTH && x - 1 >= 0 && y - 1 >= 0)
         img1Write[id - WIDTH - 1 + WIDTH * HEIGHT] = true;
       if (z + 1 < DEPTH && x + 1 < WIDTH && y + 1 < HEIGHT)
         img1Write[id + WIDTH + 1 + WIDTH * HEIGHT] = true;
       if (z + 1 < DEPTH && x - 1 >= 0 && y + 1 < HEIGHT)
         img1Write[id + WIDTH - 1 + WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x + 1 < WIDTH && y - 1 >= 0)
         img1Write[id - WIDTH + 1 - WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x - 1 >= 0 && y - 1 >= 0)
         img1Write[id - WIDTH - 1 - WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x + 1 < WIDTH && y + 1 < HEIGHT)
         img1Write[id + WIDTH + 1 - WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x - 1 >= 0 && y + 1 < HEIGHT)
         img1Write[id + WIDTH - 1 - WIDTH * HEIGHT] = true;
 #endif
 #endif
     }
 
     if (img2Read[id]) {
       if (x + 1 < WIDTH)
         img2Write[id + 1] = true;
       if (x - 1 >= 0)
         img2Write[id - 1] = true;
       if (y + 1 < HEIGHT)
         img2Write[id + WIDTH] = true;
       if (y - 1 >= 0)
         img2Write[id - WIDTH] = true;
 #if IS_3D // if working with 3d volumes, then the 3D part
       if (z + 1 < DEPTH)
         img2Write[id + WIDTH * HEIGHT] = true;
       if (z - 1 >= 0)
         img2Write[id - WIDTH * HEIGHT] = true;
 #endif
 
 #if CHEBYSHEV
       // diagonals
       if (x + 1 < WIDTH && y - 1 >= 0)
         img2Write[id - WIDTH + 1] = true;
       if (x - 1 >= 0 && y - 1 >= 0)
         img2Write[id - WIDTH - 1] = true;
       if (x + 1 < WIDTH && y + 1 < HEIGHT)
         img2Write[id + WIDTH + 1] = true;
       if (x - 1 >= 0 && y + 1 < HEIGHT)
         img2Write[id + WIDTH - 1] = true;
 #if IS_3D // if working with 3d volumes, then the 3D part
       if (z + 1 < DEPTH && x + 1 < WIDTH && y - 1 >= 0)
         img2Write[id - WIDTH + 1 + WIDTH * HEIGHT] = true;
       if (z + 1 < DEPTH && x - 1 >= 0 && y - 1 >= 0)
         img2Write[id - WIDTH - 1 + WIDTH * HEIGHT] = true;
       if (z + 1 < DEPTH && x + 1 < WIDTH && y + 1 < HEIGHT)
         img2Write[id + WIDTH + 1 + WIDTH * HEIGHT] = true;
       if (z + 1 < DEPTH && x - 1 >= 0 && y + 1 < HEIGHT)
         img2Write[id + WIDTH - 1 + WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x + 1 < WIDTH && y - 1 >= 0)
         img2Write[id - WIDTH + 1 - WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x - 1 >= 0 && y - 1 >= 0)
         img2Write[id - WIDTH - 1 - WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x + 1 < WIDTH && y + 1 < HEIGHT)
         img2Write[id + WIDTH + 1 - WIDTH * HEIGHT] = true;
       if (z - 1 >= 0 && x - 1 >= 0 && y + 1 < HEIGHT)
         img2Write[id + WIDTH - 1 - WIDTH * HEIGHT] = true;
 #endif
 #endif
     }
 
     // this is an atomic and computed to the finished global variable, if image 1 contains all of image 2 and image 2
     // contains all pixels of image 1 then finished is true
     atomicAnd(&finished, (img2Read[id] || !IMG1[id]) && (img1Read[id] || !IMG2[id]));
   }
 }
 
 #pragma once
 int HausdorffDistance::computeDistance(Volume* img1, Volume* img2, bool* d_img1, bool* d_img2) {
   const int height = (*img1).getHeight(), width = (*img1).getWidth(), depth = (*img1).getDepth();
 
   size_t size = width * height * depth * sizeof(bool);
 
   // getting details of your CUDA device
   cudaDeviceProp props;
   cudaGetDeviceProperties(
       &props, CUDA_DEVICE_INDEX); // device index = 0, you can change it if you have more CUDA devices
   const int threadsPerBlock = props.maxThreadsPerBlock / 2;
   const int blocksPerGrid = (height * width * depth + threadsPerBlock - 1) / threadsPerBlock;
 
   // copying the dimensions to the GPU
   cudaMemcpyToSymbolAsync(WIDTH, &width, sizeof(width), 0);
   cudaMemcpyToSymbolAsync(HEIGHT, &height, sizeof(height), 0);
   cudaMemcpyToSymbolAsync(DEPTH, &depth, sizeof(depth), 0);
 
   // allocating the input images on the GPU
 
   // allocating the images that will be the processing ones
   bool *d_img1Write, *d_img1Read, *d_img2Write, *d_img2Read;
   cudaMalloc(&d_img1Write, size);
   cudaMalloc(&d_img1Read, size);
   cudaMalloc(&d_img2Write, size);
   cudaMalloc(&d_img2Read, size);
 
   // cloning the input images to these two image versions (write and read)
   cudaMemcpyAsync(d_img1Read, d_img1, size, cudaMemcpyDeviceToDevice);
   cudaMemcpyAsync(d_img2Read, d_img2, size, cudaMemcpyDeviceToDevice);
   cudaMemcpyAsync(d_img1Write, d_img1, size, cudaMemcpyDeviceToDevice);
   cudaMemcpyAsync(d_img2Write, d_img2, size, cudaMemcpyDeviceToDevice);
 
   // required variables to compute the distance
   int h_finished = false, t = true;
   int distance = -1;
 
   // where the magic happens
   while (!h_finished) {
     // reset the bool variable that verifies if the processing ended
     cudaMemcpyToSymbol(finished, &t, sizeof(h_finished));
 
     // lauching the verify kernel, which verifies if the processing finished
     dilate<<<blocksPerGrid, threadsPerBlock>>>(d_img1, d_img2, d_img1Read, d_img2Read, d_img1Write, d_img2Write);
 
     // cudaDeviceSynchronize();
 
     // updating the imgRead (cloning imgWrite to imgRead)
     cudaMemcpy(d_img1Read, d_img1Write, size, cudaMemcpyDeviceToDevice);
     cudaMemcpy(d_img2Read, d_img2Write, size, cudaMemcpyDeviceToDevice);
 
     // copying the result back to host memory
     cudaMemcpyFromSymbol(&h_finished, finished, sizeof(h_finished));
 
     // incrementing the distance at each iteration
     distance++;
   }
 
   // freeing memory
   cudaFree(d_img1);
   cudaFree(d_img2);
   cudaFree(d_img1Write);
   cudaFree(d_img1Read);
   cudaFree(d_img2Write);
   cudaFree(d_img2Read);
 
   // resetting device
   // cudaDeviceReset();
 
   // print(cudaGetLastError(), "processing CUDA. Something may be wrong with your CUDA device.");
 
   return distance;
 }
 #pragma once
 inline void HausdorffDistance::print(cudaError_t error, char* msg) {
   if (error != cudaSuccess) {
     printf("Error on %s ", msg);
     fprintf(stderr, "Error code: %s!\n", cudaGetErrorString(error));
     exit(EXIT_FAILURE);
   }
 }
 
 /*
 benchmark for original code from  https://github.com/Oyatsumi/HausdorffDistanceComparison
 */
 std::tuple<int, double> benchmarkOlivieraCUDA(
     torch::Tensor goldStandardA,
     torch::Tensor algoOutputA,
     int WIDTH,
     int HEIGHT,
     int DEPTH) {
   // just originally it started for cpu so ...
 
   int lenn = WIDTH * HEIGHT * DEPTH;
   size_t sizee = sizeof(bool) * lenn;
 
   bool* goldStandard = (bool*)calloc(lenn, sizeof(bool));
   bool* algoOutput = (bool*)calloc(lenn, sizeof(bool));
 
   cudaMemcpy(goldStandard, goldStandardA.data_ptr(), sizee, cudaMemcpyDeviceToHost);
   cudaMemcpy(algoOutput, algoOutputA.data_ptr(), sizee, cudaMemcpyDeviceToHost);
 
   // auto goldStandardA.data_ptr()
 
   // bool* goldStandard = (bool*)goldStandardA.to(torch::kCPU).data_ptr();
   // bool* algoOutput = (bool*)algoOutputA.to(torch::kCPU).data_ptr();
 
   Volume img1 = Volume(WIDTH, HEIGHT, DEPTH), img2 = Volume(WIDTH, HEIGHT, DEPTH);
 
   for (int x = 0; x < WIDTH; x++) {
     for (int y = 0; y < HEIGHT; y++) {
       for (int z = 0; z < DEPTH; z++) {
         img1.setVoxelValue(algoOutput[x + y * WIDTH + z * WIDTH * HEIGHT], x, y, z);
         img2.setVoxelValue(goldStandard[x + y * WIDTH + z * WIDTH * HEIGHT], x, y, z);
       }
     }
   }
 
   size_t size = WIDTH * HEIGHT * DEPTH * sizeof(bool);
 
   bool *d_img1, *d_img2;
   cudaMalloc(&d_img1, size);
   cudaMalloc(&d_img2, size);
 
   // copying the data to the allocated memory on the GPU
   cudaMemcpyAsync(d_img1, (img1).getVolume(), size, cudaMemcpyHostToDevice);
   cudaMemcpyAsync(d_img2, (img2).getVolume(), size, cudaMemcpyHostToDevice);
 
   auto begin = std::chrono::high_resolution_clock::now();
   HausdorffDistance* hd = new HausdorffDistance();
 
   cudaDeviceSynchronize();
 
   int dist = (*hd).computeDistance(&img1, &img2, d_img1, d_img2);
   cudaDeviceSynchronize();
 
   auto end = std::chrono::high_resolution_clock::now();
 
   // std::cout << "Total elapsed time: ";
   double time =
       (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / (double)1000000000);
   // std::cout << time << "s" << std::endl;
 
   // printf("HD: %d \n", dist);
 
   // freeing memory
   img1.dispose();
   img2.dispose();
   free(goldStandard);
   free(algoOutput);
   // Datasize: 216530944
   // Datasize : 216530944
   // Total elapsed time : 2.62191s
   // HD : 234
   return {dist, time};
 }