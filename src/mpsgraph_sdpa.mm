#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mpsgraph_sdpa.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace moge {

struct RuntimeMpsSdpa {
    int64_t H=0,N=0,D=0;
    size_t elems=0, bytes=0;
    bool bound=false;
    id<MTLDevice> dev=nil;
    id<MTLCommandQueue> queue=nil;
    id<MTLBuffer> qb=nil,kb=nil,vb=nil,ob=nil;
    MPSGraph * graph=nil;
    MPSGraphExecutable * exec=nil;
    MPSGraphTensor * qt=nil,*kt=nil,*vt=nil,*ot=nil;
    MPSGraphTensorData * qd=nil,*kd=nil,*vd=nil,*od=nil;
    NSDictionary<MPSGraphTensor*,MPSGraphTensorData*> * feeds=nil;
    NSDictionary<MPSGraphTensor*,MPSGraphTensorData*> * results=nil;
    NSArray<MPSGraphTensorData*> * exec_inputs=nil,*exec_results=nil;
};

static void sync_queue(id<MTLCommandQueue> q) {
    id<MTLCommandBuffer> cb=[q commandBuffer];
    if(!cb) throw std::runtime_error("MPSGraph sync command buffer failed");
    [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError) throw std::runtime_error("MPSGraph command buffer failed");
}

bool runtime_mps_sdpa_supported() {
    if (@available(macOS 15.0, *)) return MTLCreateSystemDefaultDevice() != nil;
    return false;
}

static RuntimeMpsSdpa * create_common(int64_t H,int64_t N,int64_t D,float scale,
                                      id<MTLBuffer> qb,id<MTLBuffer> kb,id<MTLBuffer> vb,id<MTLBuffer> ob,
                                      bool bound) {
    if (@available(macOS 15.0, *)) {} else throw std::runtime_error("MPSGraph fused SDPA requires macOS 15 or newer");
    if(H<=0||N<=0||D<=0) throw std::runtime_error("invalid MPSGraph SDPA shape");
    auto * c=new RuntimeMpsSdpa(); c->H=H;c->N=N;c->D=D;c->elems=size_t(H*N*D);c->bytes=c->elems*sizeof(uint16_t);c->bound=bound;
    c->dev=MTLCreateSystemDefaultDevice(); if(!c->dev){delete c;throw std::runtime_error("no Metal device for MPSGraph SDPA");}
    c->queue=[c->dev newCommandQueue]; if(!c->queue){runtime_mps_sdpa_free(c);throw std::runtime_error("cannot create MPSGraph command queue");}
    if(bound) {
        c->qb=[qb retain]; c->kb=[kb retain]; c->vb=[vb retain]; c->ob=[ob retain];
    } else {
        c->qb=[c->dev newBufferWithLength:c->bytes options:MTLResourceStorageModeShared];
        c->kb=[c->dev newBufferWithLength:c->bytes options:MTLResourceStorageModeShared];
        c->vb=[c->dev newBufferWithLength:c->bytes options:MTLResourceStorageModeShared];
        c->ob=[c->dev newBufferWithLength:c->bytes options:MTLResourceStorageModeShared];
    }
    if(!c->qb||!c->kb||!c->vb||!c->ob){runtime_mps_sdpa_free(c);throw std::runtime_error("MPSGraph SDPA buffer allocation failed");}
    NSArray<NSNumber*> *inShape=@[@1,@(H),@(N),@(D)];
    NSArray<NSNumber*> *outShape=@[@1,@(N),@(H),@(D)];
    c->graph=[[MPSGraph alloc] init];
    c->qt=[c->graph placeholderWithShape:inShape dataType:MPSDataTypeFloat16 name:@"q"];
    c->kt=[c->graph placeholderWithShape:inShape dataType:MPSDataTypeFloat16 name:@"k"];
    c->vt=[c->graph placeholderWithShape:inShape dataType:MPSDataTypeFloat16 name:@"v"];
    MPSGraphTensor * raw=[c->graph scaledDotProductAttentionWithQueryTensor:c->qt keyTensor:c->kt valueTensor:c->vt scale:scale name:@"sdpa"];
    // MPSGraph SDPA returns B,H,N,D while ggml's projection consumes N,H,D.
    // Put the transpose on-GPU and write directly into the ggml-bound output buffer.
    c->ot=[c->graph transposeTensor:raw dimension:1 withDimension:2 name:@"sdpa_nhd"];
    c->qd=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->qb shape:inShape dataType:MPSDataTypeFloat16];
    c->kd=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->kb shape:inShape dataType:MPSDataTypeFloat16];
    c->vd=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->vb shape:inShape dataType:MPSDataTypeFloat16];
    c->od=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->ob shape:outShape dataType:MPSDataTypeFloat16];
    if(!c->qd||!c->kd||!c->vd||!c->od){runtime_mps_sdpa_free(c);throw std::runtime_error("MPSGraph SDPA tensor-data creation failed");}
    c->feeds=[@{c->qt:c->qd,c->kt:c->kd,c->vt:c->vd} retain];
    c->results=[@{c->ot:c->od} retain];

    // Compile the fused attention graph explicitly for Metal and disable
    // MPSGraph's level-1 cross-hardware placement pass.  Level 1 may probe
    // ANE candidates and print compatibility diagnostics even though the
    // graph ultimately runs on the caller's Metal queue.
    MPSGraphDevice * gd=[MPSGraphDevice deviceWithMTLDevice:c->dev];
    MPSGraphShapedType * qst=[[[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16] autorelease];
    MPSGraphShapedType * kst=[[[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16] autorelease];
    MPSGraphShapedType * vst=[[[MPSGraphShapedType alloc] initWithShape:inShape dataType:MPSDataTypeFloat16] autorelease];
    MPSGraphCompilationDescriptor * cd=[[[MPSGraphCompilationDescriptor alloc] init] autorelease];
    cd.optimizationLevel=MPSGraphOptimizationLevel0;
    cd.waitForCompilationCompletion=YES;
    c->exec=[[c->graph compileWithDevice:gd
                                  feeds:@{c->qt:qst,c->kt:kst,c->vt:vst}
                          targetTensors:@[c->ot]
                       targetOperations:nil
                  compilationDescriptor:cd] retain];
    if(!c->exec){runtime_mps_sdpa_free(c);throw std::runtime_error("MPSGraph Metal SDPA compile failed");}
    c->exec_inputs=[@[c->qd,c->kd,c->vd] retain];
    c->exec_results=[@[c->od] retain];
    return c;
}

RuntimeMpsSdpa * runtime_mps_sdpa_create(int64_t H,int64_t N,int64_t D,float scale) {
    @autoreleasepool { return create_common(H,N,D,scale,nil,nil,nil,nil,false); }
}

RuntimeMpsSdpa * runtime_mps_sdpa_create_bound(int64_t H,int64_t N,int64_t D,float scale,
                                               void * q_mtl,void * k_mtl,void * v_mtl,void * out_mtl) {
    if(!q_mtl||!k_mtl||!v_mtl||!out_mtl) throw std::runtime_error("null bound MPSGraph Metal buffer");
    @autoreleasepool {
        return create_common(H,N,D,scale,(id<MTLBuffer>)q_mtl,(id<MTLBuffer>)k_mtl,(id<MTLBuffer>)v_mtl,(id<MTLBuffer>)out_mtl,true);
    }
}

void runtime_mps_sdpa_free(RuntimeMpsSdpa * c) {
    if(!c) return;
    @autoreleasepool {
        [c->exec_results release];[c->exec_inputs release];
        [c->results release];[c->feeds release];
        [c->od release];[c->vd release];[c->kd release];[c->qd release];
        [c->exec release];[c->graph release];
        [c->ob release];[c->vb release];[c->kb release];[c->qb release];
        [c->queue release];[c->dev release];
    }
    delete c;
}

static double run_graph(RuntimeMpsSdpa * c) {
    const auto t0=std::chrono::steady_clock::now();
    [c->exec runWithMTLCommandQueue:c->queue
                          inputsArray:c->exec_inputs
                         resultsArray:c->exec_results
                  executionDescriptor:nil];
    sync_queue(c->queue);
    const auto t1=std::chrono::steady_clock::now();
    return std::chrono::duration<double,std::milli>(t1-t0).count();
}

double runtime_mps_sdpa_run_f16(RuntimeMpsSdpa * c,const uint16_t *q,const uint16_t*k,const uint16_t*v,uint16_t*out) {
    if(!c||!q||!k||!v||!out) throw std::runtime_error("null MPSGraph SDPA argument");
    if(c->bound) throw std::runtime_error("runtime_mps_sdpa_run_f16 called on bound context");
    @autoreleasepool {
        const auto t0=std::chrono::steady_clock::now();
        std::memcpy(c->qb.contents,q,c->bytes); std::memcpy(c->kb.contents,k,c->bytes); std::memcpy(c->vb.contents,v,c->bytes);
        [c->exec runWithMTLCommandQueue:c->queue
                          inputsArray:c->exec_inputs
                         resultsArray:c->exec_results
                  executionDescriptor:nil];
        sync_queue(c->queue);
        // The graph already transposes into N,H,D in ob.
        std::memcpy(out,c->ob.contents,c->bytes);
        const auto t1=std::chrono::steady_clock::now();
        return std::chrono::duration<double,std::milli>(t1-t0).count();
    }
}

double runtime_mps_sdpa_run_bound(RuntimeMpsSdpa * c) {
    if(!c||!c->bound) throw std::runtime_error("runtime_mps_sdpa_run_bound requires bound context");
    @autoreleasepool { return run_graph(c); }
}

} // namespace moge
