// Copyright (c) 2026 Kenji8965
// SPDX-License-Identifier: MIT
// Programming and implementation assistance: ChatGPT 5.6 Sol (OpenAI)
// Human testing, validation, and release acceptance: Kenji8965
// Provided "AS IS"; use at your own risk. See LICENSE.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "mpsgraph_conv.hpp"
#include <ggml.h>
#include <ggml-metal.h>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace moge {

struct RuntimeMpsConv {
    int64_t W=0,R=0,Cin=0,Cout=0;
    id<MTLDevice> dev=nil;
    id<MTLCommandQueue> queue=nil;
    id<MTLBuffer> ib=nil, wb=nil, ob=nil;
    MPSGraph * graph=nil;
    MPSGraphExecutable * exec=nil;
    MPSGraphTensor * it=nil,*wt=nil,*ot=nil;
    MPSGraphTensorData * idata=nil,*wdata=nil,*odata=nil;
    NSDictionary<MPSGraphTensor*,MPSGraphTensorData*> * feeds=nil,*results=nil;
    NSArray<MPSGraphTensorData*> * exec_inputs=nil,*exec_results=nil;
};

static void sync_queue(id<MTLCommandQueue> q) {
    id<MTLCommandBuffer> cb=[q commandBuffer];
    if(!cb) throw std::runtime_error("MPSGraph conv sync command buffer failed");
    [cb commit]; [cb waitUntilCompleted];
    if(cb.status==MTLCommandBufferStatusError) throw std::runtime_error("MPSGraph conv command buffer failed");
}

bool runtime_mps_conv_supported() {
    if (@available(macOS 15.0, *)) return MTLCreateSystemDefaultDevice() != nil;
    return false;
}

void runtime_mps_conv_free(RuntimeMpsConv * c) {
    if(!c) return;
    @autoreleasepool {
        [c->exec_results release]; [c->exec_inputs release];
        [c->results release]; [c->feeds release];
        [c->odata release]; [c->wdata release]; [c->idata release];
        [c->exec release]; [c->graph release];
        [c->ob release]; [c->wb release]; [c->ib release];
        [c->queue release]; [c->dev release];
    }
    delete c;
}

RuntimeMpsConv * runtime_mps_conv_create_bound(int64_t W,int64_t R,int64_t Cin,int64_t Cout,
                                                const uint16_t * weight_f16,
                                                ggml_tensor * input,ggml_tensor * output) {
    if (@available(macOS 15.0, *)) {} else throw std::runtime_error("MPSGraph convolution requires macOS 15+");
    if(W<=0||R<=0||Cin<=0||Cout<=0||!weight_f16||!input||!output) throw std::runtime_error("invalid bound MPSGraph conv arguments");
    if(input->type!=GGML_TYPE_F32||output->type!=GGML_TYPE_F32) throw std::runtime_error("bound MPSGraph conv requires F32 input/output tensors");
    if(input->ne[0]!=W+2||input->ne[1]!=R+2||input->ne[2]!=Cin||output->ne[0]!=W||output->ne[1]!=R||output->ne[2]!=Cout)
        throw std::runtime_error("bound MPSGraph conv tensor shape mismatch");
    size_t io=0,oo=0;
    void * imp=ggml_backend_metal_get_tensor_mtlbuffer(input,&io);
    void * omp=ggml_backend_metal_get_tensor_mtlbuffer(output,&oo);
    if(!imp||!omp) throw std::runtime_error("bound MPSGraph conv requires Metal tensors");
    if(io!=0||oo!=0) throw std::runtime_error("bound MPSGraph conv requires zero-offset dedicated Metal buffers");
    auto * c=new RuntimeMpsConv(); c->W=W;c->R=R;c->Cin=Cin;c->Cout=Cout;
    @autoreleasepool {
        c->dev=MTLCreateSystemDefaultDevice(); if(!c->dev){runtime_mps_conv_free(c);throw std::runtime_error("no Metal device for MPSGraph conv");}
        c->queue=[c->dev newCommandQueue]; if(!c->queue){runtime_mps_conv_free(c);throw std::runtime_error("cannot create MPSGraph conv command queue");}
        c->ib=[(id<MTLBuffer>)imp retain]; c->ob=[(id<MTLBuffer>)omp retain];
        const size_t wbytes=size_t(3*3*Cin*Cout)*sizeof(uint16_t);
        c->wb=[c->dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
        if(!c->ib||!c->ob||!c->wb){runtime_mps_conv_free(c);throw std::runtime_error("MPSGraph conv buffer setup failed");}
        std::memcpy(c->wb.contents,weight_f16,wbytes);
        NSArray<NSNumber*> * is=@[@1,@(Cin),@(R+2),@(W+2)];
        NSArray<NSNumber*> * ws=@[@(Cout),@(Cin),@3,@3];
        NSArray<NSNumber*> * os=@[@1,@(Cout),@(R),@(W)];
        c->graph=[[MPSGraph alloc] init];
        c->it=[c->graph placeholderWithShape:is dataType:MPSDataTypeFloat32 name:@"x"];
        c->wt=[c->graph placeholderWithShape:ws dataType:MPSDataTypeFloat16 name:@"w"];
        MPSGraphTensor * xh=[c->graph castTensor:c->it toType:MPSDataTypeFloat16 name:@"x_f16"];
        MPSGraphConvolution2DOpDescriptor * d=[MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor * yh=[c->graph convolution2DWithSourceTensor:xh weightsTensor:c->wt descriptor:d name:@"conv"];
        c->ot=[c->graph castTensor:yh toType:MPSDataTypeFloat32 name:@"y_f32"];
        c->idata=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->ib shape:is dataType:MPSDataTypeFloat32];
        c->wdata=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->wb shape:ws dataType:MPSDataTypeFloat16];
        c->odata=[[MPSGraphTensorData alloc] initWithMTLBuffer:c->ob shape:os dataType:MPSDataTypeFloat32];
        if(!c->idata||!c->wdata||!c->odata){runtime_mps_conv_free(c);throw std::runtime_error("MPSGraph conv tensor-data failed");}
        c->feeds=[@{c->it:c->idata,c->wt:c->wdata} retain];
        c->results=[@{c->ot:c->odata} retain];

        // Compile explicitly for the Metal device rather than leaving device
        // selection to the generic graph compiler.  MPSGraph can internally
        // consider several Apple compute blocks and may emit diagnostics for
        // engines that are not used by this helper. Keep this path explicitly
        // Metal/GPU reference.  Specializing an executable for the MTLDevice
        // gives us an unambiguous GPU-targeted path and keeps execution on the
        // caller's MTLCommandQueue.
        MPSGraphDevice * gd=[MPSGraphDevice deviceWithMTLDevice:c->dev];
        MPSGraphShapedType * ist=[[[MPSGraphShapedType alloc] initWithShape:is dataType:MPSDataTypeFloat32] autorelease];
        MPSGraphShapedType * wst=[[[MPSGraphShapedType alloc] initWithShape:ws dataType:MPSDataTypeFloat16] autorelease];
        MPSGraphCompilationDescriptor * cd=[[[MPSGraphCompilationDescriptor alloc] init] autorelease];
        // Level 1 enables MPSGraph's placement pass, which may probe the ANE
        // even for an executable bound to an MTLDevice.  This helper is a
        // deliberately GPU-only path, so keep compilation to the core Metal
        // optimizations and do not consider other hardware blocks.
        cd.optimizationLevel=MPSGraphOptimizationLevel0;
        cd.waitForCompilationCompletion=YES;
        c->exec=[[c->graph compileWithDevice:gd
                                      feeds:@{c->it:ist,c->wt:wst}
                              targetTensors:@[c->ot]
                           targetOperations:nil
                      compilationDescriptor:cd] retain];
        if(!c->exec){runtime_mps_conv_free(c);throw std::runtime_error("MPSGraph Metal convolution compile failed");}
        c->exec_inputs=[@[c->idata,c->wdata] retain];
        c->exec_results=[@[c->odata] retain];
    }
    return c;
}

double runtime_mps_conv_run_bound(RuntimeMpsConv * c) {
    if(!c) throw std::runtime_error("null MPSGraph conv context");
    @autoreleasepool {
        const auto t0=std::chrono::steady_clock::now();
        [c->exec runWithMTLCommandQueue:c->queue
                            inputsArray:c->exec_inputs
                           resultsArray:c->exec_results
                    executionDescriptor:nil];
        sync_queue(c->queue);
        const auto t1=std::chrono::steady_clock::now();
        return std::chrono::duration<double,std::milli>(t1-t0).count();
    }
}

} // namespace moge
