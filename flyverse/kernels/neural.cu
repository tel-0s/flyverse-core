#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

struct Lif {
    float *v, *g, *drive, *adapt, *refrac, *active, *prob, *rnd;
    float *res, *std_u, *spikes, *out, *rate, *counts, *p, *phase;
};

// p: rest, reset, threshold, a_m, dt, refractory, adapt_jump, a_ad, a_std, a_r, rate_gain
// phase, when present: a_m, dt, a_ad, a_std, a_r, rate_gain, update_mask, poisson_multiplier
__global__ void lif_update(Lif s, int n, int total, int flags) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int i = t % n;
    // Frozen optic cells normally sit at rest. Preserve their full public state,
    // and fall through immediately when stimulation or a restored state needs work.
    if ((flags & 32) && s.active[i]==0 && (!(flags & 1) || s.prob[t]==0) &&
        s.v[t]==s.p[0] && s.g[t]==0 && s.drive[t]==0 && s.adapt[t]==0 &&
        s.refrac[t]==0 && s.rate[t]==0 && s.spikes[t]==0 && s.out[t]==0 && s.res[t]==1) return;
    float am=s.p[3], dt=s.p[4], aad=s.p[7], ast=s.p[8], ar=s.p[9], rg=s.p[10], u=1, pois=1;
    if (flags & 16) {
        am=s.phase[i]; dt=s.phase[n+i]; aad=s.phase[2*n+i]; ast=s.phase[3*n+i];
        ar=s.phase[4*n+i]; rg=s.phase[5*n+i]; u=s.phase[6*n+i]; pois=s.phase[7*n+i];
    }
    float target = s.p[0] + s.g[t] + s.drive[t] - s.adapt[t];
    float v = target + (s.v[t] - target) * am;
    float ref = s.refrac[t];
    if (ref > 0) v = s.p[1];
    ref = ref - dt; if (ref < 0) ref = 0;
    float spike = (v >= s.p[2] ? 1.0f : 0.0f) * u * s.active[i];
    if ((flags & 1) && s.rnd[t] < s.prob[t] * pois) spike = fmaxf(spike,1.0f);
    bool fired = spike > 0;
    if (fired) { v = s.p[1]; ref = s.p[5]; }
    s.v[t] = v; s.refrac[t] = ref; s.spikes[t] = spike;
    // torch's add_(spikes, alpha=jump) may contract this one multiply-add.
    if (flags & 8) s.adapt[t] = fmaf(spike, s.p[6], s.adapt[t] * aad);
    if (flags & 2) {
        float resource = s.res[t];
        s.out[t] = spike * resource;
        if (fired) resource *= 1.0f - s.std_u[i];
        s.res[t] = 1.0f - (1.0f - resource) * ast;
    } else s.out[t] = spike;
    if (flags & 4) s.counts[t] += spike;
    s.rate[t] = s.rate[t] * ar + spike * rg;
}

EXPORT int launch_lif(void** p, int n, int batch, int flags, void* stream) {
    Lif s{(float*)p[0],(float*)p[1],(float*)p[2],(float*)p[3],(float*)p[4],(float*)p[5],
          (float*)p[6],(float*)p[7],(float*)p[8],(float*)p[9],(float*)p[10],(float*)p[11],
          (float*)p[12],(float*)p[13],(float*)p[14],(float*)p[15]};
    if (n*batch) lif_update<<<(n*batch+255)/256,256,0,(cudaStream_t)stream>>>(s,n,n*batch,flags);
    return (int)cudaGetLastError();
}

__device__ float clip01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
__global__ void optic_dr(const float* v, const float* b, float* dr, int n, int total) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<total) { float base=b[t%n]; dr[t]=clip01(v[t]+base)-base; }
}
struct Optic { float *v,*adapt,*dr,*y,*pr,*spk,*a,*b; };
__global__ void optic_update(Optic s,int n,int total,float gr,float ga,float aad) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=total) return;
    int i=t%n;
    float old=s.dr[t];
    float inp=gr*s.y[t]+s.pr[t]-ga*s.adapt[t];
    inp=inp+s.spk[t];
    float v=inp+(s.v[t]-inp)*s.a[i];
    s.v[t]=v;
    s.adapt[t]=old+(s.adapt[t]-old)*aad;
    s.dr[t]=clip01(v+s.b[i])-s.b[i];
}
EXPORT int launch_optic_dr(void** p,int n,int batch,void* stream) {
    if(n*batch) optic_dr<<<(n*batch+255)/256,256,0,(cudaStream_t)stream>>>((float*)p[0],(float*)p[1],(float*)p[2],n,n*batch);
    return (int)cudaGetLastError();
}
EXPORT int launch_optic(void** p,int n,int batch,float gr,float ga,float aad,void* stream) {
    Optic s{(float*)p[0],(float*)p[1],(float*)p[2],(float*)p[3],(float*)p[4],(float*)p[5],(float*)p[6],(float*)p[7]};
    if(n*batch) optic_update<<<(n*batch+255)/256,256,0,(cudaStream_t)stream>>>(s,n,n*batch,gr,ga,aad);
    return (int)cudaGetLastError();
}

// A fixed reduction tree for comparison with cuSPARSE; one warp per output row.
template<typename W> __global__ void csr_product(const int* ptr,const int* idx,const W* w,
        const float* x,float* out,int rows,int cols,int batch) {
    int lane=threadIdx.x&31, gid=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(gid>=rows*batch) return;
    int b=gid/rows,row=gid%rows;
    float sum=0;
    for(int e=ptr[row]+lane;e<ptr[row+1];e+=32) {
        float xv=x[b*cols+idx[e]];
        if(sizeof(W)==2) xv=__half2float(__float2half_rn(xv));
        sum += (float)w[e]*xv;
    }
    for(int k=16;k;k/=2) sum+=__shfl_down_sync(0xffffffff,sum,k);
    if(lane==0) out[gid]=sum;
}
EXPORT int launch_csr(void** p,int rows,int cols,int batch,int half,void* stream) {
    int blocks=(rows*batch+7)/8;
    if(blocks) {
        if(half) csr_product<<<blocks,256,0,(cudaStream_t)stream>>>((int*)p[0],(int*)p[1],(__half*)p[2],(float*)p[3],(float*)p[4],rows,cols,batch);
        else csr_product<<<blocks,256,0,(cudaStream_t)stream>>>((int*)p[0],(int*)p[1],(float*)p[2],(float*)p[3],(float*)p[4],rows,cols,batch);
    }
    return (int)cudaGetLastError();
}

// Device-only sparse activity traversal. Silent presynaptic neurons exit without visiting edges.
template<typename W> __global__ void event_scatter(const int* ptr,const int* idx,const W* w,
        const float* x,float* g,const int* pre_ids,int n,int batch,int count) {
    int lane=threadIdx.x&31,gid=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    int size=count<0 ? n : count;
    if(gid>=size*batch) return;
    int b=gid/size,pre=count<0 ? gid%size : pre_ids[gid%size];
    float value=x[b*n+pre]; if(value==0) return;
    for(int e=ptr[pre]+lane;e<ptr[pre+1];e+=32) atomicAdd(g+b*n+idx[e],(float)w[e]*value);
}
EXPORT int launch_events(void** p,int n,int batch,int half,int count,void* stream) {
    int blocks=(((count<0 ? n : count)*batch+7)/8);
    if(blocks) {
        if(half) event_scatter<<<blocks,256,0,(cudaStream_t)stream>>>((int*)p[0],(int*)p[1],(__half*)p[2],(float*)p[3],(float*)p[4],(int*)p[5],n,batch,count);
        else event_scatter<<<blocks,256,0,(cudaStream_t)stream>>>((int*)p[0],(int*)p[1],(float*)p[2],(float*)p[3],(float*)p[4],(int*)p[5],n,batch,count);
    }
    return (int)cudaGetLastError();
}

// Small motor groups: fixed-order double accumulation matches CPU bincount PN means.
__global__ void group_means(const int* ptr,const int* idx,const float* rates,double* out,int n,int groups,int batch,int stride) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=groups*batch) return;
    int b=t/groups,k=t%groups; double sum=0;
    for(int j=ptr[k];j<ptr[k+1];++j) sum+=(double)rates[b*n+idx[j]];
    int count=ptr[k+1]-ptr[k]; out[b*stride+k]=count ? sum/count : 0;
}
EXPORT int launch_means(void** p,int n,int groups,int batch,int stride,void* stream) {
    if(groups*batch) group_means<<<(groups*batch+127)/128,128,0,(cudaStream_t)stream>>>((int*)p[0],(int*)p[1],(float*)p[2],(double*)p[3],n,groups,batch,stride);
    return (int)cudaGetLastError();
}
