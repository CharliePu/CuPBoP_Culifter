#include <stdexcept>
#include "cpu_runtime.hpp"
#include <atomic>
#include <cmath>
#include "../../CuLifter/test/common/directed_fma.h"
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <sched.h>
extern "C" {
alignas(64) thread_local unsigned char const_mem[5][4096];
alignas(64) thread_local unsigned char shared_mem[32768];
thread_local int block_size,block_size_x,block_size_y,block_size_z;
thread_local int grid_size_x,grid_size_y,grid_size_z;
thread_local int block_index_x,block_index_y,block_index_z;
thread_local int intra_warp_index,inter_warp_index;
thread_local unsigned char* cpu_local_memory;
thread_local unsigned char* cpu_shared_memory;
}
static thread_local std::vector<unsigned char> local_storage;
static thread_local std::vector<unsigned char> shared_storage;
extern "C" void cpu_prepare_shared_memory(uint64_t bytes){
  if(!bytes||bytes>1048576)throw std::runtime_error("shared memory contract exceeded");
  if(shared_storage.size()<bytes)shared_storage.resize(bytes);
  cpu_shared_memory=shared_storage.data();
}
extern "C" void cpu_prepare_local_memory(uint64_t bytes){
  if(bytes>uint64_t(1024)*32768)throw std::runtime_error("local memory contract exceeded");
  if(local_storage.size()<bytes)local_storage.resize(bytes);
  std::fill(local_storage.begin(),local_storage.begin()+bytes,0);
  cpu_local_memory=local_storage.data();
}
static thread_local cpu_coarsening::Regions region_ops;
// Preserve the source wrapper's scalar helper behavior, including RCP bias.
extern "C" float rsqrt_f32(float x){return 1.f/std::sqrt(x);}
extern "C" float rcp_f32(float x){
  float r=1.f/x;uint32_t xb,b;std::memcpy(&xb,&x,4);
  if((xb&0x007fffffu)==0)return r;
  std::memcpy(&b,&r,4);if(x>0)b+=3;else if(x<0)b-=3;std::memcpy(&r,&b,4);return r;
}
extern "C" float cpu_region_sasum(int n,const float* x){
  if(!region_ops.sum)throw std::runtime_error("No lower-level reduction provider registered");
  return region_ops.sum(n,x);
}
extern "C" float cpu_region_partner(const float* x,int lane){
  float half[16];int parity=(lane^1)&1;for(int i=0;i<16;i++)half[i]=x[2*i+parity];
  for(int gap=8;gap;gap/=2)for(int i=0;i<gap;i++)half[i]+=half[i+gap];return half[0];
}
extern "C" void region_affine(uint64_t xa,uint64_t ya,int offset,int lane,int length,int stride,
    float mean,float gamma,float shift,float rstd,float alpha,float beta){
  if(lane)return;
  if(!region_ops.affine)throw std::runtime_error("No lower-level affine provider registered");
  float a=alpha*rstd*gamma,b=alpha*(shift-rstd*gamma*mean);
  region_ops.affine(length,a,reinterpret_cast<const float*>(xa)+offset,beta,reinterpret_cast<float*>(ya)+offset,b);
}
namespace cpu_coarsening {
struct Executor::Impl {
  unsigned count;std::vector<std::thread> pool;std::vector<int> cpus;
  std::mutex mutex,launch_mutex;std::condition_variable start,done;
  unsigned generation=0,remaining=0;bool stop=false;
  Kernel kernel{};Dim3 grid,block;Parameters params{};Regions regions;
  std::exception_ptr error;
  explicit Impl(unsigned n,const std::vector<int>& affinity):count(n),cpus(affinity){
    if(!n||(!cpus.empty()&&cpus.size()!=n))throw std::invalid_argument("invalid worker/affinity count");
    for(int cpu:cpus)if(cpu<0||cpu>=CPU_SETSIZE)throw std::invalid_argument("invalid CPU affinity id");
    try{for(unsigned w=0;w<n;w++)pool.emplace_back([this,w]{worker(w);});}
    catch(...){ {std::lock_guard<std::mutex> guard(mutex);stop=true;}start.notify_all();for(auto& t:pool)t.join();throw; }
  }
  ~Impl(){ {std::lock_guard<std::mutex> guard(mutex);stop=true;}start.notify_all();for(auto& t:pool)t.join(); }
  void worker(unsigned w){
    unsigned seen=0;
    while(true){
      std::unique_lock<std::mutex> lock(mutex);start.wait(lock,[&]{return stop||generation!=seen;});
      if(stop)return;seen=generation;lock.unlock();
      try{
        if(!cpus.empty()){
          cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpus[w],&mask);
          if(sched_setaffinity(0,sizeof(mask),&mask))throw std::runtime_error("worker affinity failed");
        }
        std::memcpy(const_mem,params.data(),params.size());region_ops=regions;
        block_size_x=block.x;block_size_y=block.y;block_size_z=block.z;block_size=block.x*block.y*block.z;
        grid_size_x=grid.x;grid_size_y=grid.y;grid_size_z=grid.z;
        uint64_t total=uint64_t(grid.x)*grid.y*grid.z;
        for(uint64_t linear=w;linear<total;linear+=count){
          block_index_x=linear%grid.x;block_index_y=(linear/grid.x)%grid.y;block_index_z=linear/(uint64_t(grid.x)*grid.y);
          intra_warp_index=inter_warp_index=0;
          kernel.entry();
        }
      }catch(...){std::lock_guard<std::mutex> guard(mutex);if(!error)error=std::current_exception();}
      lock.lock();if(--remaining==0)done.notify_one();
    }
  }
};
Executor::Executor(unsigned workers,const std::vector<int>& cpus):impl(new Impl(workers,cpus)){}
Executor::~Executor()=default;
unsigned Executor::workers()const{return impl->count;}
void Executor::run(Kernel kernel,Dim3 grid,Dim3 block,const Parameters& params,Regions regions){
  if(!kernel.entry||!grid.x||!grid.y||!grid.z||!block.x||!block.y||!block.z||
     uint64_t(block.x)*block.y*block.z!=kernel.block_size||kernel.block_size>1024)
    throw std::invalid_argument("launch does not match coarsened kernel contract");
  auto& s=*impl;std::lock_guard<std::mutex> serialize(s.launch_mutex);
  std::unique_lock<std::mutex> lock(s.mutex);
  s.kernel=kernel;s.grid=grid;s.block=block;s.params=params;s.regions=regions;s.error=nullptr;
  s.remaining=s.count;++s.generation;s.start.notify_all();s.done.wait(lock,[&]{return s.remaining==0;});
  if(s.error)std::rethrow_exception(s.error);
}
}
