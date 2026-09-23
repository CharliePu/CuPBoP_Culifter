#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
namespace cpu_coarsening {
struct Dim3 { unsigned x=1,y=1,z=1; };
using Parameters=std::array<unsigned char,5*4096>;
template<class T> void put(Parameters& p,unsigned offset,T value){
  if(offset+sizeof(T)>p.size())throw std::out_of_range("kernel parameter offset");
  std::memcpy(p.data()+offset,&value,sizeof(T));
}
struct Regions {
  float (*sum)(int,const float*)=nullptr;
  void (*affine)(int,float,const float*,float,float*,float)=nullptr;
};
struct Kernel { void (*entry)(); unsigned block_size; };
class Executor {
  struct Impl;std::unique_ptr<Impl> impl;
public:
  explicit Executor(unsigned workers,const std::vector<int>& cpus={});
  ~Executor();
  Executor(const Executor&)=delete;
  void run(Kernel kernel,Dim3 grid,Dim3 block,const Parameters&,Regions regions={});
  unsigned workers()const;
};
}
extern "C" {
extern thread_local unsigned char const_mem[5][4096];
extern thread_local unsigned char shared_mem[32768];
extern thread_local int block_size,block_size_x,block_size_y,block_size_z;
extern thread_local int grid_size_x,grid_size_y,grid_size_z;
extern thread_local int block_index_x,block_index_y,block_index_z;
extern thread_local int intra_warp_index,inter_warp_index;
}
