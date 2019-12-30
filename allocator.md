使用GPU训练时，一次训练任务无论是模型参数还是中间结果都需要占用大量显存。为了避免每次训练重新开辟显存带来计算之外的开销，一般框架的做法是在真正的训练任务开始前，将每个节点的输入和输出，以及模型参数的shape计算出来并全局开辟一次，例如Caffe就是这种做法。随着深度学习模型的发展和迭代，不仅模型训练的数据shape可能发生变化，就连模型本身在训练过程中也可能发生变化，那么按照固定shape一次开辟显存的做法就不能满足需求了。为此，TensorFlow重新设计了较为灵活的显存管理机制，它使用了名为BFC的分配算法，并通过BFC Allocator为每个Tensor分配满足需求的显存。

### Tensor内存分配时机

在Tensor的构造过程函数中

```c++
Tensor::Tensor(Allocator* a, DataType type, const TensorShape& shape) : shape_(shape), buf_(nullptr){
  /***/
  if (shape_.num_elements() > 0 || a->AllocatesOpaqueHandle()) {
    CASES(type, buf_ = new Buffer<T>(a, shape.num_elements()));
  }
  /***/
}
```



该代码说明了tensor的内存分配是在构造过程中完成，在训练过程中不断的有tensor被释放，tensor被创建，这样频繁的分配和回收存储区，会对训练速度造成非常大的影响，因此tensorflow使用了存储池的方法解决该问题，每次调用allocate时直接从内存池中取出内存，调用Deallocate则将内存释放至内存池中。为此需要设计一个较复杂的内存管理器，tensorflow中对gpu显存的管理就称为BFC Allocator。

### 内存管理器

核心思想是将存储区划分为块，挂载到存储池中进行管理，将存储区划分为存储块时需要满足以下要求1.块内地址是连续地址2.存储区中的块要以每个块地址升序排列，并组成双向链表3.高地址的块的大小要大于低地址的块的大小。

###### 数据结构:块chunk

chunk指向一块要么被完全使用，要么被完全释放的内存，基本数据结构定义在tensorflow/core/common_runtime/bfc_allocator.h中

```c++
struct Chunk {
  size_t size = 0; // buffer的大小
  size_t requested_size = 0; //   client请求的内存大小
  int64 allocation_id = -1; // 未被使用的chunk该值为-1，使用时会被allocator赋一个唯一值
  void* ptr = nullptr; // 指向分配的subbuffer的指针
  ChunkHandle prev = kInvalidChunkHandle; // prev指向该块使用内存的前一块内存，地址为ptr-prev->size
  ChunkHandle next = kInvalidChunkHandle; // next指向该块使用内存的后一块内存，地址为ptr+size
  BinNum bin_num = kInvaludBinNum; // 该块处于bin的序号
  /***/
}
```



###### 数据结构:bin

该数据结构是为了更好的对块进行索引，否则搜索满足条件的内存块，只能通过遍历双向链表，效率比较低，在创建chunk链表时，按照一定的顺序进行排列，将整个有序链表在逻辑上分为多个段，为每个段记录所包含的chunk的范围，这种数据结构就是bin，类似于统计直方图中的横坐标。每个bin都设有自己的bin_size，该bin_size表示该段包含的最小的chunk的size大小，bin_size = 256*2^bin_num(bin的索引号)。这样用户端就可以直接根据申请的内存大小找到相应的bin，然后在bin中遍历寻找合适的chunk。

bin的结构定义在tensorflow/core/common_runtime/bfc_allocator.h中，定义中表示bin是相似大小的空闲chunks的集合，被分配过的内存绝不会出现在bin中。

```c++
struct Bin {
  // 该bin中所有的chunks >= bin_size
  size_t bin_size = 0;
  
  // 比较chunk的大小，相同大小的比地址高低，由小到大
  class ChunkComparator {
    /***/
  }
  typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
  // 该bin中按照chunk size排序的空闲chunks链表
  FreeChunkSet free_chunks;
}
```



###### 辅助类

BFC Allocator每次分配存储区时都以Chunk为单位，指向Chunk的指针又是ChunkHandle类型（实际为数组下标），但分配存储的最终目的是把Chunk中指向存储区域的头指针ptr分配给请求方。另外，当系统回收存储区时，面对的也是存储区的头指针，那么如果不能根据头指针找到Chunk和Bin信息，回收就不能成功。因此这里显然应该设计一系列接口和函数：它能够记录每次分配的Chunk，并且能够保存分配存储区的地址ptr与Chunk之间的映射关系。AllocationRegion和RegionManager就是完成这些功能的接口。

具体而言，AllocationRegion对应一次存储区分配的记录。一次存储区分配的信息包括起始地址ptr和存储区大小memory_size，这可能包括多个Chunk，所以该结构要记录此次分配中所包含所有Chunk的信息。RegionManager是AllocationRegion的管理器，它维护了AllocationRegion的数组。在RegionManager中，AllocationRegion数组是需要按照end_ptr地址排序的。

AllocationRegion定义在tensorflow/core/common_runtime/bfc_allocator.h。

```c++
class AllocationRegion {
  /***/
  private:
 		//	申请region的元数据 	
  	void* ptr = nullptr; // 起始地址
  	size_t memory_size = 0; // 内存大小
  	void* end_ptr_ = nullptr; // 终止地址
  
  	// 数组大小为 memory_size/256，通过(p-base)/256进行索引，p为申请内存地址
  	 std::unique_ptr<ChunkHandle[]> handles_;
}
```



RegionManager定义在tensorflow/core/common_runtime/bfc_allocator.h

```c++
class RegionManager {
  public:
  	void AddAllocationRegion(void* ptr, size_t memory_size) {
      /***/
    }
  	std::vector<AllocationRegion>::iterator RemoveAllocationRegion(std::vector<AllocationRegion>::iterator it) {
      /***/
    }
  	ChunkHandle get_handle(const void* p) const {/***/}
  	void set_handle(const void* p, ChunkHandle h) {/***/}
  private:
  	std::vector<AllocationRegion> regions_;
}
```



### Allocate流程

内存申请的基本流程定义在tensorflow/core/common_runtime/bfc_allocator.cc中函数AllocateRawInternal中

```c++
void* BFCAllocator::AllocateRawInternal(/***/size_t num_bytes/***/) {
  /***/
  // 根据用户请求的内存大小做调整，使其为256*（（num_bytes + 255）/256）
  size_t rounded_bytes = RoundedBytes(num_bytes);
  
  // 计算该大小在内存中bin的索引号
  BinNum bin_num = BinNumForSize(rounded_bytes);
  
  // 在bin中找到首个符合要求的chunk，在选取chunk的过程中可能出现请求的size比所选择的chunk的size小很多的情况
  // 这时需要使用SplitChunk将chunk进行分割，用以防止过多内存未被使用。
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
  if (ptr != nullptr) {
    return ptr;
  }
  
  // 如果在已有的chunks中找不到合适的chunk,需要调用extend过程，说明现有的存储池中已经没有可以满足的存储区了
  // 需要向物理设备进行申请，创建新的chunks，然后放入bin中。
  if (Extend(unused_aligment, round_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, freed_before);
    if (ptr != nullptr) {
      return ptr;
    }
  }
 	/***/
}
```



Deallocate流程

内存释放定义在tensorflow/core/common_runtime/bfc_allocator.cc中

```c++
void BFCAllocator::DeallocateRawInternal(void* ptr) {
  /***/
  // 释放的时候只知道存储空间的首地址，通过辅助类region_manager_获取对应的chunk指针
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  
  // 将该chunk标记为已释放
  MarkFree(h);
  
  // 将空余的chunk插入到bin中
  InsertFreeChunkIntoBin(h);
  /***/
  
}
```

