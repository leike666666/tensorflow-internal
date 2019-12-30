### tensor的定义

tensorflow将高维数据统一定义的类型称为tensor。class tensor在源码中定义的位置在tensorflow/core/framework/tensor.h，该类的部分成员函数和成员变量如下

```c++
class Tensor {
  // ...
  public:
  	template <typename T>
  	typename TTypes<T>::Vec vec() {
      return tensor<T,1>();
    }
  
  	template <typename T>
  	typename TTypes<T>::Matrix matrix() {
      return tensor<T,2>();
    }
  
  	template <typename T, size_t NDIMS>
  	typename TTypes<T, NDIMS>::Tensor tensor();
  	
  
  private:
  	TensorShape shape_; // 表示tensor的shape信息
  	TensorBuffer* buffer_; // 表示tensor的底层数据信息
}
```

　

### 成员变量TensorBuffer

成员变量buffer_只是一个指向底层数据的指针，也就意味着tensor并不实际持有底层数据，实际只是对底层数据的一种试图。同样的一份底层数据，可以有多个视图，例如对于一个长度为12底层数组，可以是一个size为12的向量，也可以是shape为[2,6]的矩阵，也可以是一个[2,3,2]的张量，通过这种方式可以对同一份底层数据进行复用。避免重新申请内存空间。其中TensorBuffer的定义为

```c++
class TensorBuffer : public core::RefCounted {
  // ***
}
```



该类是一个继承自引用计数类的虚拟接口，不包含任何实现，具体实现功能类的代码如下

```c++
class BufferBase : public TensorBuffer {
  // ***
  protected:
  	Allocator* const alloc_; //内存分配器指针
}

class Buffer : public BufferBase {
  // ***
  private:
  	T* data_; // 实际数据指针
  	int64 elem_; // 数据size大小，即为包含多少个元素
}
```



### 与eigen3库的关系

在tensor的内部定义中，很多地方用到了结构TTypes,该结构的定义为tensorflow中tensor的定义与eigen3库中tensor的定义建立了联系。在tensorflow/core/framework/tensor_types.h中对该结构进行了定义

```c++
template <typename T, int NDIMS=1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned> Tensor;
  // ***
}
```



可以看到tensorflow中对于tensor的定义只是对eigen::tensor的封装，这一点从tf对于tensor的定义也能看出

```c++
template <typename T, size_t NDIMS>
typename TTypes<T,NDIMS>::Tensor Tensor::tensor() {
  /***/
  return typename TTypes<T,NDIMS>::Tensor(base<T>(), shape().AsEigenDSizes<NDIMS>());
}
```



​    这种封装相对高级，不是简单的在私有成员变量中包含后者，而是包含构造后者需要的数据，在需要后者时，对其进行构造并返回，这种方式，既能利用eigen3库对于张量的告诉计算，也能为tensor定制api。

### 成员变量TensorShape

成员变量Tensorshape定义如下

```c++
class TensorShape : public TensorShapeBase<TensorShape> {
  /***/
}

class TensorShapeBase : public TensorShapeRep {
  /***/
}

class TensorShapeRep {
  /***/
  protected:
  	struct Rep16 {
      uint16 dims_[6]; // 最多表示6维张量，每个维度不超过2^16-1
    };
    struct Rep32 {
      uint32 dims_[3]; // 最多表示3维张量，每个维度不超过2^32-1
    };
    struct Rep16 {
      gtl::InlinedVector<int64,4>* dims_;
    };
 private:
  union {
    // buf数组前12个元素用来存储形状，提供以上三种方式利用这12个字节，其中第14-16个字节中，分别存储张量中的数据
    // 类型编号，张量的维度数目，张量维度的表示类型。张量维度由一个字节表示，Tensor最多支持256维。
    uint8 buf[16];
    Rep64* unused_aligner;
  } u_;
  int num_elements_;
}
```



与TensorShape同一级别的还定义了PartialTensorShape，主要是用来处理一些未知维度tensor的操作。