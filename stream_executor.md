TensorFlow是一个异构的并行执行框架，对于异构Device的管理是一件非常复杂的事，不仅包括Device的添加、注册、删除、属性的管理，还必须要对Device的并行执行过程做进一步抽象形成统一的框架，才能实现更好的解耦。

### 简介

StreamExecutor本身就是一个在Google内部为并行编程模型开发的单独的库，StreamExecutor为TensorFlow的执行层面提供了较为统一的抽象，而在底层各种Device的执行管理细节却完全不同。我们可以看到stream_executor下面有cuda和host两个子目录，他们分别是GPU执行引擎和CPU执行引擎所使用的子模块。

### Stream

Stream存在于计算机相关的各种技术中，比如在操作系统、流式计算、计算机网络传输或是CUDA编程中都有涉及。Stream从抽象角度来看其本质是定义了一个操作序列，处于同一个Stream的操作必须按顺序执行，不同Stream之间的并无顺序关系。在TensorFlow中存在一些高性能的并行编程设备，所以需要有一套抽象框架对这些设备的执行过程管理起来，这就是StreamExecutor的用武之地了。

为了隐藏StreamExecutor框架管理的复杂性，它对外暴露的handler必须足够简单。事实也确实如此，StreamExecutor通过暴露Stream对象作为操作底层的handler。一般而言，在TensorFlow的框架中都是使用Stream对象来调用底层计算库，进行设备间数据拷贝操作等过程。比如调用Stream对象的ThenMemcpy即可完成异步的数据传输拷贝过程，调用ThenConvolveXXX等函数即可完成DNN库中的卷积调用。事实上，TensorFlow中很多Op的C++实现中，其Compute函数内就是通过使用Stream对象来完成某些实际计算或数据拷贝的过程。

Stream对象是通过持有StreamInterface的具体实现对象来获得实际平台的Stream，进而通过Stream这个统一的handler完成与底层的交互。

### StreamExecutor层次结构

总体上StreamExecutor框架由三个层次组成，从上到下依次为Platform层（平台描述）、StreamExecutor 
Core层（执行引擎）和LibrarySupport层（基础库）。如果需要为TensorFlow添加新的计算设备种类，不但要向TensorFlow中注册Device的定义，还需要在StreamExecutor框架中提供负责管理该Device计算的代码。

###### Platform层

在StreamExecutor中Platform指的是计算所使用设备平台的抽象，每种Device对应一种Platform。比如GPU对应的是CudaPlatform，而CPU对应的是HostPlatform等。一旦获得了某种Device的Platform，就可以获取和该Platform对应的StreamExecutor Core以及相应的LibrarySupport。在TensorFlow的代码实现中，所有Platform类都是通过宏定义和MultiPlatformManager管理类的静态方法主动注册到系统中的。

###### StreamExecutor core层

对于外部使用者来说，获取Platform就是为了获取对应的执行引擎。对于TensorFlow这种存在多种Platform和执行引擎的异构框架来说，必须为每一种执行引擎提供完整的实现，这具有一定的复杂度。为了让代码结构更有层次感，也为了向Platform层隐藏底层的设计复杂度，该层选择只向上层暴露StreamExecutor类，而涉及到具体实现的StreamExecutorInterface以及各种具体的实现将由StreamExecutor类统一控制，这种代理的方式让这一层的架构更加干净。

CudaExecutor和HostExecutor继承自StreamExecutorInterface后，由StreamExecutor持有，并暴露给上一层Platform使用。同各种Platform类似，每个具体的StreamExecutor也需要注册到系统中，但他们却没有依赖于任何控制类，直接通过宏定义将自己注册到全局工厂中，注册过程也是借助Initializer模块实现的。

initialize_cuda_gpu_executor函数中定义了一个创建CUDAExecutor的匿名函数，而MakeCUDAExecutorImplementation函数实际上创建了一个全局的table，中间的等号赋值操作实际上就是把该匿名函数放到了全局instance中，这实际上就是一种简单的工厂模式，在StreamExecutor中存在多种类似的工厂。

StreamExecutor框架使用Cache机制避免为同一种StreamExecutor Core被重复创建，这个Cache就是ExecutorCache，下面代码展示了Platform从Cache获取StreyinqamExecutor Core的内容，当Cache中不存在所需要的StreamExecutor时，会创建新的对象并放入cache中，并以config作为key。

###### Library层

这一层提供的是各种底层加速库的接入，当前该层主要负责接入Dnn，Blas，Rng和Fft模块，每个模块和对应的类说明如下表所示 。

| 子模块名称  | 功能说明                                                     |
| ----------- | ------------------------------------------------------------ |
| DNNSupport  | DNN计算模块，主要包含DNN计算的基本操作。在GPU实现中，它将作为CuDNN的封装 |
| RngSupport  | 随机数生成模块                                               |
| BlasSupport | 基础线性代数库模块，主要包含矩阵系列的计算，在CPU实现中它可以是Eigen，mkl等；在GPU实现中，它将作为CuBLAS的封装 |
| FFTSupport  | FFT系列运算模块                                              |

因为这些基础库同StreamExecutor类似，都具有平台属性，例如在CUDAHostPlatform中使用的Blas库应为CuBLAS，而HostPlatform中对应的可能是OpenBlas，MKL等。虽然StreamExecutorInterface创建出来的各种Library指针均由StreamExecutor持有，但是他们却由StreamExecutorInterface的实现类负责创建，所以从逻辑上看他们处于StreamExecutor Core的下一层。

Library层将这些基础库统一作为插件（Plugin）来管理，用以应对未来出现的各种各样的基础库。他们通过PluginRegister模块注册。和StreamExecutor Core中的管理方式相同，依然要先创建插件的Factory，Factory的创建也通过宏实现。以CudnnSupport为例，通过向通用初始化模块Intializer传入initialize_cudnn函数并调用，将创建CudnnSupport的函数作为DnnFactory放到PluginRegister模块中，至此完成了DnnFactory的创建。使用时，只需要拿到PluginRegister的key（即要求拿到何种插件）即可取出对应的LibrarySupport。

###### 执行过程

实际执行过程中外部通过stream对象调用卷积前向，反向等操作，在stream对象内部，通过获取支持该流操作的StreamExecutor从而得到执行引擎底层加速库的支持。

