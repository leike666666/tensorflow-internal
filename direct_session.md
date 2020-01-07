在静态图模式中，将计算图和输入交给session，由它来调度执行器，执行计算产生结果，在本地运行下，其运行时是由direct_session控制。整个计算的过程可以概括为图的创建，OP的placer编排，图的剪枝，图的分裂，图的执行。

### 图的创建

图的创建是在DirectSession的Create过程，主要是将图的protobuf格式转换为后端定义的Graph类，主要功能由函数ConvertGraphDefToGraph完成，主要构造过程在tensorflow/core/graph/graph_constructor.cc中

```c++
Status TryImport() {
    // 确保图中节点命名没有冲突
    EnsureNoNameCollisions();
    // 验证输入映射和控制依赖
    ValidateInputMapAndControlDependencies();
    // 创建节点在图中的索引
    BuildNodeIndex();
    // 初始化工作
    InitFromEdges();
    // 图的转换工作
    convert();
    /***/
}
```

### OP编排

op的编排主要是按照定义的启发式算法，定义op在哪个设备上运行，也是发生在DirectSession的create过程，在函数InitBaseGraph中会定义一个placer对象，所有工作由该对象完成，主要是根据用户要求，对每个node的placement进行调整，原则概括为以下四条：

1. 尽可能满足用户要求
2. 尽可能使用计算更快的设备
3. 保证程序可运行
4. 尽可能考虑近邻特性，尽量减少无意义的拷贝

基本功能定义在tensorflow/core/common_runtime/placer.cc中。

```c++
Status Placer::Run() {
    /***/
    // colocation_graph是节点的并查集结构，可以理解为将相同device的节点聚合在一起，此处将该结构进行初始化
    colocation_graph.Initialize();
    
    std::vector<Node*> second_pass;
    
    for (Node* node : grpah_->op_nodes()) {
        // 若该节点有被分配设备名称，限定该node所在colocation group中所有节点使用的设备
        if (node->has_assigned_device_name()) {
            colocation_graph.LimitToAssignedDevice(*node);
            continue;
        }
        
        // 启发式规则A：将类型为Generator的节点与该节点的消费者放置在一起，其中类型为Generator的
        // 节点定义为没有输入，只有一个输出的节点。先将其放在second_pass数组中，后面处理。
        if (IsGeneratorNode(node)) {
           second_pass.push_back(node);
            continue;
        }
        
        // 获得该节点的可能被放置的节点
        const std::vector<Device*>* devices;
        colocation_graph.GetDevicesForNode(node, &devices);
        
        int assigned_device =-1;
        
        // 启发式规则B:若该节点类型为Metadata时，规则期望能将其与它的输入节点放置在同一设备上。
        // 当该节点只依赖它输入的元数据(shape)时，例如"Size","Shape","Rank",该节点类型为MetaData。
        if (IsMetadata(node)) {
            const Node* input = (*node->in_edges().begin())->src();
            if (CanAssignToDevice(input->assigned_device_name(), *devices)) {
                assigned_device = input->assigned_device_name_index();
            }
        }
        
        // 若启发式规则B没有对其设备信息进行赋值，则将其设备信息赋值为候选设备列表中的第一个
        if (assigned_device == -1) {
            assigned_device = graph_->InternDeviceName((*devices)[0]->name());
        }
        AssignAndLog(assigned_device, node, &colocation_graph, log_device_placement_)
    }
    
    // 对于第一阶段跳过没被分配相应设备信息的节点在第二阶段处理
    for(Node* node : second_pass) {
        const std::vector<Device*>* devices;
        colocation_graph.GetDevicesForNode(node, &devices);
        /***/
        int assigned_device = -1;
        // 启发式规则A的应用
        if (IsGeneratorNode(node) && !node->out_edges().empty()) {
            const Node* output = (*node->out_edges().begin())->dst();
            int output_device_name = output->assigned_device_name_index();
            const bool consumers_on_same_device = std::all_of(
                node->out_edges().begin(), node->out_edges().end(),
                [output_device_name](const Edge* e) {
                    return e->dst()->assigned_device_name_index() == output_device_name;
                });
            if (consumers_on_same_device &&
                CanAssignToDevice(output->assigned_device_name(), *devices)) {
                assigned_device = output_device_name;
            }
        }
        if (assigned_device == -1) {
            assigned_device = graph_->InternDeviceName((*devices)[0]->name());
        }
        AssignAndLog(assigned_device, node, &colocation_graph, log_device_placement_));
    }
    return Status::OK();
}
```

### 图的剪枝

图的剪枝发生在DirectSession的Run过程中，主要从图上遍历找到所有输出节点依赖的输入节点，将无关节点裁剪掉，减少图的复杂度和相关计算量。在裁剪的过程中根据Session.Run传递的feeds,fetches输入输出列表，反向遍历FullGraph进行剪枝，计算本地迭代执行得到的最小依赖子图，称为ClientGraph。所用算法为广度优先遍历，在代码中的关键方法为SubGraph::RewriteGraphForExecution，一般地,对于ClientGraph输入节点,扮演了起始节点;而输出节点,扮演了终止节点。因此,关于输入和输出,存在两个比较棘手的问题:

1.输入：当ClientGraph计算开始前，外部的运行时如何传递Tensor给输入节点；

2.输出：当ClientGraph计算完成后，外部的运行时如何从输出节点获取Tensor;

存在两种媒介:FunctionCallFrame和Rendezvous,外部运行时与输入输出节点可以使用其中的一种媒介交换数据。

FunctionCallFrame用于Arg/RetVal函数调用的 OP,用于函数调用时传递函数参数值,及其返回函数值。但是,它们仅适用于单进程的运行时环境。

Rendezvous用于Send/Recv消息发送的 OP,这是一种更为通用的通信方式,适用于分布式的运行时环境。

### 图的分裂

下一步是根据Placement信息对Graph做切割，然后分发到不同的Device上去执行，主要包括一下三个核心步骤:

1.对原图的placement信息做划分，产生多个子图sub_graph;

2.为具有跨device依赖的节点对插入Send类和Recv类节点对;

3.插入必要的control edge

在单机单卡的运行过程中，DirectSession会让Graph Partition根据不同的device进行切割，而在分布式运行过程中，Graph Partition会被执行两次，一次是SplitByWorker,另一次是SplitByDevice。其主要功能主要在tensorflow/core/graph/graph_partition.cc中的Partition函数。

```c++
Status Partition(const PartitionOptions& opts, Graph* g,
                 std::unordered_map<string, GraphDef>* partitions) {
    /***/
    if (!opts.control_flow_added) {
        // 用于在分布式执行中添加控制流代码
        status = AddControlFlow(opts, g, &g_info);
        if(!status.ok) return status;
    }
    // 在该时候所有图的转变已经完成，该步为图中所有节点和边创建内存和设备类型信息
    status = BuildMemoryDeviceInfo(*g, &g_info);
  	if (!status.ok()) return status;

    std::vector<const Edge*> inputs;
    for (const Node* dst:g->op_nodes()) {
        // 从原图中取出一个节点dst,根据dst节点的location信息，得到其被放置的对应的partiotion图中
        dstp = opts.node_to_loc(dst);
        GraphDef* dst_graph = &(*partitions)[dstp];
        /***/
        for (const Edge* edge : dst->in_edges()) {
            // 遍历该节点所有的输入边
            if (edge->IsControlEdge()) {
                /***/
                inputs.push_back(edge);
            } else {
                /***/
                inputs[edge->dst_input()] = edge;
                ++num_input_edges;
            }
        }
    }
    // 按照顺序遍历该节点的数据输入边
    for (const Edge* edge : inputs) {
      // 获取该节点的上游节点及上游节点所在图
      const Node* src = edge->src();
      GraphDef* src_graph = &(*partitions)[opts.node_to_loc(src)];
      if (src_graph == dst_graph && !NeedSameDeviceSendRecv(edge, g_info)) {
        // 若src,dst所在图为相同图，而且设置相同设备下不需要send,recv节点，直接连接两个节点，
        // src->dst
        AddInput(dst_def, src->name(), edge->src_output());
        /***/
        continue;
      }
      /***/
      // 通过src,edge信息拼出所需要recv的key
     	const bool on_host = IsDstInputOnHost(edge, g_info);
      DupRecvKey key{src->id(), edge->src_output(), dst_graph, on_host};
      // 从缓存哈希表<key,recvInfo>查找信息
      auto iter = dup_recv.find(key);
      if (iter != dup_recv.end()) {
        // 若在缓存哈希表中找到，复用send,recv节点
        const string& recv_node_name = iter->second.recv->name();
        if (edge->IsControlEdge()) {
          // 为dst节点添加输入recv node
          AddInput(dst_def, recv_node_name, Graph::kControlSlot);
        } else {
          // 为dst节点添加输入recv node
          AddInput(dst_def, recv_node_name, 0);
        }
        /***/
        continue;
      }
      
      NodeDefBuilder::NodeOut send_from;
      if (edge->IsControlEdge()) {
        // 若该边类型为控制边,创建一个const dummy节点
        NodeDef* dummy = AddDummyConst(opts, src_graph, edge, &status);
        if (!status.ok()) return status;
        // 将src节点与dummy节点相连，src->dummy，dummy节点作为send节点的输入
        AddInput(dummy, src->name(), Graph::kControlSlot);
        send_from.Reset(dummy->name(), 0, DT_FLOAT);
      } else {
        // 若该类型为数据边，src节点作为send节点的输入
        send_from.Reset(src->name(), edge->src_output(), EdgeType(edge));
      }
      
      // 创建一个send节点
      NodeDef* send = AddSend(opts, g_info, src_graph, edge, send_from, 
                              send_start_time, &status);
      // 创建一个recv节点
      NodeDef* real_recv = nullptr;
      NodeDef* recv = AddRecv(opts, g_info, dst_graph, edge, &real_recv, &status);
      
      if (src_graph == dst_graph) {
        // 连接send，recv节点，send->recv,若在同一台设备上，则标记该边为控制边
        AddInput(real_recv, send->name(), Graph::kControlSlot);
      } else if (control_flow_edge != nullptr) {
        --num_control_flow_edges;
        AddInput(real_recv, control_flow_edge->src()->name(),
                 Graph::kControlSlot);
      }
      /***/
      if (edge->IsControlEdge()) {
        ++num_control;
        // 连接dst,recv节点，recv->dst，标记该边为控制边
        AddInput(dst_def, recv->name(), Graph::kControlSlot);
      } else {
        ++num_data;
        // 连接dst,recv节点，recv->dst
        AddInput(dst_def, recv->name(), 0);
      }
    }
  /***/
}
```

创建send和recv节点可以概括以下三种情况:

1.若src和dst节点在同一设备上，将两个节点相连，将连接边标记为控制边；

2.若src和dst节点在不同设备上，连接边为控制边，创建一个DummyConst节点，将其作为send节点的唯一输入，src->dummyconst->send->recv->dst;

3.若src和dst节点在不同设备上，连接边为数据边，直接连接send和recv，src->send->recv->dst。

### 图的执行

图的执行主要依靠执行器完成，执行器的应用可以概括如下

```c++
Graph* graph = ...;//构建图
Executor* executor;
NewSimpleExecutor(my_device, graph, &executor);//生成执行器
Rendezvous* rendezvous = NewNaiveRendezvous();//构建通信通道
rendezvous->Send("input", some_input_tensor);//提供输入
executor->Run({ExecutorOpts, rendezvous, nullptr});
rendezvous->Recv("output",&output_tensor);//获得输出
```

执行器的本身的接口定义在tensorflow/core/common_runtime/executor.h中

```c++
class Exexutor {
  /***/
  struct Args {
    int64 step_id = 0; // 是一个进程级别的唯一标识符，用来标识执行的步骤。当一个步骤运行了一个需要在多个设备上执行的op时，这些不同设备上的执行器将会收到相同的step_id。step_id是被用来追踪一个步骤中用到的资源的。
    RendezvousInterface* rendezvous = nullptr; // 通信接口，作为图之间通信的机制
    StepStatsCollectorInterface* stats_collector = nullptr; // 收集统计信息
    CallFrameInterface* call_frame = nullptr; // 如果该执行器用于执行一个函数，可以使用callframe，用来在调用者和被调用者之间传递参数和返回值
    CancellationManager* cancellation_manager = nullptr; // cancellation_manager用于执行一些注册一些在图被取消执行的回调函数
    SessionState* session_state = nullptr;
    // Unique session identifier. Can be empty.
    string session_handle;
    TensorStore* tensor_store = nullptr;
    ScopedStepContainer* step_container = nullptr;
    CollectiveExecutor* collective_executor = nullptr;
    thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr;

    // If true, calls Sync() on the device.
    bool sync_on_finish = false;

    typedef std::function<void()> Closure;
    typedef std::function<void(Closure)> Runner; // 将代执行的闭包交给runner执行，runner背后都有线程池支持
    Runner runner = nullptr;
  };
  
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void RunAsync(const Args& args, DoneCallback done) = 0;
  // 对RunAsync()进行封装得到的同步版本.
  virtual Status Run(const Args& args) {
    Status ret;
    Notification n;
    RunAsync(args, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
}
```

###### ExecutorBarrier

在实际应用中，可能用到多个执行器，为了使多个执行器并行执行，需要对这些执行的执行器进行管理，设置ExecutorBarrier类，相当于为多个Executor设置栅栏(barrier)，该类的定义如下:

```c++
class ExecutorBarrier {
public:
    typedef std::function<void(const Status&)> StatusCallback;
    // 为num个不同的executor创建barrier,r是一个共享的数据传输通道，如果任意一个执行器失败，rendezvous仅会崩溃一次，等最后一个执行器被执行完毕，就会调用回调函数done.
    ExecutorBarrier(size_t num, Rendezvous* r, StatusCallback done)
      : rendez_(r), done_cb_(done), pending_(num) {}
    // 获取所有执行器执行完成后调用的函数
    StatusCallback Get() {
        return std::bind(&ExecutorBarrier::WhenDone, this, std::placeholders::_1);
    }
private:
    Rendezvous* rendez_ = nullptr;
    StatusCallback done_cb_ = nullptr;
    /***/
}
```

###### NodeItem

NodeItem表示执行过程中图的一个节点，该结构体定义如下：

```c++
struct NodeItem {
    // 在执行图中的节点的index
    int node_id = -1;
    /***/
    // 该节点的kernel
    OpKernel* kernel = nullptr;
    
    int num_inputs;
  	int num_outputs;
    
    // 输入的起始索引
    int input_start = 0;
    // 输出边的数量
    size_t num_output_edges;
    /***/
}
```

###### GraphView

为了执行效率，执行器对一些基础结构进行了简化，剔除了不必要的信息，例如，对于计算图来说，在执行过程中，不需要对图结果进行更改，因此原来Graph中很多修改图的接口都没用了，所以tensorflow提供了一个不可更改的视图，用于使图的执行更加高效。该类的定义如下：

```c++
class GraphView {
public:
    GraphView() : space_(nullptr) {}
    void Initialize(const Graph* g);// 根据图Graph进行初始化
    NodeItem* node(size_t id) const;// 根据id返回相应的节点信息
    /***/
private:
    char* InitializeNode(char* ptr, const Node* n); // 初始化node
    size_t NodeItemBytes(const Node* n); // 根据node获取相应nodeitem占用字节数
    int32 num_nodes = 0; // 节点数量
    uint32* node_offsets_ = nullptr; // 节点偏置，node_offsets_[id]保存了节点id在space_中的偏置量
    char* space_; // 所有NodeItem对象的内存空间
}
```

该结构分配了一块内存空间，将图中所有的节点信息都依次存入该空间中，我们仍然能访问图中节点所有的静态信息，但无法对节点信息进行修改。

###### ExecutorImpl

Executor只是一个基类，真正执行器的实现tensorflow提供了一个子类ExecutorImpl,它的结构如下

```c++
class ExecutorImpl : public Executor {
public:
    /***/
    // 初始化过程中会为每个node的kernel进行实例化
    Status Initialize(const Graph& graph);
    void RunAsync(const Args& args, DoneCallback done) override;
private:
    struct ControlFlowInfo {
        // 包含帧的名称，提供了set和vector两种方式，set为了更方便查找某个帧的名称是否包含在内。而vector包含了帧的详细信息，主要是输入数量，以及未完成的节点计数等信息。
        gtl::FlatSet<string> unique_frame_names;
        std::vector<string> frame_names;
    };
    
    // 图执行过程中的帧信息主要是为了控制结构准备的
    struct FrameInfo {
        // 帧的输入数量
        int input_count;
        // 帧的各节点输入张量数量的总和
        int total_inputs;
        // 决定了我们最终创建的pending_counts数据结构在将要被分配的内存中的位置
        PendingCounts::Layout pending_counts_layout;
        // 每个帧都包含它自己的PendingCounts信息，只针对当前帧中的节点
        PendingCounts* pending_counts;
    }
    
    // 构建控制流信息
    static Status BuildControlFlowInfo(const Graph* graph,
                                     ControlFlowInfo* cf_info);
    // 初始化待执行计数信息
    void InitializePending(const Graph* graph, const ControlFlowInfo& cf_info);
    // 确认每个FrameInfo都已经准备好了
    FrameInfo* EnsureFrameInfo(const string& fname);
    
    // 被当前对象拥有
    LocalExecutorParams params_;
    GraphView gview_;
    
    // 不依赖任何输入边的根节点，由它们组成初始输入队列
    std::vector<const NodeItem*> root_nodes_;
    
    // frame名称与frameinfo的映射表
    gtl::FlatMap<string, FrameInfo*> frame_info_;
  
}
```

###### ExecutorState

在执行器执行图的计算过程中，需要一个结构来保存当前计算的信息，该结构即为ExecutorState，它会在一个节点准备好之后再调用该节点。

```c++
class ExecutorState {
public:
    void RunAsync(Executor::DoneCallback done);
private:
    // Entry要么是一个张量指针，要么是一个张量值，为图中的节点输入输出提供一个统一的类型
    struct Entry{}
    // 执行步骤开始时分配的设备上下文信息
    DeviceContext* device_context_ = nullptr;
    
    struct TaggedNode;
  	typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  	typedef gtl::InlinedVector<Entry, 4> EntryVector;
    
    // 代表一轮迭代时的状态
    struct IterationState {
        // 每一轮迭代都有一个单独的拷贝，对于第K轮迭代，第i个节点的第j个输入在input_tensors[k][impl_-         // >nodes[i].input_start + j]
        Entry* input_tensors;
        // 每一轮迭代中未完成的op的数量
        size_t outstanding_ops;
        
        // 每一轮迭代未完成的帧数量
        int outstanding_frame_count;
        int pending(PendingCounts::Handle h);
        int decrement_pending(PendingCounts::Handle h, int v);
        
        // 将一个merge节点标记为live
        void mark_live(PendingCounts::Handle h);
        // 标记一个节点开始处理
        void mark_started(PendingCounts::Handle h)
        // 标记一个节点处理完成
        void mark_completed(PendingCounts::Handle h)
        // 获取节点状态
        PendingCounts::NodeState node_state(PendingCounts::Handle h)
        /***/
    }
    
    // FrameState代表一个帧的状态
    struct FrameState {
        const ExecutorImpl* executor = nullptr; // 帧所在的执行器
        string frame_name; // 是由父帧，迭代轮次，frame_name字段拼合起来的
        unit64 frame_id; // 当前帧的唯一标识
        int64 parent_iter = -1; //该帧创建时父帧的迭代轮次
        FrameState* parent_frame = nullptr; // 父帧的帧状态
        const int max_parallel_iterations; // 最大允许并行迭代数量
        /***/
    }
    
    // TaggedNode代表一个有标签的节点：<frame*, iter, node*>
    struct TaggedNode {
        const NodeItem* node_item;
        FrameState* input_frame = nullptr;
        int64 input_iter = -1;
        bool is_dead = false;
        /***/
    };
    
    // TaggedNodeReadyQueue表示TaggedNode的一个队列
    class TaggedNodeReadyQueue {
        void push_back(TaggedNode node);
        void pop_front()
        /***/
    private:
        gtl::InlinedVector<TaggedNode, 16> ready_;
        /***/
    }
    
    // 当前步骤开始执行的根帧
    FrameState* root_frame_;
    
    // 在当前线程中处理一个已经准备好的节点
    void Process(TaggedNode node, int64 scheduled_nsec);
	// 在调用item->kernel之前，填入它的输入    
    Status PrepareInputs(const NodeItem& item, Entry* first_input,
                           TensorValueVec* inputs,
                           AllocatorAttributeVec* input_alloc_attrs,
                           bool* is_input_dead);
    // 在调用item->kernel之后，处理它的输出
    Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStatsInterface* stats);
    // 在处理完输出之后，将输出传递给下一个节点的输入
	void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                        EntryVector* outputs, TaggedNodeSeq* ready);
    // 节点计算结束后，接管stats
    bool NodeDone(const Status& s, const TaggedNodeSeq& ready,
                NodeExecStatsInterface* stats,
                TaggedNodeReadyQueue* inline_ready);
    // 调度ready中所有的复杂节点，然后将ready中的非复杂节点放入inline_ready中
    void ScheduleReady(const TaggedNodeSeq& ready,
                     TaggedNodeReadyQueue* inline_ready);
  	
    // 执行器执行完成后对资源进行清理
    void Finish();
  	void ScheduleFinish();
    
}
```

###### 执行过程

对于在direct_session中调用的执行器的Run或者RunAsync方法，最终调用的是ExecutorState的RunAsync方法

```c++
void ExecutorState::RunAsync(Executor::DoneCallback done) {
    // 构建准备节点队列
    TaggedNodeSeq ready;
    
    // 让设备填充设备上下文映射
    Device* device = impl_->params_.device;
    const Status get_context_status =
      device->TryGetDeviceContext(&device_context_);
    /***/
    
    // 初始化ready队列
    for (const NodeItem* item : impl_->root_nodes_) {
        ready.push_back(TaggedNode{item, root_frame_, 0, false});
    }
    if (ready.empty()) {
        // 若ready为空，直接执行完成
        delete this;
        done(Status::OK());
    } else {
        num_outstanding_ops_ = ready.size();
        {
            mutex_lock l(root_frame_->mu);
            root_frame_->GetIteration(0)->outstanding_ops = ready.size();
        }
        done_cb_ = std::move(done);
        // 在线程池中开始对ready中的节点进行调度运算
        ScheduleReady(ready, nullptr);
    }
}
```

上述函数主要是对ready队列进行初始化，然后启动ScheduleReady函数

```c++
void ExecutorState::ScheduleReady(const TaggedNodeSeq& ready,
                                  TaggedNodeReadyQueue* inline_ready) {
    /***/
    if (inline_ready == nullptr) {
        //inline_ready为空，直接在线程池中对所有的准备好的节点调度运行
        for (auto& tagged_node : ready) {
            // runner即为在Executor::Args中定义的std::function,其中Process即为它的闭包函数
            runner_([=]() { Process(tagged_node, scheduled_nsec); });
        }
        return;
    }
    
    // inline_ready不为空时
    const TaggedNode* curr_expensive_node = nullptr;
    for (auto& tagged_node : ready) {
        const NodeItem& item = *tagged_node.node_item;
        if (tagged_node.is_dead || !item.kernel->IsExpensive()) {
            // 将非复杂节点放入内联队列中
            inline_ready->push_back(tagged_node);
        } else {
            if (curr_expensive_node) {
                // 对于当前线程已经在处理一个复杂节点，将该任务分配给其他线程
                runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                          scheduled_nsec));
            }
            curr_expensive_node = &tagged_node;
        }
    }
    if (curr_expensive_node) {
        if (inline_ready->empty()) {
            inline_ready->push_back(*curr_expensive_node);
        } else {
            runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                        scheduled_nsec));
        }
    }
}
```

可以看出该函数的两个参数，一个是待执行的节点队列，一个是待执行的内联节点队列，分了两种情况进行处理：

1.内联节点队列为空时，会为ready队列中的每一个节点，单独新增一个执行线程，也就是说对于根执行队列中的节点，分别新增了一个线程来执行。

2.inline_ready不为空，该函数不会进行任何实际的执行，只会对执行进行分配，它会遍历ready中的每个节点，如果该节点已经死亡或者为非复杂节点，将其放入inline_ready队列等待执行，否则就单独开启一个线程来处理它，同时这个遍历过程执行完成后，会保留最后一个复杂节点，这是若inline_ready为空，就把这个复杂节点放入内联队列，否则开启一个线程执行。

从上面的代码可以看出，整个执行的核心在于Process函数。

```c++
void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_nsec) {
    /***/
    TaggedNodeSeq ready;
    TaggedNodeReadyQueue inline_ready;
    
    // 为OpKernel::Compute准备参数
    OpKernelContext::Params params;
    /***/
    
    inline_ready.push_back(tagged_node);
    while (!inline_ready.empty()) {
        // 当inline_ready中节点非空时，取出队头节点，获取节点，帧信息
        tagged_node = inline_ready.front();
        inline_ready.pop_front();
        const NodeItem& item = *tagged_node.node_item;
        FrameState* input_frame = tagged_node.input_frame;
        const int64 input_iter = tagged_node.input_iter;
        const int id = item.node_id;
        /***/
        // 准备输入
        s = PrepareInputs(item, first_input, &inputs, &input_alloc_attrs,
                        &is_input_dead);
        /***/
        // 若kernel是异步调用
        if (item.kernel_is_async) {
            // 构建异步调用kernel
            AsyncOpKernel* async = item.kernel->AsAsync();
            // 构建kernel执行完成后调用的lambda函数
            auto done = [this, state]() {
                /***/
                PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
                const bool completed = NodeDone(s, ready, stats, nullptr);
          		delete state;
          		if (completed) ScheduleFinish();
            }
            /***/
            // 执行kernel函数
            device->ComputeAsync(async, &state->ctx, done);
        } else {
            // 同步调用
            OpKernelContext ctx(&params, item.num_outputs);
            /***/
            device->Compute(op_kernel, &ctx);
            s = ProcessOutputs(item, &ctx, &outputs, stats);
        }
        
        if (!launched_asynchronously) {
            /***/
            PropagateOutputs(tagged_node, &item, &outputs, &ready);
            completed = NodeDone(s, ready, stats, &inline_ready);
        }
    }
    if (completed) ScheduleFinish();
}
```

可以概括为以下几个步骤：

1.准备输入

2.调用设备进行compute

3.对输出进行处理

4.将输出传递给下一个节点的输入

5.节点处理完成后调用回调函数NodeDone

其中输入，输出的处理函数比较容易理解，就是对相关参数进行填充，下面主要对PropagateOutputs函数和NodeDone函数进行分析。

```c++
void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                     const NodeItem* item, EntryVector* outputs,
                                     TaggedNodeSeq* ready) {
    /***/
    // 沿着输出边传递输出，把准备好的节点放入ready队列中，根据不同的节点类型，选择合适的处理方法
    if (!item->is_enter_exit_or_next_iter) {
        // 若节点不是enter,exit,next_iter中的一种
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        input_frame->DecrementOutstandingOpsLocked(&impl_->gview_, input_iter, ready);
    } else if (item->is_enter) {
        // 若节点类型是enter
        FindOrCreateChildFrame(input_frame, input_iter, *item, &output_frame);
        /***/
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        input_frame->DecrementOutstandingOps(&impl_->gview_, input_iter, ready);
    } else if (item->is_exit) {
        // 若节点类型是exit
        /***/
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        input_frame->DecrementOutstandingOpsLocked(&impl_->gview_, input_iter, ready);
    } else {
        // 若节点类型是next_iter
        /***/
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
        input_frame->DecrementOutstandingOpsLocked(&impl_->gview_, input_iter, ready);
    }
    
    // 节点处理完成后，判断当前帧是否执行完成，以及递归的判断父帧有没有执行完毕
    if (is_frame_done) {
        /***/
        DeleteFrame(input_frame, ready);
        if (parent_frame != nullptr) {
            CleanupFramesIterations(parent_frame, parent_iter, ready);
        }
    }
}
```

根据不同的节点类型会为其做不同的操作，但基本会包含两个操作，一个是激活节点，一个是减少当前未执行节点的数目。在激活节点函数中

```c++
void ExecutorState::FrameState::ActivateNodes(const NodeItem* item,
                                              const bool is_dead, int64 iter,
                                              EntryVector* outputs,
                                              TaggedNodeSeq* ready) {
    /***/
    // 若目标节点准备好了，将其放入ready队列中
    if (dst_ready) {
      if (dst_item->is_control_trigger) dst_dead = false;
      ready->emplace_back(dst_item, this, iter, dst_dead);
      iter_state->outstanding_ops++;
    }
}
```

最后处理的函数为NodeDone

```c++
bool ExecutorState::NodeDone(const Status& s, const TaggedNodeSeq& ready,
                             NodeExecStatsInterface* stats,
                             TaggedNodeReadyQueue* inline_ready) {
    /***/
    ScheduleReady(ready, inline_ready);
    /***/
}
```

对上述过程进行总结如下：

1.ExecutorImpl::RunAsync作为执行器的入口，其实它是把实际执行的工作交给了ExecutorState::RunAsync，这个函数进一步调用了ScheduleReady函数来调度执行，记住这个函数有两个输入，ready和inline_ready，在这里，inline_ready是空的。也就是说，这里调用ScheduleReady的作用是，给根节点队列里的节点，分别分配一个线程执行，执行的过程调用的是Process函数。

2.在Process函数内部，我们还要记住一点，这个函数的输入只有节点，没有ready和inline_ready，这两个变量都是在Process函数内部新创建的。也就是说，一旦把一个节点交给Process函数去处理，这个节点所在的队列跟Process函数就没有任何关系了。处理的过程分为输入准备、实际计算、输出准备、输出传递、节点完成五个步骤。其中只有输出传递和节点完成会对ready和inline_ready结构产生影响。

3.我们把节点分按照异步节点和同步节点分开处理。对于异步节点，NodeDone函数的最后一个参数inline_ready是空，也就是说，在异步执行时，调用NodeDone中的ScheduleReady时，跟RunAsync中的情形是一样的，直接调度ready中的节点就好了，不需要处理inline_ready的情况。对于同步节点，NodeDone函数的最后一个参数inline_ready是当前Process函数中新创建的inline_ready，也就是说，传递给ScheduleReady的inline_ready是非空的，这也就有可能对inline_ready的结构做修改，注意这里的inline_ready是从Process函数中创建的，每个Process函数都对应一个全新的线程，也就是说，每个全新的线程里面只有一个inline_ready结构，其中的函数不断的修改它的内容，然后不断的对它进行调度执行。注意Process中的while大循环是针对inline_ready队列执行的。