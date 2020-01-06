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

