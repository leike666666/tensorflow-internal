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

下一步是根据Placement信息对Graph做切割，然后分发到不同的Device上去执行，主要包括一下三个核心步骤，1.对原图的placement信息做划分，产生多个子图sub_graph;

2.为具有跨device依赖的节点对插入Send类和Recv类节点对;

3.插入必要的control edge

在单机单卡的运行过程中，DirectSession会让Graph Partition根据不同的device进行切割，而在分布式运行过程中，Graph Partition会被执行两次，一次是SplitByWorker,另一次是SplitByDevice。

