Grappler模块是Tensorflow中的图优化模块，它的执行时机是在DirectSession::Run时创建执行器Executor时进行优化。具体实现是在tensorflow/core/common_runtime/graph_execution_state.cc中

```c++
Status GraphExecutionState::OptimizeGraph(
    const BuildGraphOptions& options, std::unique_ptr<Graph>* optimized_graph,
    std::unique_ptr<FunctionLibraryDefinition>* optimized_flib) {
    /***/
    grappler::RunMetaOptimizer(
        item, session_options_->config, cpu_device, &cluster, &new_graph)
    /***/
}
```

其中MetaOptimizer.Optimize()是所有优化实现类的入口，根据优化配置决定是否调用之后的每个优化类。

### GrapplerItem

该结构在Grappler中表示一个待优化的tensorflow模型

```c++
struct GrapplerItem {
    // 根据graphdef创建一个GrapplerItem对象
    GrapplerItem WithGraph(GraphDef&& graph) const;
    string id; // 该item表示一个独一无二的id
    
    // 输入
    GraphDef graph;
    std::vector<std::pair<string, Tensor>> feed;
    std::vector<string> fetch;
    
    /***/
    // 优化选项
    struct OptimizationOptions {
        /***/
    }
private:
    std::unordered_set<string> devices_;
    OptimizationOptions optimization_options_;
}
```

### MetaOptimizer

其中MetaOptimizer.Optimize()是所有优化实现类的入口，根据优化配置决定是否调用之后的每个优化类。

```
Status MetaOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,GraphDef* optimized_graph) {
    
}
```

