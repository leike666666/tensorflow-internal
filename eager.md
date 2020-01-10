在tensorflow2.0中默认的计算模式为动态图模式，即不用在后台构建一个计算图，通过session去运行该图。而动态图的运行方式更pythonic,每一步都直接计算出结果。

本地动态图模式执行的函数在tensorflow/core/common_runtime/eager/execute.cc中，具体函数为EagerLocalExecute。

```c++
Status EagerLocalExecute(EagerOperation* op, TensorHandle** retvals, int* num_retvals) {
    EagerContext* ctx = op->EagerContext();
    auto& executor = op->Executor();
    Device* device = op->Device();
    /***/
    core::RefCountPtr<KernelAndDevice> kernel = ctx->GetCachedKernel(cache_key);
    // 当前上下文缓存中没有kernel
    if (kernel == nullptr) {
        bool run_function_with_flr = false;
        if (op->is_function()) {
            /***/
            run_function_with_flr = true;
        }
        const NodeDef& ndef = op->MutableAttrs()->BuildNodeDef();
        FunctionLibraryRuntime* flr = device == nullptr ? nullptr : ctx->func_lib(device);
        auto runner = (flr != nullptr && flr->runner() != nullptr) ? flr->runner(): ctx->runner();
        if (run_function_with_flr) {
            /***/
        } else {
            kernel.reset(new KernelAndDeviceOp(ctx->GetRendezvous(), ctx->LogMemory(), flr, runner, ctx->GetCollectiveExecutorHandle(), ctx->HostCPU()));
        }
        
        // 初始化kernel
        kernel->Init(ndef, graph_collector);
        
        // 将kernel添加到缓存中
        ctx->AddKernelToCache(cache_key, kernel.get());
    }
    
    const bool async = executor.Async();
    if (async) {
        auto node = absl::make_unique<ExecuteNode>(
        ctx, op->Inputs(), op->remote_func_params(), std::move(kernel),
        graph_collector, output_dtypes, op->GetCancellationManager(),
        executor.Async(), absl::Span<TensorHandle*>(retvals, num_outputs));
        s = executor.AddOrExecute(std::move(node));
    } else {
        ExecuteNode node(ctx, op->Inputs(), op->remote_func_params(),
                     std::move(kernel), graph_collector, output_dtypes,
                     op->GetCancellationManager(), executor.Async(),
                     {retvals, num_outputs});
        s = executor.SyncExecute(&node);
    }
}
```

若传入参数EagerOperation类型为function，则kernel类型为KernelAndDeviceFunc，对其进行初始化

```c++
Status KernelAndDeviceFunc::InstantiateFunc(const NodeDef& ndef, GraphCollector* graph_collector) {
    /***/
    pflr_->Instantiate(ndef.op(), AttrSlice(ndef), options, &handle_);
}
```

