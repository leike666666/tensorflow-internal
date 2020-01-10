函数的本质是给定输入，经过计算给出输出，这个与图中的OP的定位相同，而它的本质就是一些比较大的OP。

### FunctionDef

function通过protobuf定义，在tensorflow/core/framework/function.proto中

```protobuf
message FunctionDef {
	// 函数的名字，参数，返回值，相关属性等定义
    OpDef signature = 1;
    // 函数定义相关的属性
    map<string, AttrValue> attr = 5;
    // 函数参数的相关属性
    message ArgAttrs {
    	map<string, AttrValue> attr = 1;
  	}
  	map<uint32, ArgAttrs> arg_attr = 7;
  	// 可能包含多个nodedef，这些节点组合在一起形成了函数的内部结构
  	repeated NodeDef node_def = 3;
  	// 一个从signature中输出参数名称到node_def输出的映射
  	map<string, string> ret = 4;
  	map<string, string> control_ret = 6;
}
```

tensorflow中支持梯度计算，是因为tensorflow针对每个函数给出了它的梯度函数，为了将原函数和其梯度函数联系在一起，因此定义了GradientDef这个结构。

```protobuf
message GradientDef {
  string function_name = 1;  // 函数名称
  string gradient_func = 2;  // 函数梯度名称
}
```

在tensorflow运行时包含了一个函数定义库，需要使用某个函数时，可以去库里找。该结构的本质是一个函数定义的集合，不具备查找等功能。

```protobuf
message FunctionDefLibrary {
  repeated FunctionDef function = 1;
  repeated GradientDef gradient = 2;
}
```

### FunctionLibraryDefinition

该类给我们提供了一个方便对function进行集中管理的地方，继承自OpRegisterInterface，提供函数注册，查找等功能。该辅助类维护一个给定的FunctionDefLibrary和函数定义之间的映射。

```c++
class FunctionLibraryDefinition : public OpRegistryInterface {
    /***/
    bool Contains(const string& func) const;
    const FunctionDef* Find(const string& func) const;
    Status AddFunctionDef(const FunctionDef& fdef);
    Status AddGradientDef(const GradientDef& grad);
    /***/
}
```

### FunctionLibraryRuntime

该类是函数库运行时的结构，为函数的执行提供了很多便利的接口，它单纯的是包裹在FunctionLibraryDefinition之上。

