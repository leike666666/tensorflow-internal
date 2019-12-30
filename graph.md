### 计算图

tensorflow的核心是计算图，计算图是由节点node和边edge组成。

##### 计算图节点node

节点可以拥有零条或者多条输入/输出边，使用in_edges_和out_edges_分别表示输入和输出边的集合。同时node持有NodeDef和OpDef。其中，OpDef 描述了 OP 的静态属性信息，例如 OP 的名称， 输入/输出参数列表，属性集定义等信息。而 NodeDef 描述了 OP 的动态属性值信息，例如 属性值等信息。node定义在tensorflow/core/graph/graph.h

```c++
class node {
  /****/
  private:
  	NodeClass class_; // 节点类别
  	EdgeSet in_edges_;
  	EdgeSet out_edges_;
  	std::shared_ptr<NodeProperties> props_; // 节点属性
}

struct NodeProperties {
  public:
  	/***/
  	const OpDef* op_def;
  	NodeDef node_def;
  	/***/
}
```



其中Nodedef包含一个操作op，表示这个节点的作用。每个节点必须放置在某个设备上(cpu,gpu...)，为了减少跨设备间的数据传输造成的计算损耗，节点的放置也是一个在受限条件下的优化问题，对此tf有专门的优化算法。

NodeDef的定义在tensorflow/core/framework/node_def.proto下

```protobuf
message NodeDef {
	string name = 1; // 节点名称
	string op = 2; // 节点包含的操作名称
	repeated string input = 3; // 节点的输入
	string device = 4; // 节点所在的设备信息
	map<string, AttrValue> attr = 5; // 节点操作的参数的具体赋值
}
```



OpDef的定义在tensorflow/core/framework/op_def.proto下

```protobuf
message OpDef {
	string name = 1; // op名称
	message ArgDef {
		string name = 1; // 输入/输出名称
		string description = 2; // 相关描述
		/***/
	}
	repeated ArgDef input_arg = 2; // 输入参数
	repeated ArgDef output_arg = 3; // 输出参数
	/***/
	message AttrDef {
	// op属性定义
	/***/
	}
	repeated AttrDef attr.= 4;
	
	/***/
}
```



##### 计算图边edge

计算图的边有两种类型，一种是普通边，另一种是控制依赖边。其中普通边承载Tensor，使用TensorId进行标识，该标识由源节点名字，及其所在边的src_output唯一确定。TensorId = node_name:src_output。默认src_output为0，当src_output为-1时，表示该边为控制依赖边。图的边定义在tensorflow/core/graph/graph.h。

```c++
class Edge {
  /***/
  private:
  	Node* src_; // 该边的源节点
  	Node* dst_; // 该边的目标节点
  	int id_;
  	int src_output_; // 生产该数据的节点的输出索引index
  	int dst_input_; // 消费该数据的节点的输入索引index
}
```



