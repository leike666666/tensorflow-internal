在tensorflow中rendezvous是用于完成消息传输的通信组件。

### 消息传递标识符:ParsedKey

消息传输从发送端send到接收端recv，在多组消息同时发送接收时，需要对每对send和recv梳理一个对应关系，因为每对处理的消息都是同一个，所以只要让消息在被发送之前加上一个唯一标识符，在接收时也能按照一定规则获得该标识符，则可以获得一一对应的关系，在tensorflow中该一一对应的标识符就是ParsedKey，该结构体定义在tensorflow/core/framework/rendezvous.h

```c++
struct ParseKey {
    StringPiece src_device; // 消息发送源的字符串信息，如"/job:localhost/replica:0/task_id:0/device:GPU:0"
    DeviceNameUtils::ParsedName src; // 和src_device信息相同，只是结构体表示
    unit64 src_incarnation = 0;
    StringPiece dst_device; // 消息发送接收方的字符串信息
    DeviceNameUtils::ParsedName dst; // 和dst_device信息相同，只是结构体表示
    StringPiece edge_name; // 可以指定为任何字符串，实现不同key的区分
}
```

### 消息通信机制:rendezvous

最基本的Rendezvou接口定义在tensorflow/core/framework/rendezvous.h中，主要包括Send,Recv,RecvAsyc接口，定义如下：

```c++
class RendezvousInterface {
    /***/
    virtual Status Send(const ParsedKey& key, const Args& args, const Tensor& val, const bool is_dead);
    virtual void RecvAsync(const ParsedKey& key, const Args& args,
                         DoneCallback done) = 0;
    // Synchronous wrapper for RecvAsync.
    Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
                bool* is_dead, int64 timeout_ms);
    Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
                bool* is_dead);
    /***/
}
```

根据不同的场景有不同的实现类，对于本地传输来说，tensorflow提供了LocalRendezvous,基本定义在tensorflow/core/framework/local_rendezvous.h中

```c++
class LocalRendezvous {
    public:
    	virtual Status Send(const ParsedKey& key, const Args& args, const Tensor& val, const bool is_dead);
    	virtual void RecvAsync(const ParsedKey& key, const Args& args,DoneCallback done);
    /***/
    private:
    	// 传递消息的基本结构
    	struct Item;
    	//在通信机制内维护消息的一个队列
        struct ItemQueue {
            void push_back(Item* item);
            Item* head = nullptr;
            Item* tail = nullptr;
        }
    	// 映射表，其中key为ParsedKey的哈希值，value为对应的消息队列
    	typedef gtl::FlatMap<uint64, ItemQueue> Table;
    	Table table_;
    	/***/
}
```

具体实现实在tensorflow/core/framework/local_rendezvous.cc，其中发送消息过程如下

```c++
Status LocalRendezvous::Send(const Rendezvous::ParsedKey& key,
                             const Rendezvous::Args& send_args,
                             const Tensor& val, const bool is_dead) {
    uint64 key_hash = KeyHash(key.FullKey());
    /***/
    ItemQueue* queue = &table_[key_hash];
    // 若消息队列为空或者消息队列的第一个消息类型为send
    if (queue->head == nullptr || queue->head->type == Item::kSend) {
        /***/
        queue->push_back(new Item(send_args, val, is_dead));
        /***/
        return Status::OK();
    }
    // 若消息队列的第一个消息为recv，其中item中的recv_state.waiter(函数指针)不为空
    Item* item = queue->head;
    if (item->next == nullptr) {
        // 若队列中的最后一个元素被消费，从映射表中删除该消息队列
        table_.erase(key_hash);
    } else {
        // 移动消息队列中的头指针指向item的下一条消息
        queue->head = item->next;
    }
    /***/
    // 调用消息队列中recv消息的函数指针
    (*item->recv_state.waiter)(Status::OK(), send_args, item->args, val, is_dead);
    delete item;
    return Status::OK();
    /***/
}
```

接收消息过程如下

```c++
void LocalRendezvous::RecvAsync(const Rendezvous::ParsedKey& key,
                                const Rendezvous::Args& recv_args,
                                Rendezvous::DoneCallback done) {
    uint64 key_hash = KeyHash(key.FullKey());
    /***/
    ItemQueue* queue = &table_[key_hash];
    // 若消息队列为空或者队列的第一个消息类型为recv
    if (queue->head == nullptr || queue->head->type == Item::kRecv) {
        /***/
        queue->push_back(new Item(recv_args, std::move(done),/***/))
    }
    // 若消息第一个消息类型为send,直接将send消息当做函数指针的参数进行消费
    Item* item  = queue->head;
    if (item->next == nullptr) {
        table_.erase(key_hash);
    } else {
        queue->head = item->next;
    }
    done(Status::OK(), item->args, recv_args, *item->send_state.value,
        item->send_state.is_dead);
    delete item;
}
```

具体的传输过程可以概括如下，因为在tensorflow中,生产和消费二者的相对顺序没有办法确定，即Send和RecvAsync的调用先后顺序没有办法确定。根据消息到来的先后顺序不同，设置消息的不同参数，这点可以通过消息结构体Item的定义看出来，该结构体定义在tensorflow/core/framework/local_rendezvous.cc。

```c++
struct LocalRendezvous::Item {
    enum Type {KSend = 0, KRecv = 1};
    /***/
    const Rendezvous::Args args;
    const Type type;
    // 链接消息队列的下条消息
    Item* next = nullptr;
	// 联合体，send_state和recv_state的有效性根据item的type决定
    union {
        struct {
            // 发送消息包含类型为Tensor的value
            ManualConstructor<Tensor> value;
            bool is_dead;
        } send_state;
        struct {
            // 接收消息包括std::function的函数指针，用来消费value
            ManualConstructor<Rendezvous::DoneCallback> waiter;
            CancellationToken cancellation_token;
        } recv_state;
    };
}
```

当类型为send的消息item先到消息队列时，设置参数send_state,后续对应消费该消息的item到达队列时，对value进行消费。当类型为recv的消息item先到消息队列时，设置参数recv_state，其中包含回调函数指针waiter，等待后续对应的类型为send的消息item，使用该waiter对其进行消费。