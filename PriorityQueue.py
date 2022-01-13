class QueueNode:
    def __init__(self,data, priority):
        self.data = data
        self.priority = priority
        self.next = None

class PriorityQueue:
    def __init__(self):
        self.front = None
        self.rear = None

    def enqueue(self,data, priority):
        new_node = QueueNode(data, priority)
        if self.front is None:
            self.front = new_node
            self.rear = self.front
        else:
            if self.front.priority < priority: #假如优先级小，则要把最大的变成front node
                new_node.next = self.front
                self.front = new_node
            else:#根据优先级挨个比较插入队列中
                cur = self.front
                while cur.next is not None and cur.next.priority > priority:
                    cur = cur.next
                new_node.next = cur.next
                cur.next = new_node

    def dequeue(self):
        if self.front is None:
            print("the queue is empty")
        else:
            self.front = self.front.next
            if self.front == None:
                self.rear = None

    def peek(self):
        return self.front.data

    def isEmpty(self):
        return self.front is None

    def showAll(self):
        cur = self.front
        listAll = list()
        if self.front is None:
            return
        else:
            while cur is not None:
                listAll.append(cur.data)
                cur = cur.next
        return listAll

def easyAppend(queue_name,input_list,prio_list):
    for i in range(len(input_list)):
        queue_name.enqueue(input_list[i],prio_list[i])

if __name__ =="__main__":
    queue1 = PriorityQueue()
    input_list = [1, 3, 2, 4, 5, 3, 2, 44, 3, 1000]
    prio_list = [1, 3, 2, 4, 5, 3, 2, 44, 3, 1000]
    easyAppend(queue1, input_list,prio_list)
    print(queue1.showAll())
    queue1.dequeue()
    print(queue1.showAll())


