class ListGraph:
    def __init__(self,v):
        """:arg
        无向图，同步录入起始到终点图
        有向图，只需要只录入起始到终点，不用反过来
        以下代码为无向图
        """
        self.v = v # v: 代表有多少个顶点
        self.graphs=dict()
        self.visited = list()
        for i in range(v):
            # 初始化每个顶点为list
            self.graphs[i]=[]
            # 保证每个点都是处于未被访问的状态
            self.visited.append(False)

    def addEdge(self,start,end):
        # 加入路径
        self.graphs[start].append(end)
        self.graphs[end].append(start)

    def removeEdge(self,start,end):
        self.graphs[start].remove(end)
        self.graphs[end].remove(start)


    def DFSTraversal(self,vertex):
        if self.visited[vertex] is True:
            return
        else:
            self.visited[vertex]=True
            print("{0}->".format(vertex))
            if self.graphs[vertex]==[]:
                return
            for i in range(len(self.graphs[vertex])):
                num = self.graphs[vertex][i]
                if self.visited[num] is not True:
                    self.DFSTraversal(num)

    def DFS(self):
        for i in self.graphs.keys():
            if self.visited[i] is not True:
                self.DFSTraversal(i)

    def BFSTraversal(self,vertex):
        queue = list()
        self.visited[vertex]=True
        queue.append(vertex)
        while len(queue)!=0:
            cur=queue.pop(0)
            print("{0}->".format(cur))
            if self.graphs[cur]==[]:
                self.visited[cur]=True
                continue
            for i in range(len(self.graphs[cur])):
                num = self.graphs[cur][i]
                if self.visited[num]!=True:
                    queue.append(num)
                    self.visited[num]=True

    def BFS(self):
        for i in range(len(self.graphs.keys())):
            if self.visited[i]==False:
                self.BFSTraversal(i)

if __name__ == "__main__":
    lp = ListGraph(10)
    list_edges = [[1, 2], [1, 3], [3, 4], [5, 6], [7, 9], [0, 9], [0, 2], [8, 4], [3, 7], [2, 6]]
    for each in list_edges:
        lp.addEdge(each[0], each[1])

    lp.DFS()
    #lp.BFS()
