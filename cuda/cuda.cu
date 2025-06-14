#include <cstdio>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

#define INF 1e20f

// 边结构体
struct Edge
{
  int to;       // 目标节点
  float weight; // 边的权重
};

// 生成随机图
void generateRandomGraph(int V, int E, std::vector<int> &vertex_offsets, std::vector<Edge> &edges)
{
  std::vector<std::vector<Edge>> adj(V); // 邻接表表示图

  std::random_device rd;                                          // 随机数生成器
  std::mt19937 gen(rd());                                         // 使用随机设备作为种子
  std::uniform_int_distribution<> node_dist(0, V - 1);            // 随机节点分布
  std::uniform_real_distribution<float> weight_dist(1.0f, 20.0f); // 随机权重分布

  // 保证每节点至少一条出边, 先给每个点加一条边
  for (int u = 0; u < V; ++u)
  {
    int v = node_dist(gen); // 随机选择目标节点

    // 确保边的目标节点不等于源节点
    while (v == u)
      v = node_dist(gen);

    float w = weight_dist(gen); // 随机权重
    adj[u].push_back({v, w});   // 添加边
  }

  // 添加剩余的边，直到达到指定的边数
  int edges_added = V; // 已添加的边数从V开始，因为每个节点至少有一条边
  while (edges_added < E)
  {
    // 随机选择两个不同的节点
    int u = node_dist(gen);
    int v = node_dist(gen);

    // 确保u和v不同，如果u和v相同，跳过这次迭代
    if (u == v)
      continue;

    // 检查边(u, v)是否已存在，如果存在，跳过这次迭代
    bool exists = false;
    for (auto &e : adj[u])
    {
      if (e.to == v)
      {
        exists = true;
        break;
      }
    }
    if (exists)
      continue;

    // 如果边不存在，添加新边，随机生成权重
    float w = weight_dist(gen);
    adj[u].push_back({v, w});
    edges_added++;
  }

  // 图存储方式：BFS（边列表）,构建顶点偏移数组和边列表
  vertex_offsets.resize(V + 1); // 顶点偏移数组大小为V+1
  vertex_offsets[0] = 0;        // 初始化第一个偏移为0
  for (int i = 0; i < V; ++i)
  {
    // 计算每个顶点的偏移量
    // vertex_offsets[i]表示顶点i的边在edges中的起始索引, vertex_offsets[i + 1]表示顶点i的边在edges中的结束索引
    vertex_offsets[i + 1] = vertex_offsets[i] + (int)adj[i].size();
  }

  // 将邻接表转换为边列表, 便于后续通过顶点偏移量（vertex_offsets）快速定位每个顶点的所有出边，实现高效的图遍历和并行处理。
  // edges数组存储所有边, edges数组的大小为E
  edges.clear();
  edges.reserve(E);
  for (int i = 0; i < V; ++i)
  {
    for (auto &e : adj[i])
    {
      edges.push_back(e);
    }
  }
}

// CPU单线程Dijkstra
void cpuDijkstra(int V, int source, const std::vector<int> &vertex_offsets,
                 const std::vector<Edge> &edges, std::vector<float> &dist, std::vector<int> &prev)
{
  dist.assign(V, INF); // 初始化距离数组为无穷大
  prev.assign(V, -1);  // 初始化前驱节点数组
  dist[source] = 0.0f; // 源节点距离为0

  // 优先队列，存储距离和节点索引
  using P = std::pair<float, int>;
  std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
  pq.emplace(0.0f, source);

  // Dijkstra算法主循环
  while (!pq.empty())
  {
    auto pair = pq.top(); // 获取队列顶部元素
    pq.pop();             // 弹出队列顶部元素

    // 获取当前节点的距离和索引
    float d = pair.first;
    int u = pair.second;

    // 如果当前距离大于已知最小距离，则跳过
    if (d > dist[u])
      continue;

    // 遍历当前节点的所有出边
    for (int i = vertex_offsets[u]; i < vertex_offsets[u + 1]; ++i)
    {
      int v = edges[i].to;       // 获取边的目标节点
      float w = edges[i].weight; // 获取边的权重
      float nd = d + w;          // 计算新距离

      // 如果新距离小于已知最小距离，则更新
      if (nd < dist[v])
      {
        dist[v] = nd;
        prev[v] = u;
        pq.emplace(nd, v);
      }
    }
  }
}

// CUDA atomicMin for float
__device__ float atomicMinFloat(float *addr, float value)
{
  // 使用原子比较交换实现原子最小操作
  int *intAddr = (int *)addr;
  int old = *intAddr, assumed;

  do
  {
    // 获取当前值
    assumed = old;

    // 如果当前值小于等于新值，则不需要更新
    if (__int_as_float(assumed) <= value)
      break;

    // 尝试将新值存入地址
    old = atomicCAS(intAddr, assumed, __float_as_int(value));
  } while (assumed != old);

  // 返回最小值
  return __int_as_float(old);
}

// CUDA kernel
__global__ void relaxEdges(int *vertex_offsets, Edge *edges, float *dist,
                           char *changed_cur, char *changed_next,
                           int *prev, int V)
{
  // 每个线程处理一个顶点
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  // 检查顶点索引是否在范围内, 检查节点是否为活跃节点
  if (u >= V || !changed_cur[u])
    return;

  int start = vertex_offsets[u];   // 获取当前顶点的起始边索引
  int end = vertex_offsets[u + 1]; // 获取当前顶点的结束边索引

  // 遍历当前顶点的所有出边
  for (int i = start; i < end; ++i)
  {
    int v = edges[i].to;
    float w = edges[i].weight;
    float new_dist = dist[u] + w;

    // 快速路径过滤
    float old = atomicMinFloat(&dist[v], new_dist);

    // 如果新距离小于旧距离，则更新前驱节点
    if (new_dist < old)
    {
      changed_next[v] = 1;
      prev[v] = u;
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    printf("Usage: %s <V><E><blockSize>\n", argv[0]);
    return 1;
  }

  const int V = atoi(argv[1]);   // 节点数
  const int E = atoi(argv[2]);   // 边数
  int blockSize = atoi(argv[3]); // CUDA线程块大小

  if (blockSize <= 0 || V <= 0 || E <= 0)
  {
    printf("Invalid input.\n");
    return 1;
  }

  std::vector<int> vertex_offsets; // 顶点偏移数组
  std::vector<Edge> edges;         // 边列表

  // 生成随机有向图
  printf("Generating random graph with %d nodes, %d edges...\n", V, E);
  generateRandomGraph(V, E, vertex_offsets, edges);
  printf("Graph generated.\n");

  // CPU计算准备
  std::vector<float> cpu_dist; // CPU距离数组
  std::vector<int> cpu_prev;   // CPU前驱数组

  // 运行CPU Dijkstra算法并计时
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpuDijkstra(V, 0, vertex_offsets, edges, cpu_dist, cpu_prev);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  // 计算CPU执行时间, 使用std::chrono计算时间差, std::chrono::duration<double, std::milli>用于毫秒级别的时间计算
  double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  // 打印CPU执行时间
  printf("CPU execution time: %.3f ms\n", cpu_ms);

  // 基于CUDA的Dijkstra算法实现
  // CUDA内存分配
  float *d_dist;                        // CUDA距离数组
  Edge *d_edges;                        // CUDA边列表
  int *d_vertex_offsets, *d_prev;       // CUDA顶点偏移数组和前驱数组
  char *d_changed_cur, *d_changed_next; // CUDA活跃节点缓冲区

  // 分配CUDA内存
  cudaMalloc(&d_vertex_offsets, sizeof(int) * (V + 1));
  cudaMalloc(&d_edges, sizeof(Edge) * E);
  cudaMalloc(&d_dist, sizeof(float) * V);
  cudaMalloc(&d_changed_cur, sizeof(char) * V);
  cudaMalloc(&d_changed_next, sizeof(char) * V);
  cudaMalloc(&d_prev, sizeof(int) * V);

  float *h_dist = new float[V];
  char *h_changed_cur = new char[V];
  char *h_changed_next = new char[V];
  int *h_prev = new int[V];

  // 初始化数据
  for (int i = 0; i < V; ++i)
  {
    h_dist[i] = INF;      // 初始化距离为无穷大
    h_changed_cur[i] = 0; // 当前活跃节点标志
    h_prev[i] = -1;       // 前驱节点初始化为-1
  }
  h_dist[0] = 0.0f;     // 源节点距离为0
  h_changed_cur[0] = 1; // 源节点标记为活跃

  cudaMemcpy(d_vertex_offsets, vertex_offsets.data(), sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_edges, edges.data(), sizeof(Edge) * E, cudaMemcpyHostToDevice);
  cudaMemcpy(d_dist, h_dist, sizeof(float) * V, cudaMemcpyHostToDevice);
  cudaMemcpy(d_changed_cur, h_changed_cur, sizeof(char) * V, cudaMemcpyHostToDevice);
  cudaMemset(d_changed_next, 0, sizeof(char) * V);
  cudaMemcpy(d_prev, h_prev, sizeof(int) * V, cudaMemcpyHostToDevice);

  // CUDA kernel配置
  int gridSize = (V + blockSize - 1) / blockSize;

  // CUDA计时开始
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  bool cont = true; // 是否继续迭代
  while (cont)
  {
    // 重置changed_next缓冲区
    cudaMemset(d_changed_next, 0, sizeof(char) * V);

    // 启动CUDA kernel
    relaxEdges<<<gridSize, blockSize>>>(d_vertex_offsets, d_edges, d_dist,
                                        d_changed_cur, d_changed_next,
                                        d_prev, V);

    // 确保kernel执行完成
    cudaDeviceSynchronize();

    // 拷贝changed_next到主机
    cudaMemcpy(h_changed_next, d_changed_next, sizeof(char) * V, cudaMemcpyDeviceToHost);

    cont = false;

    // 检查是否有节点发生变化, 如果有节点在changed_next中被标记为活跃，继续迭代
    for (int i = 0; i < V; ++i)
    {
      if (h_changed_next[i])
      {
        cont = true;
        break;
      }
    }

    // 交换changed缓冲区指针
    char *tmp = d_changed_cur;
    d_changed_cur = d_changed_next;
    d_changed_next = tmp;
  }

  // CUDA计时结束
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // 计算CUDA执行时间
  float cuda_ms = 0;
  cudaEventElapsedTime(&cuda_ms, start, stop);

  // 拷贝算法结果回主机
  cudaMemcpy(h_dist, d_dist, sizeof(float) * V, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_prev, d_prev, sizeof(int) * V, cudaMemcpyDeviceToHost);

  // 打印CUDA的结果
  printf("CUDA execution time: %.3f ms\n", cuda_ms);
  printf("Speedup (CPU / CUDA): %.2fx\n\n", cpu_ms / cuda_ms);

  /*
  // 打印部分节点距离和路径，避免过长输出，限制前20个节点
  int print_limit = 20;
  printf("Distances from source 0 (CUDA) for first %d nodes:\n", print_limit);
  for (int i = 0; i < print_limit && i < V; ++i)
  {
    if (h_dist[i] == INF)
      printf("Node %d: unreachable\n", i);
    else
      printf("Node %d: %.2f\n", i, h_dist[i]);
  }

  printf("\nPaths from source 0 (CUDA) for first %d nodes:\n", print_limit);
  for (int i = 0; i < print_limit && i < V; ++i)
  {
    if (h_dist[i] == INF)
    {
      printf("Node %d: unreachable\n", i);
      continue;
    }
    int path[100], len = 0, cur = i;
    while (cur != -1 && len < 100)
    {
      path[len++] = cur;
      cur = h_prev[cur];
    }
    printf("Node %d: ", i);
    for (int j = len - 1; j >= 0; --j)
    {
      printf("%d", path[j]);
      if (j != 0)
        printf(" -> ");
    }
    printf(" (%.2f)\n", h_dist[i]);
  }
  */

  // 验证代码
  int cuda_dist_mismatch = 0; // 距离不匹配计数
  int cuda_prev_mismatch = 0; // 前驱不匹配计数
  float epsilon = 1e-3f;      // 误差容忍度

  for (int i = 0; i < V; ++i)
  {
    if (fabs(cpu_dist[i] - h_dist[i]) > epsilon)
      cuda_dist_mismatch++;

    if (cpu_prev[i] != h_prev[i])
      cuda_prev_mismatch++;
  }

  if (cuda_dist_mismatch == 0 && cuda_prev_mismatch == 0)
  {
    printf("Validation passed: CPU and CUDA results match!\n");
  }
  else
  {
    printf("Validation failed:\n");
    printf("Distance mismatches: %d nodes\n", cuda_dist_mismatch);
    printf("Predecessor mismatches: %d nodes\n", cuda_prev_mismatch);
  }

  // 释放资源
  delete[] h_dist;
  delete[] h_changed_cur;
  delete[] h_changed_next;
  delete[] h_prev;
  cudaFree(d_vertex_offsets);
  cudaFree(d_edges);
  cudaFree(d_dist);
  cudaFree(d_changed_cur);
  cudaFree(d_changed_next);
  cudaFree(d_prev);

  return 0;
}
