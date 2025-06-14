#include <cstdio>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <omp.h>
#include <limits>
#include <atomic>

#define INF 1e20f

// 边结构体
struct Edge
{
  int to;
  float weight;
};

// 生成随机图
void generateRandomGraph(int V, int E, std::vector<int> &vertex_offsets, std::vector<Edge> &edges)
{
  std::vector<std::vector<Edge>> adj(V);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> node_dist(0, V - 1);
  std::uniform_real_distribution<float> weight_dist(1.0f, 20.0f);

  // 保证每节点至少一条出边
  for (int u = 0; u < V; ++u)
  {
    int v = node_dist(gen);
    while (v == u)
      v = node_dist(gen);
    float w = weight_dist(gen);
    adj[u].push_back({v, w});
  }

  int edges_added = V;
  while (edges_added < E)
  {
    int u = node_dist(gen);
    int v = node_dist(gen);
    if (u == v)
      continue;
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

    float w = weight_dist(gen);
    adj[u].push_back({v, w});
    edges_added++;
  }

  vertex_offsets.resize(V + 1);
  vertex_offsets[0] = 0;
  for (int i = 0; i < V; ++i)
  {
    vertex_offsets[i + 1] = vertex_offsets[i] + (int)adj[i].size();
  }

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
  dist.assign(V, INF);
  prev.assign(V, -1);
  dist[source] = 0.0f;

  using P = std::pair<float, int>;
  std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
  pq.emplace(0.0f, source);

  while (!pq.empty())
  {
    auto pair = pq.top();
    pq.pop();
    float d = pair.first;
    int u = pair.second;
    if (d > dist[u])
      continue;

    for (int i = vertex_offsets[u]; i < vertex_offsets[u + 1]; ++i)
    {
      int v = edges[i].to;
      float w = edges[i].weight;
      float nd = d + w;
      if (nd < dist[v])
      {
        dist[v] = nd;
        prev[v] = u;
        pq.emplace(nd, v);
      }
    }
  }
}

// OpenMP版本的Dijkstra算法
void cpuDijkstraOMP(
    int V, int source,
    const std::vector<int> &vertex_offsets,
    const std::vector<Edge> &edges,
    std::vector<float> &dist,
    std::vector<int> &prev)
{
  constexpr float EPSILON = 1e-5f;

  dist.assign(V, INF);
  prev.assign(V, -1);
  dist[source] = 0.0f;

  // 原子类型双缓冲活跃节点管理
  std::vector<std::atomic_char> active_current(V), active_next(V);
  active_current[source].store(1, std::memory_order_relaxed);

  bool has_active = true;

  // 无锁更新函数
  auto atomic_update = [&](int v, float expected, float desired, int u)
  {
    bool success = false;
    while (!success)
    {
      float current = dist[v];
      if (current < desired - EPSILON)
        break;

      success = std::atomic_compare_exchange_weak_explicit(
          reinterpret_cast<std::atomic<float> *>(&dist[v]),
          &current, desired,
          std::memory_order_relaxed,
          std::memory_order_relaxed);

      if (success)
      {
        prev[v] = u;
        active_next[v].store(1, std::memory_order_relaxed);
      }
    }
  };

  while (has_active)
  {
    has_active = false;

#pragma omp parallel reduction(|| : has_active)
    {
// 256是OpenMP动态调度的块大小
#pragma omp for schedule(dynamic, 256) nowait
      for (int u = 0; u < V; ++u)
      {
        if (!active_current[u].load(std::memory_order_relaxed))
          continue;

        const float du = dist[u];
        const int start = vertex_offsets[u];
        const int end = vertex_offsets[u + 1];

        for (int i = start; i < end; ++i)
        {
          const int v = edges[i].to;
          const float alt = du + edges[i].weight;

          // 快速路径过滤
          if (alt >= dist[v] - EPSILON)
            continue;

          atomic_update(v, dist[v], alt, u);
        }

        active_current[u].store(0, std::memory_order_relaxed);
      }

// 合并活跃标志
#pragma omp for schedule(static)
      for (int v = 0; v < V; ++v)
      {
        if (active_next[v].load(std::memory_order_relaxed))
        {
          has_active = true;
        }
      }
    }

    // 交换活跃集（无锁操作）
    std::swap(active_current, active_next);
    std::fill(active_next.begin(), active_next.end(), 0);
  }
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    printf("Usage: %s <V><E><numThreads>\n", argv[0]);
    return 1;
  }

  const int V = atoi(argv[1]);    // 节点数
  const int E = atoi(argv[2]);    // 边数
  int numThreads = atoi(argv[3]); // OpenMP线程数

  if (numThreads <= 0 || V <= 0 || E <= 0)
  {
    printf("Invalid input.\n");
    return 1;
  }

  if (numThreads <= 0)
  {
    printf("Invalid block size or number of threads.\n");
    return 1;
  }

  std::vector<int> vertex_offsets;
  std::vector<Edge> edges;

  // 生成随机图
  printf("Generating random graph with %d nodes, %d edges...\n", V, E);
  generateRandomGraph(V, E, vertex_offsets, edges);
  printf("Graph generated.\n");

  // CPU计算准备
  std::vector<float> cpu_dist;
  std::vector<int> cpu_prev;

  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpuDijkstra(V, 0, vertex_offsets, edges, cpu_dist, cpu_prev);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  printf("CPU execution time: %.3f ms\n", cpu_ms);

  // openmp版本的Dijkstra算法
  // 初始化
  std::vector<float> omp_dist(V, std::numeric_limits<float>::infinity());
  std::vector<int> omp_prev(V, -1);

  // 线程数
  omp_set_num_threads(numThreads);

  auto omp_start = std::chrono::high_resolution_clock::now();
  cpuDijkstraOMP(V, 0, vertex_offsets, edges, omp_dist, omp_prev);
  auto omp_end = std::chrono::high_resolution_clock::now();

  double omp_ms = std::chrono::duration<double, std::milli>(omp_end - omp_start).count();

  printf("\nOpenMP execution time: %.3f ms\n", omp_ms);
  printf("Speedup (CPU / OpenMP): %.2fx\n", cpu_ms / omp_ms);

  int omp_dist_mismatch = 0;
  int omp_prev_mismatch = 0;
  const float epsilon = 1e-5f;

  for (int i = 0; i < V; ++i)
  {
    if (fabs(omp_dist[i] - cpu_dist[i]) > epsilon)
      omp_dist_mismatch++;
    if (omp_prev[i] != cpu_prev[i])
      omp_prev_mismatch++;
  }

  if (omp_dist_mismatch == 0 && omp_prev_mismatch == 0)
  {
    printf("Validation passed: OpenMP and CPU results match!\n");
  }
  else
  {
    printf("Validation failed between CPU and OpenMP:\n");
    printf("Distance mismatches: %d nodes\n", omp_dist_mismatch);
    printf("Predecessor mismatches: %d nodes\n", omp_prev_mismatch);
  }

  return 0;
}