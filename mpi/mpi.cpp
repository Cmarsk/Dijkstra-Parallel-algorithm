#include <cstdio>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <omp.h>
#include <limits>
#include <mutex>
#include "mpi.h"

#define INF 1e20f
#define ROOT 0

struct Edge
{
  int to;
  float weight;
};

struct ReduceData
{
  float dist;
  int updated;
};

void generateRandomGraph(int V, int E, std::vector<int> &vertex_offsets, std::vector<Edge> &edges)
{
  std::vector<std::vector<Edge>> adj(V);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> node_dist(0, V - 1);
  std::uniform_real_distribution<float> weight_dist(1.0f, 20.0f);

  for (int u = 0; u < V; ++u)
  {
    int v = node_dist(gen);
    while (v == u)
      v = node_dist(gen);
    adj[u].push_back({v, weight_dist(gen)});
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
    adj[u].push_back({v, weight_dist(gen)});
    edges_added++;
  }

  vertex_offsets.resize(V + 1);
  vertex_offsets[0] = 0;
  for (int i = 0; i < V; ++i)
    vertex_offsets[i + 1] = vertex_offsets[i] + adj[i].size();

  edges.clear();
  for (auto &list : adj)
    edges.insert(edges.end(), list.begin(), list.end());
}

void serialBellmanFord(int V, int source, const std::vector<int> &vertex_offsets,
                       const std::vector<Edge> &edges, std::vector<float> &dist)
{
  dist.assign(V, INF);
  dist[source] = 0.0f;

  std::vector<int> updated(V, 0);
  std::vector<int> new_updated(V, 0);
  updated[source] = 1;

  bool has_change = true;
  while (has_change)
  {
    has_change = false;
    std::fill(new_updated.begin(), new_updated.end(), 0);

    // 遍历所有被标记需要更新的顶点
    for (int u = 0; u < V; ++u)
    {
      if (!updated[u])
        continue;

      // 松弛该顶点的所有出边
      for (int i = vertex_offsets[u]; i < vertex_offsets[u + 1]; ++i)
      {
        int v = edges[i].to;
        float w = edges[i].weight;

        if (dist[u] + w < dist[v])
        {
          dist[v] = dist[u] + w;
          new_updated[v] = 1;
          has_change = true;
        }
      }
    }

    std::swap(updated, new_updated); // 交换更新标记
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

void my_reduce_op(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype)
{
  ReduceData *in = (ReduceData *)invec;
  ReduceData *inout = (ReduceData *)inoutvec;
  for (int i = 0; i < *len; ++i)
  {
    if (in[i].dist < inout[i].dist)
    {
      inout[i].dist = in[i].dist;
    }
    inout[i].updated |= in[i].updated;
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (argc < 3)
  {
    if (world_rank == 0)
    {
      fprintf(stderr, "Usage: %s <V> <E>\n", argv[0]);
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // 从命令行读取 V 和 E
  int V = atoi(argv[1]);
  int E = atoi(argv[2]);

  std::vector<int> vertex_offsets;
  std::vector<Edge> edges;
  if (world_rank == ROOT)
  {
    generateRandomGraph(V, E, vertex_offsets, edges);
  }

  // Broadcast graph data
  vertex_offsets.resize(V + 1);
  edges.resize(E);

  MPI_Bcast(vertex_offsets.data(), V + 1, MPI_INT, ROOT, MPI_COMM_WORLD);
  MPI_Bcast(edges.data(), E * sizeof(Edge), MPI_BYTE, ROOT, MPI_COMM_WORLD);

  // Initialize MPI data types
  MPI_Datatype reduce_data_type;
  int blocklengths[2] = {1, 1};
  MPI_Datatype types[2] = {MPI_FLOAT, MPI_INT};
  MPI_Aint offsets[2];
  offsets[0] = offsetof(ReduceData, dist);
  offsets[1] = offsetof(ReduceData, updated);
  MPI_Type_create_struct(2, blocklengths, offsets, types, &reduce_data_type);
  MPI_Type_commit(&reduce_data_type);

  MPI_Op my_op;
  MPI_Op_create(my_reduce_op, 1, &my_op);

  // Initialize variables
  std::vector<float> dist(V, INF);
  std::vector<int> updated(V, 0);

  int source = 0;
  dist[source] = 0.0f;
  updated[source] = 1;

  const int step = world_size;  // 步长等于进程总数
  const int start = world_rank; // 起始点等于进程号

  MPI_Barrier(MPI_COMM_WORLD);

  double comp_time = 0.0, comm_time = 0.0; // 新增计时变量
  bool global_change = true;

  double start_time = MPI_Wtime();
  while (global_change)
  {
    global_change = false;
    std::vector<ReduceData> local_reduce(V);
    for (int i = 0; i < V; ++i)
    {
      local_reduce[i].dist = dist[i];
      local_reduce[i].updated = 0;
    }

    // 处理循环分配的顶点
    double start_comp = MPI_Wtime();
    int processed = 0;
    for (int u = start; u < V; u += step)
    {
      if (!updated[u])
        continue;
      ++processed;
      for (int i = vertex_offsets[u]; i < vertex_offsets[u + 1]; ++i)
      {
        int v = edges[i].to;
        float w = edges[i].weight;
        float new_dist = dist[u] + w;
        if (new_dist < local_reduce[v].dist)
        {
          local_reduce[v].dist = new_dist;
          local_reduce[v].updated = 1;
        }
      }
    }
    comp_time += MPI_Wtime() - start_comp;
    // 调试输出：显示各进程处理顶点数
    // printf("Rank %d processed %d vertices\n", world_rank, processed);

    // Global reduction
    double start_comm = MPI_Wtime();
    std::vector<ReduceData> global_reduce(V);
    MPI_Allreduce(local_reduce.data(), global_reduce.data(), V, reduce_data_type, my_op, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - start_comm;

    // Update distances and flags
    start_comp = MPI_Wtime();
    std::vector<int> new_updated(V, 0);
    for (int i = 0; i < V; ++i)
    {
      if (global_reduce[i].dist < dist[i])
      {
        dist[i] = global_reduce[i].dist;
        new_updated[i] = 1;
        global_change = true;
      }
    }
    comp_time += MPI_Wtime() - start_comp;
    // Check global change
    int gc_flag = global_change ? 1 : 0;
    start_comm = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, &gc_flag, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - start_comm;
    global_change = (gc_flag != 0);
    std::swap(updated, new_updated);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();

  double max_comp = 0.0, max_comm = 0.0;
  MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  if (world_rank == ROOT)
  {
    // Verification and output results
    std::vector<float> serial_dist;
    std::vector<int> cpu_prev;
    auto t1 = std::chrono::high_resolution_clock::now();
    // serialBellmanFord(V, source, vertex_offsets, edges, serial_dist);
    cpuDijkstra(V, source, vertex_offsets, edges, serial_dist, cpu_prev);
    auto t2 = std::chrono::high_resolution_clock::now();
    double serial_time = std::chrono::duration<double>(t2 - t1).count();

    int correct = 1;
    for (int i = 0; i < V; ++i)
    {
      if (std::abs(dist[i] - serial_dist[i]) > 1e-3)
      {
        correct = 0;
        break;
      }
    }

    printf("\n[性能分析]\n");
    printf("进程数: %d\n", world_size);
    printf("点数：%d, 边数：%d\n", V, E);
    printf("正确性: %s\n", correct ? "Yes" : "No");
    printf("串行时间: %.4fs\n", serial_time);
    printf("并行时间: % .4fs\n", end_time - start_time);
    printf("加速比: %.2fx\n", serial_time / (end_time - start_time));
    printf("\n");

    printf("最长计算时间: %.4f秒\n", max_comp);
    printf("最长通信时间: %.4f秒\n", max_comm);
    printf("计算时间占比: %.2f%%\n", (max_comp / (max_comp + max_comm)) * 100);
    printf("通信时间占比: %.2f%%\n", (max_comm / (max_comp + max_comm)) * 100);
  }

  MPI_Type_free(&reduce_data_type);
  MPI_Op_free(&my_op);
  MPI_Finalize();
  return 0;
}

// Compile with: mpic++ -std=c++14 mpi.cpp -o mpi -fopenmp
// Run with: mpirun -np 4 ./mpi