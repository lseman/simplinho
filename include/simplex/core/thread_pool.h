#pragma once

// Lightweight thread pool for simplex pricing parallelism.
// Reuses OS threads across calls — no per-call thread creation.
// Uses a single work queue + condition variable. No work-stealing.
//
// Usage:
//   ThreadPool& pool = ThreadPool::instance();
//   pool.submit(5, [&job_data](int tid) { /* worker body */ });

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
  public:
    // Singleton — one pool per process, lazily created.
    static ThreadPool& instance() {
        static ThreadPool pool(std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
        return pool;
    }

    // Number of worker threads.
    int thread_count() const noexcept { return static_cast<int>(workers_.size()); }

    // Submit `task_count` tasks. `worker` is called once per task with [0, task_count).
    // All tasks complete before this function returns (synchronous barrier).
    // Thread-safe: can be called from multiple threads concurrently.
    void submit(int task_count, std::function<void(int)> worker) {
        if (task_count <= 0)
            return;
        if (task_count == 1) {
            worker(0);
            return;
        }

        // Partition tasks into buckets assigned to specific workers.
        std::vector<std::vector<int>> buckets(thread_count());
        for (int i = 0; i < task_count; ++i) {
            buckets[i % thread_count()].push_back(i);
        }

        // Launch one task per worker that has work.
        std::atomic<int> done{0};
        int active = 0;
        for (int t = 0; t < thread_count(); ++t) {
            if (buckets[t].empty())
                continue;
            ++active;
            workers_[t].push([this, bucket = std::move(buckets[t]), &worker, &done, &active]() {
                for (int idx : bucket)
                    worker(idx);
                if (done.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    barrier_cv_.notify_all();
            });
        }

        // If only one worker had tasks, it ran inline above — done already reached 0.
        if (active <= 1 && done.load(std::memory_order_acquire) == 0)
            return;

        // Wait for all workers to finish.
        std::unique_lock<std::mutex> lock(barrier_mutex_);
        barrier_cv_.wait(lock, [&] { return done.load(std::memory_order_acquire) == 0; });
    }

    // Schedule a single free-standing task on the next available worker.
    void enqueue(std::function<void()> fn) {
        int idx = next_worker_.fetch_add(1, std::memory_order_relaxed) % thread_count();
        workers_[idx].push(std::move(fn));
        workers_[idx].notify_one();
    }

  private:
    explicit ThreadPool(int threads)
        : workers_(threads), stop_(false) {
        for (int i = 0; i < threads; ++i) {
            thread_.emplace_back([this, i]() { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(shutdown_mutex_);
            stop_ = true;
        }
        for (auto& wh : workers_)
            wh.cv.notify_one();
        for (auto& t : thread_)
            if (t.joinable())
                t.join();
    }

    void worker_loop(int id) {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(workers_[id].mtx);
                workers_[id].cv.wait(lock, [this, id] { return stop_ || !workers_[id].q.empty(); });
                if (stop_ && workers_[id].q.empty())
                    return;
                task = std::move(workers_[id].q.front());
                workers_[id].q.pop();
            }
            task();
        }
    }

    struct WorkerHandle {
        std::queue<std::function<void()>> q;
        std::mutex mtx;
        std::condition_variable cv;
        void push(std::function<void()> fn) {
            std::lock_guard<std::mutex> lock(mtx);
            q.push(std::move(fn));
        }
        void notify_one() { cv.notify_one(); }
    };

    std::vector<WorkerHandle> workers_;
    std::vector<std::thread> thread_;
    std::atomic<bool> stop_;
    std::mutex shutdown_mutex_;

    // Barrier synchronization for submit() calls.
    std::mutex barrier_mutex_;
    std::condition_variable barrier_cv_;
    std::atomic<int> next_worker_{0};
};
