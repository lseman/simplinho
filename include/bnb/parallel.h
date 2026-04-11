#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <exec/async_scope.hpp>
#include <exec/finally.hpp>
#include <exec/static_thread_pool.hpp>
#include <functional>
#include <mutex>
#include <stdexec/execution.hpp>
#include <utility>
namespace simplex::bnb::detail {
class ParallelDispatcher {
  public:
    explicit ParallelDispatcher(int worker_count)
        : worker_count_(std::max(1, worker_count)),
          thread_pool_(static_cast<std::uint32_t>(worker_count_)) {}

    int worker_count() const { return worker_count_; }

    template <typename Task> void run(int task_count, Task&& task) {
        if (task_count <= 0) {
            return;
        }
        if (worker_count_ <= 1 || task_count == 1) {
            for (int task_index = 0; task_index < task_count; ++task_index) {
                std::invoke(std::forward<Task>(task), task_index);
            }
            return;
        }
        std::atomic<bool> saw_error{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;
        auto result =
            stdexec::bulk(stdexec::schedule(thread_pool_.get_scheduler()), STDEXEC::par,
                          task_count,
                          [task = std::forward<Task>(task), &saw_error, &first_error,
                           &error_mutex](int task_index) mutable {
                              if (saw_error.load(std::memory_order_acquire)) {
                                  return;
                              }
                              try {
                                  std::invoke(task, task_index);
                              } catch (...) {
                                  if (!saw_error.exchange(true, std::memory_order_acq_rel)) {
                                      std::lock_guard<std::mutex> lock(error_mutex);
                                      first_error = std::current_exception();
                                  }
                              }
                          });
        stdexec::sync_wait(std::move(result));
        if (saw_error.load(std::memory_order_relaxed)) {
            std::rethrow_exception(first_error);
        }
    }

  private:
    int worker_count_ = 1;
    exec::static_thread_pool thread_pool_;
};

class AsyncTaskDispatcher {
  public:
    explicit AsyncTaskDispatcher(int worker_count)
        : worker_count_(std::max(1, worker_count)),
          thread_pool_(static_cast<std::uint32_t>(worker_count_)) {}

    int worker_count() const noexcept { return worker_count_; }

    int pending_tasks() const noexcept { return pending_tasks_.load(std::memory_order_acquire); }

    bool stop_requested() const noexcept {
        return stop_requested_.load(std::memory_order_acquire);
    }

    void request_stop() noexcept {
        stop_requested_.store(true, std::memory_order_release);
        scope_.request_stop();
        thread_pool_.request_stop();
    }

    template <typename Rep, typename Period>
    bool wait_for_all(std::chrono::duration<Rep, Period> timeout) {
        std::unique_lock<std::mutex> lock(done_mutex_);
        return done_cv_.wait_for(lock, timeout, [&] {
            return pending_tasks_.load(std::memory_order_acquire) == 0;
        });
    }

    template <typename Task> void spawn(Task&& task) {
        if (stop_requested()) {
            return;
        }
        pending_tasks_.fetch_add(1, std::memory_order_acq_rel);
        try {
            auto work =
                stdexec::starts_on(thread_pool_.get_scheduler(),
                                   stdexec::just() |
                                       stdexec::then([task = std::forward<Task>(task)]() mutable {
                                           std::invoke(task);
                                       }) |
                                       stdexec::upon_error([](auto&&) noexcept {}) |
                                       stdexec::upon_stopped([]() noexcept {}));
            auto finalize = stdexec::just() |
                            stdexec::then([this]() noexcept { finish_task_(); });
            scope_.spawn(exec::finally(std::move(work), std::move(finalize)));
        } catch (...) {
            finish_task_();
            throw;
        }
    }

  private:
    void finish_task_() noexcept {
        if (pending_tasks_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(done_mutex_);
            done_cv_.notify_all();
        }
    }

    int worker_count_ = 1;
    exec::async_scope scope_;
    exec::static_thread_pool thread_pool_;
    std::atomic<int> pending_tasks_ = 0;
    std::atomic<bool> stop_requested_ = false;
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
};
} // namespace simplex::bnb::detail
