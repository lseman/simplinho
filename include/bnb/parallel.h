#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <utility>

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

namespace simplex::bnb::detail {

class ParallelDispatcher {
   public:
    explicit ParallelDispatcher(int worker_count)
        : worker_count_(std::max(1, worker_count)),
          pool_(static_cast<std::size_t>(worker_count_)) {}

    int worker_count() const { return worker_count_; }

    template <typename Task>
    void run(int task_count, Task&& task) {
        if (task_count <= 0) {
            return;
        }
        if (worker_count_ <= 1 || task_count == 1) {
            for (int task_index = 0; task_index < task_count; ++task_index) {
                std::invoke(task, task_index);
            }
            return;
        }

        std::atomic<bool> saw_error{false};
        std::exception_ptr first_error;
        std::mutex error_mutex;
        auto sender =
            stdexec::schedule(pool_.get_scheduler()) |
            stdexec::bulk(stdexec::par_unseq, static_cast<std::size_t>(task_count),
                          [&](std::size_t task_index) noexcept {
                              if (saw_error.load(std::memory_order_relaxed)) {
                                  return;
                              }
                              try {
                                  std::invoke(task, static_cast<int>(task_index));
                              } catch (...) {
                                  if (!saw_error.exchange(true, std::memory_order_acq_rel)) {
                                      std::lock_guard<std::mutex> lock(error_mutex);
                                      first_error = std::current_exception();
                                  }
                              }
                          });
        (void)stdexec::sync_wait(std::move(sender));
        if (first_error) {
            std::rethrow_exception(first_error);
        }
    }

   private:
    int worker_count_ = 1;
    exec::static_thread_pool pool_;
};

}  // namespace simplex::bnb::detail
