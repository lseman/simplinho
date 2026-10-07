#ifdef SIMPLEX_ENABLE_BNB

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "bnb/core/core.h"
#include "bnb/search/callback_telemetry.h"

namespace py = pybind11;
namespace simplex_bnb = simplex::bnb;

// ── Bindings for CallbackTelemetry ──

void bind_telemetry(py::module_& m) {
    py::class_<simplex_bnb::CallbackTelemetry>(m, "BranchingTelemetry")
        .def(py::init<>())
        .def_readonly("call_count", &simplex_bnb::CallbackTelemetry::call_count)
        .def_readonly("success_count", &simplex_bnb::CallbackTelemetry::success_count)
        .def_readonly("fallback_count", &simplex_bnb::CallbackTelemetry::fallback_count)
        .def_readonly("exception_count", &simplex_bnb::CallbackTelemetry::exception_count)
        .def_readonly("timeout_count", &simplex_bnb::CallbackTelemetry::timeout_count)
        .def_readonly("total_wall_ns", &simplex_bnb::CallbackTelemetry::total_wall_ns)
        .def_readonly("max_wall_ns", &simplex_bnb::CallbackTelemetry::max_wall_ns)
        .def_readonly("min_wall_ns", &simplex_bnb::CallbackTelemetry::min_wall_ns)
        .def_readwrite("timeout_ms", &simplex_bnb::CallbackTelemetry::timeout_ms)
        .def("summary", &simplex_bnb::CallbackTelemetry::summary)
        .def("__repr__", [](const simplex_bnb::CallbackTelemetry& t) {
            return "<BranchingTelemetry: " + t.summary() + ">";
        });
}

void bind_bnb_branching_policy(py::module_& m) {
    bind_telemetry(m);
}

#endif // SIMPLEX_ENABLE_BNB
