#include <pybind11/pybind11.h>

#include "bnb_bindings.h"

#ifndef SIMPLEX_PROJECT_VERSION
#    define SIMPLEX_PROJECT_VERSION "unknown"
#endif

namespace py = pybind11;

PYBIND11_MODULE(simplinho_bnb, module) {
    // Establish the dependency direction at the Python boundary too: BnB
    // imports the simplex extension, while simplex has no knowledge of BnB.
    module.attr("simplex") = py::module_::import("simplinho");
    module.doc() = "Branch-and-bound bindings built on top of simplinho";
    module.attr("__version__") = SIMPLEX_PROJECT_VERSION;
    bind_bnb_bindings(module);
}
