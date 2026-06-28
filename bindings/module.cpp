#include <pybind11/pybind11.h>

#include "bindings.h"

namespace py = pybind11;

#ifndef SIMPLEX_PROJECT_VERSION
#    define SIMPLEX_PROJECT_VERSION "unknown"
#endif

#ifndef SIMPLEX_GIT_DESCRIBE
#    define SIMPLEX_GIT_DESCRIBE "unknown"
#endif

#ifndef SIMPLEX_GIT_BRANCH
#    define SIMPLEX_GIT_BRANCH "unknown"
#endif

PYBIND11_MODULE(simplinho, m) {
    m.doc() = "Bindings for the revised simplex solver";
    m.attr("__version__") = SIMPLEX_PROJECT_VERSION;
    m.attr("__git_describe__") = SIMPLEX_GIT_DESCRIBE;
    m.attr("__git_branch__") = SIMPLEX_GIT_BRANCH;

#ifdef SIMPLEX_ENABLE_BNB
    bind_bnb_bindings(m);
#endif
#ifdef SIMPLEX_ENABLE_IPM
    bind_ipm_bindings(m);
#endif
    bind_simplex_bindings(m);
    bind_model_bindings(m);

    m.attr("SimplexModel") = m.attr("Model");
}
