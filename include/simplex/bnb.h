#pragma once

// Compatibility shim: pulls in the branch-and-bound solver (bnb/core.h) and
// its MIP presolve helpers so callers can include a single "simplex/bnb.h".
#include "bnb/core.h"
#include "bnb/mip_presolve.h"
