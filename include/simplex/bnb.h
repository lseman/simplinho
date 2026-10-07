#pragma once

// Public BnB interface — re-exports the core types and solver.
// This header allows the modeling layer (model_bindings.cpp) to access
// BnB types when SIMPLEX_ENABLE_BNB is defined, without pulling in
// the full include/bnb/ directory structure.

#include "../bnb/types.h"
#include "../bnb/core/core.h"
#include "../bnb/core/manager.h"
#include "../bnb/search/branching.h"
#include "../bnb/search/branching_policy.h"
#include "../bnb/search/callback_telemetry.h"
#include "../bnb/search/default_branching_policy.h"
#include "../bnb/search/python_branching_policy.h"
#include "../bnb/cuts/cuts.h"
#include "../bnb/presolve/mip_presolve.h"
#include "../bnb/heuristics/diving.h"
#include "../bnb/heuristics/heuristic.h"
#include "../bnb/heuristics/async_heuristic_manager.h"
#include "../bnb/conflict/conflict_engine.h"
#include "../bnb/parallel/parallel.h"
