#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#    include <unistd.h>
#    include <cstdio>
#endif

#include "simplex/bnb.h"

#ifndef SIMPLEX_PROJECT_VERSION
#    define SIMPLEX_PROJECT_VERSION "unknown"
#endif

#ifndef SIMPLEX_GIT_DESCRIBE
#    define SIMPLEX_GIT_DESCRIBE "unknown"
#endif

#ifndef SIMPLEX_GIT_BRANCH
#    define SIMPLEX_GIT_BRANCH "unknown"
#endif

namespace simplinho::bindings {

inline bool log_colors_enabled_() {
    static const bool enabled = []() {
        if (std::getenv("NO_COLOR") != nullptr) {
            return false;
        }
        const char* term = std::getenv("TERM");
        if (term == nullptr || std::string_view(term) == "dumb") {
            return false;
        }
#if defined(__unix__) || defined(__APPLE__)
        return ::isatty(fileno(stdout)) != 0;
#else
        return false;
#endif
    }();
    return enabled;
}

inline std::string colorize_(std::string_view text, std::string_view ansi_code) {
    if (!log_colors_enabled_()) {
        return std::string(text);
    }
    std::string out;
    out.reserve(text.size() + ansi_code.size() + 10);
    out += "\033[";
    out += ansi_code;
    out += "m";
    out += text;
    out += "\033[0m";
    return out;
}

inline std::string accent_(std::string_view text) { return colorize_(text, "38;5;39"); }
inline std::string bold_(std::string_view text) { return colorize_(text, "1"); }
inline std::string dim_(std::string_view text) { return colorize_(text, "2"); }
inline std::string good_(std::string_view text) { return colorize_(text, "38;5;40"); }
inline std::string warn_(std::string_view text) { return colorize_(text, "38;5;214"); }
inline std::string cyan_(std::string_view text) { return colorize_(text, "38;5;51"); }
inline std::string green_(std::string_view text) { return colorize_(text, "38;5;82"); }
inline std::string yellow_(std::string_view text) { return colorize_(text, "38;5;226"); }
inline std::string magenta_(std::string_view text) { return colorize_(text, "38;5;213"); }

// Width = 79 chars (Gurobi-standard terminal width)
inline constexpr int kLogWidth = 79;
inline std::string rule_(char ch = '=') { return std::string(kLogWidth, ch); }
inline std::string thin_rule_() { return dim_(std::string(kLogWidth, '-')); }

inline void print_verbose_solver_banner() {
    std::cout << dim_(rule_()) << "\n";
    // Center "Simplinho" with version and git info
    std::ostringstream title;
    title << "Simplinho " << SIMPLEX_PROJECT_VERSION;
    const std::string title_str = title.str();
    const int pad = std::max(0, (kLogWidth - static_cast<int>(title_str.size())) / 2);
    std::cout << std::string(pad, ' ') << bold_(title_str) << "\n";
    std::ostringstream git_line;
    git_line << "git " << SIMPLEX_GIT_DESCRIBE << "  branch:" << SIMPLEX_GIT_BRANCH;
    const std::string git_str = git_line.str();
    const int git_pad = std::max(0, (kLogWidth - static_cast<int>(git_str.size())) / 2);
    std::cout << dim_(std::string(git_pad, ' ') + git_str) << "\n";
    std::cout << dim_(rule_()) << "\n\n";
}

inline std::string feature_token_(std::string_view name, bool enabled) {
    return enabled ? good_(name) : dim_(name);
}

template <typename Features> std::string join_feature_tokens_(const Features& features) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& [name, enabled] : features) {
        if (!first) {
            oss << "  ";
        }
        first = false;
        oss << feature_token_(name, enabled);
    }
    return oss.str();
}

inline void print_verbose_solver_configuration(const simplex::bnb::Options& options) {
    namespace simplex_bnb = simplex::bnb;
    using DivingStrategy = simplex_bnb::DivingStrategy;

    std::cout << accent_("Search") << " " << dim_("|") << " node "
              << cyan_(simplex_bnb::to_string(options.node_selection)) << "  branch "
              << cyan_(simplex_bnb::to_string(options.branching_strategy)) << "  dive "
              << cyan_(simplex_bnb::to_string(options.diving_strategy)) << "  workers "
              << cyan_(std::to_string(options.parallel_workers)) << "  async "
              << feature_token_("heuristics", options.use_async_heuristics) << "\n";

    const std::vector<std::pair<std::string_view, bool>> heuristics = {
        {"rounding", options.use_rounding},
        {"diving", options.use_diving && options.diving_strategy != DivingStrategy::Disabled},
        {"feas-jump", options.use_feasibility_jump},
        {"feas-pump", options.use_feasibility_pump},
        {"RENS", options.use_rens},
        {"RINS", options.use_rins},
        {"local-search", options.use_local_search},
        {"local-branch", options.use_local_branching},
    };
    std::cout << accent_("Heur") << "   " << dim_("|") << " " << join_feature_tokens_(heuristics)
              << "\n";

    const std::vector<std::pair<std::string_view, bool>> cuts = {
        {"pool", options.use_cut_pool},
        {"gomory", options.use_gomory_cuts},
        {"mir", options.use_mir_cuts},
        {"cover", options.use_cover_cuts},
        {"zero-half", options.use_zero_half_cuts},
        {"impl-bound", options.use_implied_bound_cuts},
        {"clique", options.use_clique_cuts},
        {"odd-cycle", options.use_odd_cycle_cuts},
        {"probing", options.use_probing_implications},
        {"conflict", options.use_conflict_cuts},
        {"dual-proof", options.use_dual_proof_cuts},
    };
    std::cout << accent_("Cuts") << "   " << dim_("|") << " " << join_feature_tokens_(cuts) << "\n";
    std::cout << "\n" << thin_rule_() << "\n";
}

}  // namespace simplinho::bindings
