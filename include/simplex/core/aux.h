#pragma once

// Shared strategy enum (keeps AdaptivePricerLike for compatibility)
struct AdaptivePricerLike {
    enum Strategy {
        STEEPEST_EDGE = 0,
        DEVEX = 1,
        PARTIAL_PRICING = 2,
        MOST_NEGATIVE = 3
    };
};

using PricingStrategy = AdaptivePricerLike::Strategy;

namespace dm_consts {
inline constexpr double kDegenerateAlphaTol = 1e-14;
inline constexpr double kRcStallTol = 1e-12;
inline constexpr double kObjStallTol = 1e-10;
inline constexpr int kStepHistCap = 32;
inline constexpr int kRepeatWinCap = 16;
inline constexpr int kRepeatMinCount = 3;
inline constexpr int kPerfWindow = 10;  // for adaptive switching
}  // namespace dm_consts
