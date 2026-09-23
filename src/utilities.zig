const std = @import("std");
const bit_word = @import("bit_word.zig");
const WordType = bit_word.WordType;

/// Largest representable u64, used as an unbounded limit for iteration cursors.
pub const MaxInt64: u64 = std.math.maxInt(u64);
/// Default growable array, used as the base for aligned word storage.
pub const List = std.ArrayList;

/// Cache-aligned word list that keeps bit planes on 64-byte boundaries.
/// - `T` - element type stored in the list.
///
/// Return - aligned list type with reduced false sharing on scans.
pub fn ListA64(comptime T: type) type {
    return std.array_list.Aligned(T, std.mem.Alignment.@"64");
}

/// Optional per-bit visitor with a regular call convention.
/// - `Context` - caller-provided iteration context.
///
/// Return - nullable callback type returning false to stop the walk.
pub fn IteratorCallback(Context: type) type {
    return ?fn (context: Context, bit_id: u32) bool;
}

/// Optional per-bit visitor forced inline for hot iteration paths.
/// - `Context` - caller-provided iteration context.
///
/// Return - nullable inline callback type returning false to stop the walk.
pub fn InlineIteratorCallback(Context: type) type {
    return ?fn (context: Context, bit_id: u32) callconv(.@"inline") bool;
}

/// Visits every set bit of one word in ascending order.
/// - `wt` - word width that defines the backing integer type.
/// - `Context` - caller-provided iteration context.
/// - `callback` - invoked once per set bit, false stops the walk.
/// - `context` - value forwarded to each callback invocation.
/// - `start_bit_id` - global id of bit zero of the word.
/// - `word` - word whose set bits are visited.
///
/// Return - false when the callback requested early exit, true otherwise.
pub inline fn iterateActiveBitsInWord(
    comptime wt: WordType,
    comptime Context: type,
    comptime callback: fn (context: Context, bit_id: u32) bool,
    context: Context,
    start_bit_id: u32,
    word: wt.Type(),
) bool {
    var w = word;
    while (w != 0) {
        const i: u32 = @ctz(w);
        if (!callback(context, start_bit_id + i)) return false;
        w &= w - 1;
    }
    return true;
}

/// Visits every set bit of one word with an inline callback for speed.
/// - `wt` - word width that defines the backing integer type.
/// - `Context` - caller-provided iteration context.
/// - `callback` - inline visitor invoked per set bit, false stops the walk.
/// - `context` - value forwarded to each callback invocation.
/// - `start_bit_id` - global id of bit zero of the word.
/// - `word` - word whose set bits are visited.
///
/// Return - false when the callback requested early exit, true otherwise.
pub inline fn iterateActiveBitsInWordInline(
    comptime wt: WordType,
    comptime Context: type,
    comptime callback: fn (context: Context, bit_id: u32) callconv(.@"inline") bool,
    context: Context,
    start_bit_id: u32,
    word: wt.Type(),
) bool {
    var w = word;
    while (w != 0) {
        const i: u32 = @ctz(w);
        if (!callback(context, start_bit_id + i)) return false;
        w &= w - 1;
    }
    return true;
}

/// Walk order shared by every iterator in bitset, layer and tree.
/// - `forward` - visits bit ids ascending, low words and low bits first.
/// - `backward` - visits bit ids descending, high words and high bits first.
pub const Direction = enum {
    /// Ascending visit order.
    forward,
    /// Descending visit order.
    backward,
};

/// Resolved half-open bit interval shared by every ranged iteration.
/// - `lo` - first visited bit id, inclusive.
/// - `hi` - one past the last visited bit id, exclusive.
pub const ResolvedRange = struct {
    /// First visited bit id, inclusive.
    lo: u32,
    /// One past the last visited bit id, exclusive.
    hi: u32,
};

/// Normalizes nullable range bounds into a clamped half-open interval.
/// Argument order does not matter: the smaller value becomes the start.
/// - `total` - valid bits in the scanned container, clamps both bounds.
/// - `start_bit` - range edge or null for no lower limit.
/// - `end_bit` - range edge or null for no upper limit.
///
/// Return - clamped `[lo, hi)` with `lo <= hi`; empty when `lo == hi`.
pub inline fn resolveRange(total: u32, start_bit: ?u32, end_bit: ?u32) ResolvedRange {
    const a = start_bit orelse 0;
    const b = end_bit orelse total;
    var lo = @min(a, b);
    var hi = @max(a, b);
    if (lo > total) lo = total;
    if (hi > total) hi = total;
    return .{ .lo = lo, .hi = hi };
}

/// Compact address of a contiguous bit span inside one pyramid level.
pub const BitRange = struct {
    /// First bit of the span, used to seed range scans.
    start: u32,
    /// Span length in bits, bounds the scan without extra branches.
    len: u32,
    /// Pyramid level the span belongs to, selects the backing plane.
    layer: u32,
};

/// Single-bit logical value shared by bitsets, layers and trees.
pub const BitState = enum(u1) {
    const Self = @This();

    /// Cleared bit, skipped by active-only iteration.
    inactive = 0,
    /// Set bit, visited by active-only iteration.
    active = 1,

    /// Expands one logical state into a fully filled backing word.
    /// - `self` - state to replicate across all word bits.
    /// - `wt` - word width that defines the result integer type.
    ///
    /// Return - word with all bits cleared or all bits set.
    pub inline fn toWordState(self: Self, comptime wt: WordType) wt.Type() {
        return 0 -% @as(wt.Type(), @intFromEnum(self));
    }

    /// Converts the state into a plain boolean for branch conditions.
    /// - `self` - state to test for activity.
    ///
    /// Return - true for active, false for inactive.
    pub inline fn asBool(self: Self) bool {
        return self == .active;
    }
};
