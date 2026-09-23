const std = @import("std");
const math = std.math;
const utilities = @import("utilities.zig");
const bit_word = @import("bit_word.zig");
const Allocator = std.mem.Allocator;
const ListA64 = utilities.ListA64;

const IteratorCallback = utilities.IteratorCallback;
const InlineIteratorCallback = utilities.InlineIteratorCallback;
const iterateWord = utilities.iterateActiveBitsInWord;
const iterateWordInline = utilities.iterateActiveBitsInWordInline;
const BitState = utilities.BitState;
const WordType = bit_word.WordType;

/// Pyramid level that summarizes one bit plane into activity and mixed planes.
/// - `wt` - word width for backing storage.
///
/// Return - level type with counters and word-wise iteration.
pub fn Layer(comptime wt: WordType) type {
    const Word = wt.Type();
    const bw = bit_word.BitWord(wt);

    return struct {
        const Self = @This();

        /// Activity plane, one bit per summarized position.
        activity: ListA64(Word) = .empty,
        /// Mixed plane, marks positions whose children disagree.
        mixed: ListA64(Word) = .empty,
        /// Valid positions in this level, excludes padding bits.
        bits_count: u32 = 0,
        /// Per-state totals indexed by State tag, kept incrementally.
        state_counters: [4]u32 = .{ 0, 0, 0, 0 },

        /// Four-state summary of a child group used by upper levels.
        pub const State = enum(u2) {
            /// All children cleared, safe to skip for active scans.
            inactive = 0,
            /// All children set, safe to bulk-visit for active scans.
            active = 1,
            /// Children disagree above the leaf, requires descent.
            mixed = 2,
            /// Leaf word is internally mixed, requires bit-level scan.
            deep_mixed = 3,

            /// Numeric alias for inactive used in packed arithmetic.
            pub const inactive_u2: u2 = 0;
            /// Numeric alias for active used in packed arithmetic.
            pub const active_u2: u2 = 1;
            /// Numeric alias for mixed used in packed arithmetic.
            pub const mixed_u2: u2 = 2;
            /// Numeric alias for deep_mixed used in packed arithmetic.
            pub const deep_mixed_u2: u2 = 3;

            /// Array index for inactive counters.
            pub const inactive_u32: u32 = 0;
            /// Array index for active counters.
            pub const active_u32: u32 = 1;
            /// Array index for mixed counters.
            pub const mixed_u32: u32 = 2;
            /// Array index for deep_mixed counters.
            pub const deep_mixed_u32: u32 = 3;

            /// Packs two raw planes into one two-bit state code.
            /// - `activity` - raw activity bit.
            /// - `mixed` - raw mixed bit.
            ///
            /// Return - packed state value matching the enum layout.
            pub inline fn fromBits(activity: u1, mixed: u1) u2 {
                const b0: u2 = activity;
                const b1: u2 = @as(u2, mixed) << 1;
                return b0 | b1;
            }

            /// Combines two logical states into one summary state.
            /// - `activity` - logical activity value.
            /// - `mixed` - logical mixed value.
            ///
            /// Return - decoded summary state.
            pub inline fn fromBitsState(activity: BitState, mixed: BitState) State {
                const b0: u2 = @intFromEnum(activity);
                const b1: u2 = @as(u2, @intFromEnum(mixed)) << 1;
                return @enumFromInt(b0 | b1);
            }

            /// Extracts the activity lane of a summary state.
            /// - `self` - summary state to split.
            ///
            /// Return - logical activity value.
            pub inline fn activityBit(self: State) BitState {
                const b0: u1 = @truncate(@as(u2, @intFromEnum(self)));
                return @enumFromInt(b0);
            }

            /// Extracts the mixed lane of a summary state.
            /// - `self` - summary state to split.
            ///
            /// Return - logical mixed value.
            pub inline fn mixedBit(self: State) BitState {
                const b0: u1 = @truncate(@as(u2, @intFromEnum(self)) >> 1);
                return @enumFromInt(b0);
            }
        };

        /// Counted states inside one masked word for counter updates.
        pub const StateCounts = struct {
            /// Cleared positions counted in the mask.
            inactive: u32,
            /// Set positions counted in the mask.
            active: u32,
            /// Disagreeing positions counted in the mask.
            mixed: u32,
            /// Deeply mixed positions counted in the mask.
            deep: u32,
        };

        /// Bundles a level pointer with caller context for iteration.
        /// - `Context` - caller-provided iteration context.
        ///
        /// Return - pairing struct passed to every step call.
        pub fn LayerWithContext(Context: type) type {
            return struct {
                /// Level being scanned, provides activity and mixed planes.
                layer: *Self,
                /// Caller context forwarded to per-bit callbacks.
                context: Context,
            };
        }

        /// Bundles level pointers with caller context for common iteration.
        /// - `Context` - caller-provided iteration context.
        ///
        /// Return - pairing struct passed to every step call.
        pub fn LayersWithContext(comptime include_len: u32, comptime exclude_len: u32, Context: type) type {
            return struct {
                /// Levels whose states are ANDed, provide shared bounds.
                includes: [include_len]*Self,
                /// Levels whose states are ORed, provide exclusion lanes.
                excludes: [exclude_len]*Self,
                /// Caller context forwarded to per-bit callbacks.
                context: Context,
            };
        }

        /// Builds a word-wise visitor that dispatches by four-state summary.
        /// - `Context` - caller-provided iteration context.
        /// - `on_inactive` - visitor for inactive lanes, null skips them.
        /// - `on_active` - visitor for active lanes, null skips them.
        /// - `on_mixed` - visitor for mixed lanes, null skips them.
        /// - `on_deep_mixed` - visitor for deeply mixed lanes, null skips them.
        /// - `direction` - walk order for words and for lanes inside one word.
        ///
        /// Return - iterator type with ranged and single-word entry points.
        pub fn Iterator(
            comptime Context: type,
            comptime on_inactive: InlineIteratorCallback(Context),
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_mixed: IteratorCallback(Context),
            comptime on_deep_mixed: InlineIteratorCallback(Context),
            comptime direction: utilities.Direction,
        ) type {
            return struct {
                /// Visits one word and routes each valid bit to its state callback.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - level and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                pub inline fn step(data: LayerWithContext(Context), word_id: u32) bool {
                    if (on_inactive != null and on_active != null and on_mixed != null and on_deep_mixed != null) {
                        return stepFull(data, word_id);
                    } else {
                        return stepPending(data, word_id);
                    }
                }

                /// Scans words inside an optional bit range in walk order.
                /// - `data` - level and caller context.
                /// - `start_bit` - range edge or null for no lower limit.
                /// - `end_bit` - range edge or null for no upper limit.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub inline fn iterateAll(data: LayerWithContext(Context), start_bit: ?u32, end_bit: ?u32) bool {
                    const layer = data.layer;
                    if (layer.activity.items.len == 0) return true;
                    const range = utilities.resolveRange(layer.bits_count, start_bit, end_bit);
                    if (range.lo >= range.hi) return true;
                    const lo_word: usize = @intCast(bw.bitToWordId(range.lo));
                    const hi_word: usize = @intCast(bw.bitToWordId(range.hi - 1));
                    var wid: usize = if (direction == .forward) lo_word else hi_word;
                    while (true) {
                        const w_base = bw.wordToBitId(@truncate(wid));
                        const s = @max(range.lo, w_base) - w_base;
                        const e = @min(range.hi, w_base + bw.word_type_bits) - w_base;
                        if (!stepRange(data, @truncate(wid), s, e)) return false;
                        if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                        wid = if (direction == .forward) wid + 1 else wid - 1;
                    }
                    return true;
                }

                /// Visits only the `[sub_lo, sub_hi)` lanes of one word.
                /// - `data` - level and caller context.
                /// - `word_id` - word index to scan.
                /// - `sub_lo` - first visited lane, inclusive.
                /// - `sub_hi` - one past the last visited lane, exclusive.
                ///
                /// Return - false on early exit, true when the lanes completed.
                inline fn stepRange(data: LayerWithContext(Context), word_id: u32, sub_lo: u32, sub_hi: u32) bool {
                    const layer = data.layer;
                    const context = data.context;
                    std.debug.assert(word_id < layer.activity.items.len);
                    std.debug.assert(layer.activity.items.len == layer.mixed.items.len);
                    const activity_word = layer.activity.items[word_id];
                    const mixed_word = layer.mixed.items[word_id];
                    const start = bw.wordToBitId(word_id);
                    const range = rangeMask(sub_lo, sub_hi);

                    if (on_inactive != null and on_active != null and on_mixed != null and on_deep_mixed != null) {
                        if (direction == .forward) {
                            var i: u32 = sub_lo;
                            while (i < sub_hi) : (i += 1) {
                                if (!visitLane(context, start + i, activity_word, mixed_word, i)) return false;
                            }
                        } else {
                            var i: u32 = sub_hi;
                            while (i > sub_lo) {
                                i -= 1;
                                if (!visitLane(context, start + i, activity_word, mixed_word, i)) return false;
                            }
                        }
                        return true;
                    }

                    const only_inactive_word = (~activity_word) & (~mixed_word) & range;
                    const only_active_word = activity_word & (~mixed_word) & range;
                    const only_mixed_word = (~activity_word) & mixed_word & range;
                    const only_deep_mixed = activity_word & mixed_word & range;

                    var pending_word: Word = 0;
                    if (on_inactive != null) pending_word |= only_inactive_word;
                    if (on_active != null) pending_word |= only_active_word;
                    if (on_mixed != null) pending_word |= only_mixed_word;
                    if (on_deep_mixed != null) pending_word |= only_deep_mixed;

                    if (direction == .forward) {
                        while (pending_word != 0) {
                            const bit_id_in_word: u32 = @ctz(pending_word);
                            pending_word &= pending_word - 1;
                            if (!visitPendingLane(context, start + bit_id_in_word, bit_id_in_word, only_inactive_word, only_active_word, only_mixed_word, only_deep_mixed)) return false;
                        }
                    } else {
                        while (pending_word != 0) {
                            const lz: u32 = @clz(pending_word);
                            const bit_id_in_word = bw.word_type_bits - 1 - lz;
                            pending_word ^= @as(Word, 1) << @truncate(bit_id_in_word);
                            if (!visitPendingLane(context, start + bit_id_in_word, bit_id_in_word, only_inactive_word, only_active_word, only_mixed_word, only_deep_mixed)) return false;
                        }
                    }
                    return true;
                }

                /// Routes one classified lane to its state callback.
                /// - `context` - caller context for the callbacks.
                /// - `bit_id` - global id of the visited lane.
                /// - `activity_word` - raw activity plane of the word.
                /// - `mixed_word` - raw mixed plane of the word.
                /// - `i` - lane index inside the word.
                ///
                /// Return - false on early exit, true to continue.
                inline fn visitLane(context: Context, bit_id: u32, activity_word: Word, mixed_word: Word, i: u32) bool {
                    const abit: u1 = @truncate(activity_word >> @truncate(i));
                    const mbit: u1 = @truncate(mixed_word >> @truncate(i));
                    if (abit == 0 and mbit == 0) {
                        if (on_inactive) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    } else if (abit == 1 and mbit == 0) {
                        if (on_active) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    } else if (abit == 0) {
                        if (on_mixed) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    } else {
                        if (on_deep_mixed) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    }
                    return true;
                }

                /// Routes one peeled lane to the installed state callback.
                /// - `context` - caller context for the callbacks.
                /// - `bit_id` - global id of the visited lane.
                /// - `bit_id_in_word` - lane index inside the word.
                /// - `only_inactive_word` - masked inactive lanes.
                /// - `only_active_word` - masked active lanes.
                /// - `only_mixed_word` - masked mixed lanes.
                /// - `only_deep_mixed` - masked deeply mixed lanes.
                ///
                /// Return - false on early exit, true to continue.
                inline fn visitPendingLane(context: Context, bit_id: u32, bit_id_in_word: u32, only_inactive_word: Word, only_active_word: Word, only_mixed_word: Word, only_deep_mixed: Word) bool {
                    const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                    if (on_inactive) |f| {
                        if ((only_inactive_word & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    if (on_active) |f| {
                        if ((only_active_word & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    if (on_mixed) |f| {
                        if ((only_mixed_word & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    if (on_deep_mixed) |f| {
                        if ((only_deep_mixed & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    return true;
                }

                /// Builds a mask keeping exactly the `[s, e)` lanes of one word.
                /// - `s` - first kept lane, inclusive.
                /// - `e` - one past the last kept lane, exclusive.
                ///
                /// Return - mask with only the sub-range lanes set.
                inline fn rangeMask(s: u32, e: u32) Word {
                    if (s >= e) return 0;
                    var m: Word = bw.max_value;
                    if (s != 0) m &= bw.max_value << @truncate(s);
                    if (e != bw.word_type_bits) m &= ~(bw.max_value << @truncate(e));
                    return m;
                }

                /// Fast path when all four callbacks exist, scans linearly by bound.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - level and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepFull(data: LayerWithContext(Context), word_id: u32) bool {
                    const layer = data.layer;
                    const context = data.context;
                    std.debug.assert(word_id < layer.activity.items.len);
                    std.debug.assert(layer.activity.items.len == layer.mixed.items.len);
                    const activity_words = layer.activity.items;
                    const activity_word = activity_words[word_id];
                    const mixed_word = layer.mixed.items[word_id];
                    const start = bw.wordToBitId(word_id);
                    var bound: u32 = bw.word_type_bits;
                    if (word_id == activity_words.len - 1) {
                        const used = bw.bitIdInWord(layer.bits_count);
                        if (used != 0) bound = used;
                    }
                    if (direction == .forward) {
                        var i: u32 = 0;
                        while (i < bound) : (i += 1) {
                            if (!visitLane(context, start + i, activity_word, mixed_word, i)) return false;
                        }
                    } else {
                        var i: u32 = bound;
                        while (i > 0) {
                            i -= 1;
                            if (!visitLane(context, start + i, activity_word, mixed_word, i)) return false;
                        }
                    }
                    return true;
                }

                /// Selective path that peels only states with installed callbacks.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - level and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepPending(data: LayerWithContext(Context), word_id: u32) bool {
                    const layer = data.layer;
                    const context = data.context;
                    std.debug.assert(word_id < layer.activity.items.len);
                    std.debug.assert(layer.activity.items.len == layer.mixed.items.len);
                    const activity_words = layer.activity.items;
                    const activity_word = activity_words[word_id];
                    const mixed_word = layer.mixed.items[word_id];
                    const inv_activity_word = ~activity_word;
                    const inv_mixed_word = ~mixed_word;
                    const start = bw.wordToBitId(word_id);

                    var mask: Word = bw.max_value;
                    if (word_id == activity_words.len - 1) {
                        const used = bw.bitIdInWord(layer.bits_count);
                        if (used != 0) mask = bw.maskStart(used);
                    }

                    const only_inactive_word = inv_activity_word & inv_mixed_word & mask;
                    const only_active_word = activity_word & inv_mixed_word & mask;
                    const only_mixed_word = inv_activity_word & mixed_word & mask;
                    const only_deep_mixed = activity_word & mixed_word & mask;

                    var pending_word: Word = 0;
                    if (on_inactive != null) pending_word |= only_inactive_word;
                    if (on_active != null) pending_word |= only_active_word;
                    if (on_mixed != null) pending_word |= only_mixed_word;
                    if (on_deep_mixed != null) pending_word |= only_deep_mixed;

                    if (direction == .forward) {
                        while (pending_word != 0) {
                            const bit_id_in_word: u32 = @ctz(pending_word);
                            pending_word &= pending_word - 1;
                            if (!visitPendingLane(context, start + bit_id_in_word, bit_id_in_word, only_inactive_word, only_active_word, only_mixed_word, only_deep_mixed)) return false;
                        }
                    } else {
                        while (pending_word != 0) {
                            const lz: u32 = @clz(pending_word);
                            const bit_id_in_word = bw.word_type_bits - 1 - lz;
                            pending_word ^= @as(Word, 1) << @truncate(bit_id_in_word);
                            if (!visitPendingLane(context, start + bit_id_in_word, bit_id_in_word, only_inactive_word, only_active_word, only_mixed_word, only_deep_mixed)) return false;
                        }
                    }

                    return true;
                }
            };
        }

        /// Builds a word-wise visitor that dispatches by common states.
        /// - `include_len` - levels whose states are ANDed.
        /// - `exclude_len` - levels whose states are ORed into the veto mask.
        /// - `Context` - caller-provided iteration context.
        /// - `on_inactive` - visitor for common inactive lanes, null skips them.
        /// - `on_active` - visitor for common active lanes, null skips them.
        /// - `on_mixed` - visitor for common mixed lanes, null skips them.
        /// - `on_deep_mixed` - visitor for common deeply mixed lanes, null skips them.
        /// - `direction` - walk order for words and for lanes inside one word.
        ///
        /// Return - iterator type with ranged and single-word entry points.
        pub fn CommonIterator(
            comptime include_len: u32,
            comptime exclude_len: u32,
            comptime Context: type,
            comptime on_inactive: InlineIteratorCallback(Context),
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_mixed: IteratorCallback(Context),
            comptime on_deep_mixed: InlineIteratorCallback(Context),
            comptime direction: utilities.Direction,
        ) type {
            return struct {
                /// Visits one word and routes each valid bit to its state callback.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                pub inline fn step(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) bool {
                    if (include_len == 0 and exclude_len == 0) return true;
                    if (on_inactive != null and on_active != null and on_mixed != null and on_deep_mixed != null) {
                        return stepFull(data, word_id);
                    } else {
                        return stepPending(data, word_id);
                    }
                }

                /// Scans words inside an optional bit range in walk order.
                /// - `data` - levels and caller context.
                /// - `start_bit` - range edge or null for no lower limit.
                /// - `end_bit` - range edge or null for no upper limit.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub inline fn iterateAll(data: LayersWithContext(include_len, exclude_len, Context), start_bit: ?u32, end_bit: ?u32) bool {
                    if (include_len == 0 and exclude_len == 0) return true;
                    const first = if (include_len > 0) data.includes[0] else data.excludes[0];
                    if (first.activity.items.len == 0) return true;
                    const range = utilities.resolveRange(first.bits_count, start_bit, end_bit);
                    if (range.lo >= range.hi) return true;
                    const lo_word: usize = @intCast(bw.bitToWordId(range.lo));
                    const hi_word: usize = @intCast(bw.bitToWordId(range.hi - 1));
                    var wid: usize = if (direction == .forward) lo_word else hi_word;
                    while (true) {
                        const w_base = bw.wordToBitId(@truncate(wid));
                        const s = @max(range.lo, w_base) - w_base;
                        const e = @min(range.hi, w_base + bw.word_type_bits) - w_base;
                        if (!stepRange(data, @truncate(wid), s, e)) return false;
                        if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                        wid = if (direction == .forward) wid + 1 else wid - 1;
                    }
                    return true;
                }

                /// Visits only the `[sub_lo, sub_hi)` lanes of one word.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to scan.
                /// - `sub_lo` - first visited lane, inclusive.
                /// - `sub_hi` - one past the last visited lane, exclusive.
                ///
                /// Return - false on early exit, true when the lanes completed.
                inline fn stepRange(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32, sub_lo: u32, sub_hi: u32) bool {
                    const context = data.context;
                    const start = bw.wordToBitId(word_id);
                    const range = rangeMask(sub_lo, sub_hi);
                    const inactive_mask = commonInactiveWord(data, word_id) & range;
                    const active_mask = commonActiveWord(data, word_id) & range;
                    const mixed_mask = commonMixedWord(data, word_id) & range;
                    const deep_mask = commonDeepWord(data, word_id) & range;

                    if (on_inactive != null and on_active != null and on_mixed != null and on_deep_mixed != null) {
                        if (direction == .forward) {
                            var i: u32 = sub_lo;
                            while (i < sub_hi) : (i += 1) {
                                if (!visitCommonLane(context, start + i, inactive_mask, active_mask, mixed_mask, deep_mask, i)) return false;
                            }
                        } else {
                            var i: u32 = sub_hi;
                            while (i > sub_lo) {
                                i -= 1;
                                if (!visitCommonLane(context, start + i, inactive_mask, active_mask, mixed_mask, deep_mask, i)) return false;
                            }
                        }
                        return true;
                    }

                    var pending_word: Word = 0;
                    if (on_inactive != null) pending_word |= inactive_mask;
                    if (on_active != null) pending_word |= active_mask;
                    if (on_mixed != null) pending_word |= mixed_mask;
                    if (on_deep_mixed != null) pending_word |= deep_mask;

                    if (direction == .forward) {
                        while (pending_word != 0) {
                            const bit_id_in_word: u32 = @ctz(pending_word);
                            pending_word &= pending_word - 1;
                            if (!visitCommonPendingLane(context, start + bit_id_in_word, bit_id_in_word, inactive_mask, active_mask, mixed_mask, deep_mask)) return false;
                        }
                    } else {
                        while (pending_word != 0) {
                            const lz: u32 = @clz(pending_word);
                            const bit_id_in_word = bw.word_type_bits - 1 - lz;
                            pending_word ^= @as(Word, 1) << @truncate(bit_id_in_word);
                            if (!visitCommonPendingLane(context, start + bit_id_in_word, bit_id_in_word, inactive_mask, active_mask, mixed_mask, deep_mask)) return false;
                        }
                    }
                    return true;
                }

                /// Routes one classified lane to its common state callback.
                /// - `context` - caller context for the callbacks.
                /// - `bit_id` - global id of the visited lane.
                /// - `inactive_mask` - merged inactive lanes.
                /// - `active_mask` - merged active lanes.
                /// - `mixed_mask` - merged mixed lanes.
                /// - `deep_mask` - merged deeply mixed lanes.
                /// - `i` - lane index inside the word.
                ///
                /// Return - false on early exit, true to continue.
                inline fn visitCommonLane(context: Context, bit_id: u32, inactive_mask: Word, active_mask: Word, mixed_mask: Word, deep_mask: Word, i: u32) bool {
                    if (((inactive_mask >> @truncate(i)) & 1) != 0) {
                        if (on_inactive) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    } else if (((active_mask >> @truncate(i)) & 1) != 0) {
                        if (on_active) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    } else if (((mixed_mask >> @truncate(i)) & 1) != 0) {
                        if (on_mixed) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    } else if (((deep_mask >> @truncate(i)) & 1) != 0) {
                        if (on_deep_mixed) |f| {
                            if (!f(context, bit_id)) return false;
                        }
                    }
                    return true;
                }

                /// Routes one peeled lane to the installed common state callback.
                /// - `context` - caller context for the callbacks.
                /// - `bit_id` - global id of the visited lane.
                /// - `bit_id_in_word` - lane index inside the word.
                /// - `inactive_mask` - merged inactive lanes.
                /// - `active_mask` - merged active lanes.
                /// - `mixed_mask` - merged mixed lanes.
                /// - `deep_mask` - merged deeply mixed lanes.
                ///
                /// Return - false on early exit, true to continue.
                inline fn visitCommonPendingLane(context: Context, bit_id: u32, bit_id_in_word: u32, inactive_mask: Word, active_mask: Word, mixed_mask: Word, deep_mask: Word) bool {
                    const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                    if (on_inactive) |f| {
                        if ((inactive_mask & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    if (on_active) |f| {
                        if ((active_mask & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    if (on_mixed) |f| {
                        if ((mixed_mask & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    if (on_deep_mixed) |f| {
                        if ((deep_mask & bit) != 0) {
                            if (!f(context, bit_id)) return false;
                            return true;
                        }
                    }
                    return true;
                }

                /// Builds a mask keeping exactly the `[s, e)` lanes of one word.
                /// - `s` - first kept lane, inclusive.
                /// - `e` - one past the last kept lane, exclusive.
                ///
                /// Return - mask with only the sub-range lanes set.
                inline fn rangeMask(s: u32, e: u32) Word {
                    if (s >= e) return 0;
                    var m: Word = bw.max_value;
                    if (s != 0) m &= bw.max_value << @truncate(s);
                    if (e != bw.word_type_bits) m &= ~(bw.max_value << @truncate(e));
                    return m;
                }

                /// Fast path when all four callbacks exist, scans linearly by bound.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepFull(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) bool {
                    const first = if (include_len > 0) data.includes[0] else data.excludes[0];
                    const context = data.context;
                    std.debug.assert(word_id < first.activity.items.len);
                    std.debug.assert(first.activity.items.len == first.mixed.items.len);
                    const start = bw.wordToBitId(word_id);
                    const inactive_mask = commonInactiveWord(data, word_id);
                    const active_mask = commonActiveWord(data, word_id);
                    const mixed_mask = commonMixedWord(data, word_id);
                    const deep_mask = commonDeepWord(data, word_id);
                    var bound: u32 = bw.word_type_bits;
                    if (word_id == first.activity.items.len - 1) {
                        const used = bw.bitIdInWord(first.bits_count);
                        if (used != 0) bound = used;
                    }
                    if (direction == .forward) {
                        var i: u32 = 0;
                        while (i < bound) : (i += 1) {
                            if (!visitCommonLane(context, start + i, inactive_mask, active_mask, mixed_mask, deep_mask, i)) return false;
                        }
                    } else {
                        var i: u32 = bound;
                        while (i > 0) {
                            i -= 1;
                            if (!visitCommonLane(context, start + i, inactive_mask, active_mask, mixed_mask, deep_mask, i)) return false;
                        }
                    }
                    return true;
                }

                /// Selective path that peels only states with installed callbacks.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepPending(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) bool {
                    const first = if (include_len > 0) data.includes[0] else data.excludes[0];
                    const context = data.context;
                    std.debug.assert(word_id < first.activity.items.len);
                    std.debug.assert(first.activity.items.len == first.mixed.items.len);
                    const start = bw.wordToBitId(word_id);

                    var mask: Word = bw.max_value;
                    if (word_id == first.activity.items.len - 1) {
                        const used = bw.bitIdInWord(first.bits_count);
                        if (used != 0) mask = bw.maskStart(used);
                    }

                    const only_inactive_word = commonInactiveWord(data, word_id) & mask;
                    const only_active_word = commonActiveWord(data, word_id) & mask;
                    const only_mixed_word = commonMixedWord(data, word_id) & mask;
                    const only_deep_mixed = commonDeepWord(data, word_id) & mask;

                    var pending_word: Word = 0;
                    if (on_inactive != null) pending_word |= only_inactive_word;
                    if (on_active != null) pending_word |= only_active_word;
                    if (on_mixed != null) pending_word |= only_mixed_word;
                    if (on_deep_mixed != null) pending_word |= only_deep_mixed;

                    if (direction == .forward) {
                        while (pending_word != 0) {
                            const bit_id_in_word: u32 = @ctz(pending_word);
                            pending_word &= pending_word - 1;
                            if (!visitCommonPendingLane(context, start + bit_id_in_word, bit_id_in_word, only_inactive_word, only_active_word, only_mixed_word, only_deep_mixed)) return false;
                        }
                    } else {
                        while (pending_word != 0) {
                            const lz: u32 = @clz(pending_word);
                            const bit_id_in_word = bw.word_type_bits - 1 - lz;
                            pending_word ^= @as(Word, 1) << @truncate(bit_id_in_word);
                            if (!visitCommonPendingLane(context, start + bit_id_in_word, bit_id_in_word, only_inactive_word, only_active_word, only_mixed_word, only_deep_mixed)) return false;
                        }
                    }

                    return true;
                }

                /// Merges include and exclude lanes of one index into a shared inactive mask.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - lanes with all includes inactive, no excludes inactive and no mixed states.
                inline fn commonInactiveWord(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    var include_mask: Word = bw.max_value;
                    inline for (0..include_len) |k| {
                        include_mask &= (~data.includes[k].activity.items[word_id]) & (~data.includes[k].mixed.items[word_id]);
                    }
                    var exclude_mask: Word = 0;
                    inline for (0..exclude_len) |k| {
                        exclude_mask |= (~data.excludes[k].activity.items[word_id]) & (~data.excludes[k].mixed.items[word_id]);
                    }
                    return include_mask & ~exclude_mask & ~commonNonterminalWord(data, word_id);
                }

                /// Merges include and exclude lanes of one index into a shared active mask.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - lanes with all includes active, no excludes active and no mixed states.
                inline fn commonActiveWord(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    var include_mask: Word = bw.max_value;
                    inline for (0..include_len) |k| {
                        include_mask &= data.includes[k].activity.items[word_id] & (~data.includes[k].mixed.items[word_id]);
                    }
                    var exclude_mask: Word = 0;
                    inline for (0..exclude_len) |k| {
                        exclude_mask |= data.excludes[k].activity.items[word_id] & (~data.excludes[k].mixed.items[word_id]);
                    }
                    return include_mask & ~exclude_mask & ~commonNonterminalWord(data, word_id);
                }

                /// Merges every mixed plane of one index into a shared nonterminal mask.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - lanes with any mixed or deeply mixed state.
                inline fn commonNonterminalWord(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    var nonterminal_mask: Word = 0;
                    inline for (0..include_len) |k| {
                        nonterminal_mask |= data.includes[k].mixed.items[word_id];
                    }
                    inline for (0..exclude_len) |k| {
                        nonterminal_mask |= data.excludes[k].mixed.items[word_id];
                    }
                    return nonterminal_mask;
                }

                /// Merges every plane of one index into a shared mixed mask.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - lanes with any mixed state but no unanimous deep state.
                inline fn commonMixedWord(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    return commonNonterminalWord(data, word_id) & ~commonDeepWord(data, word_id);
                }

                /// Merges every plane of one index into a shared deep mask.
                /// - `data` - levels and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - lanes with every include and exclude deeply mixed.
                inline fn commonDeepWord(data: LayersWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    var deep_mask: Word = bw.max_value;
                    inline for (0..include_len) |k| {
                        deep_mask &= data.includes[k].activity.items[word_id] & data.includes[k].mixed.items[word_id];
                    }
                    inline for (0..exclude_len) |k| {
                        deep_mask &= data.excludes[k].activity.items[word_id] & data.excludes[k].mixed.items[word_id];
                    }
                    return deep_mask;
                }
            };
        }

        /// Releases both backing planes.
        /// - `self` - level to destroy.
        /// - `allocator` - allocator that owns the planes.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.activity.deinit(allocator);
            self.mixed.deinit(allocator);
        }

        /// Writes masked lanes of both planes and refreshes counters.
        /// - `self` - level to update.
        /// - `id` - word index to patch.
        /// - `activity` - new activity lanes.
        /// - `mixed` - new mixed lanes.
        /// - `mask` - selects lanes to overwrite.
        pub fn setWord(self: *Self, id: u32, activity: Word, mixed: Word, mask: Word) void {
            if (mask == 0) return;
            std.debug.assert(id < self.activity.items.len);
            std.debug.assert(self.activity.items.len == self.mixed.items.len);
            var eff: Word = mask;

            const words_count = self.activity.items.len;
            if (words_count > 0 and id == words_count - 1) {
                const used: u32 = bw.bitIdInWord(self.bits_count);
                if (used != 0) eff &= bw.maskStartClamped(used);
            }
            if (eff == 0) return;

            const old_a: Word = self.activity.items[id];
            const old_m: Word = self.mixed.items[id];
            const new_a: Word = bw.merge(old_a, activity, eff);
            const new_m: Word = bw.merge(old_m, mixed, eff);
            if (new_a == old_a and new_m == old_m) return;

            const o = countStates(old_a, old_m, eff);
            const n = countStates(activity, mixed, eff);

            self.activity.items[id] = new_a;
            self.mixed.items[id] = new_m;

            self.state_counters[State.inactive_u32] = self.state_counters[State.inactive_u32] - o.inactive + n.inactive;
            self.state_counters[State.active_u32] = self.state_counters[State.active_u32] - o.active + n.active;
            self.state_counters[State.mixed_u32] = self.state_counters[State.mixed_u32] - o.mixed + n.mixed;
            self.state_counters[State.deep_mixed_u32] = self.state_counters[State.deep_mixed_u32] - o.deep + n.deep;
        }

        /// Reads one summary position without touching counters.
        /// - `self` - level to read.
        /// - `id` - bit position to read.
        ///
        /// Return - stored summary state.
        pub fn getBit(self: *const Self, id: u32) State {
            std.debug.assert(id < self.bits_count);
            const word_id = bw.bitToWordId(id);
            const bit_id_in_word = bw.bitIdInWord(id);
            return State.fromBitsState(
                bw.readBitState(self.activity.items[word_id], bit_id_in_word),
                bw.readBitState(self.mixed.items[word_id], bit_id_in_word),
            );
        }

        /// Writes one summary position and moves its counter.
        /// - `self` - level to update.
        /// - `id` - bit position to write.
        /// - `value` - summary state to store.
        pub fn setBit(self: *Self, id: u32, value: State) void {
            std.debug.assert(id < self.bits_count);
            const word_id = bw.bitToWordId(id);
            const bit_id_in_word = bw.bitIdInWord(id);
            const old_a: Word = self.activity.items[word_id];
            const old_m: Word = self.mixed.items[word_id];
            const old_value = State.fromBitsState(
                bw.readBitState(old_a, bit_id_in_word),
                bw.readBitState(old_m, bit_id_in_word),
            );
            if (old_value == value) return;
            const bit: Word = @as(Word, 1) << bit_id_in_word;
            if (value.activityBit() == .active) {
                self.activity.items[word_id] = old_a | bit;
            } else {
                self.activity.items[word_id] = old_a & ~bit;
            }
            if (value.mixedBit() == .active) {
                self.mixed.items[word_id] = old_m | bit;
            } else {
                self.mixed.items[word_id] = old_m & ~bit;
            }
            self.state_counters[@intFromEnum(old_value)] -= 1;
            self.state_counters[@intFromEnum(value)] += 1;
        }

        /// Grows or shrinks both planes while keeping counters exact.
        /// - `self` - level to resize.
        /// - `allocator` - owns backing storage.
        /// - `new_bits_count` - target valid positions.
        /// - `created_bits_value` - state filling newly created positions.
        ///
        /// Return - error on allocation failure.
        pub fn resize(
            self: *Self,
            allocator: Allocator,
            new_bits_count: u32,
            created_bits_value: State,
        ) !void {
            const old_bits_count = self.bits_count;
            if (new_bits_count == old_bits_count) return;
            const old_words_count: usize = bw.bitsToWordsCount(old_bits_count);
            const new_words_count: usize = bw.bitsToWordsCount(new_bits_count);
            const created_a: Word = created_bits_value.activityBit().toWordState(wt);
            const created_m: Word = created_bits_value.mixedBit().toWordState(wt);

            if (new_bits_count > old_bits_count) {
                try self.activity.resize(allocator, new_words_count);
                errdefer self.activity.resize(allocator, old_words_count) catch {};
                try self.mixed.resize(allocator, new_words_count);
                errdefer self.mixed.resize(allocator, old_words_count) catch {};

                var act = self.activity.items;
                var mix = self.mixed.items;

                const old_used: u32 = bw.bitIdInWord(old_bits_count);
                const new_used: u32 = bw.bitIdInWord(new_bits_count);

                if (old_words_count < new_words_count) {
                    if (old_words_count > 0 and old_used != 0) {
                        const old_valid = bw.maskStartClamped(old_used);
                        const idx = old_words_count - 1;
                        act[idx] = bw.merge(created_a, act[idx], old_valid);
                        mix[idx] = bw.merge(created_m, mix[idx], old_valid);
                    }
                    if (new_words_count > old_words_count) {
                        @memset(act[old_words_count..new_words_count], created_a);
                        @memset(mix[old_words_count..new_words_count], created_m);
                    }
                } else {
                    const idx = new_words_count - 1;
                    const old_valid = bw.maskStartClamped(old_used);
                    const new_valid = bw.maskStartClamped(new_used);
                    const range = new_valid & ~old_valid;
                    act[idx] = (act[idx] & old_valid) | (created_a & range);
                    mix[idx] = (mix[idx] & old_valid) | (created_m & range);
                }

                if (new_used != 0 and new_words_count > 0) {
                    const new_valid = bw.maskStartClamped(new_used);
                    act[new_words_count - 1] &= new_valid;
                    mix[new_words_count - 1] &= new_valid;
                }

                self.state_counters[@intFromEnum(created_bits_value)] += new_bits_count - old_bits_count;
                self.bits_count = new_bits_count;
            } else {
                const act_old = self.activity.items;
                const mix_old = self.mixed.items;
                var c_inactive: u32 = 0;
                var c_active: u32 = 0;
                var c_mixed: u32 = 0;
                var c_deep: u32 = 0;
                const new_used: u32 = bw.bitIdInWord(new_bits_count);
                var w: usize = 0;
                if (new_words_count > 0 and new_used != 0) {
                    const rm: Word = ~bw.maskStartClamped(new_used);
                    const c = countStates(act_old[new_words_count - 1], mix_old[new_words_count - 1], rm);
                    c_inactive += c.inactive;
                    c_active += c.active;
                    c_mixed += c.mixed;
                    c_deep += c.deep;
                    w = new_words_count;
                } else {
                    w = new_words_count;
                }
                while (w < old_words_count) : (w += 1) {
                    const a: Word = act_old[w];
                    const m: Word = mix_old[w];
                    var mm: Word = bw.max_value;
                    if (w + 1 == old_words_count) {
                        const old_used: u32 = bw.bitIdInWord(old_bits_count);
                        if (old_used != 0) mm = bw.maskStartClamped(old_used);
                    }
                    const c = countStates(a, m, mm);
                    c_inactive += c.inactive;
                    c_active += c.active;
                    c_mixed += c.mixed;
                    c_deep += c.deep;
                }
                self.state_counters[State.inactive_u32] -|= c_inactive;
                self.state_counters[State.active_u32] -|= c_active;
                self.state_counters[State.mixed_u32] -|= c_mixed;
                self.state_counters[State.deep_mixed_u32] -|= c_deep;

                try self.activity.resize(allocator, new_words_count);
                try self.mixed.resize(allocator, new_words_count);

                if (new_used != 0 and new_words_count > 0) {
                    const new_valid = bw.maskStartClamped(new_used);
                    self.activity.items[new_words_count - 1] &= new_valid;
                    self.mixed.items[new_words_count - 1] &= new_valid;
                }
                self.bits_count = new_bits_count;
            }
        }

        /// Counts four states inside one masked word with popcounts.
        /// - `activity` - activity word to classify.
        /// - `mixed` - mixed word to classify.
        /// - `mask` - selects valid lanes only.
        ///
        /// Return - per-state totals for the mask.
        inline fn countStates(activity: Word, mixed: Word, mask: Word) StateCounts {
            const a = activity & mask;
            const m = mixed & mask;
            return .{
                .inactive = @popCount(~a & ~m & mask),
                .active = @popCount(a & ~m & mask),
                .mixed = @popCount(~a & m & mask),
                .deep = @popCount(a & m & mask),
            };
        }
    };
}

const t = std.testing;

fn scanCounters(comptime wt: WordType, layer: *const Layer(wt)) [4]u32 {
    const BW = bit_word.BitWord(wt);
    var out = [4]u32{ 0, 0, 0, 0 };
    var i: u32 = 0;
    while (i < layer.bits_count) : (i += 1) {
        const bit_word_id = BW.bitToWordId(i);
        const bit_id_in_word = BW.bitIdInWord(i);
        const a = BW.readBitState(layer.activity.items[bit_word_id], @truncate(bit_id_in_word));
        const m = BW.readBitState(layer.mixed.items[bit_word_id], @truncate(bit_id_in_word));
        out[@intFromEnum(Layer(wt).State.fromBitsState(a, m))] += 1;
    }
    return out;
}

fn expectLayerValid(comptime wt: WordType, layer: *const Layer(wt)) !void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    const want_words: usize = BW.bitsToWordsCount(layer.bits_count);
    try t.expectEqual(want_words, layer.activity.items.len);
    try t.expectEqual(want_words, layer.mixed.items.len);
    const used: u32 = BW.bitIdInWord(layer.bits_count);
    if (layer.bits_count > 0 and used != 0 and want_words > 0) {
        const valid = BW.maskStartClamped(used);
        try t.expectEqual(@as(Word, 0), layer.activity.items[want_words - 1] & ~valid);
        try t.expectEqual(@as(Word, 0), layer.mixed.items[want_words - 1] & ~valid);
    }
    const scanned = scanCounters(wt, layer);
    try t.expectEqualSlices(u32, &scanned, &layer.state_counters);
    var total: u32 = 0;
    for (layer.state_counters) |c| total += c;
    try t.expectEqual(layer.bits_count, total);
}

test "Layer.State int constants match enum values" {
    const S = Layer(.u64).State;
    try t.expectEqual(S.inactive_u2, @intFromEnum(S.inactive));
    try t.expectEqual(S.active_u2, @intFromEnum(S.active));
    try t.expectEqual(S.mixed_u2, @intFromEnum(S.mixed));
    try t.expectEqual(S.deep_mixed_u2, @intFromEnum(S.deep_mixed));
    try t.expectEqual(S.inactive_u32, @intFromEnum(S.inactive));
    try t.expectEqual(S.active_u32, @intFromEnum(S.active));
    try t.expectEqual(S.mixed_u32, @intFromEnum(S.mixed));
    try t.expectEqual(S.deep_mixed_u32, @intFromEnum(S.deep_mixed));
    comptime {
        if (S.inactive_u2 != @intFromEnum(S.inactive)) @compileError("inactive const mismatch");
        if (S.active_u2 != @intFromEnum(S.active)) @compileError("active const mismatch");
        if (S.mixed_u2 != @intFromEnum(S.mixed)) @compileError("mixed const mismatch");
        if (S.deep_mixed_u2 != @intFromEnum(S.deep_mixed)) @compileError("deep_mixed const mismatch");
    }
}

test "Layer(u8) countStates" {
    const L8 = Layer(.u8);

    const c = L8.countStates(0b1010, 0b1100, 0xFF);
    try t.expectEqual(@as(u32, 5), c.inactive);
    try t.expectEqual(@as(u32, 1), c.active);
    try t.expectEqual(@as(u32, 1), c.mixed);
    try t.expectEqual(@as(u32, 1), c.deep);

    const c2 = L8.countStates(0b1010, 0b1100, 0x03);
    try t.expectEqual(@as(u32, 1), c2.inactive);
    try t.expectEqual(@as(u32, 1), c2.active);
    try t.expectEqual(@as(u32, 0), c2.mixed);
    try t.expectEqual(@as(u32, 0), c2.deep);
}

test "Layer(u64) resize grow: allocation + counters" {
    var layer: Layer(.u64) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(u32, 10), layer.bits_count);
    try t.expectEqual([4]u32{ 10, 0, 0, 0 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    try layer.resize(t.allocator, 70, .active);
    try t.expectEqual(@as(u32, 70), layer.bits_count);
    try t.expectEqual([4]u32{ 10, 60, 0, 0 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    try layer.resize(t.allocator, 130, .mixed);
    try t.expectEqual([4]u32{ 10, 60, 60, 0 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    try layer.resize(t.allocator, 200, .deep_mixed);
    try t.expectEqual([4]u32{ 10, 60, 60, 70 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);
}

test "Layer(u64) resize grow within one word" {
    var layer: Layer(.u64) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 5, .active);
    try layer.resize(t.allocator, 9, .mixed);
    try t.expectEqual([4]u32{ 0, 5, 4, 0 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);
}

test "Layer(u64) resize shrink: counters follow removed bits" {
    var layer: Layer(.u64) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 200, .inactive);
    for (0..10) |i| {
        layer.activity.items[0] |= @as(u64, 1) << @intCast(i);
        layer.state_counters[Layer(.u64).State.inactive_u32] -= 1;
        layer.state_counters[Layer(.u64).State.active_u32] += 1;
    }
    for (10..20) |i| {
        layer.mixed.items[0] |= @as(u64, 1) << @intCast(i);
        layer.state_counters[Layer(.u64).State.inactive_u32] -= 1;
        layer.state_counters[Layer(.u64).State.mixed_u32] += 1;
    }
    for (20..30) |i| {
        layer.activity.items[0] |= @as(u64, 1) << @intCast(i);
        layer.mixed.items[0] |= @as(u64, 1) << @intCast(i);
        layer.state_counters[Layer(.u64).State.inactive_u32] -= 1;
        layer.state_counters[Layer(.u64).State.deep_mixed_u32] += 1;
    }
    try expectLayerValid(.u64, &layer);

    try layer.resize(t.allocator, 25, .inactive);
    try t.expectEqual(@as(u32, 25), layer.bits_count);
    try t.expectEqual([4]u32{ 0, 10, 10, 5 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    try layer.resize(t.allocator, 10, .inactive);
    try t.expectEqual([4]u32{ 0, 10, 0, 0 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    try layer.resize(t.allocator, 0, .inactive);
    try t.expectEqual([4]u32{ 0, 0, 0, 0 }, layer.state_counters);
    try t.expectEqual(@as(usize, 0), layer.activity.items.len);
    try t.expectEqual(@as(usize, 0), layer.mixed.items.len);
}

test "Layer(u8) resize small word: grow/shrink roundtrip" {
    var layer: Layer(.u8) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 3, .active);
    try layer.resize(t.allocator, 20, .mixed);
    try t.expectEqual([4]u32{ 0, 3, 17, 0 }, layer.state_counters);
    try expectLayerValid(.u8, &layer);
    try layer.resize(t.allocator, 8, .inactive);
    try t.expectEqual([4]u32{ 0, 3, 5, 0 }, layer.state_counters);
    try expectLayerValid(.u8, &layer);
    try layer.resize(t.allocator, 8, .active);
    try t.expectEqual([4]u32{ 0, 3, 5, 0 }, layer.state_counters);
}

test "Layer(u64) setWord: masked insert + counters" {
    const L64 = Layer(.u64);
    var layer: L64 = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 128, .inactive);
    try t.expectEqual([4]u32{ 128, 0, 0, 0 }, layer.state_counters);

    var act: u64 = 0;
    var mix: u64 = 0;
    for (0..32) |i| act |= @as(u64, 1) << @intCast(i);
    for (32..64) |i| mix |= @as(u64, 1) << @intCast(i);
    layer.setWord(0, act, mix, std.math.maxInt(u64));
    try t.expectEqual([4]u32{ 64, 32, 32, 0 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    layer.setWord(0, std.math.maxInt(u64), std.math.maxInt(u64), 0xF);
    try t.expectEqual([4]u32{ 64, 28, 32, 4 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    layer.setWord(0, 0, 0, 0);
    try t.expectEqual([4]u32{ 64, 28, 32, 4 }, layer.state_counters);

    layer.setWord(0, act | 0xF, mix | 0xF, std.math.maxInt(u64));
    try t.expectEqual([4]u32{ 64, 28, 32, 4 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);

    layer.setWord(1, std.math.maxInt(u64), 0, 0x1);
    try t.expectEqual([4]u32{ 63, 29, 32, 4 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);
}

test "Layer(u64) setWord: padding bits ignored" {
    const L64 = Layer(.u64);
    var layer: L64 = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 70, .inactive);
    layer.setWord(1, std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64));

    try t.expectEqual([4]u32{ 64, 0, 0, 6 }, layer.state_counters);
    try expectLayerValid(.u64, &layer);
    try t.expectEqual(@as(u64, 0x3F), layer.activity.items[1]);
    try t.expectEqual(@as(u64, 0x3F), layer.mixed.items[1]);
}

const StepStates = struct {
    inactive: [256]u32 = undefined,
    ni: usize = 0,
    active: [256]u32 = undefined,
    na: usize = 0,
    mixed: [256]u32 = undefined,
    nm: usize = 0,
    deep: [256]u32 = undefined,
    nd: usize = 0,
    stop_after: u32 = std.math.maxInt(u32),

    fn total(self: *const StepStates) usize {
        return self.ni + self.na + self.nm + self.nd;
    }
};

inline fn stepPushI(ctx: *StepStates, bit_id: u32) bool {
    ctx.inactive[ctx.ni] = bit_id;
    ctx.ni += 1;
    return ctx.total() < ctx.stop_after;
}

inline fn stepPushA(ctx: *StepStates, bit_id: u32) bool {
    ctx.active[ctx.na] = bit_id;
    ctx.na += 1;
    return ctx.total() < ctx.stop_after;
}

fn stepPushM(ctx: *StepStates, bit_id: u32) bool {
    ctx.mixed[ctx.nm] = bit_id;
    ctx.nm += 1;
    return ctx.total() < ctx.stop_after;
}

inline fn stepPushD(ctx: *StepStates, bit_id: u32) bool {
    ctx.deep[ctx.nd] = bit_id;
    ctx.nd += 1;
    return ctx.total() < ctx.stop_after;
}

fn stepCollectAll(comptime wt: WordType, layer: *Layer(wt), ctx: *StepStates) bool {
    const It = Layer(wt).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
    var wid: u32 = 0;
    while (wid < layer.activity.items.len) : (wid += 1) {
        if (!It.step(.{ .layer = layer, .context = ctx }, wid)) return false;
    }
    return true;
}

fn stepOracleStates(comptime wt: WordType, layer: *const Layer(wt), out: *[4][256]u32) [4]usize {
    const BW = bit_word.BitWord(wt);
    var ns = [4]usize{ 0, 0, 0, 0 };
    var i: u32 = 0;
    while (i < layer.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        const in_w = BW.bitIdInWord(i);
        const a = BW.readBitState(layer.activity.items[wid], in_w);
        const m = BW.readBitState(layer.mixed.items[wid], in_w);
        const s: usize = @intFromEnum(Layer(wt).State.fromBitsState(a, m));
        out[s][ns[s]] = i;
        ns[s] += 1;
    }
    return ns;
}

const StepPattern = enum { all_inactive, all_active, all_mixed, all_deep, cycle4, sparse_deep, pseudo };

fn patternState(pat: StepPattern, i: u32) Layer(.u64).State {
    return switch (pat) {
        .all_inactive => .inactive,
        .all_active => .active,
        .all_mixed => .mixed,
        .all_deep => .deep_mixed,
        .cycle4 => @enumFromInt(@as(u2, @truncate(i))),
        .sparse_deep => if (i == 0 or i % 63 == 0) .deep_mixed else .inactive,
        .pseudo => @enumFromInt(@as(u2, @truncate((i *% 2654435761) >> 29))),
    };
}

fn layerSetState(comptime wt: WordType, layer: *Layer(wt), i: u32, st: Layer(wt).State) void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    const wid = BW.bitToWordId(i);
    const in_w = BW.bitIdInWord(i);
    const m: Word = @as(Word, 1) << in_w;
    const a: Word = if (st.activityBit() == .active) m else 0;
    const mm: Word = if (st.mixedBit() == .active) m else 0;
    layer.setWord(wid, a, mm, m);
}

fn stepCheckOne(comptime wt: WordType, n: u32, pat: StepPattern) !void {
    const BW = bit_word.BitWord(wt);
    const L = Layer(wt);
    var layer: L = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const st64 = patternState(pat, i);
        layerSetState(wt, &layer, i, @enumFromInt(@intFromEnum(st64)));
    }
    try expectLayerValid(wt, &layer);

    if (n > 0) {
        const used = BW.bitIdInWord(n);
        if (used != 0) {
            const valid = BW.maskStart(used);
            const last = layer.activity.items.len - 1;
            layer.activity.items[last] |= ~valid;
            layer.mixed.items[last] |= ~valid;
        }
    }

    var ctx = StepStates{};
    try t.expect(stepCollectAll(wt, &layer, &ctx));

    var exp: [4][256]u32 = undefined;
    const ns = stepOracleStates(wt, &layer, &exp);
    const got = [_][]const u32{
        ctx.inactive[0..ctx.ni],
        ctx.active[0..ctx.na],
        ctx.mixed[0..ctx.nm],
        ctx.deep[0..ctx.nd],
    };

    for (0..4) |s| {
        try t.expectEqualSlices(u32, exp[s][0..ns[s]], got[s]);
    }

    try t.expectEqual(n, @as(u32, @intCast(ctx.ni + ctx.na + ctx.nm + ctx.nd)));
}

test "Layer step: 4 states counts + corners" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 127, 128, 129, 200 };
    const patterns = [_]StepPattern{ .all_inactive, .all_active, .all_mixed, .all_deep, .cycle4, .sparse_deep, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            try stepCheckOne(.u64, n, pat);
            try stepCheckOne(.u8, n, pat);
        }
    }
}

test "Layer step: early exit stops the walk" {
    {
        var layer: Layer(.u64) = .{};
        defer layer.deinit(t.allocator);
        try layer.resize(t.allocator, 70, .inactive);
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 3 };
        try t.expect(!It.step(.{ .layer = &layer, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 3), ctx.ni);
        try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ctx.inactive[0..ctx.ni]);
    }

    {
        var layer: Layer(.u64) = .{};
        defer layer.deinit(t.allocator);
        try layer.resize(t.allocator, 10, .inactive);
        layerSetState(.u64, &layer, 0, .active);
        layerSetState(.u64, &layer, 9, .active);
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 10 };

        try t.expect(!It.step(.{ .layer = &layer, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 8), ctx.ni);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 9 }, ctx.active[0..ctx.na]);
    }

    {
        var layer: Layer(.u64) = .{};
        defer layer.deinit(t.allocator);
        try layer.resize(t.allocator, 130, .deep_mixed);
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 70 };
        try t.expect(It.step(.{ .layer = &layer, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 64), ctx.nd);
        try t.expect(!It.step(.{ .layer = &layer, .context = &ctx }, 1));
        try t.expectEqual(@as(usize, 70), ctx.nd);
        try t.expectEqual(@as(u32, 64), ctx.deep[64]);
        try t.expectEqual(@as(u32, 69), ctx.deep[69]);
    }

    {
        var layer: Layer(.u64) = .{};
        defer layer.deinit(t.allocator);
        try layer.resize(t.allocator, 10, .mixed);
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 1 };
        try t.expect(!It.step(.{ .layer = &layer, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 1), ctx.ni + ctx.na + ctx.nm + ctx.nd);
    }
}

const CommonOrderStates = struct {
    ids: [256]u32 = undefined,
    tags: [256]u2 = undefined,
    n: usize = 0,
    stop_after: u32 = std.math.maxInt(u32),
};

inline fn commonOrderPushI(ctx: *CommonOrderStates, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.tags[ctx.n] = 0;
    ctx.n += 1;
    return ctx.n < ctx.stop_after;
}

inline fn commonOrderPushA(ctx: *CommonOrderStates, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.tags[ctx.n] = 1;
    ctx.n += 1;
    return ctx.n < ctx.stop_after;
}

fn commonOrderPushM(ctx: *CommonOrderStates, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.tags[ctx.n] = 2;
    ctx.n += 1;
    return ctx.n < ctx.stop_after;
}

inline fn commonOrderPushD(ctx: *CommonOrderStates, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.tags[ctx.n] = 3;
    ctx.n += 1;
    return ctx.n < ctx.stop_after;
}

fn commonStepCollect(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    comptime on_i: InlineIteratorCallback(*StepStates),
    comptime on_a: InlineIteratorCallback(*StepStates),
    comptime on_m: IteratorCallback(*StepStates),
    comptime on_d: InlineIteratorCallback(*StepStates),
    includes: [IL]*Layer(wt),
    excludes: [EL]*Layer(wt),
    ctx: *StepStates,
) bool {
    const It = Layer(wt).CommonIterator(IL, EL, *StepStates, on_i, on_a, on_m, on_d, .forward);
    if (IL == 0 and EL == 0) return true;
    const words_len = if (IL > 0) includes[0].activity.items.len else excludes[0].activity.items.len;
    var wid: u32 = 0;
    while (wid < words_len) : (wid += 1) {
        if (!It.step(.{ .includes = includes, .excludes = excludes, .context = ctx }, wid)) return false;
    }
    return true;
}

fn commonStepCollectOrder(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    includes: [IL]*Layer(wt),
    excludes: [EL]*Layer(wt),
    ctx: *CommonOrderStates,
) bool {
    const It = Layer(wt).CommonIterator(IL, EL, *CommonOrderStates, commonOrderPushI, commonOrderPushA, commonOrderPushM, commonOrderPushD, .forward);
    if (IL == 0 and EL == 0) return true;
    const words_len = if (IL > 0) includes[0].activity.items.len else excludes[0].activity.items.len;
    var wid: u32 = 0;
    while (wid < words_len) : (wid += 1) {
        if (!It.step(.{ .includes = includes, .excludes = excludes, .context = ctx }, wid)) return false;
    }
    return true;
}

fn commonOracleStates(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    includes: [IL]*Layer(wt),
    excludes: [EL]*Layer(wt),
    n: u32,
    out: *[4][256]u32,
) [4]usize {
    const BW = bit_word.BitWord(wt);
    var ns = [4]usize{ 0, 0, 0, 0 };
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        const in_w = BW.bitIdInWord(i);
        var any_nt = false;
        var all_deep = true;
        var all_inc_a = true;
        var all_inc_i = true;
        var any_exc_a = false;
        var any_exc_i = false;
        for (includes) |layer| {
            const a = BW.readBitState(layer.activity.items[wid], in_w);
            const m = BW.readBitState(layer.mixed.items[wid], in_w);
            if (m == .active) any_nt = true;
            if (!(a == .active and m == .active)) all_deep = false;
            if (!(a == .active and m == .inactive)) all_inc_a = false;
            if (!(a == .inactive and m == .inactive)) all_inc_i = false;
        }
        for (excludes) |layer| {
            const a = BW.readBitState(layer.activity.items[wid], in_w);
            const m = BW.readBitState(layer.mixed.items[wid], in_w);
            if (m == .active) any_nt = true;
            if (!(a == .active and m == .active)) all_deep = false;
            if (a == .active and m == .inactive) any_exc_a = true;
            if (a == .inactive and m == .inactive) any_exc_i = true;
        }
        const s: usize = if (any_nt)
            if (all_deep) 3 else 2
        else if (all_inc_i and !any_exc_i)
            0
        else if (all_inc_a and !any_exc_a)
            1
        else
            4;
        if (s < 4) {
            out[s][ns[s]] = i;
            ns[s] += 1;
        }
    }
    return ns;
}

fn commonPoisonPadding(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    includes: [IL]*Layer(wt),
    excludes: [EL]*Layer(wt),
    n: u32,
) void {
    if (n == 0) return;
    const BW = bit_word.BitWord(wt);
    const used = BW.bitIdInWord(n);
    if (used == 0) return;
    const valid = BW.maskStart(used);
    for (includes) |layer| {
        layer.activity.items[layer.activity.items.len - 1] |= ~valid;
        layer.mixed.items[layer.mixed.items.len - 1] |= ~valid;
    }
    for (excludes) |layer| {
        layer.activity.items[layer.activity.items.len - 1] |= ~valid;
        layer.mixed.items[layer.mixed.items.len - 1] |= ~valid;
    }
}

fn commonCheckOne(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    n: u32,
    pats_inc: [IL]StepPattern,
    pats_exc: [EL]StepPattern,
) !void {
    var inc_sets: [IL]Layer(wt) = undefined;
    var exc_sets: [EL]Layer(wt) = undefined;
    for (0..IL) |k| {
        inc_sets[k] = .{};
        try inc_sets[k].resize(t.allocator, n, .inactive);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            layerSetState(wt, &inc_sets[k], i, @enumFromInt(@intFromEnum(patternState(pats_inc[k], i))));
        }
    }
    for (0..EL) |k| {
        exc_sets[k] = .{};
        try exc_sets[k].resize(t.allocator, n, .inactive);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            layerSetState(wt, &exc_sets[k], i, @enumFromInt(@intFromEnum(patternState(pats_exc[k], i))));
        }
    }
    defer {
        for (0..IL) |k| inc_sets[k].deinit(t.allocator);
        for (0..EL) |k| exc_sets[k].deinit(t.allocator);
    }
    var inc_ptrs: [IL]*Layer(wt) = undefined;
    var exc_ptrs: [EL]*Layer(wt) = undefined;
    for (0..IL) |k| inc_ptrs[k] = &inc_sets[k];
    for (0..EL) |k| exc_ptrs[k] = &exc_sets[k];
    commonPoisonPadding(wt, IL, EL, inc_ptrs, exc_ptrs, n);

    var exp: [4][256]u32 = undefined;
    const ns = commonOracleStates(wt, IL, EL, inc_ptrs, exc_ptrs, n, &exp);

    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, stepPushI, stepPushA, stepPushM, stepPushD, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[0][0..ns[0]], ctx.inactive[0..ctx.ni]);
        try t.expectEqualSlices(u32, exp[1][0..ns[1]], ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, exp[2][0..ns[2]], ctx.mixed[0..ctx.nm]);
        try t.expectEqualSlices(u32, exp[3][0..ns[3]], ctx.deep[0..ctx.nd]);
    }
    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, stepPushI, null, null, null, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[0][0..ns[0]], ctx.inactive[0..ctx.ni]);
        try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
    }
    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, null, stepPushA, null, null, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[1][0..ns[1]], ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.nm + ctx.nd);
    }
    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, null, null, stepPushM, null, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[2][0..ns[2]], ctx.mixed[0..ctx.nm]);
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na + ctx.nd);
    }
    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, null, null, null, stepPushD, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[3][0..ns[3]], ctx.deep[0..ctx.nd]);
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na + ctx.nm);
    }
    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, stepPushI, stepPushA, null, null, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[0][0..ns[0]], ctx.inactive[0..ctx.ni]);
        try t.expectEqualSlices(u32, exp[1][0..ns[1]], ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.nm + ctx.nd);
    }
    {
        var ctx = StepStates{};
        try t.expect(commonStepCollect(wt, IL, EL, null, null, stepPushM, stepPushD, inc_ptrs, exc_ptrs, &ctx));
        try t.expectEqualSlices(u32, exp[2][0..ns[2]], ctx.mixed[0..ctx.nm]);
        try t.expectEqualSlices(u32, exp[3][0..ns[3]], ctx.deep[0..ctx.nd]);
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na);
    }
    for (0..4) |s| {
        for (exp[s][0..ns[s]]) |id| try t.expect(id < n);
    }
}

fn commonCheckOrder(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    n: u32,
    pats_inc: [IL]StepPattern,
    pats_exc: [EL]StepPattern,
) !void {
    var inc_sets: [IL]Layer(wt) = undefined;
    var exc_sets: [EL]Layer(wt) = undefined;
    for (0..IL) |k| {
        inc_sets[k] = .{};
        try inc_sets[k].resize(t.allocator, n, .inactive);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            layerSetState(wt, &inc_sets[k], i, @enumFromInt(@intFromEnum(patternState(pats_inc[k], i))));
        }
    }
    for (0..EL) |k| {
        exc_sets[k] = .{};
        try exc_sets[k].resize(t.allocator, n, .inactive);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            layerSetState(wt, &exc_sets[k], i, @enumFromInt(@intFromEnum(patternState(pats_exc[k], i))));
        }
    }
    defer {
        for (0..IL) |k| inc_sets[k].deinit(t.allocator);
        for (0..EL) |k| exc_sets[k].deinit(t.allocator);
    }
    var inc_ptrs: [IL]*Layer(wt) = undefined;
    var exc_ptrs: [EL]*Layer(wt) = undefined;
    for (0..IL) |k| inc_ptrs[k] = &inc_sets[k];
    for (0..EL) |k| exc_ptrs[k] = &exc_sets[k];
    commonPoisonPadding(wt, IL, EL, inc_ptrs, exc_ptrs, n);

    var exp: [4][256]u32 = undefined;
    const ns = commonOracleStates(wt, IL, EL, inc_ptrs, exc_ptrs, n, &exp);

    var ctx = CommonOrderStates{};
    try t.expect(commonStepCollectOrder(wt, IL, EL, inc_ptrs, exc_ptrs, &ctx));

    var taken = [4]usize{ 0, 0, 0, 0 };
    var prev: u32 = 0;
    var k: usize = 0;
    while (k < ctx.n) : (k += 1) {
        const id = ctx.ids[k];
        const tag: usize = ctx.tags[k];
        if (k > 0) try t.expect(id > prev);
        prev = id;
        try t.expect(id < n);
        try t.expect(taken[tag] < ns[tag]);
        try t.expectEqual(exp[tag][taken[tag]], id);
        taken[tag] += 1;
    }
    for (0..4) |s| try t.expectEqual(ns[s], taken[s]);
}

test "Layer CommonIterator: spec poles 1+1" {
    var inc: Layer(.u8) = .{};
    var exc: Layer(.u8) = .{};
    defer inc.deinit(t.allocator);
    defer exc.deinit(t.allocator);
    try inc.resize(t.allocator, 4, .inactive);
    try exc.resize(t.allocator, 4, .inactive);
    layerSetState(.u8, &inc, 0, .inactive);
    layerSetState(.u8, &inc, 1, .active);
    layerSetState(.u8, &inc, 2, .mixed);
    layerSetState(.u8, &inc, 3, .deep_mixed);
    layerSetState(.u8, &exc, 0, .active);
    layerSetState(.u8, &exc, 1, .inactive);
    layerSetState(.u8, &exc, 2, .deep_mixed);
    layerSetState(.u8, &exc, 3, .mixed);

    const It = Layer(.u8).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
    var ctx = StepStates{};
    try t.expect(It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
    try t.expectEqualSlices(u32, &[_]u32{0}, ctx.inactive[0..ctx.ni]);
    try t.expectEqualSlices(u32, &[_]u32{1}, ctx.active[0..ctx.na]);
    try t.expectEqualSlices(u32, &[_]u32{ 2, 3 }, ctx.mixed[0..ctx.nm]);
    try t.expectEqual(@as(usize, 0), ctx.nd);
}

test "Layer CommonIterator: 1+1 truth table" {
    const all = [_]Layer(.u8).State{ .inactive, .active, .mixed, .deep_mixed };
    for (all) |si| {
        for (all) |se| {
            var inc: Layer(.u8) = .{};
            var exc: Layer(.u8) = .{};
            defer inc.deinit(t.allocator);
            defer exc.deinit(t.allocator);
            try inc.resize(t.allocator, 1, .inactive);
            try exc.resize(t.allocator, 1, .inactive);
            layerSetState(.u8, &inc, 0, si);
            layerSetState(.u8, &exc, 0, se);
            const It = Layer(.u8).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
            var ctx = StepStates{};
            try t.expect(It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
            const si_nt = si == .mixed or si == .deep_mixed;
            const se_nt = se == .mixed or se == .deep_mixed;
            if (si_nt or se_nt) {
                try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na);
                if (si == .deep_mixed and se == .deep_mixed) {
                    try t.expectEqualSlices(u32, &[_]u32{0}, ctx.deep[0..ctx.nd]);
                    try t.expectEqual(@as(usize, 0), ctx.nm);
                } else {
                    try t.expectEqualSlices(u32, &[_]u32{0}, ctx.mixed[0..ctx.nm]);
                    try t.expectEqual(@as(usize, 0), ctx.nd);
                }
            } else if (si == .inactive and se == .active) {
                try t.expectEqualSlices(u32, &[_]u32{0}, ctx.inactive[0..ctx.ni]);
                try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
            } else if (si == .active and se == .inactive) {
                try t.expectEqualSlices(u32, &[_]u32{0}, ctx.active[0..ctx.na]);
                try t.expectEqual(@as(usize, 0), ctx.ni + ctx.nm + ctx.nd);
            } else {
                try t.expectEqual(@as(usize, 0), ctx.total());
            }
        }
    }
}

fn commonCheckSingleMatchesIterator(comptime wt: WordType, n: u32, pat: StepPattern) !void {
    var layer: Layer(wt) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        layerSetState(wt, &layer, i, @enumFromInt(@intFromEnum(patternState(pat, i))));
    }
    commonPoisonPadding(wt, 1, 0, .{&layer}, .{}, n);

    var exp: [4][256]u32 = undefined;
    const ns = stepOracleStates(wt, &layer, &exp);

    var got = StepStates{};
    try t.expect(commonStepCollect(wt, 1, 0, stepPushI, stepPushA, stepPushM, stepPushD, .{&layer}, .{}, &got));
    const lists = [_][]const u32{
        got.inactive[0..got.ni],
        got.active[0..got.na],
        got.mixed[0..got.nm],
        got.deep[0..got.nd],
    };
    for (0..4) |s| try t.expectEqualSlices(u32, exp[s][0..ns[s]], lists[s]);

    var ref = StepStates{};
    var wid: u32 = 0;
    const It = Layer(wt).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
    while (wid < layer.activity.items.len) : (wid += 1) {
        if (!It.step(.{ .layer = &layer, .context = &ref }, wid)) break;
    }
    try t.expectEqualSlices(u32, ref.inactive[0..ref.ni], got.inactive[0..got.ni]);
    try t.expectEqualSlices(u32, ref.active[0..ref.na], got.active[0..got.na]);
    try t.expectEqualSlices(u32, ref.mixed[0..ref.nm], got.mixed[0..got.nm]);
    try t.expectEqualSlices(u32, ref.deep[0..ref.nd], got.deep[0..got.nd]);
}

test "Layer CommonIterator: 1+0 matches Iterator" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 128, 129, 200 };
    const patterns = [_]StepPattern{ .all_inactive, .all_active, .all_mixed, .all_deep, .cycle4, .sparse_deep, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            try commonCheckSingleMatchesIterator(.u64, n, pat);
            try commonCheckSingleMatchesIterator(.u8, n, pat);
        }
    }
}

fn commonCheckSwappedDual(comptime wt: WordType, n: u32, pat: StepPattern) !void {
    var layer: Layer(wt) = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        layerSetState(wt, &layer, i, @enumFromInt(@intFromEnum(patternState(pat, i))));
    }
    commonPoisonPadding(wt, 0, 1, .{}, .{&layer}, n);

    var exp: [4][256]u32 = undefined;
    const ns = stepOracleStates(wt, &layer, &exp);

    var got = StepStates{};
    try t.expect(commonStepCollect(wt, 0, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{}, .{&layer}, &got));
    try t.expectEqualSlices(u32, exp[1][0..ns[1]], got.inactive[0..got.ni]);
    try t.expectEqualSlices(u32, exp[0][0..ns[0]], got.active[0..got.na]);
    try t.expectEqualSlices(u32, exp[2][0..ns[2]], got.mixed[0..got.nm]);
    try t.expectEqualSlices(u32, exp[3][0..ns[3]], got.deep[0..got.nd]);
}

test "Layer CommonIterator: 0+1 swapped dual" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 128, 129, 200 };
    const patterns = [_]StepPattern{ .all_inactive, .all_active, .all_mixed, .all_deep, .cycle4, .sparse_deep, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            try commonCheckSwappedDual(.u64, n, pat);
            try commonCheckSwappedDual(.u8, n, pat);
        }
    }
}

test "Layer CommonIterator: all uniform and aliasing" {
    for ([_]StepPattern{ .all_inactive, .all_active, .all_mixed, .all_deep }) |pat| {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .inactive);
        try exc.resize(t.allocator, 70, .inactive);
        var i: u32 = 0;
        while (i < 70) : (i += 1) {
            layerSetState(.u64, &inc, i, @enumFromInt(@intFromEnum(patternState(pat, i))));
            layerSetState(.u64, &exc, i, @enumFromInt(@intFromEnum(patternState(pat, i))));
        }
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{&inc}, .{&exc}, &ctx));
        switch (pat) {
            .all_inactive, .all_active => try t.expectEqual(@as(usize, 0), ctx.total()),
            .all_mixed => {
                try t.expectEqual(@as(usize, 70), ctx.nm);
                try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na + ctx.nd);
            },
            .all_deep => {
                try t.expectEqual(@as(usize, 70), ctx.nd);
                try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na + ctx.nm);
            },
            else => unreachable,
        }
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .active);
        try exc.resize(t.allocator, 70, .inactive);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 70), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.nm + ctx.nd);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .inactive);
        try exc.resize(t.allocator, 70, .active);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 70), ctx.ni);
        try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
    }
}

test "Layer CommonIterator: oracle corners 2+2" {
    const sizes_u64 = [_]u32{ 0, 1, 2, 7, 8, 9, 63, 64, 65, 70, 129, 200 };
    const sizes_u8 = [_]u32{ 0, 1, 7, 8, 9, 16, 17, 24 };
    const pats = [_]StepPattern{ .all_inactive, .all_active, .cycle4, .pseudo };
    for (sizes_u64) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                for (pats) |p2| {
                    for (pats) |p3| {
                        try commonCheckOne(.u64, 2, 2, n, .{ p0, p1 }, .{ p2, p3 });
                    }
                }
            }
        }
    }
    for (sizes_u8) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                for (pats) |p2| {
                    for (pats) |p3| {
                        try commonCheckOne(.u8, 2, 2, n, .{ p0, p1 }, .{ p2, p3 });
                    }
                }
            }
        }
    }
}

test "Layer CommonIterator: oracle corners 1+1 exhaustive patterns" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 16, 17, 63, 64, 65, 70, 129 };
    const patterns = [_]StepPattern{ .all_inactive, .all_active, .all_mixed, .all_deep, .cycle4, .sparse_deep, .pseudo };
    for (sizes) |n| {
        for (patterns) |pi| {
            for (patterns) |pe| {
                try commonCheckOne(.u64, 1, 1, n, .{pi}, .{pe});
                try commonCheckOne(.u8, 1, 1, n, .{pi}, .{pe});
            }
        }
    }
}

test "Layer CommonIterator: oracle corners mixed lens" {
    const sizes = [_]u32{ 0, 1, 8, 9, 17, 64, 65, 70, 130 };
    const pats = [_]StepPattern{ .all_inactive, .all_active, .cycle4, .pseudo };
    for (sizes) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                try commonCheckOne(.u64, 2, 1, n, .{ p0, p1 }, .{p0});
                try commonCheckOne(.u64, 1, 2, n, .{p0}, .{ p0, p1 });
                try commonCheckOne(.u64, 3, 1, n, .{ p0, p1, p0 }, .{p1});
                try commonCheckOne(.u64, 1, 3, n, .{p0}, .{ p0, p1, p0 });
                try commonCheckOne(.u64, 3, 3, n, .{ p0, p1, p0 }, .{ p1, p0, p1 });
                try commonCheckOne(.u8, 2, 0, n, .{ p0, p1 }, .{});
                try commonCheckOne(.u8, 0, 2, n, .{}, .{ p0, p1 });
                try commonCheckOne(.u8, 3, 0, n, .{ p0, p1, p0 }, .{});
                try commonCheckOne(.u8, 0, 3, n, .{}, .{ p0, p1, p0 });
            }
        }
    }
}

test "Layer CommonIterator: oracle word types u16/u32" {
    const sizes16 = [_]u32{ 0, 1, 15, 16, 17, 33 };
    const sizes32 = [_]u32{ 0, 1, 31, 32, 33, 65 };
    const pats = [_]StepPattern{ .all_inactive, .all_active, .cycle4, .pseudo };
    for (sizes16) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                try commonCheckOne(.u16, 2, 2, n, .{ p0, p1 }, .{ p1, p0 });
                try commonCheckOne(.u16, 1, 1, n, .{p0}, .{p1});
            }
        }
    }
    for (sizes32) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                try commonCheckOne(.u32, 2, 2, n, .{ p0, p1 }, .{ p1, p0 });
                try commonCheckOne(.u32, 1, 1, n, .{p0}, .{p1});
            }
        }
    }
}

test "Layer CommonIterator: boundary oracle u64" {
    const bounds = [_]u32{ 0, 1, 2, 63, 64, 65, 70, 100, 126, 127, 128, 129, 130, 191, 192, 193, 200 };
    const pats = [_]StepPattern{ .all_inactive, .all_active, .all_mixed, .all_deep, .cycle4, .sparse_deep, .pseudo };
    for (bounds) |n| {
        for (pats) |pi| {
            try commonCheckOne(.u64, 2, 2, n, .{ pi, .pseudo }, .{ .cycle4, pi });
            try commonCheckOne(.u64, 1, 1, n, .{pi}, .{.pseudo});
        }
    }
}

fn commonRandU32(state: *u64) u32 {
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @truncate((x *% 0x2545F4914F6CDD1D) >> 32);
}

test "Layer CommonIterator: fuzz vs oracle u64 2+2" {
    var rng: u64 = 0x9E3779B97F4A7C15;
    var step: usize = 0;
    while (step < 300) : (step += 1) {
        const n: u32 = commonRandU32(&rng) % 201;
        errdefer std.debug.print("COMMON LAYER FUZZ u64 2+2 fail step={} n={}\n", .{ step, n });
        var inc_sets: [2]Layer(.u64) = .{ .{}, .{} };
        var exc_sets: [2]Layer(.u64) = .{ .{}, .{} };
        defer {
            for (0..2) |k| inc_sets[k].deinit(t.allocator);
            for (0..2) |k| exc_sets[k].deinit(t.allocator);
        }
        for (0..2) |k| try inc_sets[k].resize(t.allocator, n, .inactive);
        for (0..2) |k| try exc_sets[k].resize(t.allocator, n, .inactive);
        for (0..2) |k| {
            var w: usize = 0;
            while (w < inc_sets[k].activity.items.len) : (w += 1) {
                inc_sets[k].activity.items[w] = commonRandU32(&rng);
                inc_sets[k].activity.items[w] |= @as(u64, commonRandU32(&rng)) << 32;
                inc_sets[k].mixed.items[w] = commonRandU32(&rng);
                inc_sets[k].mixed.items[w] |= @as(u64, commonRandU32(&rng)) << 32;
            }
        }
        for (0..2) |k| {
            var w: usize = 0;
            while (w < exc_sets[k].activity.items.len) : (w += 1) {
                exc_sets[k].activity.items[w] = commonRandU32(&rng);
                exc_sets[k].activity.items[w] |= @as(u64, commonRandU32(&rng)) << 32;
                exc_sets[k].mixed.items[w] = commonRandU32(&rng);
                exc_sets[k].mixed.items[w] |= @as(u64, commonRandU32(&rng)) << 32;
            }
        }
        if (n > 0) {
            const BW = bit_word.BitWord(.u64);
            const used = BW.bitIdInWord(n);
            if (used != 0) {
                const valid = BW.maskStart(used);
                for (0..2) |k| {
                    inc_sets[k].activity.items[inc_sets[k].activity.items.len - 1] &= valid;
                    inc_sets[k].mixed.items[inc_sets[k].mixed.items.len - 1] &= valid;
                    exc_sets[k].activity.items[exc_sets[k].activity.items.len - 1] &= valid;
                    exc_sets[k].mixed.items[exc_sets[k].mixed.items.len - 1] &= valid;
                }
            }
        }
        if ((step & 1) == 0) {
            commonPoisonPadding(.u64, 2, 2, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, n);
        }
        var exp: [4][256]u32 = undefined;
        const ns = commonOracleStates(.u64, 2, 2, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, n, &exp);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 2, 2, stepPushI, stepPushA, stepPushM, stepPushD, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, &ctx));
        try t.expectEqualSlices(u32, exp[0][0..ns[0]], ctx.inactive[0..ctx.ni]);
        try t.expectEqualSlices(u32, exp[1][0..ns[1]], ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, exp[2][0..ns[2]], ctx.mixed[0..ctx.nm]);
        try t.expectEqualSlices(u32, exp[3][0..ns[3]], ctx.deep[0..ctx.nd]);
    }
}

test "Layer CommonIterator: fuzz vs oracle u8 3+3" {
    var rng: u64 = 0x123456789ABCDEF;
    var step: usize = 0;
    while (step < 200) : (step += 1) {
        const n: u32 = commonRandU32(&rng) % 41;
        errdefer std.debug.print("COMMON LAYER FUZZ u8 3+3 fail step={} n={}\n", .{ step, n });
        var inc_sets: [3]Layer(.u8) = .{ .{}, .{}, .{} };
        var exc_sets: [3]Layer(.u8) = .{ .{}, .{}, .{} };
        defer {
            for (0..3) |k| inc_sets[k].deinit(t.allocator);
            for (0..3) |k| exc_sets[k].deinit(t.allocator);
        }
        for (0..3) |k| try inc_sets[k].resize(t.allocator, n, .inactive);
        for (0..3) |k| try exc_sets[k].resize(t.allocator, n, .inactive);
        for (0..3) |k| {
            var w: usize = 0;
            while (w < inc_sets[k].activity.items.len) : (w += 1) {
                inc_sets[k].activity.items[w] = @truncate(commonRandU32(&rng));
                inc_sets[k].mixed.items[w] = @truncate(commonRandU32(&rng));
            }
        }
        for (0..3) |k| {
            var w: usize = 0;
            while (w < exc_sets[k].activity.items.len) : (w += 1) {
                exc_sets[k].activity.items[w] = @truncate(commonRandU32(&rng));
                exc_sets[k].mixed.items[w] = @truncate(commonRandU32(&rng));
            }
        }
        if (n > 0) {
            const BW = bit_word.BitWord(.u8);
            const used = BW.bitIdInWord(n);
            if (used != 0) {
                const valid = BW.maskStart(used);
                for (0..3) |k| {
                    inc_sets[k].activity.items[inc_sets[k].activity.items.len - 1] &= valid;
                    inc_sets[k].mixed.items[inc_sets[k].mixed.items.len - 1] &= valid;
                    exc_sets[k].activity.items[exc_sets[k].activity.items.len - 1] &= valid;
                    exc_sets[k].mixed.items[exc_sets[k].mixed.items.len - 1] &= valid;
                }
            }
        }
        if ((step & 1) == 0) {
            commonPoisonPadding(.u8, 3, 3, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, n);
        }
        var exp: [4][256]u32 = undefined;
        const ns = commonOracleStates(.u8, 3, 3, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, n, &exp);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u8, 3, 3, stepPushI, stepPushA, stepPushM, stepPushD, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, &ctx));
        try t.expectEqualSlices(u32, exp[0][0..ns[0]], ctx.inactive[0..ctx.ni]);
        try t.expectEqualSlices(u32, exp[1][0..ns[1]], ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, exp[2][0..ns[2]], ctx.mixed[0..ctx.nm]);
        try t.expectEqualSlices(u32, exp[3][0..ns[3]], ctx.deep[0..ctx.nd]);
    }
}

test "Layer CommonIterator: null side is skipped" {
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .inactive);
        try exc.resize(t.allocator, 70, .inactive);
        layerSetState(.u64, &inc, 5, .active);
        layerSetState(.u64, &inc, 69, .active);
        layerSetState(.u64, &exc, 69, .active);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, null, stepPushA, null, null, .{&inc}, .{&exc}, &ctx));
        try t.expectEqualSlices(u32, &[_]u32{5}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.nm + ctx.nd);
    }
    {
        var inc: Layer(.u8) = .{};
        var exc: Layer(.u8) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        try exc.resize(t.allocator, 10, .inactive);
        layerSetState(.u8, &exc, 3, .active);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u8, 1, 1, stepPushI, null, null, null, .{&inc}, .{&exc}, &ctx));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx.inactive[0..ctx.ni]);
        try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .mixed);
        try exc.resize(t.allocator, 10, .mixed);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, null, null, stepPushM, stepPushD, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.na);
        try t.expectEqual(@as(usize, 10), ctx.nm);
        try t.expectEqual(@as(usize, 0), ctx.nd);
    }
}

test "Layer CommonIterator: early exit stops the walk" {
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .inactive);
        try exc.resize(t.allocator, 70, .active);
        const It = Layer(.u64).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 3 };
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 3), ctx.ni);
        try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ctx.inactive[0..ctx.ni]);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        try exc.resize(t.allocator, 10, .active);
        layerSetState(.u64, &inc, 0, .active);
        layerSetState(.u64, &inc, 9, .active);
        layerSetState(.u64, &exc, 0, .inactive);
        const It = Layer(.u64).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 4 };
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, ctx.inactive[0..ctx.ni]);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 130, .deep_mixed);
        try exc.resize(t.allocator, 130, .deep_mixed);
        const It = Layer(.u64).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 70 };
        try t.expect(It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 64), ctx.nd);
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 1));
        try t.expectEqual(@as(usize, 70), ctx.nd);
        try t.expectEqual(@as(u32, 64), ctx.deep[64]);
        try t.expectEqual(@as(u32, 69), ctx.deep[69]);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .mixed);
        try exc.resize(t.allocator, 10, .deep_mixed);
        const It = Layer(.u64).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 1 };
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 1), ctx.total());
        try t.expectEqual(@as(u32, 0), ctx.mixed[0]);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .active);
        try exc.resize(t.allocator, 10, .inactive);
        const It = Layer(.u64).CommonIterator(1, 1, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{ .stop_after = 50 };
        try t.expect(It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 10), ctx.na);
        var only_d = StepStates{ .stop_after = 1 };
        const ItD = Layer(.u64).CommonIterator(1, 1, *StepStates, null, null, null, stepPushD, .forward);
        try t.expect(ItD.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &only_d }, 0));
        try t.expectEqual(@as(usize, 0), only_d.total());
    }
}

test "Layer CommonIterator: empty sets and zero lens" {
    {
        var inc0: Layer(.u64) = .{};
        var inc1: Layer(.u64) = .{};
        var exc0: Layer(.u64) = .{};
        var exc1: Layer(.u64) = .{};
        defer inc0.deinit(t.allocator);
        defer inc1.deinit(t.allocator);
        defer exc0.deinit(t.allocator);
        defer exc1.deinit(t.allocator);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 2, 2, stepPushI, stepPushA, stepPushM, stepPushD, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.total());
    }
    {
        const It00 = Layer(.u64).CommonIterator(0, 0, *StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
        var ctx = StepStates{};
        try t.expect(It00.step(.{ .includes = .{}, .excludes = .{}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 0), ctx.total());
    }
    {
        var exc: Layer(.u8) = .{};
        defer exc.deinit(t.allocator);
        try exc.resize(t.allocator, 10, .inactive);
        layerSetState(.u8, &exc, 5, .active);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u8, 0, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 9), ctx.na);
        try t.expectEqualSlices(u32, &[_]u32{5}, ctx.inactive[0..ctx.ni]);
        for (ctx.active[0..ctx.na]) |id| try t.expect(id != 5);
    }
    {
        var inc: Layer(.u8) = .{};
        defer inc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        layerSetState(.u8, &inc, 3, .active);
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u8, 1, 0, stepPushI, stepPushA, stepPushM, stepPushD, .{&inc}, .{}, &ctx));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 9), ctx.ni);
        for (ctx.inactive[0..ctx.ni]) |id| try t.expect(id != 3);
    }
}

test "Layer CommonIterator: tail bound never leaks padding" {
    const tails = [_]u32{ 1, 2, 63, 65, 70, 127, 129, 130, 193, 200 };
    for (tails) |n| {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, n, .active);
        try exc.resize(t.allocator, n, .inactive);
        const BW = bit_word.BitWord(.u64);
        const valid = BW.maskStart(BW.bitIdInWord(n));
        inc.activity.items[inc.activity.items.len - 1] |= ~valid;
        inc.mixed.items[inc.mixed.items.len - 1] |= ~valid;
        exc.activity.items[exc.activity.items.len - 1] |= ~valid;
        exc.mixed.items[exc.mixed.items.len - 1] |= ~valid;
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(n, @as(u32, @intCast(ctx.na)));
        try t.expectEqual(@as(usize, 0), ctx.ni + ctx.nm + ctx.nd);
        try t.expectEqual(@as(u32, n - 1), ctx.active[ctx.na - 1]);
    }
    {
        var inc: Layer(.u64) = .{};
        var exc: Layer(.u64) = .{};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 65, .inactive);
        try exc.resize(t.allocator, 65, .active);
        const BW = bit_word.BitWord(.u64);
        const valid = BW.maskStart(BW.bitIdInWord(65));
        inc.activity.items[inc.activity.items.len - 1] |= ~valid;
        inc.mixed.items[inc.mixed.items.len - 1] |= ~valid;
        exc.activity.items[exc.activity.items.len - 1] |= ~valid;
        exc.mixed.items[exc.mixed.items.len - 1] |= ~valid;
        var ctx = StepStates{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushI, stepPushA, stepPushM, stepPushD, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.na + ctx.nm + ctx.nd);
        try t.expectEqual(@as(u32, 65), @as(u32, @intCast(ctx.ni)));
        try t.expectEqual(@as(u32, 64), ctx.inactive[ctx.ni - 1]);
    }
}

test "Layer CommonIterator: global order is ascending" {
    const sizes = [_]u32{ 1, 8, 9, 17, 64, 65, 70, 130 };
    const pats = [_]StepPattern{ .cycle4, .sparse_deep, .pseudo, .all_mixed };
    for (sizes) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                try commonCheckOrder(.u64, 2, 2, n, .{ p0, p1 }, .{ p1, p0 });
                try commonCheckOrder(.u8, 1, 1, n % 25, .{p0}, .{p1});
            }
        }
    }
    try commonCheckOrder(.u64, 1, 0, 70, .{.pseudo}, .{});
    try commonCheckOrder(.u64, 0, 1, 70, .{}, .{.pseudo});
    try commonCheckOrder(.u64, 3, 3, 130, .{ .cycle4, .sparse_deep, .pseudo }, .{ .pseudo, .all_deep, .all_inactive });
}

test "Layer getBit: defaults and set/get roundtrip" {
    const L64 = Layer(.u64);
    var layer: L64 = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 130, .inactive);

    try t.expectEqual(L64.State.inactive, layer.getBit(0));
    try t.expectEqual(L64.State.inactive, layer.getBit(63));
    try t.expectEqual(L64.State.inactive, layer.getBit(64));
    try t.expectEqual(L64.State.inactive, layer.getBit(129));

    layer.setBit(0, .active);
    layer.setBit(1, .mixed);
    layer.setBit(63, .deep_mixed);
    layer.setBit(64, .active);
    layer.setBit(129, .mixed);

    try t.expectEqual(L64.State.active, layer.getBit(0));
    try t.expectEqual(L64.State.mixed, layer.getBit(1));
    try t.expectEqual(L64.State.deep_mixed, layer.getBit(63));
    try t.expectEqual(L64.State.active, layer.getBit(64));
    try t.expectEqual(L64.State.mixed, layer.getBit(129));

    try t.expectEqual(L64.State.inactive, layer.getBit(2));
    try t.expectEqual(L64.State.inactive, layer.getBit(65));

    layer.setBit(1, .deep_mixed);
    try t.expectEqual(L64.State.deep_mixed, layer.getBit(1));

    const before = layer.state_counters;
    _ = layer.getBit(0);
    _ = layer.getBit(64);
    try t.expectEqualSlices(u32, &before, &layer.state_counters);

    const clayer: *const L64 = &layer;
    try t.expectEqual(L64.State.active, clayer.getBit(0));
    try t.expectEqual(L64.State.deep_mixed, clayer.getBit(1));
}

test "Layer getBit: u8 all four states" {
    const L8 = Layer(.u8);
    var layer: L8 = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, 4, .inactive);

    layer.setBit(0, .inactive);
    layer.setBit(1, .active);
    layer.setBit(2, .mixed);
    layer.setBit(3, .deep_mixed);

    try t.expectEqual(L8.State.inactive, layer.getBit(0));
    try t.expectEqual(L8.State.active, layer.getBit(1));
    try t.expectEqual(L8.State.mixed, layer.getBit(2));
    try t.expectEqual(L8.State.deep_mixed, layer.getBit(3));
}

fn rangeCheckLayer(comptime wt: WordType, n: u32, lo: ?u32, hi: ?u32) !void {
    const L = Layer(wt);
    var layer: L = .{};
    defer layer.deinit(t.allocator);
    try layer.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        layerSetState(wt, &layer, i, @enumFromInt(@as(u2, @truncate((i *% 2654435761) >> 29))));
    }

    const r_lo: u32 = @min(lo orelse 0, hi orelse n);
    const r_hi: u32 = @max(lo orelse 0, hi orelse n);
    const c_lo: u32 = @min(r_lo, n);
    const c_hi: u32 = @min(r_hi, n);

    var exp: [4][256]u32 = undefined;
    var ns = [4]usize{ 0, 0, 0, 0 };
    var k: u32 = c_lo;
    while (k < c_hi) : (k += 1) {
        const s: usize = @intFromEnum(layer.getBit(k));
        exp[s][ns[s]] = k;
        ns[s] += 1;
    }

    const ItF = L.Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .forward);
    const ItB = L.Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD, .backward);
    var fwd = StepStates{};
    var bwd = StepStates{};
    try t.expect(ItF.iterateAll(.{ .layer = &layer, .context = &fwd }, lo, hi));
    try t.expect(ItB.iterateAll(.{ .layer = &layer, .context = &bwd }, lo, hi));
    const got_f = [_][]const u32{
        fwd.inactive[0..fwd.ni],
        fwd.active[0..fwd.na],
        fwd.mixed[0..fwd.nm],
        fwd.deep[0..fwd.nd],
    };
    for (0..4) |s| try t.expectEqualSlices(u32, exp[s][0..ns[s]], got_f[s]);
    const got_b = [_][]const u32{
        bwd.inactive[0..bwd.ni],
        bwd.active[0..bwd.na],
        bwd.mixed[0..bwd.nm],
        bwd.deep[0..bwd.nd],
    };
    for (0..4) |s| {
        var rev: [256]u32 = undefined;
        for (0..ns[s]) |j| rev[j] = exp[s][ns[s] - 1 - j];
        try t.expectEqualSlices(u32, rev[0..ns[s]], got_b[s]);
    }
}

test "Layer iterateAll: forward/backward ranges vs oracle" {
    const bounds = [_][2]?u32{
        .{ null, null },
        .{ 0, 130 },
        .{ 5, 70 },
        .{ 70, 5 },
        .{ 0, 1 },
        .{ 63, 65 },
        .{ 64, 128 },
        .{ 129, 130 },
        .{ 130, 130 },
        .{ 500, 600 },
        .{ null, 10 },
        .{ 120, null },
    };
    for (bounds) |b| {
        try rangeCheckLayer(.u64, 130, b[0], b[1]);
        try rangeCheckLayer(.u8, 70, b[0], b[1]);
        try rangeCheckLayer(.u64, 0, b[0], b[1]);
        try rangeCheckLayer(.u64, 1, b[0], b[1]);
    }
}

test "Layer CommonIterator iterateAll: ranges vs oracle" {
    var inc: Layer(.u64) = .{};
    var exc: Layer(.u64) = .{};
    defer inc.deinit(t.allocator);
    defer exc.deinit(t.allocator);
    try inc.resize(t.allocator, 130, .inactive);
    try exc.resize(t.allocator, 130, .inactive);
    var i: u32 = 0;
    while (i < 130) : (i += 1) {
        layerSetState(.u64, &inc, i, if (i % 2 == 0) .active else .inactive);
        layerSetState(.u64, &exc, i, if (i % 5 == 0) .active else .inactive);
    }
    const bounds = [_][2]?u32{
        .{ null, null },
        .{ 10, 100 },
        .{ 100, 10 },
        .{ 0, 64 },
        .{ 129, 130 },
    };
    for (bounds) |b| {
        const r_lo: u32 = @min(b[0] orelse 0, b[1] orelse 130);
        const r_hi: u32 = @max(b[0] orelse 0, b[1] orelse 130);
        const c_lo: u32 = @min(r_lo, 130);
        const c_hi: u32 = @min(r_hi, 130);
        var exp: [256]u32 = undefined;
        var n_exp: usize = 0;
        var k: u32 = c_lo;
        while (k < c_hi) : (k += 1) {
            if (k % 2 == 0 and k % 5 != 0) {
                exp[n_exp] = k;
                n_exp += 1;
            }
        }
        const ItF = Layer(.u64).CommonIterator(1, 1, *StepStates, null, stepPushA, null, null, .forward);
        const ItB = Layer(.u64).CommonIterator(1, 1, *StepStates, null, stepPushA, null, null, .backward);
        var fwd = StepStates{};
        var bwd = StepStates{};
        try t.expect(ItF.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &fwd }, b[0], b[1]));
        try t.expect(ItB.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &bwd }, b[0], b[1]));
        try t.expectEqualSlices(u32, exp[0..n_exp], fwd.active[0..fwd.na]);
        var rev: [256]u32 = undefined;
        for (0..n_exp) |j| rev[j] = exp[n_exp - 1 - j];
        try t.expectEqualSlices(u32, rev[0..n_exp], bwd.active[0..bwd.na]);
    }
}
