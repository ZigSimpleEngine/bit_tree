const std = @import("std");
const utilities = @import("utilities.zig");
const bit_word = @import("bit_word.zig");
const Allocator = std.mem.Allocator;
const ListA64 = utilities.ListA64;

const InlineIteratorCallback = utilities.InlineIteratorCallback;
const iterateActiveBitsInWordInline = utilities.iterateActiveBitsInWordInline;
const BitRange = utilities.BitRange;
const BitState = utilities.BitState;
const WordType = bit_word.WordType;

/// Flat bitset with word storage and exact active-bit counts.
/// - `wt` - word width for backing storage.
///
/// Return - bitset type with word-wise iteration.
pub fn Bitset(comptime wt: WordType) type {
    const Word = wt.Type();
    const bw = bit_word.BitWord(wt);

    return struct {
        const Self = @This();

        /// Backing words, padding bits always zeroed.
        words: ListA64(Word) = .empty,
        /// Valid bits in the set, excludes padding.
        bits_count: u32 = 0,
        /// Cached active total, updated on every mutation.
        active_bits_counter: u32 = 0,

        /// Bundles a bitset pointer with caller context for iteration.
        /// - `Context` - caller-provided iteration context.
        ///
        /// Return - pairing struct passed to every step call.
        pub fn BitsetWithContext(Context: type) type {
            return struct {
                /// Bitset being scanned, provides words and bounds.
                bitset: *Self,
                /// Caller context forwarded to per-bit callbacks.
                context: Context,
            };
        }

        /// Bundles bitset pointers with caller context for common iteration.
        /// - `Context` - caller-provided iteration context.
        ///
        /// Return - pairing struct passed to every step call.
        pub fn BitsetsWithContext(comptime include_len: u32, comptime exclude_len: u32, Context: type) type {
            return struct {
                /// Bitsets whose words are ANDed, provide shared bounds.
                includes: [include_len]*Self,
                /// Bitsets whose words are ORed, provide exclusion lanes.
                excludes: [exclude_len]*Self,
                /// Caller context forwarded to per-bit callbacks.
                context: Context,
            };
        }

        /// Builds a word-wise visitor that dispatches by bit value.
        /// - `Context` - caller-provided iteration context.
        /// - `on_active` - visitor for set bits, null skips them.
        /// - `on_inactive` - visitor for cleared bits, null skips them.
        /// - `direction` - walk order for words and for bits inside one word.
        ///
        /// Return - iterator type with ranged and single-word entry points.
        pub fn Iterator(
            comptime Context: type,
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_inactive: InlineIteratorCallback(Context),
            comptime direction: utilities.Direction,
        ) type {
            return struct {
                /// Visits one word and routes each valid bit to its callback.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - bitset and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                pub inline fn step(data: BitsetWithContext(Context), word_id: u32) bool {
                    if (on_active != null and on_inactive != null) {
                        return stepFull(data, word_id);
                    } else {
                        return stepPending(data, word_id);
                    }
                }
                /// Scans words inside an optional bit range in walk order.
                /// - `data` - bitset and caller context.
                /// - `start_bit` - range edge or null for no lower limit.
                /// - `end_bit` - range edge or null for no upper limit.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub inline fn iterateAll(data: BitsetWithContext(Context), start_bit: ?u32, end_bit: ?u32) bool {
                    const bitset = data.bitset;
                    const context = data.context;
                    const words = bitset.words.items;
                    if (words.len == 0) return true;
                    const range = utilities.resolveRange(bitset.bits_count, start_bit, end_bit);
                    if (range.lo >= range.hi) return true;
                    const lo_word: usize = @intCast(bw.bitToWordId(range.lo));
                    const hi_word: usize = @intCast(bw.bitToWordId(range.hi - 1));

                    if (on_active) |f| {
                        if (on_inactive == null) {
                            var wid: usize = if (direction == .forward) lo_word else hi_word;
                            while (true) {
                                const w = words[wid] & wordRangeMask(wid, range.lo, range.hi);
                                const base = bw.wordToBitId(@truncate(wid));
                                const ok = if (direction == .forward)
                                    flatPeelWord(context, base, w, f)
                                else
                                    flatPeelWordReverse(context, base, w, f);
                                if (!ok) return false;
                                if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                                wid = if (direction == .forward) wid + 1 else wid - 1;
                            }
                            return true;
                        }
                    }

                    if (on_inactive) |f| {
                        if (on_active == null) {
                            var wid: usize = if (direction == .forward) lo_word else hi_word;
                            while (true) {
                                const w = ~words[wid] & wordRangeMask(wid, range.lo, range.hi);
                                const base = bw.wordToBitId(@truncate(wid));
                                const ok = if (direction == .forward)
                                    flatPeelWord(context, base, w, f)
                                else
                                    flatPeelWordReverse(context, base, w, f);
                                if (!ok) return false;
                                if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                                wid = if (direction == .forward) wid + 1 else wid - 1;
                            }
                            return true;
                        }
                    }

                    var wid: usize = if (direction == .forward) lo_word else hi_word;
                    while (true) {
                        const w_base = bw.wordToBitId(@truncate(wid));
                        const s = @max(range.lo, w_base) - w_base;
                        const e = @min(range.hi, w_base + bw.word_type_bits) - w_base;
                        if (!flatMergeWord(context, w_base, words[wid], s, e)) return false;
                        if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                        wid = if (direction == .forward) wid + 1 else wid - 1;
                    }
                    return true;
                }

                /// Keeps only range-covered lanes of one word, padding excluded.
                /// - `word_id` - word index to mask.
                /// - `lo` - resolved range start, inclusive.
                /// - `hi` - resolved range end, exclusive.
                ///
                /// Return - mask with exactly the visited lanes set.
                inline fn wordRangeMask(word_id: usize, lo: u32, hi: u32) Word {
                    const w_base = bw.wordToBitId(@truncate(word_id));
                    const s = @max(lo, w_base) - w_base;
                    const e = @min(hi, w_base + bw.word_type_bits) - w_base;
                    return rangeMask(s, e);
                }

                /// Fast path when both callbacks exist, merges by bound linearly.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - bitset and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepFull(data: BitsetWithContext(Context), word_id: u32) bool {
                    const bitset = data.bitset;
                    const context = data.context;
                    std.debug.assert(word_id < bitset.words.items.len);
                    const words = bitset.words.items;
                    const start = bw.wordToBitId(word_id);
                    const word = words[word_id];
                    var bound: u32 = bw.word_type_bits;
                    if (word_id == words.len - 1) {
                        const used = bw.bitIdInWord(bitset.bits_count);
                        if (used != 0) bound = used;
                    }
                    return flatMergeWord(context, start, word, 0, bound);
                }

                /// Selective path that peels only sides with installed callbacks.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - bitset and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepPending(data: BitsetWithContext(Context), word_id: u32) bool {
                    const bitset = data.bitset;
                    const context = data.context;
                    std.debug.assert(word_id < bitset.words.items.len);

                    const words = bitset.words.items;
                    const start = bw.wordToBitId(word_id);
                    const word = words[word_id];

                    var mask: Word = bw.max_value;
                    if (word_id == words.len - 1) {
                        const used = bw.bitIdInWord(bitset.bits_count);
                        if (used != 0) mask = bw.maskStart(used);
                    }

                    if (on_active) |f| {
                        if (on_inactive == null) {
                            const only_active_word = word & mask;
                            const ok = if (direction == .forward)
                                flatPeelWord(context, start, only_active_word, f)
                            else
                                flatPeelWordReverse(context, start, only_active_word, f);
                            return ok;
                        }
                    }

                    if (on_inactive) |f| {
                        if (on_active == null) {
                            const only_inactive_word = (~word) & mask;
                            const ok = if (direction == .forward)
                                flatPeelWord(context, start, only_inactive_word, f)
                            else
                                flatPeelWordReverse(context, start, only_inactive_word, f);
                            return ok;
                        }
                    }

                    const only_active_word = word & mask;
                    const only_inactive_word = (~word) & mask;

                    var pending_word: Word = 0;
                    if (on_active != null) pending_word |= only_active_word;
                    if (on_inactive != null) pending_word |= only_inactive_word;

                    if (direction == .forward) {
                        while (pending_word != 0) {
                            const bit_id_in_word: u32 = @ctz(pending_word);
                            const bit_id = start + bit_id_in_word;
                            const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                            pending_word &= pending_word - 1;

                            if (on_active) |f| {
                                if ((only_active_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }

                            if (on_inactive) |f| {
                                if ((only_inactive_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }
                        }
                    } else {
                        while (pending_word != 0) {
                            const lz: u32 = @clz(pending_word);
                            const bit_id_in_word = bw.word_type_bits - 1 - lz;
                            const bit_id = start + bit_id_in_word;
                            const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                            pending_word ^= bit;

                            if (on_active) |f| {
                                if ((only_active_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }

                            if (on_inactive) |f| {
                                if ((only_inactive_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }
                        }
                    }

                    return true;
                }

                /// Peels one masked word into a single callback in ascending bit order.
                /// - `context` - caller context for the callback.
                /// - `base` - global id of the word start.
                /// - `word` - pre-masked word to peel.
                /// - `f` - inline visitor, false stops the walk.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatPeelWord(
                    context: Context,
                    base: u32,
                    word: Word,
                    comptime f: fn (context: Context, bit_id: u32) callconv(.@"inline") bool,
                ) bool {
                    var w = word;
                    while (w != 0) {
                        const bit_id_in_word: u32 = @ctz(w);
                        w &= w - 1;
                        if (!f(context, base + bit_id_in_word)) return false;
                    }
                    return true;
                }

                /// Peels one masked word into a single callback in reverse bit order.
                /// - `context` - caller context for the callback.
                /// - `base` - global id of the word start.
                /// - `word` - pre-masked word to peel.
                /// - `f` - inline visitor, false stops the walk.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatPeelWordReverse(
                    context: Context,
                    base: u32,
                    word: Word,
                    comptime f: fn (context: Context, bit_id: u32) callconv(.@"inline") bool,
                ) bool {
                    var w = word;
                    while (w != 0) {
                        const lz: u32 = @clz(w);
                        const bit_id_in_word = bw.word_type_bits - 1 - lz;
                        w ^= @as(Word, 1) << @truncate(bit_id_in_word);
                        if (!f(context, base + bit_id_in_word)) return false;
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

                /// Merges one bounded word into active and inactive callbacks linearly.
                /// - `context` - caller context for the callbacks.
                /// - `base` - global id of the word start.
                /// - `word` - raw word to classify bit by bit.
                /// - `sub_lo` - first classified lane, inclusive.
                /// - `sub_hi` - one past the last classified lane, exclusive.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatMergeWord(
                    context: Context,
                    base: u32,
                    word: Word,
                    sub_lo: u32,
                    sub_hi: u32,
                ) bool {
                    if (direction == .forward) {
                        var i: u32 = sub_lo;
                        while (i < sub_hi) : (i += 1) {
                            if (((word >> @truncate(i)) & 1) != 0) {
                                if (on_active) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            } else {
                                if (on_inactive) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            }
                        }
                    } else {
                        var i: u32 = sub_hi;
                        while (i > sub_lo) {
                            i -= 1;
                            if (((word >> @truncate(i)) & 1) != 0) {
                                if (on_active) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            } else {
                                if (on_inactive) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            }
                        }
                    }
                    return true;
                }
            };
        }

        /// Builds a word-wise visitor that dispatches by common masks.
        /// - `include_len` - bitsets whose words are ANDed.
        /// - `exclude_len` - bitsets whose words are ORed into the veto mask.
        /// - `Context` - caller-provided iteration context.
        /// - `on_active` - visitor for common set bits, null skips them.
        /// - `on_inactive` - visitor for common cleared bits, null skips them.
        /// - `direction` - walk order for words and for bits inside one word.
        ///
        /// Return - iterator type with ranged and single-word entry points.
        pub fn CommonIterator(
            comptime include_len: u32,
            comptime exclude_len: u32,
            comptime Context: type,
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_inactive: InlineIteratorCallback(Context),
            comptime direction: utilities.Direction,
        ) type {
            return struct {
                /// Visits one word and routes each valid bit to its callback.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - bitsets and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                pub inline fn step(data: BitsetsWithContext(include_len, exclude_len, Context), word_id: u32) bool {
                    if (include_len == 0 and exclude_len == 0) return true;
                    if (on_active != null and on_inactive != null) {
                        return stepFull(data, word_id);
                    } else {
                        return stepPending(data, word_id);
                    }
                }
                /// Scans words inside an optional bit range in walk order.
                /// - `data` - bitsets and caller context.
                /// - `start_bit` - range edge or null for no lower limit.
                /// - `end_bit` - range edge or null for no upper limit.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub inline fn iterateAll(data: BitsetsWithContext(include_len, exclude_len, Context), start_bit: ?u32, end_bit: ?u32) bool {
                    if (include_len == 0 and exclude_len == 0) return true;
                    const first = if (include_len > 0) data.includes[0] else data.excludes[0];
                    const context = data.context;
                    if (first.words.items.len == 0) return true;
                    const range = utilities.resolveRange(first.bits_count, start_bit, end_bit);
                    if (range.lo >= range.hi) return true;
                    const lo_word: usize = @intCast(bw.bitToWordId(range.lo));
                    const hi_word: usize = @intCast(bw.bitToWordId(range.hi - 1));

                    if (on_active) |f| {
                        if (on_inactive == null) {
                            var wid: usize = if (direction == .forward) lo_word else hi_word;
                            while (true) {
                                const word_id: u32 = @truncate(wid);
                                const w = commonActiveWord(data, word_id) & wordRangeMask(wid, range.lo, range.hi);
                                const base = bw.wordToBitId(word_id);
                                const ok = if (direction == .forward)
                                    flatPeelWord(context, base, w, f)
                                else
                                    flatPeelWordReverse(context, base, w, f);
                                if (!ok) return false;
                                if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                                wid = if (direction == .forward) wid + 1 else wid - 1;
                            }
                            return true;
                        }
                    }

                    if (on_inactive) |f| {
                        if (on_active == null) {
                            var wid: usize = if (direction == .forward) lo_word else hi_word;
                            while (true) {
                                const word_id: u32 = @truncate(wid);
                                const w = commonInactiveWord(data, word_id) & wordRangeMask(wid, range.lo, range.hi);
                                const base = bw.wordToBitId(word_id);
                                const ok = if (direction == .forward)
                                    flatPeelWord(context, base, w, f)
                                else
                                    flatPeelWordReverse(context, base, w, f);
                                if (!ok) return false;
                                if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                                wid = if (direction == .forward) wid + 1 else wid - 1;
                            }
                            return true;
                        }
                    }

                    var wid: usize = if (direction == .forward) lo_word else hi_word;
                    while (true) {
                        const word_id: u32 = @truncate(wid);
                        const w_base = bw.wordToBitId(word_id);
                        const s = @max(range.lo, w_base) - w_base;
                        const e = @min(range.hi, w_base + bw.word_type_bits) - w_base;
                        if (!flatMergeWord(context, w_base, commonActiveWord(data, word_id), commonInactiveWord(data, word_id), s, e)) return false;
                        if (wid == (if (direction == .forward) hi_word else lo_word)) break;
                        wid = if (direction == .forward) wid + 1 else wid - 1;
                    }
                    return true;
                }

                /// Keeps only range-covered lanes of one word, padding excluded.
                /// - `word_id` - word index to mask.
                /// - `lo` - resolved range start, inclusive.
                /// - `hi` - resolved range end, exclusive.
                ///
                /// Return - mask with exactly the visited lanes set.
                inline fn wordRangeMask(word_id: usize, lo: u32, hi: u32) Word {
                    const w_base = bw.wordToBitId(@truncate(word_id));
                    const s = @max(lo, w_base) - w_base;
                    const e = @min(hi, w_base + bw.word_type_bits) - w_base;
                    return rangeMask(s, e);
                }

                /// Fast path when both callbacks exist, merges by bound linearly.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - bitsets and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepFull(data: BitsetsWithContext(include_len, exclude_len, Context), word_id: u32) bool {
                    const first = if (include_len > 0) data.includes[0] else data.excludes[0];
                    const context = data.context;
                    std.debug.assert(word_id < first.words.items.len);
                    const start = bw.wordToBitId(word_id);
                    const active_mask = commonActiveWord(data, word_id);
                    const inactive_mask = commonInactiveWord(data, word_id);
                    var bound: u32 = bw.word_type_bits;
                    if (word_id == first.words.items.len - 1) {
                        const used = bw.bitIdInWord(first.bits_count);
                        if (used != 0) bound = used;
                    }
                    return flatMergeWord(context, start, active_mask, inactive_mask, 0, bound);
                }

                /// Selective path that peels only sides with installed callbacks.
                /// Lanes are visited in factory `direction` order.
                /// - `data` - bitsets and caller context.
                /// - `word_id` - word index to scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn stepPending(data: BitsetsWithContext(include_len, exclude_len, Context), word_id: u32) bool {
                    const first = if (include_len > 0) data.includes[0] else data.excludes[0];
                    const context = data.context;
                    std.debug.assert(word_id < first.words.items.len);

                    const start = bw.wordToBitId(word_id);

                    var mask: Word = bw.max_value;
                    if (word_id == first.words.items.len - 1) {
                        const used = bw.bitIdInWord(first.bits_count);
                        if (used != 0) mask = bw.maskStart(used);
                    }

                    if (on_active) |f| {
                        if (on_inactive == null) {
                            const only_active_word = commonActiveWord(data, word_id) & mask;
                            const ok = if (direction == .forward)
                                flatPeelWord(context, start, only_active_word, f)
                            else
                                flatPeelWordReverse(context, start, only_active_word, f);
                            return ok;
                        }
                    }

                    if (on_inactive) |f| {
                        if (on_active == null) {
                            const only_inactive_word = commonInactiveWord(data, word_id) & mask;
                            const ok = if (direction == .forward)
                                flatPeelWord(context, start, only_inactive_word, f)
                            else
                                flatPeelWordReverse(context, start, only_inactive_word, f);
                            return ok;
                        }
                    }

                    const only_active_word = commonActiveWord(data, word_id) & mask;
                    const only_inactive_word = commonInactiveWord(data, word_id) & mask;

                    var pending_word: Word = 0;
                    if (on_active != null) pending_word |= only_active_word;
                    if (on_inactive != null) pending_word |= only_inactive_word;

                    if (direction == .forward) {
                        while (pending_word != 0) {
                            const bit_id_in_word: u32 = @ctz(pending_word);
                            const bit_id = start + bit_id_in_word;
                            const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                            pending_word &= pending_word - 1;

                            if (on_active) |f| {
                                if ((only_active_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }

                            if (on_inactive) |f| {
                                if ((only_inactive_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }
                        }
                    } else {
                        while (pending_word != 0) {
                            const lz: u32 = @clz(pending_word);
                            const bit_id_in_word = bw.word_type_bits - 1 - lz;
                            const bit_id = start + bit_id_in_word;
                            const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                            pending_word ^= bit;

                            if (on_active) |f| {
                                if ((only_active_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }

                            if (on_inactive) |f| {
                                if ((only_inactive_word & bit) != 0) {
                                    if (!f(context, bit_id)) return false;
                                    continue;
                                }
                            }
                        }
                    }

                    return true;
                }

                /// Merges include and exclude words of one index into a shared active mask.
                /// - `data` - bitsets and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - active lanes with all includes set and no excludes set.
                inline fn commonActiveWord(data: BitsetsWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    var include_mask: Word = bw.max_value;
                    inline for (0..include_len) |k| {
                        include_mask &= data.includes[k].words.items[word_id];
                    }
                    var exclude_mask: Word = 0;
                    inline for (0..exclude_len) |k| {
                        exclude_mask |= data.excludes[k].words.items[word_id];
                    }
                    return include_mask & ~exclude_mask;
                }

                /// Merges inverted include and exclude words into a shared inactive mask.
                /// - `data` - bitsets and caller context.
                /// - `word_id` - word index to merge.
                ///
                /// Return - inactive lanes with all includes cleared and excludes merged.
                inline fn commonInactiveWord(data: BitsetsWithContext(include_len, exclude_len, Context), word_id: u32) Word {
                    var include_inv_mask: Word = bw.max_value;
                    inline for (0..include_len) |k| {
                        include_inv_mask &= ~data.includes[k].words.items[word_id];
                    }
                    var exclude_inv_mask: Word = 0;
                    inline for (0..exclude_len) |k| {
                        exclude_inv_mask |= ~data.excludes[k].words.items[word_id];
                    }
                    return include_inv_mask & ~exclude_inv_mask;
                }

                /// Peels one masked word into a single callback in ascending bit order.
                /// - `context` - caller context for the callback.
                /// - `base` - global id of the word start.
                /// - `word` - pre-masked word to peel.
                /// - `f` - inline visitor, false stops the walk.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatPeelWord(
                    context: Context,
                    base: u32,
                    word: Word,
                    comptime f: fn (context: Context, bit_id: u32) callconv(.@"inline") bool,
                ) bool {
                    var w = word;
                    while (w != 0) {
                        const bit_id_in_word: u32 = @ctz(w);
                        w &= w - 1;
                        if (!f(context, base + bit_id_in_word)) return false;
                    }
                    return true;
                }

                /// Peels one masked word into a single callback in reverse bit order.
                /// - `context` - caller context for the callback.
                /// - `base` - global id of the word start.
                /// - `word` - pre-masked word to peel.
                /// - `f` - inline visitor, false stops the walk.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatPeelWordReverse(
                    context: Context,
                    base: u32,
                    word: Word,
                    comptime f: fn (context: Context, bit_id: u32) callconv(.@"inline") bool,
                ) bool {
                    var w = word;
                    while (w != 0) {
                        const lz: u32 = @clz(w);
                        const bit_id_in_word = bw.word_type_bits - 1 - lz;
                        w ^= @as(Word, 1) << @truncate(bit_id_in_word);
                        if (!f(context, base + bit_id_in_word)) return false;
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

                /// Merges one bounded word into active and inactive callbacks linearly.
                /// - `context` - caller context for the callbacks.
                /// - `base` - global id of the word start.
                /// - `active_mask` - pre-merged active lanes to visit.
                /// - `inactive_mask` - pre-merged inactive lanes to visit.
                /// - `sub_lo` - first classified lane, inclusive.
                /// - `sub_hi` - one past the last classified lane, exclusive.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatMergeWord(
                    context: Context,
                    base: u32,
                    active_mask: Word,
                    inactive_mask: Word,
                    sub_lo: u32,
                    sub_hi: u32,
                ) bool {
                    if (direction == .forward) {
                        var i: u32 = sub_lo;
                        while (i < sub_hi) : (i += 1) {
                            if (((active_mask >> @truncate(i)) & 1) != 0) {
                                if (on_active) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            } else if (((inactive_mask >> @truncate(i)) & 1) != 0) {
                                if (on_inactive) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            }
                        }
                    } else {
                        var i: u32 = sub_hi;
                        while (i > sub_lo) {
                            i -= 1;
                            if (((active_mask >> @truncate(i)) & 1) != 0) {
                                if (on_active) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            } else if (((inactive_mask >> @truncate(i)) & 1) != 0) {
                                if (on_inactive) |f| {
                                    if (!f(context, base + i)) return false;
                                }
                            }
                        }
                    }
                    return true;
                }
            };
        }

        /// Releases backing word storage.
        /// - `self` - bitset to destroy.
        /// - `allocator` - allocator that owns the words.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.words.deinit(allocator);
        }

        /// Writes masked lanes of one word and refreshes the active count.
        /// - `self` - bitset to update.
        /// - `id` - word index to patch.
        /// - `word` - new lanes to install.
        /// - `mask` - selects lanes to overwrite, padding is ignored.
        pub fn setWord(self: *Self, id: u32, word: Word, mask: Word) void {
            if (mask == 0) return;
            std.debug.assert(id < self.words.items.len);
            var eff: Word = mask;
            const words_count = self.words.items.len;
            if (words_count > 0 and id == words_count - 1) {
                const used = bw.bitIdInWord(self.bits_count);
                if (used != 0) eff &= bw.maskStart(used);
            }
            if (eff == 0) return;
            const old_word: Word = self.words.items[id];
            const new_word: Word = bw.merge(old_word, word, eff);
            if (new_word == old_word) return;
            const old_active: u32 = @popCount(old_word & eff);
            const new_active: u32 = @popCount(word & eff);
            self.words.items[id] = new_word;
            self.active_bits_counter = self.active_bits_counter - old_active + new_active;
        }

        /// Reads one bit without touching the active counter.
        /// - `self` - bitset to read.
        /// - `id` - global bit position to read.
        ///
        /// Return - stored bit state.
        pub fn getBit(self: *const Self, id: u32) BitState {
            std.debug.assert(id < self.bits_count);
            const word_id = bw.bitToWordId(id);
            const bit_id_in_word = bw.bitIdInWord(id);
            return bw.readBitState(self.words.items[word_id], bit_id_in_word);
        }

        /// Writes one bit and moves the active counter.
        /// - `self` - bitset to update.
        /// - `id` - global bit position to write.
        /// - `value` - state to store.
        pub fn setBit(self: *Self, id: u32, value: BitState) void {
            std.debug.assert(id < self.bits_count);
            const word_id = bw.bitToWordId(id);
            const bit_id_in_word = bw.bitIdInWord(id);
            const old_word: Word = self.words.items[word_id];
            const old_value = bw.readBitState(old_word, bit_id_in_word);
            if (old_value == value) return;
            if (value == .active) {
                self.words.items[word_id] = old_word | (@as(Word, 1) << bit_id_in_word);
                self.active_bits_counter += 1;
            } else {
                self.words.items[word_id] = old_word & ~(@as(Word, 1) << bit_id_in_word);
                self.active_bits_counter -= 1;
            }
        }

        /// Grows or shrinks the set while keeping counts and padding exact.
        /// - `self` - bitset to resize.
        /// - `allocator` - owns backing storage.
        /// - `new_bits_count` - target valid bits.
        /// - `created_bits_value` - state filling newly created bits.
        ///
        /// Return - error on allocation failure.
        pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, created_bits_value: BitState) !void {
            const old_bits_count = self.bits_count;
            if (old_bits_count == new_bits_count) return;

            self.bits_count = new_bits_count;
            const words = self.words.items;
            const old_words_count: u32 = @truncate(words.len);
            const new_words_count = bw.bitsToWordsCount(new_bits_count);

            if (old_bits_count < new_bits_count) {
                const bits_count_delta = new_bits_count - old_bits_count;
                if (created_bits_value == .active) {
                    self.active_bits_counter += bits_count_delta;
                }

                const created_word_value = created_bits_value.toWordState(wt);
                if (bw.bitIdInWord(old_bits_count) != 0) {
                    const last_old_bit_id = old_bits_count - 1;
                    const last_old_word_id = bw.bitToWordId(last_old_bit_id);
                    const last_old_word = words[last_old_word_id];
                    const end_mask = bw.maskEnd(@truncate(bw.remainBitsInWord(bw.bitIdInWord(last_old_bit_id))));
                    words[last_old_word_id] = bw.merge(last_old_word, created_word_value, end_mask);
                }

                if (new_words_count != old_words_count) {
                    try self.words.resize(allocator, new_words_count);
                    const new_words = self.words.items;

                    for (old_words_count..new_words_count) |i| {
                        new_words[i] = created_word_value;
                    }
                }

                if (bw.bitIdInWord(new_bits_count) != 0) {
                    const last_new_bit_id = new_bits_count - 1;
                    const last_new_word_id = bw.bitToWordId(last_new_bit_id);
                    self.words.items[last_new_word_id] &= bw.maskStart(bw.bitIdInWord(new_bits_count));
                }
            } else {
                if (bw.bitIdInWord(new_bits_count) != 0) {
                    const last_new_bit_id = new_bits_count - 1;
                    const last_old_bit_id = old_bits_count - 1;
                    const last_new_word_id = bw.bitToWordId(last_new_bit_id);
                    const last_old_word_id = bw.bitToWordId(last_old_bit_id);
                    const last_new_word = words[last_new_word_id];
                    if (new_words_count == old_words_count) {
                        const start_end_mask = bw.maskStartEnd(
                            bw.bitIdInWord(new_bits_count),
                            bw.remainBitsInWord(last_old_bit_id),
                        );
                        const new_word = bw.merge(last_new_word, 0, start_end_mask);
                        self.active_bits_counter -= @popCount(new_word);
                    } else {
                        const start_mask = bw.maskStart(bw.bitIdInWord(new_bits_count));
                        const end_mask = bw.maskEnd(bw.remainBitsInWord(last_old_bit_id));
                        const masked_last_new_word = bw.merge(last_new_word, 0, start_mask);
                        const masked_last_old_word = bw.merge(words[last_old_word_id], 0, end_mask);
                        self.active_bits_counter -= (@popCount(masked_last_new_word) + @popCount(masked_last_old_word));
                    }
                    words[last_new_word_id] &= bw.maskStart(bw.bitIdInWord(new_bits_count));
                }

                if (new_words_count != old_words_count) {
                    const skip_last_word: u32 = @intFromBool(bw.bitIdInWord(new_bits_count) != 0);
                    for (new_words_count..old_words_count - skip_last_word) |i| {
                        self.active_bits_counter -= @popCount(words[i]);
                    }

                    try self.words.resize(allocator, new_words_count);
                }
            }
        }
    };
}

const t = std.testing;

fn bitsetScanActiveCount(comptime wt: WordType, bs: *const Bitset(wt)) u32 {
    const BW = bit_word.BitWord(wt);
    var acc: u32 = 0;
    var i: u32 = 0;
    while (i < bs.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        const bid = BW.bitIdInWord(i);
        if (BW.readBitState(bs.words.items[wid], bid) == .active) acc += 1;
    }
    return acc;
}

fn expectBitsetInvariants(comptime wt: WordType, bs: *const Bitset(wt)) !void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    const want_words: usize = @intCast(BW.bitsToWordsCount(bs.bits_count));
    try t.expectEqual(want_words, bs.words.items.len);
    if (bs.bits_count == 0) {
        try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
        return;
    }
    const used = BW.bitIdInWord(bs.bits_count);
    if (used != 0) {
        const valid = BW.maskStart(used);
        const last = bs.words.items[want_words - 1];
        try t.expectEqual(@as(Word, 0), last & ~valid);
    }
    const scanned = bitsetScanActiveCount(wt, bs);
    try t.expectEqual(scanned, bs.active_bits_counter);
    try t.expect(bs.active_bits_counter <= bs.bits_count);
}

const ResizePattern = enum { zero, one, alt01, alt10, every3, pseudo };

fn patternBit(p: ResizePattern, i: u32) BitState {
    return switch (p) {
        .zero => .inactive,
        .one => .active,
        .alt01 => if ((i & 1) == 0) .active else .inactive,
        .alt10 => if ((i & 1) == 0) .inactive else .active,
        .every3 => if ((i % 3) == 0) .active else .inactive,
        .pseudo => if ((((i *% 1664525) +% 1013904223) >> 15) & 1 == 1) .active else .inactive,
    };
}

fn initBitsetPattern(comptime wt: WordType, allocator: Allocator, bits: u32, pat: ResizePattern) !Bitset(wt) {
    const Word = wt.Type();
    var bs = Bitset(wt){};
    try bs.resize(allocator, bits, .inactive);
    const BW = bit_word.BitWord(wt);
    var want_active: u32 = 0;
    var i: u32 = 0;
    while (i < bits) : (i += 1) {
        if (patternBit(pat, i) == .active) {
            const wid: usize = @intCast(BW.bitToWordId(i));
            const bid = BW.bitIdInWord(i);
            bs.words.items[wid] |= (@as(Word, 1) << bid);
            want_active += 1;
        }
    }
    bs.active_bits_counter = want_active;
    try expectBitsetInvariants(wt, &bs);
    return bs;
}

fn checkOneResize(comptime wt: WordType, allocator: Allocator, old: u32, new: u32, created: BitState, pat: ResizePattern) !void {
    var bs = try initBitsetPattern(wt, allocator, old, pat);
    defer bs.deinit(allocator);

    var expected_active: u32 = 0;
    const min_len = @min(old, new);
    var k: u32 = 0;
    while (k < min_len) : (k += 1) {
        if (patternBit(pat, k) == .active) expected_active += 1;
    }
    if (new > old and created == .active) expected_active += new - old;

    errdefer std.debug.print(
        "CTX Word={s} old={} new={} created={s} pat={s} bits={} counter={} expected_active={}\n",
        .{ @tagName(wt), old, new, @tagName(created), @tagName(pat), bs.bits_count, bs.active_bits_counter, expected_active },
    );

    try bs.resize(allocator, new, created);

    const BW = bit_word.BitWord(wt);
    try t.expectEqual(new, bs.bits_count);
    const want_words: usize = @intCast(BW.bitsToWordsCount(new));
    try t.expectEqual(want_words, bs.words.items.len);
    try t.expectEqual(expected_active, bs.active_bits_counter);

    var j: u32 = 0;
    while (j < new) : (j += 1) {
        const want: BitState = if (j < old) patternBit(pat, j) else created;
        const wid: usize = @intCast(BW.bitToWordId(j));
        const bid = BW.bitIdInWord(j);
        const got = BW.readBitState(bs.words.items[wid], bid);
        try t.expectEqual(want, got);
    }
    try expectBitsetInvariants(wt, &bs);
}

fn nextRandU32(state: *u64) u32 {
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @truncate((x *% 0x2545F4914F6CDD1D) >> 32);
}

test "Bitset resize" {
    var bs = Bitset(.u64){};
    defer bs.deinit(t.allocator);

    try bs.resize(t.allocator, 100, BitState.active);
    try t.expectEqual(100, bs.bits_count);
    try t.expectEqual(100, bs.active_bits_counter);
    try t.expectEqual(2, bs.words.items.len);

    try bs.resize(t.allocator, 127, BitState.inactive);
    try t.expectEqual(127, bs.bits_count);
    try t.expectEqual(100, bs.active_bits_counter);
    try t.expectEqual(2, bs.words.items.len);

    try bs.resize(t.allocator, 80, BitState.inactive);
    try t.expectEqual(80, bs.bits_count);
    try t.expectEqual(80, bs.active_bits_counter);
    try t.expectEqual(2, bs.words.items.len);

    try bs.resize(t.allocator, 10, BitState.inactive);
    try t.expectEqual(10, bs.bits_count);
    try t.expectEqual(10, bs.active_bits_counter);
    try t.expectEqual(1, bs.words.items.len);

    try bs.resize(t.allocator, 60, BitState.inactive);
    try t.expectEqual(60, bs.bits_count);
    try t.expectEqual(10, bs.active_bits_counter);
    try t.expectEqual(1, bs.words.items.len);

    try bs.resize(t.allocator, 200, BitState.active);
    try t.expectEqual(200, bs.bits_count);
    try t.expectEqual(150, bs.active_bits_counter);
    try t.expectEqual(4, bs.words.items.len);

    try bs.resize(t.allocator, 10000, BitState.inactive);
    try t.expectEqual(10000, bs.bits_count);
    try t.expectEqual(150, bs.active_bits_counter);
    try t.expectEqual(157, bs.words.items.len);

    try bs.resize(t.allocator, 2000, BitState.active);
    try t.expectEqual(2000, bs.bits_count);
    try t.expectEqual(150, bs.active_bits_counter);
    try t.expectEqual(32, bs.words.items.len);

    try bs.resize(t.allocator, 3000, BitState.active);
    try t.expectEqual(3000, bs.bits_count);
    try t.expectEqual(1150, bs.active_bits_counter);
    try t.expectEqual(47, bs.words.items.len);
}

test "Bitset resize minimal: grow active unaligned must keep padding zero" {
    var bs = Bitset(.u64){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .active);
    try t.expectEqual(@as(u32, 10), bs.bits_count);
    try t.expectEqual(@as(u32, 10), bs.active_bits_counter);
    try t.expectEqual(@as(usize, 1), bs.words.items.len);
    const BW = bit_word.BitWord(.u64);
    try t.expectEqual(@as(u64, 0), bs.words.items[0] & ~BW.maskStart(10));
    try expectBitsetInvariants(.u64, &bs);

    var bs2 = Bitset(.u64){};
    defer bs2.deinit(t.allocator);
    try bs2.resize(t.allocator, 65, .active);
    try t.expectEqual(@as(u32, 65), bs2.active_bits_counter);
    try t.expectEqual(@as(u64, 0), bs2.words.items[1] & ~BW.maskStart(1));
    try expectBitsetInvariants(.u64, &bs2);
}

test "Bitset resize minimal: shrink to aligned must fix counter" {
    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 128, .active);
        try t.expectEqual(@as(u32, 128), bs.active_bits_counter);
        try bs.resize(t.allocator, 64, .inactive);
        try t.expectEqual(@as(u32, 64), bs.bits_count);
        try t.expectEqual(@as(usize, 1), bs.words.items.len);
        try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
        try expectBitsetInvariants(.u64, &bs);
    }
    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 64, .active);
        try bs.resize(t.allocator, 0, .inactive);
        try t.expectEqual(@as(u32, 0), bs.bits_count);
        try t.expectEqual(@as(usize, 0), bs.words.items.len);
        try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
    }
    {
        var bs = try initBitsetPattern(.u64, t.allocator, 65, .one);
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 64, .inactive);
        try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
        try expectBitsetInvariants(.u64, &bs);
    }
    {
        var bs = try initBitsetPattern(.u8, t.allocator, 9, .one);
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 8, .inactive);
        try t.expectEqual(@as(u32, 8), bs.active_bits_counter);
        try expectBitsetInvariants(.u8, &bs);
    }
}

test "Bitset resize minimal: shrink same-word must clear tail" {
    var bs = try initBitsetPattern(.u64, t.allocator, 20, .one);
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(u32, 10), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);
}

test "Bitset(u8) resize exhaustive 0..20 oracle" {
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (0..21) |old_usize| {
        for (0..21) |new_usize| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(.u8, t.allocator, @intCast(old_usize), @intCast(new_usize), created, pat);
                }
            }
        }
    }
}

test "Bitset(u64) resize boundary oracle" {
    const bounds = [_]u32{ 0, 1, 2, 63, 64, 65, 70, 100, 126, 127, 128, 129, 130, 191, 192, 193, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (bounds) |old| {
        for (bounds) |new| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(.u64, t.allocator, old, new, created, pat);
                }
            }
        }
    }
}

test "Bitset(u32) resize boundary oracle" {
    const bounds = [_]u32{ 0, 1, 2, 31, 32, 33, 40, 63, 64, 65, 95, 96, 97, 100 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (bounds) |old| {
        for (bounds) |new| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(.u32, t.allocator, old, new, created, pat);
                }
            }
        }
    }
}

test "Bitset(u16) resize boundary oracle" {
    const bounds = [_]u32{ 0, 1, 2, 15, 16, 17, 24, 31, 32, 33, 47, 48, 49 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    const createds = [_]BitState{ .inactive, .active };
    for (bounds) |old| {
        for (bounds) |new| {
            for (createds) |created| {
                for (patterns) |pat| {
                    try checkOneResize(.u16, t.allocator, old, new, created, pat);
                }
            }
        }
    }
}

test "Bitset resize explicit corner table u64" {
    const Case = struct { old: u32, new: u32, created: BitState, pat: ResizePattern };
    const cases = [_]Case{
        .{ .old = 0, .new = 0, .created = .active, .pat = .zero },
        .{ .old = 0, .new = 1, .created = .active, .pat = .zero },
        .{ .old = 0, .new = 1, .created = .inactive, .pat = .zero },
        .{ .old = 0, .new = 63, .created = .active, .pat = .zero },
        .{ .old = 0, .new = 64, .created = .active, .pat = .zero },
        .{ .old = 0, .new = 65, .created = .active, .pat = .zero },
        .{ .old = 0, .new = 128, .created = .active, .pat = .zero },
        .{ .old = 1, .new = 0, .created = .inactive, .pat = .one },
        .{ .old = 64, .new = 0, .created = .inactive, .pat = .one },
        .{ .old = 65, .new = 0, .created = .inactive, .pat = .one },
        .{ .old = 10, .new = 10, .created = .active, .pat = .alt01 },
        .{ .old = 64, .new = 64, .created = .active, .pat = .one },
        .{ .old = 10, .new = 20, .created = .active, .pat = .zero },
        .{ .old = 20, .new = 10, .created = .inactive, .pat = .one },
        .{ .old = 10, .new = 64, .created = .inactive, .pat = .one },
        .{ .old = 64, .new = 10, .created = .inactive, .pat = .one },
        .{ .old = 60, .new = 65, .created = .active, .pat = .alt01 },
        .{ .old = 65, .new = 60, .created = .inactive, .pat = .one },
        .{ .old = 10, .new = 70, .created = .active, .pat = .zero },
        .{ .old = 70, .new = 10, .created = .inactive, .pat = .one },
        .{ .old = 64, .new = 65, .created = .active, .pat = .one },
        .{ .old = 65, .new = 64, .created = .inactive, .pat = .one },
        .{ .old = 63, .new = 64, .created = .active, .pat = .alt10 },
        .{ .old = 64, .new = 63, .created = .inactive, .pat = .one },
        .{ .old = 127, .new = 128, .created = .active, .pat = .one },
        .{ .old = 128, .new = 127, .created = .inactive, .pat = .one },
        .{ .old = 128, .new = 129, .created = .active, .pat = .pseudo },
        .{ .old = 129, .new = 128, .created = .inactive, .pat = .pseudo },
        .{ .old = 64, .new = 128, .created = .active, .pat = .alt01 },
        .{ .old = 128, .new = 64, .created = .inactive, .pat = .one },
        .{ .old = 192, .new = 64, .created = .inactive, .pat = .one },
        .{ .old = 200, .new = 64, .created = .inactive, .pat = .one },
        .{ .old = 70, .new = 64, .created = .inactive, .pat = .one },
        .{ .old = 100, .new = 70, .created = .inactive, .pat = .one },
        .{ .old = 2000, .new = 3000, .created = .active, .pat = .pseudo },
        .{ .old = 3000, .new = 2000, .created = .inactive, .pat = .one },
        .{ .old = 10000, .new = 2000, .created = .inactive, .pat = .pseudo },
    };
    for (cases) |c| {
        try checkOneResize(.u64, t.allocator, c.old, c.new, c.created, c.pat);
    }
}

test "Bitset(u64) resize sequential fuzz vs oracle" {
    var bs = Bitset(.u64){};
    defer bs.deinit(t.allocator);
    var expected: [512]u1 = [_]u1{0} ** 512;
    var expected_len: u32 = 0;
    var expected_active: u32 = 0;
    var rng: u64 = 0x9E3779B97F4A7C15;
    const BW = bit_word.BitWord(.u64);
    var step: usize = 0;
    while (step < 500) : (step += 1) {
        const r1 = nextRandU32(&rng);
        const r2 = nextRandU32(&rng);
        const new_len: u32 = r1 % 201;
        const created: BitState = if ((r2 & 1) == 1) .active else .inactive;
        if (new_len > expected_len) {
            for (expected_len..new_len) |i| {
                expected[i] = @intFromEnum(created);
                if (created == .active) expected_active += 1;
            }
        } else if (new_len < expected_len) {
            for (new_len..expected_len) |i| {
                if (expected[i] == 1) expected_active -= 1;
            }
        }
        expected_len = new_len;
        errdefer std.debug.print("SEQ u64 fail step={} new={} created={s}\n", .{ step, new_len, @tagName(created) });
        try bs.resize(t.allocator, new_len, created);
        try t.expectEqual(expected_len, bs.bits_count);
        try t.expectEqual(@as(usize, @intCast(BW.bitsToWordsCount(expected_len))), bs.words.items.len);
        try t.expectEqual(expected_active, bs.active_bits_counter);
        for (0..expected_len) |idx| {
            const i: u32 = @intCast(idx);
            const want: BitState = @enumFromInt(expected[idx]);
            const wid: usize = @intCast(BW.bitToWordId(i));
            const got = BW.readBitState(bs.words.items[wid], BW.bitIdInWord(i));
            try t.expectEqual(want, got);
        }
        try expectBitsetInvariants(.u64, &bs);
    }
}

test "Bitset(u8) resize sequential fuzz vs oracle" {
    var bs = Bitset(.u8){};
    defer bs.deinit(t.allocator);
    var expected: [64]u1 = [_]u1{0} ** 64;
    var expected_len: u32 = 0;
    var expected_active: u32 = 0;
    var rng: u64 = 0x123456789ABCDEF;
    const BW = bit_word.BitWord(.u8);
    var step: usize = 0;
    while (step < 300) : (step += 1) {
        const r1 = nextRandU32(&rng);
        const r2 = nextRandU32(&rng);
        const new_len: u32 = r1 % 41;
        const created: BitState = if ((r2 & 1) == 1) .active else .inactive;
        if (new_len > expected_len) {
            for (expected_len..new_len) |i| {
                expected[i] = @intFromEnum(created);
                if (created == .active) expected_active += 1;
            }
        } else if (new_len < expected_len) {
            for (new_len..expected_len) |i| {
                if (expected[i] == 1) expected_active -= 1;
            }
        }
        expected_len = new_len;
        errdefer std.debug.print("SEQ u8 fail step={} new={} created={s}\n", .{ step, new_len, @tagName(created) });
        try bs.resize(t.allocator, new_len, created);
        try t.expectEqual(expected_len, bs.bits_count);
        try t.expectEqual(@as(usize, @intCast(BW.bitsToWordsCount(expected_len))), bs.words.items.len);
        try t.expectEqual(expected_active, bs.active_bits_counter);
        for (0..expected_len) |idx| {
            const i: u32 = @intCast(idx);
            const want: BitState = @enumFromInt(expected[idx]);
            const wid: usize = @intCast(BW.bitToWordId(i));
            const got = BW.readBitState(bs.words.items[wid], BW.bitIdInWord(i));
            try t.expectEqual(want, got);
        }
        try expectBitsetInvariants(.u8, &bs);
    }
}

test "Bitset resize same-size no-op keeps storage" {
    const sizes = [_]u32{ 0, 1, 7, 8, 9, 64, 65, 128 };
    for (sizes) |n| {
        var bs = try initBitsetPattern(.u64, t.allocator, n, .pseudo);
        defer bs.deinit(t.allocator);
        const old_counter = bs.active_bits_counter;
        const old_len = bs.words.items.len;
        try bs.resize(t.allocator, n, .active);
        try t.expectEqual(n, bs.bits_count);
        try t.expectEqual(old_counter, bs.active_bits_counter);
        try t.expectEqual(old_len, bs.words.items.len);
        try bs.resize(t.allocator, n, .inactive);
        try t.expectEqual(old_counter, bs.active_bits_counter);
        try expectBitsetInvariants(.u64, &bs);
    }
}

test "Bitset resize large sizes oracle" {
    try checkOneResize(.u64, t.allocator, 0, 10000, .active, .zero);
    try checkOneResize(.u64, t.allocator, 0, 10000, .inactive, .zero);
    try checkOneResize(.u64, t.allocator, 10000, 2000, .inactive, .one);
    try checkOneResize(.u64, t.allocator, 2000, 3000, .active, .pseudo);
    try checkOneResize(.u64, t.allocator, 3000, 64, .inactive, .one);
    try checkOneResize(.u64, t.allocator, 64, 10000, .active, .alt01);
}

test "Bitset(u64) setWord: masked insert + counters" {
    const B64 = Bitset(.u64);
    var bs: B64 = .{};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 128, .inactive);
    try t.expectEqual(@as(u32, 0), bs.active_bits_counter);

    var low: u64 = 0;
    for (0..32) |i| low |= @as(u64, 1) << @intCast(i);
    bs.setWord(0, low, std.math.maxInt(u64));
    try t.expectEqual(@as(u32, 32), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);

    bs.setWord(0, std.math.maxInt(u64), @as(u64, 0xFFFFFFFF) << 32);
    try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);

    bs.setWord(0, 0, 0);
    try t.expectEqual(@as(u32, 64), bs.active_bits_counter);

    bs.setWord(0, bs.words.items[0], std.math.maxInt(u64));
    try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);

    bs.setWord(1, std.math.maxInt(u64), 0x1);
    try t.expectEqual(@as(u32, 65), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);
}

test "Bitset(u64) setWord: padding bits ignored" {
    const B64 = Bitset(.u64);
    var bs: B64 = .{};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 70, .inactive);
    bs.setWord(1, std.math.maxInt(u64), std.math.maxInt(u64));
    try t.expectEqual(@as(u32, 6), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);
    try t.expectEqual(@as(u64, 0x3F), bs.words.items[1]);
}

test "Bitset(u64) setBit: flip + counters" {
    const B64 = Bitset(.u64);
    var bs: B64 = .{};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 130, .inactive);
    try t.expectEqual(@as(u32, 0), bs.active_bits_counter);

    bs.setBit(0, .active);
    bs.setBit(63, .active);
    bs.setBit(64, .active);
    bs.setBit(129, .active);
    try t.expectEqual(@as(u32, 4), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);

    bs.setBit(0, .active);
    try t.expectEqual(@as(u32, 4), bs.active_bits_counter);

    bs.setBit(63, .inactive);
    try t.expectEqual(@as(u32, 3), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);

    bs.setBit(129, .inactive);
    bs.setBit(64, .inactive);
    bs.setBit(0, .inactive);
    try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
    try expectBitsetInvariants(.u64, &bs);
}

const StepIds = struct {
    active: [256]u32 = undefined,
    na: usize = 0,
    inactive: [256]u32 = undefined,
    ni: usize = 0,
    stop_after: u32 = std.math.maxInt(u32),
};

inline fn stepPushA(ctx: *StepIds, bit_id: u32) bool {
    ctx.active[ctx.na] = bit_id;
    ctx.na += 1;
    return ctx.na + ctx.ni < ctx.stop_after;
}

inline fn stepPushI(ctx: *StepIds, bit_id: u32) bool {
    ctx.inactive[ctx.ni] = bit_id;
    ctx.ni += 1;
    return ctx.na + ctx.ni < ctx.stop_after;
}

fn stepCollectAll(comptime wt: WordType, bs: *Bitset(wt), ctx: *StepIds) bool {
    const It = Bitset(wt).Iterator(*StepIds, stepPushA, stepPushI, .forward);
    var wid: u32 = 0;
    while (wid < bs.words.items.len) : (wid += 1) {
        if (!It.step(.{ .bitset = bs, .context = ctx }, wid)) return false;
    }
    return true;
}

fn stepOracleIds(comptime wt: WordType, bs: *const Bitset(wt), target: BitState, out: *[256]u32) usize {
    const BW = bit_word.BitWord(wt);
    var n: usize = 0;
    var i: u32 = 0;
    while (i < bs.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        if (BW.readBitState(bs.words.items[wid], BW.bitIdInWord(i)) == target) {
            out[n] = i;
            n += 1;
        }
    }
    return n;
}

fn stepCheckOne(comptime wt: WordType, n: u32, pat: ResizePattern) !void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    var bs = Bitset(wt){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        bs.setBit(i, patternBit(pat, i));
    }

    if (n > 0) {
        const used = BW.bitIdInWord(n);
        const valid: Word = if (used == 0) std.math.maxInt(Word) else BW.maskStart(used);
        bs.words.items[bs.words.items.len - 1] |= ~valid;
    }

    var ctx = StepIds{};
    try t.expect(stepCollectAll(wt, &bs, &ctx));

    var exp_a: [256]u32 = undefined;
    var exp_i: [256]u32 = undefined;
    const n_a = stepOracleIds(wt, &bs, .active, &exp_a);
    const n_i = stepOracleIds(wt, &bs, .inactive, &exp_i);

    try t.expectEqualSlices(u32, exp_a[0..n_a], ctx.active[0..ctx.na]);
    try t.expectEqualSlices(u32, exp_i[0..n_i], ctx.inactive[0..ctx.ni]);

    try t.expectEqual(n, @as(u32, @intCast(ctx.na + ctx.ni)));
}

test "Bitset step: active/inactive counts + corners" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 127, 128, 129, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            try stepCheckOne(.u64, n, pat);
            try stepCheckOne(.u8, n, pat);
        }
    }
}

test "Bitset step: null side is skipped" {
    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 70, .inactive);
        bs.setBit(5, .active);
        bs.setBit(69, .active);
        const It = Bitset(.u64).Iterator(*StepIds, stepPushA, null, .forward);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 1));
        try t.expectEqualSlices(u32, &[_]u32{ 5, 69 }, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }

    {
        var bs = Bitset(.u8){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .active);
        bs.setBit(3, .inactive);
        const It = Bitset(.u8).Iterator(*StepIds, null, stepPushI, .forward);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 1));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx.inactive[0..ctx.ni]);
        try t.expectEqual(@as(usize, 0), ctx.na);
    }
}

test "Bitset step: early exit stops the walk" {
    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 70, .active);
        const It = Bitset(.u64).Iterator(*StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 3 };
        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 3), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ctx.active[0..ctx.na]);
    }

    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .inactive);
        bs.setBit(0, .active);
        bs.setBit(9, .active);
        const It = Bitset(.u64).Iterator(*StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 4 };

        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, ctx.inactive[0..ctx.ni]);
    }

    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 130, .active);
        const It = Bitset(.u64).Iterator(*StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 70 };
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 64), ctx.na);
        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 1));
        try t.expectEqual(@as(usize, 70), ctx.na);
        try t.expectEqual(@as(u32, 64), ctx.active[64]);
        try t.expectEqual(@as(u32, 69), ctx.active[69]);
    }

    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .active);
        const It = Bitset(.u64).Iterator(*StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 1 };
        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 1), ctx.na);
    }
}

fn iterateAllCollect(comptime wt: WordType, bs: *Bitset(wt), ctx: *StepIds) bool {
    const It = Bitset(wt).Iterator(*StepIds, stepPushA, stepPushI, .forward);
    return It.iterateAll(.{ .bitset = bs, .context = ctx }, null, null);
}

test "Bitset iterateAll: parity with per-word step + oracle" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 63, 64, 65, 70, 128, 129, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            var bs = try initBitsetPattern(.u64, t.allocator, n, pat);
            defer bs.deinit(t.allocator);
            var a = StepIds{};
            var b = StepIds{};
            try t.expect(stepCollectAll(.u64, &bs, &a));
            try t.expect(iterateAllCollect(.u64, &bs, &b));
            try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
            try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
            var exp_a: [256]u32 = undefined;
            var exp_i: [256]u32 = undefined;
            const n_a = stepOracleIds(.u64, &bs, .active, &exp_a);
            const n_i = stepOracleIds(.u64, &bs, .inactive, &exp_i);
            try t.expectEqualSlices(u32, exp_a[0..n_a], b.active[0..b.na]);
            try t.expectEqualSlices(u32, exp_i[0..n_i], b.inactive[0..b.ni]);
        }
    }
    {
        var bs = Bitset(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .inactive);
        bs.setBit(0, .active);
        bs.setBit(9, .active);
        var a = StepIds{ .stop_after = 4 };
        var b = StepIds{ .stop_after = 4 };
        try t.expect(!stepCollectAll(.u64, &bs, &a));
        try t.expect(!iterateAllCollect(.u64, &bs, &b));
        try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
        try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
        try t.expectEqualSlices(u32, &[_]u32{0}, b.active[0..b.na]);
        try t.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, b.inactive[0..b.ni]);
    }
}

const CommonOrderIds = struct {
    ids: [256]u32 = undefined,
    tags: [256]u1 = undefined,
    n: usize = 0,
    stop_after: u32 = std.math.maxInt(u32),
};

inline fn commonOrderPushA(ctx: *CommonOrderIds, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.tags[ctx.n] = 1;
    ctx.n += 1;
    return ctx.n < ctx.stop_after;
}

inline fn commonOrderPushI(ctx: *CommonOrderIds, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.tags[ctx.n] = 0;
    ctx.n += 1;
    return ctx.n < ctx.stop_after;
}

fn commonStepCollect(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    comptime on_a: InlineIteratorCallback(*StepIds),
    comptime on_i: InlineIteratorCallback(*StepIds),
    includes: [IL]*Bitset(wt),
    excludes: [EL]*Bitset(wt),
    ctx: *StepIds,
) bool {
    const It = Bitset(wt).CommonIterator(IL, EL, *StepIds, on_a, on_i, .forward);
    if (IL == 0 and EL == 0) return true;
    const words_len = if (IL > 0) includes[0].words.items.len else excludes[0].words.items.len;
    var wid: u32 = 0;
    while (wid < words_len) : (wid += 1) {
        if (!It.step(.{ .includes = includes, .excludes = excludes, .context = ctx }, wid)) return false;
    }
    return true;
}

fn commonIterateAllCollect(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    comptime on_a: InlineIteratorCallback(*StepIds),
    comptime on_i: InlineIteratorCallback(*StepIds),
    includes: [IL]*Bitset(wt),
    excludes: [EL]*Bitset(wt),
    ctx: *StepIds,
) bool {
    const It = Bitset(wt).CommonIterator(IL, EL, *StepIds, on_a, on_i, .forward);
    return It.iterateAll(.{ .includes = includes, .excludes = excludes, .context = ctx }, null, null);
}

fn commonStepCollectOrder(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    includes: [IL]*Bitset(wt),
    excludes: [EL]*Bitset(wt),
    ctx: *CommonOrderIds,
) bool {
    const It = Bitset(wt).CommonIterator(IL, EL, *CommonOrderIds, commonOrderPushA, commonOrderPushI, .forward);
    if (IL == 0 and EL == 0) return true;
    const words_len = if (IL > 0) includes[0].words.items.len else excludes[0].words.items.len;
    var wid: u32 = 0;
    while (wid < words_len) : (wid += 1) {
        if (!It.step(.{ .includes = includes, .excludes = excludes, .context = ctx }, wid)) return false;
    }
    return true;
}

fn commonOracleIds(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    includes: [IL]*Bitset(wt),
    excludes: [EL]*Bitset(wt),
    n: u32,
    out_a: *[256]u32,
    out_i: *[256]u32,
) struct { na: usize, ni: usize } {
    const BW = bit_word.BitWord(wt);
    var na: usize = 0;
    var ni: usize = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        const bid = BW.bitIdInWord(i);
        var is_a = true;
        var is_i = true;
        for (includes) |bs| {
            const st = BW.readBitState(bs.words.items[wid], bid);
            if (st != .active) is_a = false;
            if (st != .inactive) is_i = false;
        }
        for (excludes) |bs| {
            const st = BW.readBitState(bs.words.items[wid], bid);
            if (st != .inactive) is_a = false;
            if (st != .active) is_i = false;
        }
        if (is_a) {
            out_a[na] = i;
            na += 1;
        }
        if (is_i) {
            out_i[ni] = i;
            ni += 1;
        }
    }
    return .{ .na = na, .ni = ni };
}

fn commonPoisonPadding(comptime wt: WordType, comptime IL: u32, comptime EL: u32, includes: [IL]*Bitset(wt), excludes: [EL]*Bitset(wt), n: u32) void {
    if (n == 0) return;
    const BW = bit_word.BitWord(wt);
    const used = BW.bitIdInWord(n);
    if (used == 0) return;
    const valid = BW.maskStart(used);
    for (includes) |bs| {
        bs.words.items[bs.words.items.len - 1] |= ~valid;
    }
    for (excludes) |bs| {
        bs.words.items[bs.words.items.len - 1] |= ~valid;
    }
}

fn commonCheckOne(
    comptime wt: WordType,
    comptime IL: u32,
    comptime EL: u32,
    n: u32,
    pats_inc: [IL]ResizePattern,
    pats_exc: [EL]ResizePattern,
) !void {
    var inc_sets: [IL]Bitset(wt) = undefined;
    var exc_sets: [EL]Bitset(wt) = undefined;
    for (0..IL) |k| inc_sets[k] = try initBitsetPattern(wt, t.allocator, n, pats_inc[k]);
    for (0..EL) |k| exc_sets[k] = try initBitsetPattern(wt, t.allocator, n, pats_exc[k]);
    defer {
        for (0..IL) |k| inc_sets[k].deinit(t.allocator);
        for (0..EL) |k| exc_sets[k].deinit(t.allocator);
    }
    var inc_ptrs: [IL]*Bitset(wt) = undefined;
    var exc_ptrs: [EL]*Bitset(wt) = undefined;
    for (0..IL) |k| inc_ptrs[k] = &inc_sets[k];
    for (0..EL) |k| exc_ptrs[k] = &exc_sets[k];
    commonPoisonPadding(wt, IL, EL, inc_ptrs, exc_ptrs, n);

    var exp_a: [256]u32 = undefined;
    var exp_i: [256]u32 = undefined;
    const exp = commonOracleIds(wt, IL, EL, inc_ptrs, exc_ptrs, n, &exp_a, &exp_i);

    {
        var a = StepIds{};
        var b = StepIds{};
        try t.expect(commonStepCollect(wt, IL, EL, stepPushA, stepPushI, inc_ptrs, exc_ptrs, &a));
        try t.expect(commonIterateAllCollect(wt, IL, EL, stepPushA, stepPushI, inc_ptrs, exc_ptrs, &b));
        try t.expectEqualSlices(u32, exp_a[0..exp.na], a.active[0..a.na]);
        try t.expectEqualSlices(u32, exp_i[0..exp.ni], a.inactive[0..a.ni]);
        try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
        try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
    }
    {
        var a = StepIds{};
        var b = StepIds{};
        try t.expect(commonStepCollect(wt, IL, EL, stepPushA, null, inc_ptrs, exc_ptrs, &a));
        try t.expect(commonIterateAllCollect(wt, IL, EL, stepPushA, null, inc_ptrs, exc_ptrs, &b));
        try t.expectEqualSlices(u32, exp_a[0..exp.na], a.active[0..a.na]);
        try t.expectEqualSlices(u32, exp_a[0..exp.na], b.active[0..b.na]);
        try t.expectEqual(@as(usize, 0), a.ni);
        try t.expectEqual(@as(usize, 0), b.ni);
    }
    {
        var a = StepIds{};
        var b = StepIds{};
        try t.expect(commonStepCollect(wt, IL, EL, null, stepPushI, inc_ptrs, exc_ptrs, &a));
        try t.expect(commonIterateAllCollect(wt, IL, EL, null, stepPushI, inc_ptrs, exc_ptrs, &b));
        try t.expectEqualSlices(u32, exp_i[0..exp.ni], a.inactive[0..a.ni]);
        try t.expectEqualSlices(u32, exp_i[0..exp.ni], b.inactive[0..b.ni]);
        try t.expectEqual(@as(usize, 0), a.na);
        try t.expectEqual(@as(usize, 0), b.na);
    }
    for (exp_a[0..exp.na]) |id| try t.expect(id < n);
    for (exp_i[0..exp.ni]) |id| try t.expect(id < n);
}

fn commonCheckOrder(comptime wt: WordType, comptime IL: u32, comptime EL: u32, n: u32, pats_inc: [IL]ResizePattern, pats_exc: [EL]ResizePattern) !void {
    var inc_sets: [IL]Bitset(wt) = undefined;
    var exc_sets: [EL]Bitset(wt) = undefined;
    for (0..IL) |k| inc_sets[k] = try initBitsetPattern(wt, t.allocator, n, pats_inc[k]);
    for (0..EL) |k| exc_sets[k] = try initBitsetPattern(wt, t.allocator, n, pats_exc[k]);
    defer {
        for (0..IL) |k| inc_sets[k].deinit(t.allocator);
        for (0..EL) |k| exc_sets[k].deinit(t.allocator);
    }
    var inc_ptrs: [IL]*Bitset(wt) = undefined;
    var exc_ptrs: [EL]*Bitset(wt) = undefined;
    for (0..IL) |k| inc_ptrs[k] = &inc_sets[k];
    for (0..EL) |k| exc_ptrs[k] = &exc_sets[k];
    commonPoisonPadding(wt, IL, EL, inc_ptrs, exc_ptrs, n);

    var exp_a: [256]u32 = undefined;
    var exp_i: [256]u32 = undefined;
    const exp = commonOracleIds(wt, IL, EL, inc_ptrs, exc_ptrs, n, &exp_a, &exp_i);

    var ctx = CommonOrderIds{};
    try t.expect(commonStepCollectOrder(wt, IL, EL, inc_ptrs, exc_ptrs, &ctx));

    var ia: usize = 0;
    var ii: usize = 0;
    var prev: u32 = 0;
    var k: usize = 0;
    while (k < ctx.n) : (k += 1) {
        const id = ctx.ids[k];
        if (k > 0) try t.expect(id > prev);
        prev = id;
        try t.expect(id < n);
        if (ctx.tags[k] == 1) {
            try t.expect(ia < exp.na);
            try t.expectEqual(exp_a[ia], id);
            ia += 1;
        } else {
            try t.expect(ii < exp.ni);
            try t.expectEqual(exp_i[ii], id);
            ii += 1;
        }
    }
    try t.expectEqual(exp.na, ia);
    try t.expectEqual(exp.ni, ii);
}

test "Bitset CommonIterator: spec example 2+2" {
    var inc0 = Bitset(.u8){};
    var inc1 = Bitset(.u8){};
    var exc0 = Bitset(.u8){};
    var exc1 = Bitset(.u8){};
    defer inc0.deinit(t.allocator);
    defer inc1.deinit(t.allocator);
    defer exc0.deinit(t.allocator);
    defer exc1.deinit(t.allocator);
    try inc0.resize(t.allocator, 8, .inactive);
    try inc1.resize(t.allocator, 8, .inactive);
    try exc0.resize(t.allocator, 8, .inactive);
    try exc1.resize(t.allocator, 8, .inactive);
    inc0.words.items[0] = 0b0101_1010;
    inc1.words.items[0] = 0b0110_1110;
    exc0.words.items[0] = 0b1100_0101;
    exc1.words.items[0] = 0b0001_1101;
    inc0.active_bits_counter = @popCount(inc0.words.items[0]);
    inc1.active_bits_counter = @popCount(inc1.words.items[0]);
    exc0.active_bits_counter = @popCount(exc0.words.items[0]);
    exc1.active_bits_counter = @popCount(exc1.words.items[0]);

    const inc_ptrs: [2]*Bitset(.u8) = .{ &inc0, &inc1 };
    const exc_ptrs: [2]*Bitset(.u8) = .{ &exc0, &exc1 };

    {
        const It = Bitset(.u8).CommonIterator(2, 2, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .includes = inc_ptrs, .excludes = exc_ptrs, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{1}, ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.inactive[0..ctx.ni]);
    }
    {
        const It = Bitset(.u8).CommonIterator(2, 2, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{};
        try t.expect(It.iterateAll(.{ .includes = inc_ptrs, .excludes = exc_ptrs, .context = &ctx }, null, null));
        try t.expectEqualSlices(u32, &[_]u32{1}, ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.inactive[0..ctx.ni]);
    }
    {
        const It = Bitset(.u8).CommonIterator(2, 2, *StepIds, stepPushA, null, .forward);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .includes = inc_ptrs, .excludes = exc_ptrs, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{1}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        var ctx2 = StepIds{};
        try t.expect(It.iterateAll(.{ .includes = inc_ptrs, .excludes = exc_ptrs, .context = &ctx2 }, null, null));
        try t.expectEqualSlices(u32, &[_]u32{1}, ctx2.active[0..ctx2.na]);
    }
    {
        const It = Bitset(.u8).CommonIterator(2, 2, *StepIds, null, stepPushI, .forward);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .includes = inc_ptrs, .excludes = exc_ptrs, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.inactive[0..ctx.ni]);
        try t.expectEqual(@as(usize, 0), ctx.na);
        var ctx2 = StepIds{};
        try t.expect(It.iterateAll(.{ .includes = inc_ptrs, .excludes = exc_ptrs, .context = &ctx2 }, null, null));
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx2.inactive[0..ctx2.ni]);
    }
}

test "Bitset CommonIterator: 1+1 truth table" {
    var inc = Bitset(.u8){};
    var exc = Bitset(.u8){};
    defer inc.deinit(t.allocator);
    defer exc.deinit(t.allocator);
    try inc.resize(t.allocator, 4, .inactive);
    try exc.resize(t.allocator, 4, .inactive);
    inc.words.items[0] = 0b0000_1100;
    exc.words.items[0] = 0b0000_1010;
    inc.active_bits_counter = @popCount(inc.words.items[0]);
    exc.active_bits_counter = @popCount(exc.words.items[0]);

    const It = Bitset(.u8).CommonIterator(1, 1, *StepIds, stepPushA, stepPushI, .forward);
    var ctx = StepIds{};
    try t.expect(It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
    try t.expectEqualSlices(u32, &[_]u32{2}, ctx.active[0..ctx.na]);
    try t.expectEqualSlices(u32, &[_]u32{1}, ctx.inactive[0..ctx.ni]);

    const combos = [_]struct { inc_bit: u1, exc_bit: u1, want_a: bool, want_i: bool }{
        .{ .inc_bit = 0, .exc_bit = 0, .want_a = false, .want_i = false },
        .{ .inc_bit = 1, .exc_bit = 0, .want_a = true, .want_i = false },
        .{ .inc_bit = 0, .exc_bit = 1, .want_a = false, .want_i = true },
        .{ .inc_bit = 1, .exc_bit = 1, .want_a = false, .want_i = false },
    };
    for (combos) |c| {
        var single_inc = Bitset(.u8){};
        var single_exc = Bitset(.u8){};
        defer single_inc.deinit(t.allocator);
        defer single_exc.deinit(t.allocator);
        try single_inc.resize(t.allocator, 1, .inactive);
        try single_exc.resize(t.allocator, 1, .inactive);
        if (c.inc_bit == 1) single_inc.setBit(0, .active);
        if (c.exc_bit == 1) single_exc.setBit(0, .active);
        var one = StepIds{};
        try t.expect(It.step(.{ .includes = .{&single_inc}, .excludes = .{&single_exc}, .context = &one }, 0));
        try t.expectEqual(@as(usize, @intFromBool(c.want_a)), one.na);
        try t.expectEqual(@as(usize, @intFromBool(c.want_i)), one.ni);
        if (c.want_a) try t.expectEqual(@as(u32, 0), one.active[0]);
        if (c.want_i) try t.expectEqual(@as(u32, 0), one.inactive[0]);
    }
}

test "Bitset CommonIterator: 1+0 matches Iterator" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 128, 129, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            for ([_]WordType{ .u8, .u64 }) |wt| {
                switch (wt) {
                    .u8 => try commonCheckSingleMatchesIterator(.u8, n, pat),
                    .u64 => try commonCheckSingleMatchesIterator(.u64, n, pat),
                    else => unreachable,
                }
            }
        }
    }
}

fn commonCheckSingleMatchesIterator(comptime wt: WordType, n: u32, pat: ResizePattern) !void {
    var bs = try initBitsetPattern(wt, t.allocator, n, pat);
    defer bs.deinit(t.allocator);
    commonPoisonPadding(wt, 1, 0, .{&bs}, .{}, n);

    var exp_a: [256]u32 = undefined;
    var exp_i: [256]u32 = undefined;
    const n_a = stepOracleIds(wt, &bs, .active, &exp_a);
    const n_i = stepOracleIds(wt, &bs, .inactive, &exp_i);

    {
        var a = StepIds{};
        var b = StepIds{};
        try t.expect(commonStepCollect(wt, 1, 0, stepPushA, stepPushI, .{&bs}, .{}, &a));
        try t.expect(commonIterateAllCollect(wt, 1, 0, stepPushA, stepPushI, .{&bs}, .{}, &b));
        try t.expectEqualSlices(u32, exp_a[0..n_a], a.active[0..a.na]);
        try t.expectEqualSlices(u32, exp_i[0..n_i], a.inactive[0..a.ni]);
        try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
        try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
    }
    {
        const It = Bitset(wt).Iterator(*StepIds, stepPushA, stepPushI, .forward);
        var ref = StepIds{};
        var wid: u32 = 0;
        while (wid < bs.words.items.len) : (wid += 1) {
            if (!It.step(.{ .bitset = &bs, .context = &ref }, wid)) break;
        }
        var got = StepIds{};
        try t.expect(commonStepCollect(wt, 1, 0, stepPushA, stepPushI, .{&bs}, .{}, &got));
        try t.expectEqualSlices(u32, ref.active[0..ref.na], got.active[0..got.na]);
        try t.expectEqualSlices(u32, ref.inactive[0..ref.ni], got.inactive[0..got.ni]);
    }
}

test "Bitset CommonIterator: 0+1 swapped Iterator" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 128, 129, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            try commonCheckSwappedMatchesIterator(.u64, n, pat);
            try commonCheckSwappedMatchesIterator(.u8, n, pat);
        }
    }
}

fn commonCheckSwappedMatchesIterator(comptime wt: WordType, n: u32, pat: ResizePattern) !void {
    var bs = try initBitsetPattern(wt, t.allocator, n, pat);
    defer bs.deinit(t.allocator);
    commonPoisonPadding(wt, 0, 1, .{}, .{&bs}, n);

    var exp_a: [256]u32 = undefined;
    var exp_i: [256]u32 = undefined;
    const n_a = stepOracleIds(wt, &bs, .active, &exp_a);
    const n_i = stepOracleIds(wt, &bs, .inactive, &exp_i);

    var got = StepIds{};
    try t.expect(commonStepCollect(wt, 0, 1, stepPushA, stepPushI, .{}, .{&bs}, &got));
    try t.expectEqualSlices(u32, exp_i[0..n_i], got.active[0..got.na]);
    try t.expectEqualSlices(u32, exp_a[0..n_a], got.inactive[0..got.ni]);

    var got_all = StepIds{};
    try t.expect(commonIterateAllCollect(wt, 0, 1, stepPushA, stepPushI, .{}, .{&bs}, &got_all));
    try t.expectEqualSlices(u32, got.active[0..got.na], got_all.active[0..got_all.na]);
    try t.expectEqualSlices(u32, got.inactive[0..got.ni], got_all.inactive[0..got_all.ni]);
}

test "Bitset CommonIterator: all active, all inactive, all gap" {
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .active);
        try exc.resize(t.allocator, 70, .inactive);
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 70), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        try t.expectEqual(@as(u32, 0), ctx.active[0]);
        try t.expectEqual(@as(u32, 69), ctx.active[69]);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .inactive);
        try exc.resize(t.allocator, 70, .active);
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 70), ctx.ni);
        try t.expectEqual(@as(u32, 0), ctx.inactive[0]);
        try t.expectEqual(@as(u32, 69), ctx.inactive[69]);
    }
    {
        var bs = try initBitsetPattern(.u64, t.allocator, 70, .pseudo);
        defer bs.deinit(t.allocator);
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 1, 1, stepPushA, stepPushI, .{&bs}, .{&bs}, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        var ctx_step = StepIds{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushA, stepPushI, .{&bs}, .{&bs}, &ctx_step));
        try t.expectEqual(@as(usize, 0), ctx_step.na);
        try t.expectEqual(@as(usize, 0), ctx_step.ni);
    }
    {
        var inc = Bitset(.u8){};
        var exc = Bitset(.u8){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 8, .active);
        try exc.resize(t.allocator, 8, .active);
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u8, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }
}

test "Bitset CommonIterator: oracle corners 2+2" {
    const sizes_u64 = [_]u32{ 0, 1, 2, 7, 8, 9, 63, 64, 65, 70, 129, 200 };
    const sizes_u8 = [_]u32{ 0, 1, 7, 8, 9, 16, 17, 24 };
    const pats = [_]ResizePattern{ .zero, .one, .alt01, .pseudo };
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

test "Bitset CommonIterator: oracle corners 1+1 exhaustive patterns" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 16, 17, 63, 64, 65, 70, 129 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pi| {
            for (patterns) |pe| {
                try commonCheckOne(.u64, 1, 1, n, .{pi}, .{pe});
                try commonCheckOne(.u8, 1, 1, n, .{pi}, .{pe});
            }
        }
    }
}

test "Bitset CommonIterator: oracle corners mixed lens" {
    const sizes = [_]u32{ 0, 1, 8, 9, 17, 64, 65, 70, 130 };
    const pats = [_]ResizePattern{ .zero, .one, .alt01, .pseudo };
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

test "Bitset CommonIterator: oracle word types u16/u32" {
    const sizes16 = [_]u32{ 0, 1, 15, 16, 17, 33 };
    const sizes32 = [_]u32{ 0, 1, 31, 32, 33, 65 };
    const pats = [_]ResizePattern{ .zero, .one, .alt01, .pseudo };
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

test "Bitset CommonIterator: boundary oracle u64" {
    const bounds = [_]u32{ 0, 1, 2, 63, 64, 65, 70, 100, 126, 127, 128, 129, 130, 191, 192, 193, 200 };
    const pats = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (bounds) |n| {
        for (pats) |pi| {
            try commonCheckOne(.u64, 2, 2, n, .{ pi, .pseudo }, .{ .alt01, pi });
            try commonCheckOne(.u64, 1, 1, n, .{pi}, .{.pseudo});
        }
    }
}

test "Bitset CommonIterator: fuzz vs oracle" {
    var rng: u64 = 0x9E3779B97F4A7C15;
    var step: usize = 0;
    while (step < 300) : (step += 1) {
        const n: u32 = nextRandU32(&rng) % 201;
        errdefer std.debug.print("COMMON FUZZ u64 2+2 fail step={} n={}\n", .{ step, n });
        var inc_sets: [2]Bitset(.u64) = undefined;
        var exc_sets: [2]Bitset(.u64) = undefined;
        for (0..2) |k| inc_sets[k] = Bitset(.u64){};
        for (0..2) |k| exc_sets[k] = Bitset(.u64){};
        defer {
            for (0..2) |k| inc_sets[k].deinit(t.allocator);
            for (0..2) |k| exc_sets[k].deinit(t.allocator);
        }
        for (0..2) |k| try inc_sets[k].resize(t.allocator, n, .inactive);
        for (0..2) |k| try exc_sets[k].resize(t.allocator, n, .inactive);
        for (0..2) |k| {
            var w: usize = 0;
            while (w < inc_sets[k].words.items.len) : (w += 1) {
                inc_sets[k].words.items[w] = nextRandU32(&rng);
                inc_sets[k].words.items[w] |= @as(u64, nextRandU32(&rng)) << 32;
            }
            inc_sets[k].active_bits_counter = 0;
            for (inc_sets[k].words.items) |word| inc_sets[k].active_bits_counter += @popCount(word);
        }
        for (0..2) |k| {
            var w: usize = 0;
            while (w < exc_sets[k].words.items.len) : (w += 1) {
                exc_sets[k].words.items[w] = nextRandU32(&rng);
                exc_sets[k].words.items[w] |= @as(u64, nextRandU32(&rng)) << 32;
            }
            exc_sets[k].active_bits_counter = 0;
            for (exc_sets[k].words.items) |word| exc_sets[k].active_bits_counter += @popCount(word);
        }
        if (n > 0) {
            const BW = bit_word.BitWord(.u64);
            const used = BW.bitIdInWord(n);
            if (used != 0) {
                const valid = BW.maskStart(used);
                for (0..2) |k| inc_sets[k].words.items[inc_sets[k].words.items.len - 1] &= valid;
                for (0..2) |k| exc_sets[k].words.items[exc_sets[k].words.items.len - 1] &= valid;
                for (0..2) |k| inc_sets[k].active_bits_counter = 0;
                for (0..2) |k| exc_sets[k].active_bits_counter = 0;
                for (0..2) |k| for (inc_sets[k].words.items) |word| {
                    inc_sets[k].active_bits_counter += @popCount(word);
                };
                for (0..2) |k| for (exc_sets[k].words.items) |word| {
                    exc_sets[k].active_bits_counter += @popCount(word);
                };
            }
        }
        if ((step & 1) == 0) {
            commonPoisonPadding(.u64, 2, 2, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, n);
        }
        var exp_a: [256]u32 = undefined;
        var exp_i: [256]u32 = undefined;
        const exp = commonOracleIds(.u64, 2, 2, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, n, &exp_a, &exp_i);
        var a = StepIds{};
        var b = StepIds{};
        try t.expect(commonStepCollect(.u64, 2, 2, stepPushA, stepPushI, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, &a));
        try t.expect(commonIterateAllCollect(.u64, 2, 2, stepPushA, stepPushI, .{ &inc_sets[0], &inc_sets[1] }, .{ &exc_sets[0], &exc_sets[1] }, &b));
        try t.expectEqualSlices(u32, exp_a[0..exp.na], a.active[0..a.na]);
        try t.expectEqualSlices(u32, exp_i[0..exp.ni], a.inactive[0..a.ni]);
        try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
        try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
    }
}

test "Bitset CommonIterator: fuzz vs oracle u8 3+3" {
    var rng: u64 = 0x123456789ABCDEF;
    var step: usize = 0;
    while (step < 200) : (step += 1) {
        const n: u32 = nextRandU32(&rng) % 41;
        errdefer std.debug.print("COMMON FUZZ u8 3+3 fail step={} n={}\n", .{ step, n });
        var inc_sets: [3]Bitset(.u8) = undefined;
        var exc_sets: [3]Bitset(.u8) = undefined;
        for (0..3) |k| inc_sets[k] = Bitset(.u8){};
        for (0..3) |k| exc_sets[k] = Bitset(.u8){};
        defer {
            for (0..3) |k| inc_sets[k].deinit(t.allocator);
            for (0..3) |k| exc_sets[k].deinit(t.allocator);
        }
        for (0..3) |k| try inc_sets[k].resize(t.allocator, n, .inactive);
        for (0..3) |k| try exc_sets[k].resize(t.allocator, n, .inactive);
        for (0..3) |k| {
            var w: usize = 0;
            while (w < inc_sets[k].words.items.len) : (w += 1) {
                inc_sets[k].words.items[w] = @truncate(nextRandU32(&rng));
            }
        }
        for (0..3) |k| {
            var w: usize = 0;
            while (w < exc_sets[k].words.items.len) : (w += 1) {
                exc_sets[k].words.items[w] = @truncate(nextRandU32(&rng));
            }
        }
        if (n > 0) {
            const BW = bit_word.BitWord(.u8);
            const used = BW.bitIdInWord(n);
            if (used != 0) {
                const valid = BW.maskStart(used);
                for (0..3) |k| inc_sets[k].words.items[inc_sets[k].words.items.len - 1] &= valid;
                for (0..3) |k| exc_sets[k].words.items[exc_sets[k].words.items.len - 1] &= valid;
            }
        }
        if ((step & 1) == 0) {
            commonPoisonPadding(.u8, 3, 3, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, n);
        }
        var exp_a: [256]u32 = undefined;
        var exp_i: [256]u32 = undefined;
        const exp = commonOracleIds(.u8, 3, 3, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, n, &exp_a, &exp_i);
        var a = StepIds{};
        var b = StepIds{};
        try t.expect(commonStepCollect(.u8, 3, 3, stepPushA, stepPushI, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, &a));
        try t.expect(commonIterateAllCollect(.u8, 3, 3, stepPushA, stepPushI, .{ &inc_sets[0], &inc_sets[1], &inc_sets[2] }, .{ &exc_sets[0], &exc_sets[1], &exc_sets[2] }, &b));
        try t.expectEqualSlices(u32, exp_a[0..exp.na], a.active[0..a.na]);
        try t.expectEqualSlices(u32, exp_i[0..exp.ni], a.inactive[0..a.ni]);
        try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
        try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
    }
}

test "Bitset CommonIterator: null side is skipped" {
    {
        var inc0 = Bitset(.u64){};
        var inc1 = Bitset(.u64){};
        var exc0 = Bitset(.u64){};
        var exc1 = Bitset(.u64){};
        defer inc0.deinit(t.allocator);
        defer inc1.deinit(t.allocator);
        defer exc0.deinit(t.allocator);
        defer exc1.deinit(t.allocator);
        try inc0.resize(t.allocator, 70, .inactive);
        try inc1.resize(t.allocator, 70, .inactive);
        try exc0.resize(t.allocator, 70, .inactive);
        try exc1.resize(t.allocator, 70, .inactive);
        inc0.setBit(5, .active);
        inc1.setBit(5, .active);
        inc0.setBit(69, .active);
        inc1.setBit(69, .active);
        exc0.setBit(69, .active);
        var ctx = StepIds{};
        try t.expect(commonStepCollect(.u64, 2, 2, stepPushA, null, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx));
        try t.expectEqualSlices(u32, &[_]u32{5}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        var ctx_all = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 2, 2, stepPushA, null, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx_all));
        try t.expectEqualSlices(u32, &[_]u32{5}, ctx_all.active[0..ctx_all.na]);
    }
    {
        var inc = Bitset(.u8){};
        var exc = Bitset(.u8){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        try exc.resize(t.allocator, 10, .inactive);
        exc.setBit(3, .active);
        var ctx = StepIds{};
        try t.expect(commonStepCollect(.u8, 1, 1, null, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx.inactive[0..ctx.ni]);
        try t.expectEqual(@as(usize, 0), ctx.na);
        var ctx_all = StepIds{};
        try t.expect(commonIterateAllCollect(.u8, 1, 1, null, stepPushI, .{&inc}, .{&exc}, &ctx_all));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx_all.inactive[0..ctx_all.ni]);
    }
}

test "Bitset CommonIterator: early exit stops the walk" {
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 70, .active);
        try exc.resize(t.allocator, 70, .inactive);
        const It = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 3 };
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 3), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ctx.active[0..ctx.na]);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        try exc.resize(t.allocator, 10, .inactive);
        inc.setBit(0, .active);
        exc.setBit(5, .active);
        const It = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 2 };
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, &[_]u32{5}, ctx.inactive[0..ctx.ni]);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 130, .active);
        try exc.resize(t.allocator, 130, .inactive);
        const It = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 70 };
        try t.expect(It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 64), ctx.na);
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 1));
        try t.expectEqual(@as(usize, 70), ctx.na);
        try t.expectEqual(@as(u32, 64), ctx.active[64]);
        try t.expectEqual(@as(u32, 69), ctx.active[69]);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .active);
        try exc.resize(t.allocator, 10, .inactive);
        const It = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 1 };
        try t.expect(!It.step(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 1), ctx.na);
    }
    {
        var inc0 = Bitset(.u8){};
        var inc1 = Bitset(.u8){};
        var exc0 = Bitset(.u8){};
        var exc1 = Bitset(.u8){};
        defer inc0.deinit(t.allocator);
        defer inc1.deinit(t.allocator);
        defer exc0.deinit(t.allocator);
        defer exc1.deinit(t.allocator);
        try inc0.resize(t.allocator, 8, .inactive);
        try inc1.resize(t.allocator, 8, .inactive);
        try exc0.resize(t.allocator, 8, .inactive);
        try exc1.resize(t.allocator, 8, .inactive);
        inc0.words.items[0] = 0b0101_1010;
        inc1.words.items[0] = 0b0110_1110;
        exc0.words.items[0] = 0b1100_0101;
        exc1.words.items[0] = 0b0001_1101;
        const It = Bitset(.u8).CommonIterator(2, 2, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 1 };
        try t.expect(!It.step(.{ .includes = .{ &inc0, &inc1 }, .excludes = .{ &exc0, &exc1 }, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 1), ctx.ni);
        try t.expectEqual(@as(u32, 0), ctx.inactive[0]);
        var ctx2 = StepIds{ .stop_after = 5 };
        try t.expect(It.step(.{ .includes = .{ &inc0, &inc1 }, .excludes = .{ &exc0, &exc1 }, .context = &ctx2 }, 0));
        try t.expectEqual(@as(usize, 1), ctx2.na);
        try t.expectEqual(@as(usize, 1), ctx2.ni);
        var ctx3 = StepIds{ .stop_after = 5 };
        try t.expect(It.iterateAll(.{ .includes = .{ &inc0, &inc1 }, .excludes = .{ &exc0, &exc1 }, .context = &ctx3 }, null, null));
        try t.expectEqual(@as(usize, 1), ctx3.na);
        try t.expectEqual(@as(usize, 1), ctx3.ni);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 130, .active);
        try exc.resize(t.allocator, 130, .inactive);
        const It = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{ .stop_after = 70 };
        try t.expect(!It.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &ctx }, null, null));
        try t.expectEqual(@as(usize, 70), ctx.na);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 130, .inactive);
        try exc.resize(t.allocator, 130, .active);
        const ItI = Bitset(.u64).CommonIterator(1, 1, *StepIds, null, stepPushI, .forward);
        var only_i = StepIds{ .stop_after = 1 };
        try t.expect(!ItI.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &only_i }, null, null));
        try t.expectEqual(@as(usize, 1), only_i.ni);
        try t.expectEqual(@as(u32, 0), only_i.inactive[0]);
        var only_i_full = StepIds{};
        try t.expect(ItI.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &only_i_full }, null, null));
        try t.expectEqual(@as(usize, 130), only_i_full.ni);
        const ItA = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, null, .forward);
        var only_a_full = StepIds{};
        try t.expect(ItA.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &only_a_full }, null, null));
        try t.expectEqual(@as(usize, 0), only_a_full.na);
    }
}

test "Bitset CommonIterator: iterateAll parity with per-word step + oracle" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 63, 64, 65, 70, 128, 129, 200 };
    const pats = [_]ResizePattern{ .zero, .one, .alt01, .pseudo };
    for (sizes) |n| {
        for (pats) |p0| {
            for (pats) |p1| {
                var inc_sets: [2]Bitset(.u64) = .{
                    try initBitsetPattern(.u64, t.allocator, n, p0),
                    try initBitsetPattern(.u64, t.allocator, n, p1),
                };
                var exc_sets: [2]Bitset(.u64) = .{
                    try initBitsetPattern(.u64, t.allocator, n, p1),
                    try initBitsetPattern(.u64, t.allocator, n, p0),
                };
                defer {
                    for (0..2) |k| inc_sets[k].deinit(t.allocator);
                    for (0..2) |k| exc_sets[k].deinit(t.allocator);
                }
                const inc_ptrs: [2]*Bitset(.u64) = .{ &inc_sets[0], &inc_sets[1] };
                const exc_ptrs: [2]*Bitset(.u64) = .{ &exc_sets[0], &exc_sets[1] };
                commonPoisonPadding(.u64, 2, 2, inc_ptrs, exc_ptrs, n);
                var a = StepIds{};
                var b = StepIds{};
                try t.expect(commonStepCollect(.u64, 2, 2, stepPushA, stepPushI, inc_ptrs, exc_ptrs, &a));
                try t.expect(commonIterateAllCollect(.u64, 2, 2, stepPushA, stepPushI, inc_ptrs, exc_ptrs, &b));
                try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
                try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
                var exp_a: [256]u32 = undefined;
                var exp_i: [256]u32 = undefined;
                const exp = commonOracleIds(.u64, 2, 2, inc_ptrs, exc_ptrs, n, &exp_a, &exp_i);
                try t.expectEqualSlices(u32, exp_a[0..exp.na], b.active[0..b.na]);
                try t.expectEqualSlices(u32, exp_i[0..exp.ni], b.inactive[0..b.ni]);
            }
        }
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        try exc.resize(t.allocator, 10, .inactive);
        inc.setBit(0, .active);
        exc.setBit(5, .active);
        var a = StepIds{ .stop_after = 2 };
        var b = StepIds{ .stop_after = 2 };
        try t.expect(!commonStepCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &a));
        try t.expect(!commonIterateAllCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &b));
        try t.expectEqualSlices(u32, a.active[0..a.na], b.active[0..b.na]);
        try t.expectEqualSlices(u32, a.inactive[0..a.ni], b.inactive[0..b.ni]);
        try t.expectEqualSlices(u32, &[_]u32{0}, b.active[0..b.na]);
        try t.expectEqualSlices(u32, &[_]u32{5}, b.inactive[0..b.ni]);
    }
}

test "Bitset CommonIterator: empty sets and zero lens" {
    {
        var inc0 = Bitset(.u64){};
        var inc1 = Bitset(.u64){};
        var exc0 = Bitset(.u64){};
        var exc1 = Bitset(.u64){};
        defer inc0.deinit(t.allocator);
        defer inc1.deinit(t.allocator);
        defer exc0.deinit(t.allocator);
        defer exc1.deinit(t.allocator);
        var ctx = StepIds{};
        try t.expect(commonStepCollect(.u64, 2, 2, stepPushA, stepPushI, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        var ctx2 = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 2, 2, stepPushA, stepPushI, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx2));
        try t.expectEqual(@as(usize, 0), ctx2.na);
        var ctx3 = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 2, 2, stepPushA, null, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx3));
        try t.expectEqual(@as(usize, 0), ctx3.na);
        var ctx4 = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 2, 2, null, stepPushI, .{ &inc0, &inc1 }, .{ &exc0, &exc1 }, &ctx4));
        try t.expectEqual(@as(usize, 0), ctx4.ni);
    }
    {
        const It00 = Bitset(.u64).CommonIterator(0, 0, *StepIds, stepPushA, stepPushI, .forward);
        var ctx = StepIds{};
        try t.expect(It00.step(.{ .includes = .{}, .excludes = .{}, .context = &ctx }, 0));
        try t.expect(It00.iterateAll(.{ .includes = .{}, .excludes = .{}, .context = &ctx }, null, null));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }
    {
        var exc = Bitset(.u8){};
        defer exc.deinit(t.allocator);
        try exc.resize(t.allocator, 10, .inactive);
        exc.setBit(5, .active);
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u8, 0, 1, stepPushA, stepPushI, .{}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 9), ctx.na);
        try t.expectEqualSlices(u32, &[_]u32{5}, ctx.inactive[0..ctx.ni]);
        for (ctx.active[0..ctx.na]) |id| try t.expect(id != 5);
    }
    {
        var inc = Bitset(.u8){};
        defer inc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .inactive);
        inc.setBit(3, .active);
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u8, 1, 0, stepPushA, stepPushI, .{&inc}, .{}, &ctx));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 9), ctx.ni);
        for (ctx.inactive[0..ctx.ni]) |id| try t.expect(id != 3);
    }
}

test "Bitset CommonIterator: tail bound never leaks padding" {
    const tails = [_]u32{ 1, 2, 63, 65, 70, 127, 129, 130, 193, 200 };
    for (tails) |n| {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, n, .active);
        try exc.resize(t.allocator, n, .inactive);
        inc.words.items[inc.words.items.len - 1] |= ~bit_word.BitWord(.u64).maskStart(bit_word.BitWord(.u64).bitIdInWord(n));
        exc.words.items[exc.words.items.len - 1] |= ~bit_word.BitWord(.u64).maskStart(bit_word.BitWord(.u64).bitIdInWord(n));
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(n, @as(u32, @intCast(ctx.na)));
        try t.expectEqual(@as(usize, 0), ctx.ni);
        try t.expectEqual(@as(u32, n - 1), ctx.active[ctx.na - 1]);
        var step_ctx = StepIds{};
        try t.expect(commonStepCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &step_ctx));
        try t.expectEqualSlices(u32, ctx.active[0..ctx.na], step_ctx.active[0..step_ctx.na]);
    }
    {
        var inc = Bitset(.u64){};
        var exc = Bitset(.u64){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 65, .inactive);
        try exc.resize(t.allocator, 65, .active);
        inc.words.items[inc.words.items.len - 1] |= ~bit_word.BitWord(.u64).maskStart(bit_word.BitWord(.u64).bitIdInWord(65));
        exc.words.items[exc.words.items.len - 1] |= ~bit_word.BitWord(.u64).maskStart(bit_word.BitWord(.u64).bitIdInWord(65));
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u64, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(u32, 65), @as(u32, @intCast(ctx.ni)));
        try t.expectEqual(@as(u32, 64), ctx.inactive[ctx.ni - 1]);
    }
    {
        var inc = Bitset(.u8){};
        var exc = Bitset(.u8){};
        defer inc.deinit(t.allocator);
        defer exc.deinit(t.allocator);
        try inc.resize(t.allocator, 10, .active);
        try exc.resize(t.allocator, 10, .inactive);
        inc.words.items[inc.words.items.len - 1] |= 0xFC;
        exc.words.items[exc.words.items.len - 1] |= 0xFC;
        var ctx = StepIds{};
        try t.expect(commonIterateAllCollect(.u8, 1, 1, stepPushA, stepPushI, .{&inc}, .{&exc}, &ctx));
        try t.expectEqual(@as(usize, 10), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }
}

test "Bitset CommonIterator: global order is ascending" {
    const sizes = [_]u32{ 1, 8, 9, 17, 64, 65, 70, 130 };
    const pats = [_]ResizePattern{ .alt01, .alt10, .every3, .pseudo };
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
    try commonCheckOrder(.u64, 3, 3, 130, .{ .alt01, .every3, .pseudo }, .{ .pseudo, .alt10, .zero });
}

test "Bitset getBit: defaults and set/get roundtrip" {
    var bs = Bitset(.u64){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 130, .inactive);

    try t.expectEqual(BitState.inactive, bs.getBit(0));
    try t.expectEqual(BitState.inactive, bs.getBit(63));
    try t.expectEqual(BitState.inactive, bs.getBit(64));
    try t.expectEqual(BitState.inactive, bs.getBit(129));

    bs.setBit(0, .active);
    bs.setBit(63, .active);
    bs.setBit(64, .active);
    bs.setBit(129, .active);

    try t.expectEqual(BitState.active, bs.getBit(0));
    try t.expectEqual(BitState.active, bs.getBit(63));
    try t.expectEqual(BitState.active, bs.getBit(64));
    try t.expectEqual(BitState.active, bs.getBit(129));

    try t.expectEqual(BitState.inactive, bs.getBit(1));
    try t.expectEqual(BitState.inactive, bs.getBit(62));
    try t.expectEqual(BitState.inactive, bs.getBit(65));
    try t.expectEqual(BitState.inactive, bs.getBit(128));

    bs.setBit(63, .inactive);
    try t.expectEqual(BitState.inactive, bs.getBit(63));
    try t.expectEqual(BitState.active, bs.getBit(64));

    const before = bs.active_bits_counter;
    _ = bs.getBit(0);
    _ = bs.getBit(64);
    try t.expectEqual(before, bs.active_bits_counter);

    const cbs: *const Bitset(.u64) = &bs;
    try t.expectEqual(BitState.active, cbs.getBit(0));
    try t.expectEqual(BitState.inactive, cbs.getBit(63));
}

test "Bitset getBit: u8 small word" {
    var bs = Bitset(.u8){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .inactive);

    bs.setBit(0, .active);
    bs.setBit(7, .active);
    bs.setBit(8, .active);
    bs.setBit(9, .active);

    try t.expectEqual(BitState.active, bs.getBit(0));
    try t.expectEqual(BitState.active, bs.getBit(7));
    try t.expectEqual(BitState.active, bs.getBit(8));
    try t.expectEqual(BitState.active, bs.getBit(9));
    try t.expectEqual(BitState.inactive, bs.getBit(1));
    try t.expectEqual(BitState.inactive, bs.getBit(6));
}

fn rangeCheckOne(comptime wt: WordType, n: u32, lo: ?u32, hi: ?u32) !void {
    const BW = bit_word.BitWord(wt);
    var bs = Bitset(wt){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (i % 3 == 0 or i % 7 == 1) bs.setBit(i, .active);
    }

    const r_lo: u32 = @min(lo orelse 0, hi orelse n);
    const r_hi: u32 = @max(lo orelse 0, hi orelse n);
    const c_lo: u32 = @min(r_lo, n);
    const c_hi: u32 = @min(r_hi, n);

    var exp_a: [256]u32 = undefined;
    var exp_i: [256]u32 = undefined;
    var n_a: usize = 0;
    var n_i: usize = 0;
    var k: u32 = c_lo;
    while (k < c_hi) : (k += 1) {
        const wid: usize = @intCast(BW.bitToWordId(k));
        if (BW.readBitState(bs.words.items[wid], BW.bitIdInWord(k)) == .active) {
            exp_a[n_a] = k;
            n_a += 1;
        } else {
            exp_i[n_i] = k;
            n_i += 1;
        }
    }

    const ItF = Bitset(wt).Iterator(*StepIds, stepPushA, stepPushI, .forward);
    const ItB = Bitset(wt).Iterator(*StepIds, stepPushA, stepPushI, .backward);
    var fwd = StepIds{};
    var bwd = StepIds{};
    try t.expect(ItF.iterateAll(.{ .bitset = &bs, .context = &fwd }, lo, hi));
    try t.expect(ItB.iterateAll(.{ .bitset = &bs, .context = &bwd }, lo, hi));
    try t.expectEqualSlices(u32, exp_a[0..n_a], fwd.active[0..fwd.na]);
    try t.expectEqualSlices(u32, exp_i[0..n_i], fwd.inactive[0..fwd.ni]);

    var rev_a: [256]u32 = undefined;
    var rev_i: [256]u32 = undefined;
    for (0..n_a) |j| rev_a[j] = exp_a[n_a - 1 - j];
    for (0..n_i) |j| rev_i[j] = exp_i[n_i - 1 - j];
    try t.expectEqualSlices(u32, rev_a[0..n_a], bwd.active[0..bwd.na]);
    try t.expectEqualSlices(u32, rev_i[0..n_i], bwd.inactive[0..bwd.ni]);

    const ItAF = Bitset(wt).Iterator(*StepIds, stepPushA, null, .forward);
    const ItAB = Bitset(wt).Iterator(*StepIds, stepPushA, null, .backward);
    var fa = StepIds{};
    var ba = StepIds{};
    try t.expect(ItAF.iterateAll(.{ .bitset = &bs, .context = &fa }, lo, hi));
    try t.expect(ItAB.iterateAll(.{ .bitset = &bs, .context = &ba }, lo, hi));
    try t.expectEqualSlices(u32, exp_a[0..n_a], fa.active[0..fa.na]);
    try t.expectEqualSlices(u32, rev_a[0..n_a], ba.active[0..ba.na]);
}

test "Bitset iterateAll: forward/backward ranges vs oracle" {
    const bounds = [_][2]?u32{
        .{ null, null },
        .{ 0, 200 },
        .{ 5, 70 },
        .{ 70, 5 },
        .{ 0, 1 },
        .{ 63, 65 },
        .{ 64, 128 },
        .{ 129, 200 },
        .{ 200, 200 },
        .{ 0, 0 },
        .{ 500, 600 },
        .{ null, 10 },
        .{ 120, null },
    };
    for (bounds) |b| {
        try rangeCheckOne(.u64, 130, b[0], b[1]);
        try rangeCheckOne(.u8, 130, b[0], b[1]);
        try rangeCheckOne(.u64, 0, b[0], b[1]);
        try rangeCheckOne(.u64, 1, b[0], b[1]);
    }
}

test "Bitset CommonIterator: forward/backward ranges vs oracle" {
    var inc = Bitset(.u64){};
    defer inc.deinit(t.allocator);
    var exc = Bitset(.u64){};
    defer exc.deinit(t.allocator);
    try inc.resize(t.allocator, 130, .inactive);
    try exc.resize(t.allocator, 130, .inactive);
    var i: u32 = 0;
    while (i < 130) : (i += 1) {
        if (i % 2 == 0) inc.setBit(i, .active);
        if (i % 5 == 0) exc.setBit(i, .active);
    }
    const bounds = [_][2]?u32{
        .{ null, null },
        .{ 10, 100 },
        .{ 100, 10 },
        .{ 0, 64 },
        .{ 64, 65 },
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
        const ItF = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, null, .forward);
        const ItB = Bitset(.u64).CommonIterator(1, 1, *StepIds, stepPushA, null, .backward);
        var fwd = StepIds{};
        var bwd = StepIds{};
        try t.expect(ItF.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &fwd }, b[0], b[1]));
        try t.expect(ItB.iterateAll(.{ .includes = .{&inc}, .excludes = .{&exc}, .context = &bwd }, b[0], b[1]));
        try t.expectEqualSlices(u32, exp[0..n_exp], fwd.active[0..fwd.na]);
        var rev: [256]u32 = undefined;
        for (0..n_exp) |j| rev[j] = exp[n_exp - 1 - j];
        try t.expectEqualSlices(u32, rev[0..n_exp], bwd.active[0..bwd.na]);
    }
}
