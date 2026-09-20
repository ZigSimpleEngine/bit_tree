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
pub fn BitSet(comptime wt: WordType) type {
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

        /// Builds a word-wise visitor that dispatches by bit value.
        pub fn Iterator(
            comptime Context: type,
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_inactive: InlineIteratorCallback(Context),
        ) type {
            return struct {
                /// Visits one word and routes each valid bit to its callback.
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
                /// Scans all words with split main and tail loops for speed.
                /// - `data` - bitset and caller context.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub inline fn iterateAll(data: BitsetWithContext(Context)) bool {
                    const bitset = data.bitset;
                    const context = data.context;
                    const words = bitset.words.items;
                    const total = bitset.bits_count;
                    if (words.len == 0) return true;

                    if (on_active) |f| {
                        if (on_inactive == null) {
                            var wid: usize = 0;
                            while (wid + 1 < words.len) : (wid += 1) {
                                const base: u32 = @truncate(wid * bw.word_type_bits);
                                if (!flatPeelWord(context, base, words[wid], f)) return false;
                            }
                            const last = words.len - 1;
                            const base: u32 = @truncate(last * bw.word_type_bits);
                            var w = words[last];
                            const used = bw.bitIdInWord(total);
                            if (used != 0) w &= bw.maskStart(used);
                            return flatPeelWord(context, base, w, f);
                        }
                    }

                    if (on_inactive) |f| {
                        if (on_active == null) {
                            var wid: usize = 0;
                            while (wid + 1 < words.len) : (wid += 1) {
                                const base: u32 = @truncate(wid * bw.word_type_bits);
                                if (!flatPeelWord(context, base, ~words[wid], f)) return false;
                            }
                            const last = words.len - 1;
                            const base: u32 = @truncate(last * bw.word_type_bits);
                            var w = ~words[last];
                            const used = bw.bitIdInWord(total);
                            if (used != 0) w &= bw.maskStart(used);
                            return flatPeelWord(context, base, w, f);
                        }
                    }

                    var wid: usize = 0;
                    while (wid + 1 < words.len) : (wid += 1) {
                        const base: u32 = @truncate(wid * bw.word_type_bits);
                        if (!flatMergeWord(context, base, words[wid], bw.word_type_bits)) return false;
                    }
                    const last = words.len - 1;
                    const base: u32 = @truncate(last * bw.word_type_bits);
                    const used = bw.bitIdInWord(total);
                    const bound: u32 = if (used != 0) used else bw.word_type_bits;
                    return flatMergeWord(context, base, words[last], bound);
                }

                /// Fast path when both callbacks exist, merges by bound linearly.
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
                    return flatMergeWord(context, start, word, bound);
                }

                /// Selective path that peels only sides with installed callbacks.
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
                            var only_active_word = word & mask;
                            while (only_active_word != 0) {
                                const bit_id_in_word: u32 = @ctz(only_active_word);
                                only_active_word &= only_active_word - 1;
                                if (!f(context, start + bit_id_in_word)) return false;
                            }
                            return true;
                        }
                    }

                    if (on_inactive) |f| {
                        if (on_active == null) {
                            var only_inactive_word = (~word) & mask;
                            while (only_inactive_word != 0) {
                                const bit_id_in_word: u32 = @ctz(only_inactive_word);
                                only_inactive_word &= only_inactive_word - 1;
                                if (!f(context, start + bit_id_in_word)) return false;
                            }
                            return true;
                        }
                    }

                    const only_active_word = word & mask;
                    const only_inactive_word = (~word) & mask;

                    var pending_word: Word = 0;
                    if (on_active != null) pending_word |= only_active_word;
                    if (on_inactive != null) pending_word |= only_inactive_word;

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

                    return true;
                }

                /// Peels one masked word into a single callback in bit order.
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

                /// Merges one bounded word into active and inactive callbacks linearly.
                /// - `context` - caller context for the callbacks.
                /// - `base` - global id of the word start.
                /// - `word` - raw word to classify bit by bit.
                /// - `valid_len` - valid bits in the word, bounds the scan.
                ///
                /// Return - false on early exit, true when the word completed.
                inline fn flatMergeWord(
                    context: Context,
                    base: u32,
                    word: Word,
                    valid_len: u32,
                ) bool {
                    var i: u32 = 0;
                    while (i < valid_len) : (i += 1) {
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

fn bitSetScanActiveCount(comptime wt: WordType, bs: *const BitSet(wt)) u32 {
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

fn expectBitSetInvariants(comptime wt: WordType, bs: *const BitSet(wt)) !void {
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
    const scanned = bitSetScanActiveCount(wt, bs);
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

fn initBitSetPattern(comptime wt: WordType, allocator: Allocator, bits: u32, pat: ResizePattern) !BitSet(wt) {
    const Word = wt.Type();
    var bs = BitSet(wt){};
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
    try expectBitSetInvariants(wt, &bs);
    return bs;
}

fn checkOneResize(comptime wt: WordType, allocator: Allocator, old: u32, new: u32, created: BitState, pat: ResizePattern) !void {
    var bs = try initBitSetPattern(wt, allocator, old, pat);
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
    try expectBitSetInvariants(wt, &bs);
}

fn nextRandU32(state: *u64) u32 {
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @truncate((x *% 0x2545F4914F6CDD1D) >> 32);
}

test "BitSet resize" {
    var bs = BitSet(.u64){};
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

test "BitSet resize minimal: grow active unaligned must keep padding zero" {
    var bs = BitSet(.u64){};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .active);
    try t.expectEqual(@as(u32, 10), bs.bits_count);
    try t.expectEqual(@as(u32, 10), bs.active_bits_counter);
    try t.expectEqual(@as(usize, 1), bs.words.items.len);
    const BW = bit_word.BitWord(.u64);
    try t.expectEqual(@as(u64, 0), bs.words.items[0] & ~BW.maskStart(10));
    try expectBitSetInvariants(.u64, &bs);

    var bs2 = BitSet(.u64){};
    defer bs2.deinit(t.allocator);
    try bs2.resize(t.allocator, 65, .active);
    try t.expectEqual(@as(u32, 65), bs2.active_bits_counter);
    try t.expectEqual(@as(u64, 0), bs2.words.items[1] & ~BW.maskStart(1));
    try expectBitSetInvariants(.u64, &bs2);
}

test "BitSet resize minimal: shrink to aligned must fix counter" {
    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 128, .active);
        try t.expectEqual(@as(u32, 128), bs.active_bits_counter);
        try bs.resize(t.allocator, 64, .inactive);
        try t.expectEqual(@as(u32, 64), bs.bits_count);
        try t.expectEqual(@as(usize, 1), bs.words.items.len);
        try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
        try expectBitSetInvariants(.u64, &bs);
    }
    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 64, .active);
        try bs.resize(t.allocator, 0, .inactive);
        try t.expectEqual(@as(u32, 0), bs.bits_count);
        try t.expectEqual(@as(usize, 0), bs.words.items.len);
        try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
    }
    {
        var bs = try initBitSetPattern(.u64, t.allocator, 65, .one);
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 64, .inactive);
        try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
        try expectBitSetInvariants(.u64, &bs);
    }
    {
        var bs = try initBitSetPattern(.u8, t.allocator, 9, .one);
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 8, .inactive);
        try t.expectEqual(@as(u32, 8), bs.active_bits_counter);
        try expectBitSetInvariants(.u8, &bs);
    }
}

test "BitSet resize minimal: shrink same-word must clear tail" {
    var bs = try initBitSetPattern(.u64, t.allocator, 20, .one);
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(u32, 10), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);
}

test "BitSet(u8) resize exhaustive 0..20 oracle" {
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

test "BitSet(u64) resize boundary oracle" {
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

test "BitSet(u32) resize boundary oracle" {
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

test "BitSet(u16) resize boundary oracle" {
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

test "BitSet resize explicit corner table u64" {
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

test "BitSet(u64) resize sequential fuzz vs oracle" {
    var bs = BitSet(.u64){};
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
        try expectBitSetInvariants(.u64, &bs);
    }
}

test "BitSet(u8) resize sequential fuzz vs oracle" {
    var bs = BitSet(.u8){};
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
        try expectBitSetInvariants(.u8, &bs);
    }
}

test "BitSet resize same-size no-op keeps storage" {
    const sizes = [_]u32{ 0, 1, 7, 8, 9, 64, 65, 128 };
    for (sizes) |n| {
        var bs = try initBitSetPattern(.u64, t.allocator, n, .pseudo);
        defer bs.deinit(t.allocator);
        const old_counter = bs.active_bits_counter;
        const old_len = bs.words.items.len;
        try bs.resize(t.allocator, n, .active);
        try t.expectEqual(n, bs.bits_count);
        try t.expectEqual(old_counter, bs.active_bits_counter);
        try t.expectEqual(old_len, bs.words.items.len);
        try bs.resize(t.allocator, n, .inactive);
        try t.expectEqual(old_counter, bs.active_bits_counter);
        try expectBitSetInvariants(.u64, &bs);
    }
}

test "BitSet resize large sizes oracle" {
    try checkOneResize(.u64, t.allocator, 0, 10000, .active, .zero);
    try checkOneResize(.u64, t.allocator, 0, 10000, .inactive, .zero);
    try checkOneResize(.u64, t.allocator, 10000, 2000, .inactive, .one);
    try checkOneResize(.u64, t.allocator, 2000, 3000, .active, .pseudo);
    try checkOneResize(.u64, t.allocator, 3000, 64, .inactive, .one);
    try checkOneResize(.u64, t.allocator, 64, 10000, .active, .alt01);
}

test "BitSet(u64) setWord: masked insert + counters" {
    const B64 = BitSet(.u64);
    var bs: B64 = .{};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 128, .inactive);
    try t.expectEqual(@as(u32, 0), bs.active_bits_counter);

    var low: u64 = 0;
    for (0..32) |i| low |= @as(u64, 1) << @intCast(i);
    bs.setWord(0, low, std.math.maxInt(u64));
    try t.expectEqual(@as(u32, 32), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);

    bs.setWord(0, std.math.maxInt(u64), @as(u64, 0xFFFFFFFF) << 32);
    try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);

    bs.setWord(0, 0, 0);
    try t.expectEqual(@as(u32, 64), bs.active_bits_counter);

    bs.setWord(0, bs.words.items[0], std.math.maxInt(u64));
    try t.expectEqual(@as(u32, 64), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);

    bs.setWord(1, std.math.maxInt(u64), 0x1);
    try t.expectEqual(@as(u32, 65), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);
}

test "BitSet(u64) setWord: padding bits ignored" {
    const B64 = BitSet(.u64);
    var bs: B64 = .{};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 70, .inactive);
    bs.setWord(1, std.math.maxInt(u64), std.math.maxInt(u64));
    try t.expectEqual(@as(u32, 6), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);
    try t.expectEqual(@as(u64, 0x3F), bs.words.items[1]);
}

test "BitSet(u64) setBit: flip + counters" {
    const B64 = BitSet(.u64);
    var bs: B64 = .{};
    defer bs.deinit(t.allocator);
    try bs.resize(t.allocator, 130, .inactive);
    try t.expectEqual(@as(u32, 0), bs.active_bits_counter);

    bs.setBit(0, .active);
    bs.setBit(63, .active);
    bs.setBit(64, .active);
    bs.setBit(129, .active);
    try t.expectEqual(@as(u32, 4), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);

    bs.setBit(0, .active);
    try t.expectEqual(@as(u32, 4), bs.active_bits_counter);

    bs.setBit(63, .inactive);
    try t.expectEqual(@as(u32, 3), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);

    bs.setBit(129, .inactive);
    bs.setBit(64, .inactive);
    bs.setBit(0, .inactive);
    try t.expectEqual(@as(u32, 0), bs.active_bits_counter);
    try expectBitSetInvariants(.u64, &bs);
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

fn stepCollectAll(comptime wt: WordType, bs: *BitSet(wt), ctx: *StepIds) bool {
    const It = BitSet(wt).Iterator(*StepIds, stepPushA, stepPushI);
    var wid: u32 = 0;
    while (wid < bs.words.items.len) : (wid += 1) {
        if (!It.step(.{ .bitset = bs, .context = ctx }, wid)) return false;
    }
    return true;
}

fn stepOracleIds(comptime wt: WordType, bs: *const BitSet(wt), target: BitState, out: *[256]u32) usize {
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
    var bs = BitSet(wt){};
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

test "BitSet step: active/inactive counts + corners" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 15, 16, 17, 63, 64, 65, 70, 127, 128, 129, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            try stepCheckOne(.u64, n, pat);
            try stepCheckOne(.u8, n, pat);
        }
    }
}

test "BitSet step: null side is skipped" {
    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 70, .inactive);
        bs.setBit(5, .active);
        bs.setBit(69, .active);
        const It = BitSet(.u64).Iterator(*StepIds, stepPushA, null);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 1));
        try t.expectEqualSlices(u32, &[_]u32{ 5, 69 }, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }

    {
        var bs = BitSet(.u8){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .active);
        bs.setBit(3, .inactive);
        const It = BitSet(.u8).Iterator(*StepIds, null, stepPushI);
        var ctx = StepIds{};
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 1));
        try t.expectEqualSlices(u32, &[_]u32{3}, ctx.inactive[0..ctx.ni]);
        try t.expectEqual(@as(usize, 0), ctx.na);
    }
}

test "BitSet step: early exit stops the walk" {
    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 70, .active);
        const It = BitSet(.u64).Iterator(*StepIds, stepPushA, stepPushI);
        var ctx = StepIds{ .stop_after = 3 };
        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 3), ctx.na);
        try t.expectEqual(@as(usize, 0), ctx.ni);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, ctx.active[0..ctx.na]);
    }

    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .inactive);
        bs.setBit(0, .active);
        bs.setBit(9, .active);
        const It = BitSet(.u64).Iterator(*StepIds, stepPushA, stepPushI);
        var ctx = StepIds{ .stop_after = 4 };

        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqualSlices(u32, &[_]u32{0}, ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, &[_]u32{ 1, 2, 3 }, ctx.inactive[0..ctx.ni]);
    }

    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 130, .active);
        const It = BitSet(.u64).Iterator(*StepIds, stepPushA, stepPushI);
        var ctx = StepIds{ .stop_after = 70 };
        try t.expect(It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 64), ctx.na);
        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 1));
        try t.expectEqual(@as(usize, 70), ctx.na);
        try t.expectEqual(@as(u32, 64), ctx.active[64]);
        try t.expectEqual(@as(u32, 69), ctx.active[69]);
    }

    {
        var bs = BitSet(.u64){};
        defer bs.deinit(t.allocator);
        try bs.resize(t.allocator, 10, .active);
        const It = BitSet(.u64).Iterator(*StepIds, stepPushA, stepPushI);
        var ctx = StepIds{ .stop_after = 1 };
        try t.expect(!It.step(.{ .bitset = &bs, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 1), ctx.na);
    }
}

fn iterateAllCollect(comptime wt: WordType, bs: *BitSet(wt), ctx: *StepIds) bool {
    const It = BitSet(wt).Iterator(*StepIds, stepPushA, stepPushI);
    return It.iterateAll(.{ .bitset = bs, .context = ctx });
}

test "BitSet iterateAll: parity with per-word step + oracle" {
    const sizes = [_]u32{ 0, 1, 2, 7, 8, 9, 63, 64, 65, 70, 128, 129, 200 };
    const patterns = [_]ResizePattern{ .zero, .one, .alt01, .alt10, .every3, .pseudo };
    for (sizes) |n| {
        for (patterns) |pat| {
            var bs = try initBitSetPattern(.u64, t.allocator, n, pat);
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
        var bs = BitSet(.u64){};
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
