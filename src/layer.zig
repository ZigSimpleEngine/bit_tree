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

pub fn Layer(comptime wt: WordType) type {
    const Word = wt.Type();
    const bw = bit_word.BitWord(wt);

    return struct {
        const Self = @This();

        activity: ListA64(Word) = .empty,
        mixed: ListA64(Word) = .empty,
        bits_count: u32 = 0,
        state_counters: [4]u32 = .{ 0, 0, 0, 0 },

        pub const State = enum(u2) {
            inactive = 0,
            active = 1,
            mixed = 2,
            deep_mixed = 3,

            pub const inactive_u2: u2 = 0;
            pub const active_u2: u2 = 1;
            pub const mixed_u2: u2 = 2;
            pub const deep_mixed_u2: u2 = 3;

            pub const inactive_u32: u32 = 0;
            pub const active_u32: u32 = 1;
            pub const mixed_u32: u32 = 2;
            pub const deep_mixed_u32: u32 = 3;

            pub inline fn fromBits(activity: u1, mixed: u1) u2 {
                const b0: u2 = activity;
                const b1: u2 = @as(u2, mixed) << 1;
                return b0 | b1;
            }

            pub inline fn fromBitsState(activity: BitState, mixed: BitState) State {
                const b0: u2 = @intFromEnum(activity);
                const b1: u2 = @as(u2, @intFromEnum(mixed)) << 1;
                return @enumFromInt(b0 | b1);
            }

            pub inline fn activityBit(self: State) BitState {
                const b0: u1 = @truncate(@as(u2, @intFromEnum(self)));
                return @enumFromInt(b0);
            }

            pub inline fn mixedBit(self: State) BitState {
                const b0: u1 = @truncate(@as(u2, @intFromEnum(self)) >> 1);
                return @enumFromInt(b0);
            }
        };

        pub const StateCounts = struct { inactive: u32, active: u32, mixed: u32, deep: u32 };

        pub fn LayerWithContext(Context: type) type {
            return struct {
                layer: *Self,
                context: Context,
            };
        }

        pub fn Iterator(
            comptime Context: type,
            comptime on_inactive: InlineIteratorCallback(Context),
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_mixed: IteratorCallback(Context),
            comptime on_deep_mixed: InlineIteratorCallback(Context),
        ) type {
            return struct {
                pub inline fn step(data: LayerWithContext(Context), word_id: u32) bool {
                    if (on_inactive != null and on_active != null and on_mixed != null and on_deep_mixed != null) {
                        return stepFull(data, word_id);
                    } else {
                        return stepPending(data, word_id);
                    }
                }

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
                    var i: u32 = 0;
                    while (i < bound) : (i += 1) {
                        const abit: u1 = @truncate(activity_word >> @truncate(i));
                        const mbit: u1 = @truncate(mixed_word >> @truncate(i));
                        const bit_id = start + i;
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
                    }
                    return true;
                }

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

                    while (pending_word != 0) {
                        const bit_id_in_word: u32 = @ctz(pending_word);
                        const bit_id = start + bit_id_in_word;
                        const bit: Word = @as(Word, 1) << @truncate(bit_id_in_word);
                        pending_word &= pending_word - 1;

                        if (on_inactive) |f| {
                            if ((only_inactive_word & bit) != 0) {
                                if (!f(context, bit_id)) return false;
                                continue;
                            }
                        }

                        if (on_active) |f| {
                            if ((only_active_word & bit) != 0) {
                                if (!f(context, bit_id)) return false;
                                continue;
                            }
                        }

                        if (on_mixed) |f| {
                            if ((only_mixed_word & bit) != 0) {
                                if (!f(context, bit_id)) return false;
                                continue;
                            }
                        }

                        if (on_deep_mixed) |f| {
                            if ((only_deep_mixed & bit) != 0) {
                                if (!f(context, bit_id)) return false;
                                continue;
                            }
                        }
                    }

                    return true;
                }
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.activity.deinit(allocator);
            self.mixed.deinit(allocator);
        }

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
    const It = Layer(wt).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD);
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
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD);
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
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD);
        var ctx = StepStates{ .stop_after = 10 };

        try t.expect(!It.step(.{ .layer = &layer, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 8), ctx.ni);
        try t.expectEqualSlices(u32, &[_]u32{ 0, 9 }, ctx.active[0..ctx.na]);
    }

    {
        var layer: Layer(.u64) = .{};
        defer layer.deinit(t.allocator);
        try layer.resize(t.allocator, 130, .deep_mixed);
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD);
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
        const It = Layer(.u64).Iterator(*StepStates, stepPushI, stepPushA, stepPushM, stepPushD);
        var ctx = StepStates{ .stop_after = 1 };
        try t.expect(!It.step(.{ .layer = &layer, .context = &ctx }, 0));
        try t.expectEqual(@as(usize, 1), ctx.ni + ctx.na + ctx.nm + ctx.nd);
    }
}
