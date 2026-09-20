const std = @import("std");
const utilities = @import("utilities.zig");
const bit_word = @import("bit_word.zig");
const BitSet = @import("bit_set.zig").BitSet;
const Layer = @import("layer.zig").Layer;

const InlineIteratorCallback = utilities.InlineIteratorCallback;
const Allocator = std.mem.Allocator;
const ListA64 = utilities.ListA64;
const BitState = utilities.BitState;
const WordType = bit_word.WordType;

/// Forces the iteration path instead of auto prediction.
pub const PredictForce = enum {
    /// Chooses flat or tree automatically from density.
    auto,
    /// Always scans the flat bitset directly.
    flat,
    /// Always descends the summary pyramid.
    tree,
};

/// Thresholds and costs that tune the flat versus tree choice.
pub const PredictConfig = struct {
    /// Hit density above which active-only scans prefer flat.
    t_active: f32 = 0.01,
    /// Hit density above which inactive-only scans prefer flat.
    t_inactive: f32 = 0.9,
    /// Short-circuits to flat when upper levels lack mixed states.
    use_uniformity: bool = true,
    /// Enables the linear cost model after density checks.
    use_cost: bool = false,
    /// Per-word cost of the flat scan in the cost model.
    c0: f32 = 1.0,
    /// Per-hit cost of the flat path in the cost model.
    c1f: f32 = 1.0,
    /// Per-top-word cost of the tree descent.
    c2: f32 = 8.0,
    /// Per-descent cost through mixed states.
    c3: f32 = 12.0,
    /// Per-hit cost of the tree path in the cost model.
    c1t: f32 = 1.4,
    /// Manual override applied before density checks.
    force: PredictForce = .auto,
};

/// Global overrides checked after per-width rows.
pub var predict_config: PredictConfig = .{};

/// Calibrated per-width rows indexed by word type for auto mode.
pub var predict_table: [4]PredictConfig = .{
    .{ .t_active = 0.002, .t_inactive = 0.93 },
    .{ .t_active = 0.002, .t_inactive = 0.7 },
    .{ .t_active = 0.002, .t_inactive = 0.7 },
    .{ .t_active = 0.002, .t_inactive = 0.9 },
};

/// Selects the calibration row for one word width.
/// - `wt` - word width whose row is needed.
///
/// Return - mutable pointer to the width-specific config.
fn predictRow(wt: WordType) *PredictConfig {
    return &predict_table[@intFromEnum(wt)];
}

/// Records the last auto choice for benchmarks and tests.
pub var last_predict_used_flat: bool = false;

/// Hierarchical bitset with pyramid summaries and adaptive iteration.
/// - `wt` - word width for backing storage.
///
/// Return - tree type with flat and tree iteration arms.
pub fn BitTree(comptime wt: WordType) type {
    const Word = wt.Type();
    const bw = bit_word.BitWord(wt);

    return struct {
        const Self = @This();

        /// Flat leaf storage holding every valid bit.
        bitset: BitSet(wt) = .{},
        /// Summary pyramid where level zero mirrors the bitset.
        layers: ListA64(L) = .empty,

        /// Bundles a tree pointer with caller context for iteration.
        /// - `Context` - caller-provided iteration context.
        ///
        /// Return - pairing struct passed to every iterate call.
        pub fn TreeWithContext(Context: type) type {
            return struct {
                /// Tree being scanned, provides leaves and pyramid.
                tree: *Self,
                /// Caller context forwarded to per-bit callbacks.
                context: Context,
            };
        }

        /// Builds an adaptive visitor that picks flat or tree traversal.
        /// - `Context` - caller-provided iteration context.
        /// - `on_active` - visitor for set bits, null skips them.
        /// - `on_inactive` - visitor for cleared bits, null skips them.
        ///
        /// Return - iterator type with flat and tree arms.
        pub fn Iterator(
            comptime Context: type,
            comptime on_active: InlineIteratorCallback(Context),
            comptime on_inactive: InlineIteratorCallback(Context),
        ) type {
            return struct {
                /// Level cursor pairing a tree with the current depth.
                const LayerWithContext = TreeWithContext(struct {
                    /// Current pyramid depth, decremented on descent.
                    layer_id: u32,
                    /// Caller context forwarded to per-bit callbacks.
                    context: Context,
                });

                /// Adapts a bit callback into a layer visitor over one word span.
                /// - `callback` - per-bit visitor to invoke for the span.
                ///
                /// Return - wrapper type exposing the unwrap entry point.
                fn ActivityUnwrapper(comptime callback: InlineIteratorCallback(Context)) type {
                    return struct {
                        /// Expands one summary word into per-bit visits honoring bounds.
                        /// - `data` - tree and depth cursor.
                        /// - `word_id` - summary word index to expand.
                        ///
                        /// Return - false on early exit, true when the span completed.
                        inline fn unwrap(data: *LayerWithContext, word_id: u32) bool {
                            const layer_id = data.context.layer_id;
                            const context = data.context.context;
                            const end_word_id = word_id + 1;
                            const shift: u32 = bw.shift_type_bits * layer_id;
                            const s5: u5 = @truncate(shift);
                            const start_bit_id = word_id << s5;
                            var end_bit_id = end_word_id << s5;
                            const total = data.tree.bitset.bits_count;
                            if (end_bit_id > total) end_bit_id = total;
                            if (start_bit_id >= total) return true;

                            for (start_bit_id..end_bit_id) |i| {
                                if (callback) |f| {
                                    if (!f(context, @truncate(i))) return false;
                                }
                            }
                            return true;
                        }
                    };
                }

                /// Adapted visitor that expands inactive summary words.
                const inactive_unwrapper = ActivityUnwrapper(on_inactive).unwrap;

                /// Adapted visitor that expands active summary words.
                const active_unwrapper = ActivityUnwrapper(on_active).unwrap;

                /// Layer visitor that drives pyramid descent for both states.
                const LI = L.Iterator(
                    *LayerWithContext,
                    if (on_inactive != null) inactive_unwrapper else null,
                    if (on_active != null) active_unwrapper else null,
                    mixed_unwrapper,
                    deep_mixed_unwrapper,
                );

                /// Runs the predicted path and records the choice for inspection.
                /// - `data` - tree and caller context.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub fn iterateAll(data: TreeWithContext(Context)) bool {
                    const use_flat = predictsFlat(data.tree);
                    last_predict_used_flat = use_flat;
                    if (use_flat) return iterateFlat(data);
                    return iterateTree(data);
                }

                /// Predicts whether a flat scan beats pyramid descent.
                /// - `tree` - tree whose density and uniformity are scored.
                ///
                /// Return - true for flat, false for tree.
                pub fn predictsFlat(tree: *const Self) bool {
                    const cfg = predictRow(wt);
                    switch (cfg.force) {
                        .flat => return true,
                        .tree => return false,
                        .auto => {},
                    }
                    switch (predict_config.force) {
                        .flat => return true,
                        .tree => return false,
                        .auto => {},
                    }
                    const n = tree.bitset.bits_count;
                    if (n == 0) return true;
                    const a = tree.bitset.active_bits_counter;
                    const want_active = on_active != null;
                    const want_inactive = on_inactive != null;
                    var hits: u32 = 0;
                    if (want_active) hits += a;
                    if (want_inactive) hits += n - a;
                    if (hits == 0) return false;
                    const layers = tree.layers.items;
                    if (cfg.use_uniformity and layers.len >= 2) {
                        const l1 = &layers[1];
                        if (l1.state_counters[L.State.mixed_u32] == 0 and l1.state_counters[L.State.deep_mixed_u32] == 0) return true;
                        const top = &layers[layers.len - 1];
                        if (top.state_counters[L.State.mixed_u32] == 0 and top.state_counters[L.State.deep_mixed_u32] == 0) return true;
                    }
                    const hd: f32 = @as(f32, @floatFromInt(hits)) / @as(f32, @floatFromInt(n));
                    var thresh: f32 = @min(cfg.t_active, cfg.t_inactive);
                    if (want_active and !want_inactive) thresh = cfg.t_active;
                    if (want_inactive and !want_active) thresh = cfg.t_inactive;
                    if (hd > thresh) return true;
                    if (cfg.use_cost and layers.len > 0) {
                        const w: f32 = @floatFromInt(bw.bitsToWordsCount(n));
                        const h: f32 = @floatFromInt(hits);
                        const top_words: f32 = @floatFromInt(layers[layers.len - 1].activity.items.len);
                        var descents: f32 = 0;
                        var li: usize = 1;
                        while (li < layers.len) : (li += 1) {
                            descents += @floatFromInt(layers[li].state_counters[L.State.mixed_u32] + layers[li].state_counters[L.State.deep_mixed_u32]);
                        }
                        const est_flat = cfg.c0 * w + cfg.c1f * h;
                        const est_tree = cfg.c2 * top_words + cfg.c3 * descents + cfg.c1t * h;
                        if (est_flat < est_tree) return true;
                    }
                    return false;
                }

                /// Scans leaves directly without touching summaries.
                /// - `data` - tree and caller context.
                ///
                /// Return - false on early exit, true when the scan completed.
                pub fn iterateFlat(data: TreeWithContext(Context)) bool {
                    const BI = B.Iterator(Context, on_active, on_inactive);
                    return BI.iterateAll(.{ .bitset = &data.tree.bitset, .context = data.context });
                }

                /// Walks the pyramid from the top word down to leaves.
                /// - `data` - tree and caller context.
                ///
                /// Return - false on early exit, true when the walk completed.
                pub inline fn iterateTree(data: TreeWithContext(Context)) bool {
                    const layers = data.tree.layers.items;
                    if (layers.len == 0) return true;
                    const layer_id = layers.len - 1;
                    var context: LayerWithContext = .{
                        .context = .{
                            .context = data.context,
                            .layer_id = @truncate(layer_id),
                        },
                        .tree = data.tree,
                    };

                    const top = &data.tree.layers.items[layer_id];
                    const words: u32 = @truncate(top.activity.items.len);
                    var wid: u32 = 0;
                    while (wid < words) : (wid += 1) {
                        if (!LI.step(.{ .layer = top, .context = &context }, wid)) return false;
                    }

                    return true;
                }

                /// Descends one level through a mixed summary word.
                /// - `data` - tree and depth cursor.
                /// - `word_id` - mixed word index to descend.
                ///
                /// Return - false on early exit, true when the descent completed.
                fn mixed_unwrapper(data: *LayerWithContext, word_id: u32) bool {
                    data.context.layer_id -= 1;
                    const lower = &data.tree.layers.items[data.context.layer_id];
                    const ok = LI.step(.{ .layer = lower, .context = data }, word_id);
                    data.context.layer_id += 1;
                    if (!ok) return false;
                    return true;
                }

                /// Scans leaf words covered by a deeply mixed summary word.
                /// - `data` - tree and depth cursor.
                /// - `word_id` - deep-mixed word index to scan.
                ///
                /// Return - false on early exit, true when the scan completed.
                inline fn deep_mixed_unwrapper(data: *LayerWithContext, word_id: u32) bool {
                    const layer_id = data.context.layer_id;
                    const end_word_id = word_id + 1;
                    const shift: u32 = bw.shift_type_bits * (layer_id - 1);
                    const s5: u5 = @truncate(shift);
                    const start_bit_id = word_id << s5;
                    var end_bit_id = end_word_id << s5;
                    const words_len: u32 = @truncate(data.tree.bitset.words.items.len);
                    if (end_bit_id > words_len) end_bit_id = words_len;
                    if (start_bit_id >= words_len) return true;

                    const BI = B.Iterator(Context, on_active, on_inactive);
                    const bi_data: B.BitsetWithContext(Context) = .{
                        .bitset = &data.tree.bitset,
                        .context = data.context.context,
                    };
                    for (start_bit_id..end_bit_id) |i| {
                        if (!BI.step(bi_data, @truncate(i))) return false;
                    }
                    return true;
                }
            };
        }

        /// Level type alias that shortens pyramid declarations.
        const L = Layer(wt);
        /// Leaf type alias that shortens flat-path declarations.
        const B = BitSet(wt);

        /// Releases leaves and every pyramid level.
        /// - `self` - tree to destroy.
        /// - `allocator` - allocator that owns all planes.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.bitset.deinit(allocator);
            for (self.layers.items) |*layer| {
                layer.deinit(allocator);
            }
            self.layers.deinit(allocator);
        }

        /// Writes masked lanes into leaves and refreshes summaries upward.
        /// - `self` - tree to update.
        /// - `id` - leaf word index to patch.
        /// - `word` - new lanes to install.
        /// - `mask` - selects lanes to overwrite, padding is ignored.
        pub fn setWord(self: *Self, id: u32, word: Word, mask: Word) void {
            self.bitset.setWord(id, word, mask);
            self.layers.items[0].setWord(id, word, 0, mask);
            var wid = id;
            var li: usize = 1;
            while (li < self.layers.items.len) : (li += 1) {
                self.refreshWord(li, wid);
                wid = bw.bitToWordId(wid);
            }
        }

        /// Writes one bit and refreshes summaries along its pyramid path.
        /// - `self` - tree to update.
        /// - `id` - global bit position to write.
        /// - `value` - state to store.
        pub fn setBit(self: *Self, id: u32, value: BitState) void {
            self.bitset.setBit(id, value);
            const leaf: Layer(wt).State = if (value == .active) .active else .inactive;
            self.layers.items[0].setBit(id, leaf);
            var wid = bw.bitToWordId(id);
            var li: usize = 1;
            while (li < self.layers.items.len) : (li += 1) {
                self.refreshWord(li, wid);
                wid = bw.bitToWordId(wid);
            }
        }

        /// Grows or shrinks leaves and pyramid while keeping summaries exact.
        /// - `self` - tree to resize.
        /// - `allocator` - owns backing storage.
        /// - `new_bits_count` - target valid bits.
        /// - `created_bits_value` - state filling newly created bits.
        ///
        /// Return - error on allocation failure.
        pub fn resize(self: *Self, allocator: Allocator, new_bits_count: u32, created_bits_value: BitState) !void {
            try self.bitset.resize(allocator, new_bits_count, created_bits_value);

            var want: usize = 0;
            if (new_bits_count > 0) {
                want = 1;
                var s = new_bits_count;
                while (s > 1) {
                    s = bw.bitsToWordsCount(s);
                    want += 1;
                }
            }
            while (self.layers.items.len > want) {
                const idx = self.layers.items.len - 1;
                var gone = self.layers.orderedRemove(idx);
                gone.deinit(allocator);
            }
            while (self.layers.items.len < want) {
                try self.layers.append(allocator, Layer(wt){});
            }

            const leaf_created: Layer(wt).State = if (created_bits_value == .active) .active else .inactive;
            var s = new_bits_count;
            for (self.layers.items, 0..) |*layer, li| {
                try layer.resize(allocator, s, if (li == 0) leaf_created else .inactive);
                s = bw.bitsToWordsCount(s);
            }
            var li: usize = 1;
            while (li < self.layers.items.len) : (li += 1) {
                const lower_words: u32 = @truncate(self.layers.items[li - 1].activity.items.len);
                var wid: u32 = 0;
                while (wid < lower_words) : (wid += 1) {
                    self.refreshWord(li, wid);
                }
            }
        }

        /// Recomputes one summary position from its lower word.
        /// - `self` - tree holding the pyramid.
        /// - `upper_layer` - level index to refresh.
        /// - `word_id` - word position inside the upper level.
        fn refreshWord(self: *Self, upper_layer: usize, word_id: u32) void {
            const lower = &self.layers.items[upper_layer - 1];
            const upper = &self.layers.items[upper_layer];
            const lower_used = bw.bitIdInWord(lower.bits_count);
            const last = lower.activity.items.len - 1;
            const valid: Word = if (word_id == last and lower_used != 0) bw.maskStart(lower_used) else bw.max_value;
            const st: L.State = if (upper_layer == 1) blk: {
                const lw: Word = lower.activity.items[word_id] & valid;
                break :blk if (lw == 0)
                    .inactive
                else if (lw == valid)
                    .active
                else
                    .deep_mixed;
            } else blk: {
                const aw: Word = lower.activity.items[word_id] & valid;
                const mw: Word = lower.mixed.items[word_id] & valid;
                const a_uni = aw == 0 or aw == valid;
                const m_uni = mw == 0 or mw == valid;
                break :blk if (a_uni and m_uni)
                    L.State.fromBitsState(
                        if (aw == 0) BitState.inactive else BitState.active,
                        if (mw == 0) BitState.inactive else BitState.active,
                    )
                else
                    .mixed;
            };
            upper.setBit(word_id, st);
        }
    };
}

const t = std.testing;

const TreeStepIds = struct {
    active: [512]u32 = undefined,
    na: usize = 0,
    inactive: [512]u32 = undefined,
    ni: usize = 0,
    stop_after: u32 = std.math.maxInt(u32),

    fn total(self: *const TreeStepIds) usize {
        return self.na + self.ni;
    }
};

inline fn treePushA(ctx: *TreeStepIds, bit_id: u32) bool {
    ctx.active[ctx.na] = bit_id;
    ctx.na += 1;
    return ctx.total() < ctx.stop_after;
}

inline fn treePushI(ctx: *TreeStepIds, bit_id: u32) bool {
    ctx.inactive[ctx.ni] = bit_id;
    ctx.ni += 1;
    return ctx.total() < ctx.stop_after;
}

fn treeStepCollectAll(comptime wt: WordType, tree: *BitTree(wt), ctx: *TreeStepIds) void {
    const It = BitTree(wt).Iterator(*TreeStepIds, treePushA, treePushI);
    _ = It.iterateAll(.{ .tree = tree, .context = ctx });
}

fn treeOracleIds(comptime wt: WordType, tree: *const BitTree(wt), target: BitState, out: *[512]u32) usize {
    const BW = bit_word.BitWord(wt);
    var n: usize = 0;
    var i: u32 = 0;
    while (i < tree.bitset.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        if (BW.readBitState(tree.bitset.words.items[wid], BW.bitIdInWord(i)) == target) {
            out[n] = i;
            n += 1;
        }
    }
    return n;
}

fn expectSummariesValid(comptime wt: WordType, tree: *const BitTree(wt)) !void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    const L = Layer(wt);
    var li: usize = 1;
    while (li < tree.layers.items.len) : (li += 1) {
        const lower = &tree.layers.items[li - 1];
        const upper = &tree.layers.items[li];
        const lower_used = BW.bitIdInWord(lower.bits_count);
        const lower_last: usize = lower.activity.items.len - 1;
        var j: u32 = 0;
        while (j < upper.bits_count) : (j += 1) {
            const w: Word = lower.activity.items[j];
            const m: Word = lower.mixed.items[j];
            const valid: Word = if (j == lower_last and lower_used != 0) BW.maskStart(lower_used) else BW.max_value;
            var want: L.State = undefined;
            if (li == 1) {
                const lw = w & valid;
                want = if (lw == 0) .inactive else if (lw == valid) .active else .deep_mixed;
            } else {
                var first: L.State = .inactive;
                var seen = false;
                var uniform = true;
                var b: u32 = 0;
                while (b < BW.word_type_bits) : (b += 1) {
                    const bit: Word = @as(Word, 1) << @truncate(b);
                    if (valid & bit == 0) continue;
                    const cur = L.State.fromBitsState(
                        if ((w & bit) != 0) BitState.active else BitState.inactive,
                        if ((m & bit) != 0) BitState.active else BitState.inactive,
                    );
                    if (!seen) {
                        first = cur;
                        seen = true;
                    } else if (cur != first) {
                        uniform = false;
                    }
                }
                want = if (uniform) first else .mixed;
            }
            const uwid = BW.bitToWordId(j);
            const ubit = BW.bitIdInWord(j);
            const got = L.State.fromBitsState(
                BW.readBitState(upper.activity.items[uwid], ubit),
                BW.readBitState(upper.mixed.items[uwid], ubit),
            );
            try t.expectEqual(want, got);
        }
    }
}

fn treeLayerState(comptime wt: WordType, tree: *const BitTree(wt), li: usize, j: u32) Layer(wt).State {
    const BW = bit_word.BitWord(wt);
    const layer = &tree.layers.items[li];
    return Layer(wt).State.fromBitsState(
        BW.readBitState(layer.activity.items[BW.bitToWordId(j)], BW.bitIdInWord(j)),
        BW.readBitState(layer.mixed.items[BW.bitToWordId(j)], BW.bitIdInWord(j)),
    );
}

test "BitTree summary rule: unanimity + layer1 deep" {
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 128, .inactive);
        tree.setWord(1, std.math.maxInt(u64), std.math.maxInt(u64));
        try t.expectEqual(Layer(.u64).State.inactive, treeLayerState(.u64, &tree, 1, 0));
        try t.expectEqual(Layer(.u64).State.active, treeLayerState(.u64, &tree, 1, 1));
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 128, .inactive);
        tree.setWord(0, 0xFFFF_FFFF, 0xFFFF_FFFF);
        try t.expectEqual(Layer(.u64).State.deep_mixed, treeLayerState(.u64, &tree, 1, 0));
        try t.expectEqual(Layer(.u64).State.inactive, treeLayerState(.u64, &tree, 1, 1));
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 8192, .inactive);
        var wid: u32 = 0;
        while (wid < 32) : (wid += 1) {
            tree.setWord(wid, std.math.maxInt(u64), std.math.maxInt(u64));
        }
        try t.expectEqual(Layer(.u64).State.active, treeLayerState(.u64, &tree, 1, 0));
        try t.expectEqual(Layer(.u64).State.inactive, treeLayerState(.u64, &tree, 1, 40));
        try t.expectEqual(Layer(.u64).State.mixed, treeLayerState(.u64, &tree, 2, 0));
        try t.expectEqual(Layer(.u64).State.inactive, treeLayerState(.u64, &tree, 2, 1));
        var tree2 = BitTree(.u64){};
        defer tree2.deinit(t.allocator);
        try tree2.resize(t.allocator, 8192, .inactive);
        var w2: u32 = 0;
        while (w2 < 64) : (w2 += 1) {
            tree2.setBit(w2 * 64, .active);
        }
        try t.expectEqual(Layer(.u64).State.deep_mixed, treeLayerState(.u64, &tree2, 1, 0));
        try t.expectEqual(Layer(.u64).State.inactive, treeLayerState(.u64, &tree2, 1, 64));
        try t.expectEqual(Layer(.u64).State.deep_mixed, treeLayerState(.u64, &tree2, 2, 0));
        try t.expectEqual(Layer(.u64).State.inactive, treeLayerState(.u64, &tree2, 2, 1));
    }
}

fn expectTreeCoherent(comptime wt: WordType, tree: *const BitTree(wt)) !void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    const L = Layer(wt);
    const n = tree.bitset.bits_count;
    const words = tree.bitset.words.items;
    if (n == 0) {
        try t.expectEqual(@as(usize, 0), words.len);
        try t.expectEqual(@as(usize, 0), tree.layers.items.len);
        return;
    }
    try t.expectEqual(words.len, tree.layers.items[0].activity.items.len);
    var wid: usize = 0;
    while (wid < words.len) : (wid += 1) {
        const used = BW.bitIdInWord(n);
        const valid: Word = if (wid == words.len - 1 and used != 0) BW.maskStart(used) else BW.max_value;
        try t.expectEqual(words[wid] & valid, tree.layers.items[0].activity.items[wid] & valid);
        try t.expectEqual(@as(Word, 0), tree.layers.items[0].mixed.items[wid] & valid);
    }
    var k: usize = 0;
    while (k < tree.layers.items.len) : (k += 1) {
        const layer = &tree.layers.items[k];
        var scanned = [4]u32{ 0, 0, 0, 0 };
        var i: u32 = 0;
        while (i < layer.bits_count) : (i += 1) {
            const w2: usize = @intCast(BW.bitToWordId(i));
            const in_w = BW.bitIdInWord(i);
            const a = BW.readBitState(layer.activity.items[w2], in_w);
            const m = BW.readBitState(layer.mixed.items[w2], in_w);
            scanned[@intFromEnum(L.State.fromBitsState(a, m))] += 1;
        }
        try t.expectEqualSlices(u32, &scanned, &layer.state_counters);
    }
    try t.expectEqual(tree.bitset.active_bits_counter, tree.layers.items[0].state_counters[L.State.active_u32]);
    try t.expectEqual(@as(u32, 0), tree.layers.items[0].state_counters[L.State.mixed_u32]);
    try t.expectEqual(@as(u32, 0), tree.layers.items[0].state_counters[L.State.deep_mixed_u32]);
    try expectSummariesValid(wt, tree);
}

fn treeStepCheckOne(comptime wt: WordType, n: u32, active_every: u32, active_offset: u32) !void {
    const Word = wt.Type();
    const BW = bit_word.BitWord(wt);
    var tree = BitTree(wt){};
    defer tree.deinit(t.allocator);
    try tree.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (active_every != 0 and (i + active_offset) % active_every == 0) {
            tree.setBit(i, .active);
        }
    }
    try expectTreeCoherent(wt, &tree);

    if (n > 0) {
        const used = BW.bitIdInWord(n);
        const valid: Word = if (used == 0) std.math.maxInt(Word) else BW.maskStart(used);
        tree.bitset.words.items[tree.bitset.words.items.len - 1] |= ~valid;
    }

    var ctx = TreeStepIds{};
    treeStepCollectAll(wt, &tree, &ctx);
    var exp_a: [512]u32 = undefined;
    var exp_i: [512]u32 = undefined;
    const n_a = treeOracleIds(wt, &tree, .active, &exp_a);
    const n_i = treeOracleIds(wt, &tree, .inactive, &exp_i);
    std.mem.sort(u32, ctx.active[0..ctx.na], {}, std.sort.asc(u32));
    std.mem.sort(u32, ctx.inactive[0..ctx.ni], {}, std.sort.asc(u32));
    try t.expectEqualSlices(u32, exp_a[0..n_a], ctx.active[0..ctx.na]);
    try t.expectEqualSlices(u32, exp_i[0..n_i], ctx.inactive[0..ctx.ni]);
    try t.expectEqual(n, @as(u32, @intCast(ctx.na + ctx.ni)));
}

test "BitTree step: counts vs oracle + corners" {
    const sizes = [_]u32{ 0, 1, 2, 63, 64, 65, 70, 127, 128, 129, 200, 300 };
    const mods = [_]u32{ 0, 1, 2, 3, 7, 64 };
    for (sizes) |n| {
        for (mods) |m| {
            try treeStepCheckOne(.u64, n, m, 0);
            try treeStepCheckOne(.u8, n, m, 1);
        }
    }
}

test "BitTree step: early exit" {
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 70, .active);
        const It = BitTree(.u64).Iterator(*TreeStepIds, treePushA, treePushI);
        var ctx = TreeStepIds{ .stop_after = 3 };
        _ = It.iterateAll(.{ .tree = &tree, .context = &ctx });
        try t.expectEqual(@as(usize, 3), ctx.na + ctx.ni);
        var seen: [70]bool = [_]bool{false} ** 70;
        for (ctx.active[0..ctx.na]) |id| {
            try t.expect(id < 70);
            try t.expect(!seen[id]);
            seen[id] = true;
        }
        for (ctx.inactive[0..ctx.ni]) |id| {
            try t.expect(id < 70);
            try t.expect(!seen[id]);
            seen[id] = true;
        }
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 130, .inactive);
        tree.setBit(100, .active);
        const It = BitTree(.u64).Iterator(*TreeStepIds, treePushA, treePushI);
        var ctx = TreeStepIds{ .stop_after = 65 };
        _ = It.iterateAll(.{ .tree = &tree, .context = &ctx });
        try t.expectEqual(@as(usize, 65), ctx.na + ctx.ni);
        var seen: [130]bool = [_]bool{false} ** 130;
        for (ctx.active[0..ctx.na]) |id| {
            try t.expect(id < 130);
            try t.expect(!seen[id]);
            seen[id] = true;
            try t.expectEqual(@as(u32, 100), id);
        }
        for (ctx.inactive[0..ctx.ni]) |id| {
            try t.expect(id < 130);
            try t.expect(!seen[id]);
            seen[id] = true;
            try t.expect(id != 100);
        }
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 10, .inactive);
        const It2 = BitTree(.u64).Iterator(*TreeStepIds, treePushA, treePushI);
        var ctx = TreeStepIds{ .stop_after = 1 };
        _ = It2.iterateAll(.{ .tree = &tree, .context = &ctx });
        try t.expectEqual(@as(usize, 1), ctx.ni + ctx.na);
        const only = if (ctx.na == 1) ctx.active[0] else ctx.inactive[0];
        try t.expect(only < 10);
    }
}

test "BitTree step: null side skips opposite uniform regions" {
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 130, .inactive);
        tree.setBit(100, .active);
        const It = BitTree(.u64).Iterator(*TreeStepIds, treePushA, null);
        var ctx = TreeStepIds{};
        _ = It.iterateAll(.{ .tree = &tree, .context = &ctx });
        try t.expectEqualSlices(u32, &[_]u32{100}, ctx.active[0..ctx.na]);
        try t.expectEqual(@as(usize, 0), ctx.ni);
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 130, .inactive);
        tree.setBit(100, .active);
        const It = BitTree(.u64).Iterator(*TreeStepIds, null, treePushI);
        var ctx = TreeStepIds{};
        _ = It.iterateAll(.{ .tree = &tree, .context = &ctx });
        try t.expectEqual(@as(usize, 129), ctx.ni);
        try t.expectEqual(@as(usize, 0), ctx.na);
        std.mem.sort(u32, ctx.inactive[0..ctx.ni], {}, std.sort.asc(u32));
        try t.expectEqual(@as(u32, 0), ctx.inactive[0]);
        try t.expectEqual(@as(u32, 129), ctx.inactive[128]);
    }
}

test "BitTree pyramid shape + coherence under ops" {
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 70, .inactive);
        try t.expectEqual(@as(usize, 3), tree.layers.items.len);
        try t.expectEqual(@as(u32, 70), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 2), tree.layers.items[1].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[2].bits_count);
        try expectTreeCoherent(.u64, &tree);
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 64, .active);
        try t.expectEqual(@as(usize, 2), tree.layers.items.len);
        try expectTreeCoherent(.u64, &tree);
        try tree.resize(t.allocator, 10, .inactive);
        try t.expectEqual(@as(usize, 2), tree.layers.items.len);
        try t.expectEqual(@as(u32, 10), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[1].bits_count);
        try expectTreeCoherent(.u64, &tree);
        try tree.resize(t.allocator, 0, .inactive);
        try t.expectEqual(@as(usize, 0), tree.layers.items.len);
        try t.expectEqual(@as(usize, 0), tree.bitset.words.items.len);
    }
    {
        var tree = BitTree(.u8){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 20, .inactive);
        tree.setBit(0, .active);
        tree.setBit(7, .active);
        tree.setBit(8, .active);
        tree.setBit(19, .active);
        tree.setWord(1, 0xF0, 0x0F);
        tree.setBit(8, .inactive);
        try expectTreeCoherent(.u8, &tree);
        var ctx = TreeStepIds{};
        treeStepCollectAll(.u8, &tree, &ctx);
        try t.expectEqual(@as(u32, 20), tree.bitset.bits_count);
        try t.expectEqual(tree.bitset.active_bits_counter, @as(u32, @intCast(ctx.na)));
    }
}

const StepTotals = struct {
    na: usize = 0,
    ni: usize = 0,
};

inline fn totalPushA(ctx: *StepTotals, bit_id: u32) bool {
    _ = bit_id;
    ctx.na += 1;
    return true;
}

inline fn totalPushI(ctx: *StepTotals, bit_id: u32) bool {
    _ = bit_id;
    ctx.ni += 1;
    return true;
}

fn treeStepTotals(comptime wt: WordType, tree: *BitTree(wt), ctx: *StepTotals) void {
    const It = BitTree(wt).Iterator(*StepTotals, totalPushA, totalPushI);
    _ = It.iterateAll(.{ .tree = tree, .context = ctx });
}

fn treeOracleTotals(comptime wt: WordType, tree: *const BitTree(wt)) [2]u64 {
    const BW = bit_word.BitWord(wt);
    var out = [2]u64{ 0, 0 };
    var i: u32 = 0;
    while (i < tree.bitset.bits_count) : (i += 1) {
        const wid: usize = @intCast(BW.bitToWordId(i));
        if (BW.readBitState(tree.bitset.words.items[wid], BW.bitIdInWord(i)) == .active) {
            out[0] += 1;
        } else {
            out[1] += 1;
        }
    }
    return out;
}

fn treeExpectTotals(comptime wt: WordType, tree: *BitTree(wt)) !void {
    var ctx = StepTotals{};
    treeStepTotals(wt, tree, &ctx);
    const want = treeOracleTotals(wt, tree);
    try t.expectEqual(want[0], ctx.na);
    try t.expectEqual(want[1], ctx.ni);
    try t.expectEqual(@as(u64, tree.bitset.bits_count), ctx.na + ctx.ni);
}

test "BitTree deep pyramid 4+ levels" {
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 20000, .inactive);
        try t.expectEqual(@as(usize, 4), tree.layers.items.len);
        try t.expectEqual(@as(u32, 20000), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 313), tree.layers.items[1].bits_count);
        try t.expectEqual(@as(u32, 5), tree.layers.items[2].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[3].bits_count);
        var i: u32 = 0;
        while (i < 20000) : (i += 997) {
            tree.setBit(i, .active);
        }
        try expectTreeCoherent(.u64, &tree);
        try treeExpectTotals(.u64, &tree);
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 20000, .active);
        var i: u32 = 0;
        while (i < 20000) : (i += 3) {
            tree.setBit(i, .inactive);
        }
        try expectTreeCoherent(.u64, &tree);
        try treeExpectTotals(.u64, &tree);
        try t.expectEqual(@as(u32, 20000 - 6667), tree.bitset.active_bits_counter);
    }
    {
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 100000, .inactive);
        try t.expectEqual(@as(usize, 4), tree.layers.items.len);
        try t.expectEqual(@as(u32, 100000), tree.layers.items[0].bits_count);
        try t.expectEqual(@as(u32, 1563), tree.layers.items[1].bits_count);
        try t.expectEqual(@as(u32, 25), tree.layers.items[2].bits_count);
        try t.expectEqual(@as(u32, 1), tree.layers.items[3].bits_count);
        var k: usize = 0;
        while (k < 2000) : (k += 1) {
            tree.setBit(@truncate((7 + k * 100003) % 100000), .active);
        }
        try expectTreeCoherent(.u64, &tree);
        try treeExpectTotals(.u64, &tree);
    }
}

test "BitTree u16/u32 spot checks" {
    const sizes = [_]u32{ 0, 1, 2, 15, 16, 17, 31, 32, 33, 64, 100, 129, 300 };
    const mods = [_]u32{ 0, 1, 2, 5, 16 };
    for (sizes) |n| {
        for (mods) |m| {
            try treeStepCheckOne(.u16, n, m, 0);
            try treeStepCheckOne(.u16, n, m, 3);
            try treeStepCheckOne(.u32, n, m, 0);
            try treeStepCheckOne(.u32, n, m, 1);
        }
    }
    {
        var tree = BitTree(.u16){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 100, .inactive);
        tree.setBit(0, .active);
        tree.setBit(15, .active);
        tree.setBit(16, .active);
        tree.setBit(99, .active);
        tree.setWord(3, 0xFF00, 0x00FF);
        try expectTreeCoherent(.u16, &tree);
        try treeExpectTotals(.u16, &tree);
    }
    {
        var tree = BitTree(.u32){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 100, .active);
        tree.setBit(31, .inactive);
        tree.setBit(32, .inactive);
        tree.setWord(2, 0, 0xFFFF_FFFF);
        try expectTreeCoherent(.u32, &tree);
        try treeExpectTotals(.u32, &tree);
    }
}

test "BitTree multi-level shrink/grow" {
    var tree = BitTree(.u64){};
    defer tree.deinit(t.allocator);
    try tree.resize(t.allocator, 20000, .active);
    try t.expectEqual(@as(usize, 4), tree.layers.items.len);
    try t.expectEqual(@as(u32, 20000), tree.bitset.active_bits_counter);
    try expectTreeCoherent(.u64, &tree);
    try tree.resize(t.allocator, 10, .inactive);
    try t.expectEqual(@as(usize, 2), tree.layers.items.len);
    try t.expectEqual(@as(u32, 10), tree.layers.items[0].bits_count);
    try t.expectEqual(@as(u32, 1), tree.layers.items[1].bits_count);
    try t.expectEqual(@as(u32, 10), tree.bitset.active_bits_counter);
    try expectTreeCoherent(.u64, &tree);
    try treeExpectTotals(.u64, &tree);
    try tree.resize(t.allocator, 5000, .active);
    try t.expectEqual(@as(usize, 4), tree.layers.items.len);
    try t.expectEqual(@as(u32, 5000), tree.layers.items[0].bits_count);
    try t.expectEqual(@as(u32, 79), tree.layers.items[1].bits_count);
    try t.expectEqual(@as(u32, 2), tree.layers.items[2].bits_count);
    try t.expectEqual(@as(u32, 1), tree.layers.items[3].bits_count);
    try t.expectEqual(@as(u32, 5000), tree.bitset.active_bits_counter);
    try expectTreeCoherent(.u64, &tree);
    try treeExpectTotals(.u64, &tree);
    try tree.resize(t.allocator, 0, .inactive);
    try t.expectEqual(@as(usize, 0), tree.layers.items.len);
    try t.expectEqual(@as(usize, 0), tree.bitset.words.items.len);
    try tree.resize(t.allocator, 64, .inactive);
    try t.expectEqual(@as(usize, 2), tree.layers.items.len);
    try expectTreeCoherent(.u64, &tree);
    try treeExpectTotals(.u64, &tree);
}

const OrderDebugIds = struct {
    ids: [512]u32 = undefined,
    n: usize = 0,
};

inline fn orderDebugPushA(ctx: *OrderDebugIds, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.n += 1;
    return true;
}

inline fn orderDebugPushI(ctx: *OrderDebugIds, bit_id: u32) bool {
    ctx.ids[ctx.n] = bit_id;
    ctx.n += 1;
    return true;
}

fn orderDebugIsSorted(ctx: *const OrderDebugIds) bool {
    var k: usize = 1;
    while (k < ctx.n) : (k += 1) {
        if (ctx.ids[k] <= ctx.ids[k - 1]) return false;
    }
    return true;
}

fn orderDebugFirstViolation(ctx: *const OrderDebugIds) [3]u32 {
    var k: usize = 1;
    while (k < ctx.n) : (k += 1) {
        if (ctx.ids[k] <= ctx.ids[k - 1]) return .{ @truncate(k), ctx.ids[k - 1], ctx.ids[k] };
    }
    return .{ 0, 0, 0 };
}

fn orderDebugCheckOne(comptime wt: WordType, n: u32, active_every: u32, active_offset: u32) !void {
    var tree = BitTree(wt){};
    defer tree.deinit(t.allocator);
    try tree.resize(t.allocator, n, .inactive);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        if (active_every != 0 and (i + active_offset) % active_every == 0) {
            tree.setBit(i, .active);
        }
    }

    var ctx = OrderDebugIds{};
    const It = BitTree(wt).Iterator(*OrderDebugIds, orderDebugPushA, orderDebugPushI);
    _ = It.iterateAll(.{ .tree = &tree, .context = &ctx });

    if (orderDebugIsSorted(&ctx)) {
        std.debug.print("BitTree iteration order: OK (n={d} every={d} offset={d} total={d})\n", .{ n, active_every, active_offset, ctx.n });
    } else {
        const v = orderDebugFirstViolation(&ctx);
        std.debug.print("BitTree iteration order: VIOLATED (n={d} every={d} offset={d} total={d} first_violation_at={d} prev={d} cur={d})\n", .{ n, active_every, active_offset, ctx.n, v[0], v[1], v[2] });
    }
}

test "BitTree step: debug iteration order" {
    const sizes = [_]u32{ 0, 1, 10, 64, 65, 70, 128, 130, 200, 300 };
    const mods = [_]u32{ 0, 1, 2, 3, 7 };
    for (sizes) |n| {
        for (mods) |m| {
            try orderDebugCheckOne(.u64, n, m, 0);
            try orderDebugCheckOne(.u8, n, m, 1);
        }
    }
}

fn predictParityCheckOne(comptime wt: WordType, n: u32, active_every: u32, active_offset: u32) !void {
    const modes = [_]PredictForce{ .auto, .flat, .tree };
    var first_a: [512]u32 = undefined;
    var first_i: [512]u32 = undefined;
    var have_first = false;
    var first_na: usize = 0;
    var first_ni: usize = 0;
    for (modes) |m| {
        const saved = predict_config;
        predict_config.force = m;
        defer predict_config = saved;
        var tree = BitTree(wt){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, n, .inactive);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (active_every != 0 and (i + active_offset) % active_every == 0) {
                tree.setBit(i, .active);
            }
        }
        var ctx = TreeStepIds{};
        treeStepCollectAll(wt, &tree, &ctx);
        std.mem.sort(u32, ctx.active[0..ctx.na], {}, std.sort.asc(u32));
        std.mem.sort(u32, ctx.inactive[0..ctx.ni], {}, std.sort.asc(u32));
        var exp_a: [512]u32 = undefined;
        var exp_i: [512]u32 = undefined;
        const n_a = treeOracleIds(wt, &tree, .active, &exp_a);
        const n_i = treeOracleIds(wt, &tree, .inactive, &exp_i);
        try t.expectEqualSlices(u32, exp_a[0..n_a], ctx.active[0..ctx.na]);
        try t.expectEqualSlices(u32, exp_i[0..n_i], ctx.inactive[0..ctx.ni]);
        if (!have_first) {
            std.mem.copyForwards(u32, first_a[0..ctx.na], ctx.active[0..ctx.na]);
            std.mem.copyForwards(u32, first_i[0..ctx.ni], ctx.inactive[0..ctx.ni]);
            first_na = ctx.na;
            first_ni = ctx.ni;
            have_first = true;
        } else {
            try t.expectEqualSlices(u32, first_a[0..first_na], ctx.active[0..ctx.na]);
            try t.expectEqualSlices(u32, first_i[0..first_ni], ctx.inactive[0..ctx.ni]);
        }
    }
}

test "BitTree predict: auto/flat/tree parity vs oracle" {
    const saved = predict_config;
    defer predict_config = saved;
    const sizes = [_]u32{ 0, 1, 10, 64, 65, 70, 130, 200, 300 };
    const mods = [_]u32{ 0, 1, 2, 3, 7 };
    for (sizes) |n| {
        for (mods) |m| {
            try predictParityCheckOne(.u64, n, m, 0);
        }
    }
    for ([_]PredictForce{ .auto, .flat, .tree }) |m| {
        predict_config.force = m;
        var tree = BitTree(.u64){};
        defer tree.deinit(t.allocator);
        try tree.resize(t.allocator, 64, .inactive);
        var ctx = TreeStepIds{};
        treeStepCollectAll(.u64, &tree, &ctx);
        try t.expectEqual(@as(usize, 0), ctx.na);
        try t.expectEqual(@as(usize, 64), ctx.ni);
    }
}

fn treeHoistedCollectAll(comptime wt: WordType, tree: *BitTree(wt), ctx: *TreeStepIds) void {
    const It = BitTree(wt).Iterator(*TreeStepIds, treePushA, treePushI);
    if (It.predictsFlat(tree)) {
        _ = It.iterateFlat(.{ .tree = tree, .context = ctx });
    } else {
        _ = It.iterateTree(.{ .tree = tree, .context = ctx });
    }
}

test "BitTree hoisted arms parity vs oracle" {
    const saved = predict_config;
    defer predict_config = saved;
    const sizes = [_]u32{ 0, 1, 10, 64, 70, 130, 200, 300 };
    const mods = [_]u32{ 0, 1, 2, 3, 7 };
    for ([_]PredictForce{ .auto, .flat, .tree }) |m| {
        predict_config.force = m;
        for (sizes) |n| {
            for (mods) |mm| {
                var tree = BitTree(.u64){};
                defer tree.deinit(t.allocator);
                try tree.resize(t.allocator, n, .inactive);
                var i: u32 = 0;
                while (i < n) : (i += 1) {
                    if (mm != 0 and (i + 1) % mm == 0) tree.setBit(i, .active);
                }
                var ctx = TreeStepIds{};
                treeHoistedCollectAll(.u64, &tree, &ctx);
                std.mem.sort(u32, ctx.active[0..ctx.na], {}, std.sort.asc(u32));
                std.mem.sort(u32, ctx.inactive[0..ctx.ni], {}, std.sort.asc(u32));
                var exp_a: [512]u32 = undefined;
                var exp_i: [512]u32 = undefined;
                const n_a = treeOracleIds(.u64, &tree, .active, &exp_a);
                const n_i = treeOracleIds(.u64, &tree, .inactive, &exp_i);
                try t.expectEqualSlices(u32, exp_a[0..n_a], ctx.active[0..ctx.na]);
                try t.expectEqualSlices(u32, exp_i[0..n_i], ctx.inactive[0..ctx.ni]);
            }
        }
    }
}

const CrossIds = struct {
    alloc: Allocator,
    sums: [4]*Layer(.u64),
    active: std.ArrayList(u32) = .empty,
    inactive: std.ArrayList(u32) = .empty,
};

inline fn crossNT(ctx: *CrossIds, pos: u32) bool {
    const BW = bit_word.BitWord(.u64);
    const wid: usize = @intCast(BW.bitToWordId(pos));
    const in_w = BW.bitIdInWord(pos);
    for (ctx.sums) |layer| {
        if (BW.readBitState(layer.mixed.items[wid], in_w) == .active) return true;
    }
    return false;
}

inline fn crossPushA(ctx: *CrossIds, bit_id: u32) bool {
    if (crossNT(ctx, bit_id >> 6)) return true;
    ctx.active.append(ctx.alloc, bit_id) catch unreachable;
    return true;
}

inline fn crossPushI(ctx: *CrossIds, bit_id: u32) bool {
    if (crossNT(ctx, bit_id >> 6)) return true;
    ctx.inactive.append(ctx.alloc, bit_id) catch unreachable;
    return true;
}

const CrossLock = struct {
    active: []const u32,
    inactive: []const u32,
    ca: usize = 0,
    ci: usize = 0,
};

inline fn crossLockPushI(ctx: *CrossLock, pos: u32) bool {
    std.debug.assert(ctx.ci + 64 <= ctx.inactive.len);
    std.debug.assert(ctx.inactive[ctx.ci] == pos * 64);
    ctx.ci += 64;
    return true;
}

inline fn crossLockPushA(ctx: *CrossLock, pos: u32) bool {
    std.debug.assert(ctx.ca + 64 <= ctx.active.len);
    std.debug.assert(ctx.active[ctx.ca] == pos * 64);
    ctx.ca += 64;
    return true;
}

fn crossLockPushM(ctx: *CrossLock, pos: u32) bool {
    _ = ctx;
    _ = pos;
    return true;
}

inline fn crossLockPushD(ctx: *CrossLock, pos: u32) bool {
    _ = ctx;
    _ = pos;
    return true;
}

fn crossRandU32(state: *u64) u32 {
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @truncate((x *% 0x2545F4914F6CDD1D) >> 32);
}

test "BitTree CommonIterator cross: summary layer vs bitset 262144" {
    const N: u32 = 262144;
    const WORDS: u32 = N / 64;
    var rng: u64 = 0x9E3779B97F4A7C15;
    var trees: [4]BitTree(.u64) = .{ .{}, .{}, .{}, .{} };
    defer {
        for (0..4) |k| trees[k].deinit(t.allocator);
    }
    for (0..4) |k| try trees[k].resize(t.allocator, N, .inactive);
    var w: u32 = 0;
    while (w < WORDS) : (w += 1) {
        for (0..4) |k| {
            const hi = crossRandU32(&rng);
            const lo = crossRandU32(&rng);
            const pick = crossRandU32(&rng);
            const word: u64 = if ((w & 1) == 0)
                (@as(u64, hi) << 32) | lo
            else if ((pick & 1) == 1)
                0
            else
                std.math.maxInt(u64);
            trees[k].setWord(w, word, std.math.maxInt(u64));
        }
    }

    const sum0 = &trees[0].layers.items[1];
    const sum1 = &trees[1].layers.items[1];
    const sum2 = &trees[2].layers.items[1];
    const sum3 = &trees[3].layers.items[1];
    try t.expectEqual(N / 64, sum0.bits_count);
    try t.expectEqual(@as(usize, WORDS / 64), sum0.activity.items.len);

    // Phase 1: walk the bitsets, skipping bits whose word is mixed or
    // deeply mixed in any summary layer.
    var bctx = CrossIds{ .alloc = t.allocator, .sums = .{ sum0, sum1, sum2, sum3 } };
    defer bctx.active.deinit(bctx.alloc);
    defer bctx.inactive.deinit(bctx.alloc);
    try bctx.active.ensureTotalCapacity(bctx.alloc, N);
    try bctx.inactive.ensureTotalCapacity(bctx.alloc, N);
    const BI = BitSet(.u64).CommonIterator(2, 2, *CrossIds, crossPushA, crossPushI);
    try t.expect(BI.iterateAll(.{
        .includes = .{ &trees[0].bitset, &trees[1].bitset },
        .excludes = .{ &trees[2].bitset, &trees[3].bitset },
        .context = &bctx,
    }));

    // Phase 2: walk the layers lockstep with the phase 1 lists. Every
    // visited active/inactive position must open a full 64-bit run.
    var lock = CrossLock{ .active = bctx.active.items, .inactive = bctx.inactive.items };
    const LI = Layer(.u64).CommonIterator(2, 2, *CrossLock, crossLockPushI, crossLockPushA, crossLockPushM, crossLockPushD);
    const ldata: Layer(.u64).LayersWithContext(2, 2, *CrossLock) = .{
        .includes = .{ sum0, sum1 },
        .excludes = .{ sum2, sum3 },
        .context = &lock,
    };
    var wid: u32 = 0;
    while (wid < sum0.activity.items.len) : (wid += 1) {
        try t.expect(LI.step(ldata, wid));
    }
    try t.expectEqual(bctx.active.items.len, lock.ca);
    try t.expectEqual(bctx.inactive.items.len, lock.ci);
    std.debug.print("cross 262144: active={} inactive={}\n", .{ bctx.active.items.len, bctx.inactive.items.len });
}
