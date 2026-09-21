const std = @import("std");
const bit_set = @import("bit_set.zig");
const utilities = @import("utilities.zig");
const bit_tree = @import("bit_tree.zig");

/// Allocator type that owns benchmark and history buffers.
const Allocator = std.mem.Allocator;
/// Reference flat set used only to verify tree results.
const Oracle = bit_set.BitSet(.u64);
/// Logical bit value selecting active or inactive benchmarks.
const BitState = utilities.BitState;
/// Measured hierarchical tree built from identical inputs.
const Tree = bit_tree.BitTree(.u64);

/// History file holding past ms/rep columns for trend comparison.
const HISTORY_PATH = "bench_bit_tree_history.txt";
/// Maximum history columns shown before old runs roll off.
const MAX_PREV_COLS: usize = 8;

/// Monotonic timer that stamps benchmark intervals in nanoseconds.
const Stopwatch = struct {
    /// I/O context providing the monotonic clock source.
    io: std.Io,
    /// Start stamp captured when the interval begins.
    t0: std.Io.Timestamp,
    /// Captures the start stamp for one measured interval.
    /// - `io` - clock source for timestamps.
    ///
    /// Return - running timer holding the start point.
    fn start(io: std.Io) @This() {
        return .{ .io = io, .t0 = std.Io.Timestamp.now(io, .awake) };
    }
    /// Reads elapsed nanoseconds since start.
    /// - `self` - running timer to sample.
    ///
    /// Return - elapsed nanoseconds as an integer.
    fn read(self: *@This()) u64 {
        const t1 = std.Io.Timestamp.now(self.io, .awake);
        return @intCast(std.Io.Timestamp.durationTo(self.t0, t1).nanoseconds);
    }
};

/// Fills the oracle with a strided active pattern.
/// - `oracle` - reference set to populate.
/// - `stride` - distance between active bits.
/// - `offset` - first active position.
fn fillStride(oracle: *Oracle, stride: u32, offset: u32) void {
    var b: u32 = offset;
    while (b < oracle.bits_count) : (b += stride) {
        oracle.setBit(b, .active);
    }
}

/// Fills the measured tree with the same strided pattern as the oracle.
/// - `tree` - tree to populate.
/// - `stride` - distance between active bits.
/// - `offset` - first active position.
fn fillTreeStride(tree: *Tree, stride: u32, offset: u32) void {
    var b: u32 = offset;
    while (b < tree.bitset.bits_count) : (b += stride) {
        tree.setBit(b, .active);
    }
}

/// Fills the oracle with clustered runs separated by gaps.
/// - `oracle` - reference set to populate.
/// - `run` - active bits per cluster.
/// - `gap` - inactive bits between clusters.
/// - `offset` - first cluster start.
fn fillClustered(oracle: *Oracle, run: u32, gap: u32, offset: u32) void {
    const total: u64 = oracle.bits_count;
    var b: u64 = offset;
    while (b < total) {
        const end: u64 = @min(b + run, total);
        var i: u32 = @intCast(b);
        while (i < end) : (i += 1) {
            oracle.setBit(i, .active);
        }
        b = end + gap;
    }
}

/// Fills the measured tree with clustered runs matching the oracle.
/// - `tree` - tree to populate.
/// - `run` - active bits per cluster.
/// - `gap` - inactive bits between clusters.
fn fillTreeClustered(tree: *Tree, run: u32, gap: u32) void {
    const total: u64 = tree.bitset.bits_count;
    var b: u64 = 0;
    while (b < total) {
        const end: u64 = @min(b + run, total);
        var i: u32 = @intCast(b);
        while (i < end) : (i += 1) {
            tree.setBit(i, .active);
        }
        b = end + gap;
    }
}

/// Shortens large counts for fixed-width row names.
/// - `buf` - scratch buffer receiving the formatted name.
/// - `n` - raw count to shorten.
///
/// Return - slice of the buffer with the short name.
fn fmtCount(buf: []u8, n: u32) ![]u8 {
    if (n >= 1_000_000 and n % 1_000_000 == 0) {
        return std.fmt.bufPrint(buf, "{d}M", .{n / 1_000_000});
    }
    if (n >= 1_000 and n % 1_000 == 0) {
        return std.fmt.bufPrint(buf, "{d}K", .{n / 1_000});
    }
    return std.fmt.bufPrint(buf, "{d}", .{n});
}

/// Computes reference sums by scanning oracle words once.
/// - `oracle` - reference set to scan.
/// - `want` - which state contributes to the totals.
///
/// Return - summed ids and matching counts.
fn oracleSumOnce(oracle: *const Oracle, want: BitState) struct { sum: u64, count: u64 } {
    var sum: u64 = 0;
    var count: u64 = 0;
    const words = oracle.words.items;
    var w: usize = 0;
    while (w < words.len) : (w += 1) {
        var bits: u64 = words[w];
        if (want == .inactive) bits = ~bits;
        if (w + 1 == words.len and (oracle.bits_count & 63) != 0) {
            const rem: u6 = @intCast(oracle.bits_count & 63);
            bits &= (@as(u64, 1) << rem) - 1;
        }
        while (bits != 0) {
            const s: u32 = @ctz(bits);
            bits &= bits - 1;
            sum += @as(u64, w) * 64 + s;
            count += 1;
        }
    }
    return .{ .sum = sum, .count = count };
}

/// Pinned per-id accumulator that prevents bulk-sum folding.
const SumCtx = struct {
    /// Summed ids, pinned so every visit must materialize.
    sum: u64 = 0,
    /// Visited count, validates cardinalities against the oracle.
    count: u64 = 0,
    /// Adds one visited id while pinning it against optimization.
    /// - `self` - accumulator receiving the id.
    /// - `id` - visited bit id.
    ///
    /// Return - always true to continue the scan.
    inline fn addInline(self: *SumCtx, id: u32) bool {
        std.mem.doNotOptimizeAway(id);
        self.sum += id;
        self.count += 1;
        return true;
    }
};

/// Times the predicted tree arm with per-id pinning over repetitions.
/// - `io` - clock source for the interval.
/// - `tree` - measured tree, immutable during timing.
/// - `want` - which state is benchmarked.
/// - `reps` - timed repetitions.
///
/// Return - pinned sums, counts and total nanoseconds.
fn benchTree(io: std.Io, tree: *Tree, comptime want: BitState, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    const It = Tree.Iterator(*SumCtx, SumCtx.addInline, null);
    const ItI = Tree.Iterator(*SumCtx, null, SumCtx.addInline);
    const use_flat = if (want == .active) It.predictsFlat(tree) else ItI.predictsFlat(tree);
    bit_tree.last_predict_used_flat = use_flat;
    var watch = Stopwatch.start(io);
    var sum: u64 = 0;
    var count: u64 = 0;
    if (use_flat) {
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            var c = SumCtx{};
            const done = if (want == .active)
                It.iterateFlat(.{ .tree = tree, .context = &c })
            else
                ItI.iterateFlat(.{ .tree = tree, .context = &c });
            std.mem.doNotOptimizeAway(done);
            sum += c.sum;
            count += c.count;
        }
    } else {
        var r: u32 = 0;
        while (r < reps) : (r += 1) {
            var c = SumCtx{};
            const done = if (want == .active)
                It.iterateTree(.{ .tree = tree, .context = &c })
            else
                ItI.iterateTree(.{ .tree = tree, .context = &c });
            std.mem.doNotOptimizeAway(done);
            sum += c.sum;
            count += c.count;
        }
    }
    std.mem.doNotOptimizeAway(sum);
    std.mem.doNotOptimizeAway(count);
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

/// One benchmark row with timing and winning path.
const Row = struct {
    /// Scenario label shown in the result table.
    name: []const u8,
    /// Matching elements, validates workload size.
    elements: u64,
    /// Milliseconds per repetition, the compared metric.
    ms: f64,
    /// Predicted path flag, null when the scenario has no choice.
    flat: ?bool = null,
};

/// Heap-backed collector that records visited ids for set comparison.
const VerifyCtx = struct {
    /// Output list receiving every visited id.
    list: *std.ArrayListUnmanaged(u32),
    /// Appends one visited id without reallocation checks in hot code.
    /// - `ctx` - collector receiving the id.
    /// - `id` - visited bit id.
    ///
    /// Return - always true to continue the scan.
    inline fn push(ctx: *VerifyCtx, id: u32) bool {
        ctx.list.appendAssumeCapacity(id);
        return true;
    }
};

/// Checks tree ids and sums against the oracle as unordered sets.
/// - `alloc` - owns temporary id lists.
/// - `oracle` - reference set defining expected ids.
/// - `tree` - measured tree to verify.
/// - `want` - which state is compared.
/// - `name` - scenario label for mismatch reports.
///
/// Return - error on id or sum mismatch.
fn verifyTree(
    alloc: Allocator,
    oracle: *const Oracle,
    tree: *Tree,
    comptime want: BitState,
    name: []const u8,
) !void {
    var exp: std.ArrayListUnmanaged(u32) = .empty;
    defer exp.deinit(alloc);
    var exp_sum: u64 = 0;
    const words = oracle.words.items;
    var w: usize = 0;
    while (w < words.len) : (w += 1) {
        var bits: u64 = words[w];
        if (want == .inactive) bits = ~bits;
        if (w + 1 == words.len and (oracle.bits_count & 63) != 0) {
            const rem: u6 = @intCast(oracle.bits_count & 63);
            bits &= (@as(u64, 1) << rem) - 1;
        }
        while (bits != 0) {
            const s: u32 = @ctz(bits);
            bits &= bits - 1;
            const id: u32 = @truncate(@as(u64, w) * 64 + s);
            try exp.append(alloc, id);
            exp_sum += id;
        }
    }
    var got: std.ArrayListUnmanaged(u32) = .empty;
    defer got.deinit(alloc);
    try got.ensureTotalCapacity(alloc, exp.items.len);
    {
        var vc = VerifyCtx{ .list = &got };
        const It = Tree.Iterator(*VerifyCtx, VerifyCtx.push, null);
        const ItI = Tree.Iterator(*VerifyCtx, null, VerifyCtx.push);
        if (want == .active) {
            _ = It.iterateAll(.{ .tree = tree, .context = &vc });
        } else {
            _ = ItI.iterateAll(.{ .tree = tree, .context = &vc });
        }
    }
    if (got.items.len != exp.items.len) {
        std.debug.print("verify {s}: count mismatch got={d} exp={d}\n", .{ name, got.items.len, exp.items.len });
        return error.VerifyMismatch;
    }
    std.mem.sort(u32, got.items, {}, std.sort.asc(u32));
    if (!std.mem.eql(u32, got.items, exp.items)) {
        std.debug.print("verify {s}: id set mismatch (len={d})\n", .{ name, exp.items.len });
        return error.VerifyMismatch;
    }
    var got_sum: u64 = 0;
    for (got.items) |id| got_sum += id;
    if (got_sum != exp_sum) {
        std.debug.print("verify {s}: sum mismatch got={d} exp={d}\n", .{ name, got_sum, exp_sum });
        return error.VerifyMismatch;
    }
}

/// Verifies once then times a single scenario and appends its row.
/// - `io` - clock source for timing.
/// - `alloc` - owns row names.
/// - `rows` - table collecting every scenario result.
/// - `name` - scenario label for the table.
/// - `oracle` - reference set for verification.
/// - `tree` - measured tree for timing.
/// - `want` - which state is benchmarked.
/// - `reps` - timed repetitions.
///
/// Return - error on verification or allocation failure.
fn benchOne(
    io: std.Io,
    alloc: Allocator,
    rows: *std.ArrayListUnmanaged(Row),
    name: []const u8,
    oracle: *const Oracle,
    tree: *Tree,
    comptime want: BitState,
    reps: u32,
) !void {
    try verifyTree(alloc, oracle, tree, want, name);
    {
        var c = SumCtx{};
        const It = Tree.Iterator(*SumCtx, SumCtx.addInline, null);
        const ItI = Tree.Iterator(*SumCtx, null, SumCtx.addInline);
        const done = if (want == .active)
            It.iterateAll(.{ .tree = tree, .context = &c })
        else
            ItI.iterateAll(.{ .tree = tree, .context = &c });
        std.mem.doNotOptimizeAway(done);
    }
    const exp = oracleSumOnce(oracle, want);
    const got = benchTree(io, tree, want, reps);
    std.debug.assert(exp.sum * reps == got.sum and exp.count * reps == got.count);
    const ms: f64 = @as(f64, @floatFromInt(got.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0;
    try rows.append(alloc, .{ .name = try alloc.dupe(u8, name), .elements = exp.count, .ms = ms, .flat = bit_tree.last_predict_used_flat });
}

/// Builds identical strided inputs and benchmarks one density point.
/// - `io` - clock source for timing.
/// - `alloc` - owns row names.
/// - `rows` - table collecting results.
/// - `bits_total` - total bits in both containers.
/// - `stride` - distance between active bits.
/// - `reps` - timed repetitions.
///
/// Return - error on verification or allocation failure.
fn benchThreshold(
    io: std.Io,
    alloc: Allocator,
    rows: *std.ArrayListUnmanaged(Row),
    bits_total: u32,
    stride: u32,
    reps: u32,
) !void {
    var oracle = Oracle{};
    defer oracle.deinit(alloc);
    var tree = Tree{};
    defer tree.deinit(alloc);
    try oracle.resize(alloc, bits_total, .inactive);
    try tree.resize(alloc, bits_total, .inactive);
    fillStride(&oracle, stride, 0);
    fillTreeStride(&tree, stride, 0);
    var name_buf: [64]u8 = undefined;
    var sbuf: [16]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "gap{s}-N{s}", .{
        try fmtCount(&sbuf, stride),
        try fmtCount(&nbuf, bits_total),
    });
    try benchOne(io, alloc, rows, name, &oracle, &tree, .active, reps);
}

/// Builds identical clustered inputs and benchmarks one shape point.
/// - `io` - clock source for timing.
/// - `alloc` - owns row names.
/// - `rows` - table collecting results.
/// - `bits_total` - total bits in both containers.
/// - `run` - active bits per cluster.
/// - `gap` - inactive bits between clusters.
/// - `reps` - timed repetitions.
///
/// Return - error on verification or allocation failure.
fn benchCluster(
    io: std.Io,
    alloc: Allocator,
    rows: *std.ArrayListUnmanaged(Row),
    bits_total: u32,
    run: u32,
    gap: u32,
    reps: u32,
) !void {
    var oracle = Oracle{};
    defer oracle.deinit(alloc);
    var tree = Tree{};
    defer tree.deinit(alloc);
    try oracle.resize(alloc, bits_total, .inactive);
    try tree.resize(alloc, bits_total, .inactive);
    fillClustered(&oracle, run, gap, 0);
    fillTreeClustered(&tree, run, gap);
    var name_buf: [64]u8 = undefined;
    var rbuf: [16]u8 = undefined;
    var gbuf: [16]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "cluster-r{s}-g{s}-N{s}", .{
        try fmtCount(&rbuf, run),
        try fmtCount(&gbuf, gap),
        try fmtCount(&nbuf, bits_total),
    });
    try benchOne(io, alloc, rows, name, &oracle, &tree, .active, reps);
}

/// Times fresh-tree construction with per-bit inserts.
/// - `io` - clock source for timing.
/// - `alloc` - owns temporary trees.
/// - `rows` - table collecting results.
/// - `bits_total` - tree size per repetition.
/// - `stride` - distance between inserted bits.
/// - `offset` - first inserted position.
/// - `reps` - timed repetitions.
///
/// Return - error on allocation failure.
fn benchFill(
    io: std.Io,
    alloc: Allocator,
    rows: *std.ArrayListUnmanaged(Row),
    bits_total: u32,
    stride: u32,
    offset: u32,
    reps: u32,
) !void {
    var watch = Stopwatch.start(io);
    var check: u64 = 0;
    var r: u32 = 0;
    while (r < reps) : (r += 1) {
        var tree = Tree{};
        try tree.resize(alloc, bits_total, .inactive);
        var b: u32 = offset;
        while (b < bits_total) : (b += stride) {
            tree.setBit(b, .active);
        }
        check += tree.bitset.active_bits_counter;
        tree.deinit(alloc);
    }
    std.mem.doNotOptimizeAway(check);
    const ns: f64 = @as(f64, @floatFromInt(watch.read())) / @as(f64, @floatFromInt(reps));
    var name_buf: [64]u8 = undefined;
    var sbuf: [16]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "fill-s{s}-N{s}", .{
        try fmtCount(&sbuf, stride),
        try fmtCount(&nbuf, bits_total),
    });
    try rows.append(alloc, .{ .name = try alloc.dupe(u8, name), .elements = check / reps, .ms = ns / 1_000_000.0 });
}

/// Callback mode selecting installed common visitors.
const CommonMode = enum { active_only, inactive_only, both };

/// Pinned per-id collector for common walks with dual lists.
const CommonCtx = struct {
    /// Active ids in visit order, capacity reserved upfront.
    active: std.ArrayListUnmanaged(u32) = .empty,
    /// Inactive ids in visit order, capacity reserved upfront.
    inactive: std.ArrayListUnmanaged(u32) = .empty,
    /// Appends one active id without reallocation checks in hot code.
    ///
    /// Return - always true to continue the scan.
    inline fn pushA(ctx: *CommonCtx, id: u32) bool {
        ctx.active.appendAssumeCapacity(id);
        return true;
    }
    /// Appends one inactive id without reallocation checks in hot code.
    ///
    /// Return - always true to continue the scan.
    inline fn pushI(ctx: *CommonCtx, id: u32) bool {
        ctx.inactive.appendAssumeCapacity(id);
        return true;
    }
};

/// Median of per-repetition timings, robust against noise spikes.
/// - `samples` - nanosecond samples, sorted in place.
///
/// Return - middle sample.
fn medianNs(samples: []u64) u64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return samples[samples.len / 2];
}

/// Verifies one common walk against word-built oracle sets.
/// - `alloc` - owns temporary id lists.
/// - `includes` - trees whose leaves are ANDed.
/// - `excludes` - trees whose leaves are ORed.
/// - `n` - shared valid bits.
/// - `name` - scenario label for mismatch reports.
///
/// Return - oracle active and inactive totals.
fn verifyCommon(
    comptime IL: u32,
    comptime EL: u32,
    alloc: Allocator,
    includes: [IL]*Tree,
    excludes: [EL]*Tree,
    n: u32,
    name: []const u8,
) !struct { na: usize, ni: usize } {
    var exp_a: std.ArrayListUnmanaged(u32) = .empty;
    defer exp_a.deinit(alloc);
    var exp_i: std.ArrayListUnmanaged(u32) = .empty;
    defer exp_i.deinit(alloc);
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const wid = i >> 6;
        const bit: u6 = @truncate(i);
        const m: u64 = @as(u64, 1) << bit;
        var is_a = true;
        var is_i = true;
        for (includes) |tree| {
            const b = (tree.bitset.words.items[wid] & m) != 0;
            if (!b) is_a = false;
            if (b) is_i = false;
        }
        for (excludes) |tree| {
            const b = (tree.bitset.words.items[wid] & m) != 0;
            if (b) is_a = false;
            if (!b) is_i = false;
        }
        if (is_a) try exp_a.append(alloc, i);
        if (is_i) try exp_i.append(alloc, i);
    }
    var vc = CommonCtx{};
    defer vc.active.deinit(alloc);
    defer vc.inactive.deinit(alloc);
    try vc.active.ensureTotalCapacity(alloc, exp_a.items.len);
    try vc.inactive.ensureTotalCapacity(alloc, exp_i.items.len);
    {
        const It = Tree.CommonIterator(IL, EL, *CommonCtx, CommonCtx.pushA, CommonCtx.pushI);
        _ = It.iterateAll(.{
            .includes = includes,
            .excludes = excludes,
            .context = &vc,
        });
    }
    if (!std.mem.eql(u32, vc.active.items, exp_a.items)) {
        std.debug.print("verify {s}: active set mismatch got={d} exp={d}\n", .{ name, vc.active.items.len, exp_a.items.len });
        return error.VerifyMismatch;
    }
    if (!std.mem.eql(u32, vc.inactive.items, exp_i.items)) {
        std.debug.print("verify {s}: inactive set mismatch got={d} exp={d}\n", .{ name, vc.inactive.items.len, exp_i.items.len });
        return error.VerifyMismatch;
    }
    return .{ .na = exp_a.items.len, .ni = exp_i.items.len };
}

/// Times one common arm once, pinning every visited id.
/// - `io` - clock source for the interval.
/// - `includes` - trees whose leaves are ANDed.
/// - `excludes` - trees whose leaves are ORed.
/// - `mode` - installed callback subset.
/// - `ctx` - reused collector with reserved capacity.
/// - `arm` - forced flat, forced tree, or predicted auto.
///
/// Return - elapsed nanoseconds.
fn timeCommonArm(
    comptime IL: u32,
    comptime EL: u32,
    comptime mode: CommonMode,
    io: std.Io,
    includes: [IL]*Tree,
    excludes: [EL]*Tree,
    ctx: *CommonCtx,
    arm: u32,
) u64 {
    const on_a = if (mode == .inactive_only) null else CommonCtx.pushA;
    const on_i = if (mode == .active_only) null else CommonCtx.pushI;
    const It = Tree.CommonIterator(IL, EL, *CommonCtx, on_a, on_i);
    const data: Tree.TreesWithContext(IL, EL, *CommonCtx) = .{ .includes = includes, .excludes = excludes, .context = ctx };
    ctx.active.clearRetainingCapacity();
    ctx.inactive.clearRetainingCapacity();
    var watch = Stopwatch.start(io);
    if (arm == 0) {
        _ = It.iterateFlat(data);
    } else if (arm == 1) {
        _ = It.iterateTree(data);
    } else {
        _ = It.iterateAll(data);
    }
    const ns = watch.read();
    std.mem.doNotOptimizeAway(ctx.active.items.len + ctx.inactive.items.len);
    return ns;
}

/// One common scenario result for the flat/tree/auto table.
const CommonBenchResult = struct {
    /// Scenario label, static storage.
    name: []const u8,
    /// Milliseconds per repetition per arm.
    ms: [3]f64,
    /// Auto choice, true for flat.
    choice_flat: bool,
    /// Faster arm, true for flat.
    faster_flat: bool,
    /// Race inside the noise margin, choice not judged.
    tied: bool,
    /// Judged choice matches the faster arm.
    ok: bool,
};

/// Benchmarks flat/tree/auto arms of one common scenario with medians.
/// - `io` - clock source for timing.
/// - `alloc` - owns temporary id lists.
/// - `name` - scenario label, static storage.
/// - `includes` - trees whose leaves are ANDed.
/// - `excludes` - trees whose leaves are ORed.
/// - `n` - shared valid bits.
/// - `mode` - installed callback subset.
///
/// Return - medians with the auto choice and match flag.
fn benchCommonMode(
    comptime IL: u32,
    comptime EL: u32,
    comptime mode: CommonMode,
    io: std.Io,
    alloc: Allocator,
    name: []const u8,
    includes: [IL]*Tree,
    excludes: [EL]*Tree,
    n: u32,
) !CommonBenchResult {
    _ = try verifyCommon(IL, EL, alloc, includes, excludes, n, name);
    var lists = CommonCtx{};
    defer lists.active.deinit(alloc);
    defer lists.inactive.deinit(alloc);
    try lists.active.ensureTotalCapacity(alloc, n);
    try lists.inactive.ensureTotalCapacity(alloc, n);
    const on_a = if (mode == .inactive_only) null else CommonCtx.pushA;
    const on_i = if (mode == .active_only) null else CommonCtx.pushI;
    const It = Tree.CommonIterator(IL, EL, *CommonCtx, on_a, on_i);
    var ms: [3]f64 = undefined;
    var a: usize = 0;
    while (a < 3) : (a += 1) {
        _ = timeCommonArm(IL, EL, mode, io, includes, excludes, &lists, @truncate(a));
        var samples: [7]u64 = undefined;
        for (0..7) |r| samples[r] = timeCommonArm(IL, EL, mode, io, includes, excludes, &lists, @truncate(a));
        ms[a] = @as(f64, @floatFromInt(medianNs(&samples))) / 1_000_000.0;
    }
    const choice_flat = It.predictsFlat(includes, excludes);
    bit_tree.last_predict_used_flat = choice_flat;
    const faster_flat = ms[0] < ms[1];
    const lo = if (faster_flat) ms[0] else ms[1];
    const margin = if (lo > 0) @abs(ms[0] - ms[1]) / lo else 0;
    const tied = margin < 0.25;
    return .{
        .name = name,
        .ms = ms,
        .choice_flat = choice_flat,
        .faster_flat = faster_flat,
        .tied = tied,
        .ok = tied or choice_flat == faster_flat,
    };
}

/// Prints one compact flat/tree/auto table without history columns.
/// - `results` - per-scenario medians with choices.
fn printCommonTable(results: []const CommonBenchResult) void {
    std.debug.print("common flat/tree/auto (ms/rep median, match: ok, MISS, - tie)\n", .{});
    std.debug.print("{s:<24} {s:>10} {s:>10} {s:>10} {s:>6} {s:>5}\n", .{ "scenario", "flat", "tree", "auto", "choice", "match" });
    for (results) |r| {
        const mark: []const u8 = if (r.tied) "-" else if (r.ok) "ok" else "MISS";
        std.debug.print("{s:<24} {d:>10.3} {d:>10.3} {d:>10.3} {s:>6} {s:>5}\n", .{
            r.name,
            r.ms[0],
            r.ms[1],
            r.ms[2],
            if (r.choice_flat) "flat" else "tree",
            mark,
        });
    }
}

/// Xorshift generator for deterministic common scenario fills.
/// - `state` - evolving generator state.
///
/// Return - next pseudorandom word halves.
fn commonBenchRandU32(state: *u64) u32 {
    var x = state.*;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state.* = x;
    return @truncate((x *% 0x2545F4914F6CDD1D) >> 32);
}

/// Resizes every tree inactive, first group active instead.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillUniA_UniI(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    for (trees, 0..) |tree, k| {
        try tree.resize(alloc, n, if (k < inc_len) .active else .inactive);
    }
}

/// Resizes every tree inactive, trailing group active instead.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillUniI_UniA(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    for (trees, 0..) |tree, k| {
        try tree.resize(alloc, n, if (k < inc_len) .inactive else .active);
    }
}

/// Resizes every tree active for inactive-heavy queries.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count, unused.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillUniA_All(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    _ = inc_len;
    for (trees) |tree| try tree.resize(alloc, n, .active);
}

/// Writes one strided active pattern into every tree.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count, unused.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillStride997(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    _ = inc_len;
    for (trees) |tree| {
        try tree.resize(alloc, n, .inactive);
        var b: u32 = 0;
        while (b < n) : (b += 997) tree.setBit(b, .active);
    }
}

/// Writes a single active bit into the first tree only.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count, unused.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillSingleBit(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    _ = inc_len;
    for (trees) |tree| try tree.resize(alloc, n, .inactive);
    trees[0].setBit(0, .active);
}

/// Writes alternating random, cleared and filled words into every tree.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count, unused.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillAlternation(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    _ = inc_len;
    var rng: u64 = 0x9E3779B97F4A7C15;
    for (trees) |tree| try tree.resize(alloc, n, .inactive);
    const words = n >> 6;
    var w: u32 = 0;
    while (w < words) : (w += 1) {
        for (trees) |tree| {
            const hi = commonBenchRandU32(&rng);
            const lo = commonBenchRandU32(&rng);
            const pick = commonBenchRandU32(&rng);
            const word: u64 = if ((w & 1) == 0)
                (@as(u64, hi) << 32) | lo
            else if ((pick & 1) == 1)
                0
            else
                std.math.maxInt(u64);
            tree.setWord(w, word, std.math.maxInt(u64));
        }
    }
}

/// Writes checkerboard words into every tree for deep summaries.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count, unused.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillChecker(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    _ = inc_len;
    for (trees) |tree| try tree.resize(alloc, n, .inactive);
    const words = n >> 6;
    var w: u32 = 0;
    while (w < words) : (w += 1) {
        for (trees) |tree| tree.setWord(w, 0xAAAAAAAAAAAAAAAA, std.math.maxInt(u64));
    }
}

/// Fills the first half of every tree, leaves the second cleared.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count, unused.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillHalf(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    _ = inc_len;
    for (trees) |tree| try tree.resize(alloc, n, .inactive);
    const words = n >> 6;
    var w: u32 = 0;
    while (w < words / 2) : (w += 1) {
        for (trees) |tree| tree.setWord(w, std.math.maxInt(u64), std.math.maxInt(u64));
    }
}

/// Writes disjoint active blocks per include, excludes cleared.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillDisjoint(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    for (trees) |tree| try tree.resize(alloc, n, .inactive);
    const lane: u32 = 2000;
    var k: u32 = 0;
    while (k < inc_len) : (k += 1) {
        var b: u32 = k * lane;
        const end: u32 = @min(b + lane, n);
        while (b < end) : (b += 1) trees[k].setBit(b, .active);
    }
}

/// Writes alternating inactive-friendly lanes, even bits suit excludes.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillCheckerInactive(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    for (trees, 0..) |tree, k| {
        try tree.resize(alloc, n, .inactive);
        const words = n >> 6;
        var w: u32 = 0;
        while (w < words) : (w += 1) {
            const word: u64 = if (k < inc_len) 0xAAAAAAAAAAAAAAAA else 0x5555555555555555;
            tree.setWord(w, word, std.math.maxInt(u64));
        }
    }
}

/// Writes full includes with strided holes cleared in excludes.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillHoles(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    for (trees, 0..) |tree, k| {
        try tree.resize(alloc, n, .active);
        if (k >= inc_len) {
            var b: u32 = 0;
            while (b < n) : (b += 997) tree.setBit(b, .inactive);
        }
    }
}

/// Writes strided includes with fully set excludes vetoing everything.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillVetoSparse(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    for (trees, 0..) |tree, k| {
        try tree.resize(alloc, n, .inactive);
        if (k < inc_len) {
            var b: u32 = 0;
            while (b < n) : (b += 997) tree.setBit(b, .active);
        } else {
            var b: u32 = 0;
            while (b < n) : (b += 1) tree.setBit(b, .active);
        }
    }
}

/// Writes random includes with fully set excludes vetoing everything.
/// - `trees` - scenario trees, includes first.
/// - `inc_len` - leading include count.
/// - `alloc` - owns backing storage.
/// - `n` - shared valid bits.
fn fillVeto(trees: []*Tree, inc_len: u32, alloc: Allocator, n: u32) !void {
    var rng: u64 = 0x123456789ABCDEF;
    for (trees, 0..) |tree, k| {
        if (k < inc_len) {
            try tree.resize(alloc, n, .inactive);
            const words = n >> 6;
            var w: u32 = 0;
            while (w < words) : (w += 1) {
                const word: u64 = (@as(u64, commonBenchRandU32(&rng)) << 32) | commonBenchRandU32(&rng);
                tree.setWord(w, word, std.math.maxInt(u64));
            }
        } else {
            try tree.resize(alloc, n, .active);
        }
    }
}

/// Builds, fills and benchmarks one common scenario end to end.
/// - `io` - clock source for timing.
/// - `alloc` - owns trees.
/// - `name` - scenario label, static storage.
/// - `n` - shared valid bits.
/// - `mode` - installed callback subset.
/// - `fill` - scenario word distribution.
///
/// Return - medians with the auto choice and match flag.
fn benchCommonBuilt(
    comptime IL: u32,
    comptime EL: u32,
    comptime mode: CommonMode,
    io: std.Io,
    alloc: Allocator,
    name: []const u8,
    n: u32,
    fill: fn ([]*Tree, u32, Allocator, u32) anyerror!void,
) !CommonBenchResult {
    var trees: [IL + EL]Tree = [_]Tree{.{}} ** (IL + EL);
    defer {
        for (0..IL + EL) |k| trees[k].deinit(alloc);
    }
    var ptrs: [IL + EL]*Tree = undefined;
    for (0..IL + EL) |k| ptrs[k] = &trees[k];
    const slice: []*Tree = &ptrs;
    try fill(slice, IL, alloc, n);
    var inc: [IL]*Tree = undefined;
    var exc: [EL]*Tree = undefined;
    for (0..IL) |k| inc[k] = &trees[k];
    for (0..EL) |k| exc[k] = &trees[IL + k];
    return benchCommonMode(IL, EL, mode, io, alloc, name, inc, exc, n);
}

/// Past timings for one scenario kept oldest-first for trends.
const HistEntry = struct {
    /// Scenario label matching current row names.
    name: []const u8,
    /// Previous ms/rep values, capped to recent runs.
    vals: std.ArrayListUnmanaged(f64),
};

/// Finds history for one scenario by name.
/// - `hists` - loaded history entries.
/// - `name` - scenario label to locate.
///
/// Return - index when found, null otherwise.
fn findHist(hists: []const HistEntry, name: []const u8) ?usize {
    for (hists, 0..) |h, i| {
        if (std.mem.eql(u8, h.name, name)) return i;
    }
    return null;
}

/// Loads prior timings, ignoring missing or non-v2 history files.
/// - `io` - file-system context for reading.
/// - `alloc` - owns history names and values.
///
/// Return - history entries, empty on fresh start.
fn loadHistory(io: std.Io, alloc: Allocator) std.ArrayListUnmanaged(HistEntry) {
    var hists: std.ArrayListUnmanaged(HistEntry) = .empty;
    const content = std.Io.Dir.cwd().readFileAlloc(io, HISTORY_PATH, alloc, .limited(4 * 1024 * 1024)) catch |err| {
        if (err != error.FileNotFound) {
            std.debug.print("warn: cannot read {s}: {t} (history ignored)\n", .{ HISTORY_PATH, err });
        }
        return hists;
    };
    defer alloc.free(content);
    if (!std.mem.containsAtLeast(u8, content, 1, "# bench_bit_tree history v2")) {
        std.debug.print("note: {s} is not v2 history, starting fresh\n", .{HISTORY_PATH});
        return hists;
    }
    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        var toks = std.mem.tokenizeAny(u8, line, " \t\r");
        const name_tok = toks.next() orelse continue;
        var e = HistEntry{ .name = alloc.dupe(u8, name_tok) catch continue, .vals = .empty };
        var ok = true;
        while (toks.next()) |tok| {
            const v = std.fmt.parseFloat(f64, tok) catch continue;
            e.vals.append(alloc, v) catch {
                ok = false;
                break;
            };
        }
        if (!ok) {
            alloc.free(e.name);
            e.vals.deinit(alloc);
            continue;
        }
        hists.append(alloc, e) catch {
            alloc.free(e.name);
            e.vals.deinit(alloc);
            break;
        };
    }
    return hists;
}

/// Prints current timings aligned with up to eight history columns.
/// - `rows` - current benchmark results.
/// - `hists` - prior timings for comparison.
fn printTable(rows: []const Row, hists: []const HistEntry) void {
    var prev_cols: usize = 0;
    for (rows) |row| {
        const n: usize = if (findHist(hists, row.name)) |idx| hists[idx].vals.items.len else 0;
        prev_cols = @max(prev_cols, @min(n, MAX_PREV_COLS));
    }
    std.debug.print("tree bench (ms/rep), history: {s} (prev runs kept: up to {d})\n", .{ HISTORY_PATH, MAX_PREV_COLS });
    std.debug.print("{s:<26} {s:>10} {s:>6} ", .{ "pattern", "elements", "path" });
    var i: usize = 0;
    while (i < prev_cols) : (i += 1) {
        var lb: [8]u8 = undefined;
        const l = std.fmt.bufPrint(&lb, "p-{d}", .{prev_cols - i}) catch "?";
        std.debug.print("{s:>10} ", .{l});
    }
    std.debug.print("{s:>10} {s:>8}\n", .{ "cur(ms)", "xPrev" });
    std.debug.print("{s:<26} {s:>10} ", .{ "--------------------------", "----------" });
    i = 0;
    while (i < prev_cols) : (i += 1) {
        std.debug.print("{s:>10} ", .{"----------"});
    }
    std.debug.print("{s:>10} {s:>8}\n", .{ "----------", "--------" });
    for (rows) |row| {
        const all: []const f64 = if (findHist(hists, row.name)) |idx| hists[idx].vals.items else &.{};
        const start: usize = if (all.len > prev_cols) all.len - prev_cols else 0;
        const shown = all[start..];
        std.debug.print("{s:<26} {d:>10} {s:>6} ", .{ row.name, row.elements, if (row.flat) |f| (if (f) "flat" else "tree") else "-" });
        var pad: usize = 0;
        while (pad < prev_cols - shown.len) : (pad += 1) {
            std.debug.print("{s:>10} ", .{"---"});
        }
        for (shown) |v| {
            std.debug.print("{d:>10.3} ", .{v});
        }
        std.debug.print("{d:>10.3} ", .{row.ms});
        if (all.len > 0) {
            const prev = all[all.len - 1];
            if (prev > 0 and row.ms > 0) {
                std.debug.print("{d:>7.2}x\n", .{prev / row.ms});
            } else {
                std.debug.print("{s:>8}\n", .{"---"});
            }
        } else {
            std.debug.print("{s:>8}\n", .{"---"});
        }
    }
}

/// Merges current timings into history and persists the capped file.
/// - `io` - file-system context for writing.
/// - `alloc` - owns the serialized buffer.
/// - `rows` - current benchmark results.
/// - `hists` - prior timings to preserve.
///
/// Return - error on allocation or write failure.
fn saveHistory(io: std.Io, alloc: Allocator, rows: []const Row, hists: []const HistEntry) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try buf.print(alloc, "# bench_bit_tree history v2: <name> <ms/rep oldest-first, up to {d}>\n", .{MAX_PREV_COLS});
    for (rows) |row| {
        try buf.print(alloc, "{s}", .{row.name});
        if (findHist(hists, row.name)) |idx| {
            const old = hists[idx].vals.items;
            const keep_from: usize = if (old.len + 1 > MAX_PREV_COLS) old.len + 1 - MAX_PREV_COLS else 0;
            for (old[keep_from..]) |v| {
                try buf.print(alloc, " {d:.6}", .{v});
            }
        }
        try buf.print(alloc, " {d:.6}\n", .{row.ms});
    }
    for (hists) |h| {
        var seen = false;
        for (rows) |row| {
            if (std.mem.eql(u8, h.name, row.name)) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        try buf.print(alloc, "{s}", .{h.name});
        for (h.vals.items) |v| {
            try buf.print(alloc, " {d:.6}", .{v});
        }
        try buf.print(alloc, "\n", .{});
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = HISTORY_PATH, .data = buf.items });
}

/// Builds shared scenarios, runs all benchmarks and updates history.
/// - `init` - process context providing I/O and arguments.
///
/// Return - error on allocation, verification or I/O failure.
pub fn main(init: std.process.Init) !void {
    const io: std.Io = init.io;
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc: Allocator = gpa.allocator();

    var arg_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, alloc);
    defer arg_it.deinit();
    _ = arg_it.next();
    const only: []const u8 = arg_it.next() orelse "all";
    const run_classic = !std.mem.eql(u8, only, "common");
    const run_common = !std.mem.eql(u8, only, "classic");

    var rows: std.ArrayListUnmanaged(Row) = .empty;
    defer {
        for (rows.items) |row| alloc.free(row.name);
        rows.deinit(alloc);
    }

    if (run_classic) {
        var sparse = Oracle{};
        defer sparse.deinit(alloc);
        try sparse.resize(alloc, 1_000_000, .inactive);
        fillStride(&sparse, 997, 1);

        var dense = Oracle{};
        defer dense.deinit(alloc);
        try dense.resize(alloc, 100_000, .active);

        var strided = Oracle{};
        defer strided.deinit(alloc);
        try strided.resize(alloc, 200_000, .inactive);
        fillStride(&strided, 3, 0);

        var ultra = Oracle{};
        defer ultra.deinit(alloc);
        try ultra.resize(alloc, 10_000_000, .inactive);
        fillStride(&ultra, 100_003, 7);

        var sparse_tree = Tree{};
        defer sparse_tree.deinit(alloc);
        try sparse_tree.resize(alloc, 1_000_000, .inactive);
        fillTreeStride(&sparse_tree, 997, 1);

        var dense_tree = Tree{};
        defer dense_tree.deinit(alloc);
        try dense_tree.resize(alloc, 100_000, .active);

        var strided_tree = Tree{};
        defer strided_tree.deinit(alloc);
        try strided_tree.resize(alloc, 200_000, .inactive);
        fillTreeStride(&strided_tree, 3, 0);

        var ultra_tree = Tree{};
        defer ultra_tree.deinit(alloc);
        try ultra_tree.resize(alloc, 10_000_000, .inactive);
        fillTreeStride(&ultra_tree, 100_003, 7);

        try benchOne(io, alloc, &rows, "sparse1M", &sparse, &sparse_tree, .active, 500);
        try benchOne(io, alloc, &rows, "sparse1M-inactive", &sparse, &sparse_tree, .inactive, 5);
        try benchOne(io, alloc, &rows, "dense100k", &dense, &dense_tree, .active, 20);
        try benchOne(io, alloc, &rows, "every3rd200k", &strided, &strided_tree, .active, 20);
        try benchOne(io, alloc, &rows, "ultra10M", &ultra, &ultra_tree, .active, 200);

        const strides = [_]u32{ 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536 };
        for ([_]u32{1_000_000}) |total| {
            for (strides) |stride| {
                try benchThreshold(io, alloc, &rows, total, stride, 100);
            }
        }
        for (strides) |stride| {
            try benchThreshold(io, alloc, &rows, 10_000_000, stride, 5);
        }

        try benchCluster(io, alloc, &rows, 2_000_000, 1000, 1000, 20);
        try benchCluster(io, alloc, &rows, 2_000_000, 1000, 10000, 20);
        try benchCluster(io, alloc, &rows, 500_000, 50, 50, 50);
        try benchCluster(io, alloc, &rows, 500_000, 50, 500, 50);

        try benchFill(io, alloc, &rows, 1_000_000, 1, 0, 3);
        try benchFill(io, alloc, &rows, 1_000_000, 997, 1, 5);
    }

    if (run_common) {
        var common_results: std.ArrayListUnmanaged(CommonBenchResult) = .empty;
        defer common_results.deinit(alloc);
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "uni-A-262k", 262144, fillUniA_UniI));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .inactive_only, io, alloc, "uni-I-262k", 262144, fillUniI_UniA));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "uni-I-Aonly-262k", 262144, fillUniI_UniA));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "sparse997-262k", 262144, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "single-bit-262k", 262144, fillSingleBit));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "alt-262k", 262144, fillAlternation));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .both, io, alloc, "alt-both-262k", 262144, fillAlternation));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "checker-262k", 262144, fillChecker));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "half-262k", 262144, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "veto-262k", 262144, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "veto-sparse-262k", 262144, fillVetoSparse));
        try common_results.append(alloc, try benchCommonBuilt(1, 1, .active_only, io, alloc, "holes-262k", 262144, fillHoles));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .inactive_only, io, alloc, "inact-heavy-262k", 262144, fillUniA_All));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "small130-alt", 130, fillAlternation));
        try common_results.append(alloc, try benchCommonBuilt(3, 3, .active_only, io, alloc, "three-three-alt-262k", 262144, fillAlternation));
        try common_results.append(alloc, try benchCommonBuilt(1, 0, .active_only, io, alloc, "one-zero-sparse-262k", 262144, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(1, 0, .active_only, io, alloc, "1x0-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(0, 1, .active_only, io, alloc, "0x1-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(1, 1, .active_only, io, alloc, "1x1-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(1, 5, .active_only, io, alloc, "1x5-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(5, 1, .active_only, io, alloc, "5x1-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(10, 0, .active_only, io, alloc, "10x0-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(0, 10, .active_only, io, alloc, "0x10-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(3, 7, .active_only, io, alloc, "3x7-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(10, 10, .active_only, io, alloc, "10x10-half-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(1, 0, .active_only, io, alloc, "1x0-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(0, 1, .active_only, io, alloc, "0x1-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(1, 1, .active_only, io, alloc, "1x1-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(1, 5, .active_only, io, alloc, "1x5-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(5, 1, .active_only, io, alloc, "5x1-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(10, 0, .active_only, io, alloc, "10x0-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(0, 10, .active_only, io, alloc, "0x10-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(3, 7, .active_only, io, alloc, "3x7-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(10, 10, .active_only, io, alloc, "10x10-veto-100k", 100000, fillVeto));
        try common_results.append(alloc, try benchCommonBuilt(1, 0, .active_only, io, alloc, "1x0-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(0, 1, .active_only, io, alloc, "0x1-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(1, 1, .active_only, io, alloc, "1x1-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(1, 5, .active_only, io, alloc, "1x5-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(5, 1, .active_only, io, alloc, "5x1-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(10, 0, .active_only, io, alloc, "10x0-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(0, 10, .active_only, io, alloc, "0x10-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(3, 7, .active_only, io, alloc, "3x7-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(10, 10, .active_only, io, alloc, "10x10-sparse-100k", 100000, fillStride997));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-checker-100k", 100000, fillChecker));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-checker-100k", 100000, fillChecker));
        try common_results.append(alloc, try benchCommonBuilt(10, 10, .active_only, io, alloc, "10x10-checker-100k", 100000, fillChecker));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-disjoint-100k", 100000, fillDisjoint));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-disjoint-100k", 100000, fillDisjoint));
        try common_results.append(alloc, try benchCommonBuilt(3, 7, .active_only, io, alloc, "3x7-disjoint-100k", 100000, fillDisjoint));
        try common_results.append(alloc, try benchCommonBuilt(10, 10, .active_only, io, alloc, "10x10-disjoint-100k", 100000, fillDisjoint));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-uni-A-100k", 100000, fillUniA_UniI));
        try common_results.append(alloc, try benchCommonBuilt(10, 10, .active_only, io, alloc, "10x10-uni-A-100k", 100000, fillUniA_UniI));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-single-100k", 100000, fillSingleBit));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-single-100k", 100000, fillSingleBit));
        try common_results.append(alloc, try benchCommonBuilt(1, 1, .active_only, io, alloc, "1x1-holes-100k", 100000, fillHoles));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .active_only, io, alloc, "2x2-holes-100k", 100000, fillHoles));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .active_only, io, alloc, "5x5-holes-100k", 100000, fillHoles));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .inactive_only, io, alloc, "2x2-half-I-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(5, 5, .inactive_only, io, alloc, "5x5-half-I-100k", 100000, fillHalf));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .inactive_only, io, alloc, "2x2-checker-I-100k", 100000, fillCheckerInactive));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .both, io, alloc, "2x2-alt-both-100k", 100000, fillAlternation));
        try common_results.append(alloc, try benchCommonBuilt(2, 2, .both, io, alloc, "2x2-half-both-100k", 100000, fillHalf));
        printCommonTable(common_results.items);
    }

    if (run_classic) {
        var hists = loadHistory(io, alloc);
        defer {
            for (hists.items) |*h| {
                alloc.free(h.name);
                h.vals.deinit(alloc);
            }
            hists.deinit(alloc);
        }
        printTable(rows.items, hists.items);
        saveHistory(io, alloc, rows.items, hists.items) catch |err| {
            std.debug.print("warn: cannot write {s}: {t}\n", .{ HISTORY_PATH, err });
        };
    }
}
