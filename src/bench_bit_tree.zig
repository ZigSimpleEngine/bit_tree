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

    var rows: std.ArrayListUnmanaged(Row) = .empty;
    defer {
        for (rows.items) |row| alloc.free(row.name);
        rows.deinit(alloc);
    }

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
