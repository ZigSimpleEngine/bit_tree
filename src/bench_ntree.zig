/// ntree-only пробег сценариев: замеряется только NewTree
/// (new_bit_tree.zig). BitSet (bit_set.zig) строится рядом лишь как
/// независимый оракул: сверяем только результат (множество id +
/// суммы/счётчики, порядок обхода и код возврата step не проверяются),
/// чтобы ловить расхождения.
/// fill-строки меряют fill именно ntree.
///
/// История: после каждого прогона ms/ряд дописывается в текстовый файл
/// bench_ntree_history.txt (формат v2: `#`-комментарии, далее строки
/// `<name> <ms_prev...>`, ms на rep, 6 знаков, oldest-first, хранится
/// максимум 8 последних; файл без v2-шапки игнорируется — чистый старт). При следующем прогоне таблица показывает
/// до 8 предыдущих колонок + текущую + xPrev = prev_last / cur
/// (>1 = стало быстрее).
/// zig build bench_ntree -Doptimize=ReleaseFast
const std = @import("std");
const bit_set = @import("bit_set.zig");
const utilities = @import("utilities.zig");
const new_bit_tree = @import("new_bit_tree.zig");

const Allocator = std.mem.Allocator;
const Oracle = bit_set.BitSet(.u64);
const BitState = utilities.BitState;
const NewTree = new_bit_tree.BitTree(.u64);

const HISTORY_PATH = "bench_ntree_history.txt";
const MAX_PREV_COLS: usize = 8;

const Stopwatch = struct {
    io: std.Io,
    t0: std.Io.Timestamp,
    fn start(io: std.Io) @This() {
        return .{ .io = io, .t0 = std.Io.Timestamp.now(io, .awake) };
    }
    fn read(self: *@This()) u64 {
        const t1 = std.Io.Timestamp.now(self.io, .awake);
        return @intCast(std.Io.Timestamp.durationTo(self.t0, t1).nanoseconds);
    }
};

fn fillStride(oracle: *Oracle, stride: u32, offset: u32) void {
    var b: u32 = offset;
    while (b < oracle.bits_count) : (b += stride) {
        oracle.setBit(b, .active);
    }
}

fn fillNewStride(tree: *NewTree, stride: u32, offset: u32) void {
    var b: u32 = offset;
    while (b < tree.bitset.bits_count) : (b += stride) {
        tree.setBit(b, .active);
    }
}

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

fn fillNewClustered(tree: *NewTree, run: u32, gap: u32) void {
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

/// Короткие числа в именах тестов, чтобы не ломать layout колонок:
/// 1000->1K, 1000000->1M; неделимые на 1000 — как есть (1024, 997).
fn fmtCount(buf: []u8, n: u32) ![]u8 {
    if (n >= 1_000_000 and n % 1_000_000 == 0) {
        return std.fmt.bufPrint(buf, "{d}M", .{n / 1_000_000});
    }
    if (n >= 1_000 and n % 1_000 == 0) {
        return std.fmt.bufPrint(buf, "{d}K", .{n / 1_000});
    }
    return std.fmt.bufPrint(buf, "{d}", .{n});
}

/// Нетаймированный однопроходный оракул по BitSet: только для сверки сумм.
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

const SumCtx = struct {
    sum: u64 = 0,
    count: u64 = 0,
    // Каждый id — через opaque use: bulk-путь step отдаёт id плотным
    // range-циклом с прозрачным колбэком, и LLVM сворачивает
    // sum += id в O(1)-арифметику (корректные суммы за ~15нс вместо
    // обхода). Пин каждого id делает свёртку невозможной: цикл обязан
    // материализовать все id. См. verify: там honesty дают heap-записи.
    inline fn addInline(self: *SumCtx, id: u32) bool {
        std.mem.doNotOptimizeAway(id);
        self.sum += id;
        self.count += 1;
        return true;
    }
};

/// Замеряемый leg: NewTree Iterator стримит id в суммирующий колбэк.
/// Решение predictsFlat хоистится один раз на сценарий (дерево во время
/// замера неменяемо), ветка — ВНЕ rep-цикла: решение внутри hot-функции
/// рядом с word loop душит оптимизацию за opaque use (см. NOTE в new_bit_tree).
fn benchNtree(io: std.Io, tree: *NewTree, comptime want: BitState, reps: u32) struct { sum: u64, count: u64, ns: u64 } {
    const It = NewTree.Iterator(*SumCtx, SumCtx.addInline, null);
    const ItI = NewTree.Iterator(*SumCtx, null, SumCtx.addInline);
    const use_flat = if (want == .active) It.predictsFlat(tree) else ItI.predictsFlat(tree);
    new_bit_tree.last_predict_used_flat = use_flat;
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
    // Пиним дата-зависимые итоги, а не только bool: иначе LLVM доказывает
    // done=true (колбэк всегда true) и выкидывает весь цикл в ReleaseFast,
    // а суммы уходят лишь в выключенный assert. Итоги — функция от слов
    // в памяти, их материализация требует реального обхода.
    std.mem.doNotOptimizeAway(sum);
    std.mem.doNotOptimizeAway(count);
    return .{ .sum = sum, .count = count, .ns = watch.read() };
}

const Row = struct {
    name: []const u8,
    elements: u64,
    ms: f64,
    flat: ?bool = null,
};

const VerifyCtx = struct {
    list: *std.ArrayListUnmanaged(u32),
    fn push(ctx: *VerifyCtx, id: u32) callconv(.@"inline") bool {
        ctx.list.appendAssumeCapacity(id);
        return true;
    }
};

/// Разовая нетаймированная проверка результата сценария: id, отданные
/// итератором, как множество (порядок обхода не важен) + суммы/счётчики —
/// против независимого оракула из BitSet. Код возврата step не проверяется.
fn verifyNtree(
    alloc: Allocator,
    oracle: *const Oracle,
    ntree: *NewTree,
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
        const It = NewTree.Iterator(*VerifyCtx, VerifyCtx.push, null);
        const ItI = NewTree.Iterator(*VerifyCtx, null, VerifyCtx.push);
        if (want == .active) {
            _ = It.iterateAll(.{ .tree = ntree, .context = &vc });
        } else {
            _ = ItI.iterateAll(.{ .tree = ntree, .context = &vc });
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

/// Один сценарий на идентичном oracle/ntree содержимом: сначала разовая
/// сверка результата (id + суммы, без привязки к step), затем таймится
/// только ntree. Результат (ms на rep) кладётся в rows, печать — общей
/// таблицей после прогона.
fn benchOne(
    io: std.Io,
    alloc: Allocator,
    rows: *std.ArrayListUnmanaged(Row),
    name: []const u8,
    oracle: *const Oracle,
    ntree: *NewTree,
    comptime want: BitState,
    reps: u32,
) !void {
    try verifyNtree(alloc, oracle, ntree, want, name);
    {
        var c = SumCtx{};
        const It = NewTree.Iterator(*SumCtx, SumCtx.addInline, null);
        const ItI = NewTree.Iterator(*SumCtx, null, SumCtx.addInline);
        const done = if (want == .active)
            It.iterateAll(.{ .tree = ntree, .context = &c })
        else
            ItI.iterateAll(.{ .tree = ntree, .context = &c });
        std.mem.doNotOptimizeAway(done);
    }
    const exp = oracleSumOnce(oracle, want);
    const got = benchNtree(io, ntree, want, reps);
    std.debug.assert(exp.sum * reps == got.sum and exp.count * reps == got.count);
    const ms: f64 = @as(f64, @floatFromInt(got.ns)) / @as(f64, @floatFromInt(reps)) / 1_000_000.0;
    try rows.append(alloc, .{ .name = try alloc.dupe(u8, name), .elements = exp.count, .ms = ms, .flat = new_bit_tree.last_predict_used_flat });
}

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
    var ntree = NewTree{};
    defer ntree.deinit(alloc);
    try oracle.resize(alloc, bits_total, .inactive);
    try ntree.resize(alloc, bits_total, .inactive);
    fillStride(&oracle, stride, 0);
    fillNewStride(&ntree, stride, 0);
    var name_buf: [64]u8 = undefined;
    var sbuf: [16]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "gap{s}-N{s}", .{
        try fmtCount(&sbuf, stride),
        try fmtCount(&nbuf, bits_total),
    });
    try benchOne(io, alloc, rows, name, &oracle, &ntree, .active, reps);
}

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
    var ntree = NewTree{};
    defer ntree.deinit(alloc);
    try oracle.resize(alloc, bits_total, .inactive);
    try ntree.resize(alloc, bits_total, .inactive);
    fillClustered(&oracle, run, gap, 0);
    fillNewClustered(&ntree, run, gap);
    var name_buf: [64]u8 = undefined;
    var rbuf: [16]u8 = undefined;
    var gbuf: [16]u8 = undefined;
    var nbuf: [16]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "cluster-r{s}-g{s}-N{s}", .{
        try fmtCount(&rbuf, run),
        try fmtCount(&gbuf, gap),
        try fmtCount(&nbuf, bits_total),
    });
    try benchOne(io, alloc, rows, name, &oracle, &ntree, .active, reps);
}

/// Write path ntree: свежее дерево + per-bit setBit. Средний fill и
/// число элементов — строкой в общую таблицу (та же ms/ряд семантика).
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
        var tree = NewTree{};
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

const HistEntry = struct {
    name: []const u8,
    vals: std.ArrayListUnmanaged(f64),
};

fn findHist(hists: []const HistEntry, name: []const u8) ?usize {
    for (hists, 0..) |h, i| {
        if (std.mem.eql(u8, h.name, name)) return i;
    }
    return null;
}

fn loadHistory(io: std.Io, alloc: Allocator) std.ArrayListUnmanaged(HistEntry) {
    var hists: std.ArrayListUnmanaged(HistEntry) = .empty;
    const content = std.Io.Dir.cwd().readFileAlloc(io, HISTORY_PATH, alloc, .limited(4 * 1024 * 1024)) catch |err| {
        if (err != error.FileNotFound) {
            std.debug.print("warn: cannot read {s}: {t} (history ignored)\n", .{ HISTORY_PATH, err });
        }
        return hists;
    };
    defer alloc.free(content);
    // v1 недействительна: там смесь чисел от старой реализации step и
    // свёрнутых (~нс) замеров до per-id пиннинга. Без v2-шапки — чистый старт.
    if (!std.mem.containsAtLeast(u8, content, 1, "# bench_ntree history v2")) {
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

fn printTable(rows: []const Row, hists: []const HistEntry) void {
    var prev_cols: usize = 0;
    for (rows) |row| {
        const n: usize = if (findHist(hists, row.name)) |idx| hists[idx].vals.items.len else 0;
        prev_cols = @max(prev_cols, @min(n, MAX_PREV_COLS));
    }
    std.debug.print("ntree bench (ms/rep), history: {s} (prev runs kept: up to {d})\n", .{ HISTORY_PATH, MAX_PREV_COLS });
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

fn saveHistory(io: std.Io, alloc: Allocator, rows: []const Row, hists: []const HistEntry) !void {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try buf.print(alloc, "# bench_ntree history v2: <name> <ms/rep oldest-first, up to {d}>\n", .{MAX_PREV_COLS});
    for (rows) |row| {
        try buf.print(alloc, "{s}", .{row.name});
        if (findHist(hists, row.name)) |idx| {
            const old = hists[idx].vals.items;
            // old ++ [cur], храним последние MAX_PREV_COLS.
            const keep_from: usize = if (old.len + 1 > MAX_PREV_COLS) old.len + 1 - MAX_PREV_COLS else 0;
            for (old[keep_from..]) |v| {
                try buf.print(alloc, " {d:.6}", .{v});
            }
        }
        try buf.print(alloc, " {d:.6}\n", .{row.ms});
    }
    // Строки из истории, которых не было в текущем прогоне, не теряем.
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

    var sparse_ntree = NewTree{};
    defer sparse_ntree.deinit(alloc);
    try sparse_ntree.resize(alloc, 1_000_000, .inactive);
    fillNewStride(&sparse_ntree, 997, 1);

    var dense_ntree = NewTree{};
    defer dense_ntree.deinit(alloc);
    try dense_ntree.resize(alloc, 100_000, .active);

    var strided_ntree = NewTree{};
    defer strided_ntree.deinit(alloc);
    try strided_ntree.resize(alloc, 200_000, .inactive);
    fillNewStride(&strided_ntree, 3, 0);

    var ultra_ntree = NewTree{};
    defer ultra_ntree.deinit(alloc);
    try ultra_ntree.resize(alloc, 10_000_000, .inactive);
    fillNewStride(&ultra_ntree, 100_003, 7);

    // Те же сценарии и reps, что раньше (без arith).
    try benchOne(io, alloc, &rows, "sparse1M", &sparse, &sparse_ntree, .active, 500);
    try benchOne(io, alloc, &rows, "sparse1M-inactive", &sparse, &sparse_ntree, .inactive, 5);
    try benchOne(io, alloc, &rows, "dense100k", &dense, &dense_ntree, .active, 20);
    try benchOne(io, alloc, &rows, "every3rd200k", &strided, &strided_ntree, .active, 20);
    try benchOne(io, alloc, &rows, "ultra10M", &ultra, &ultra_ntree, .active, 200);

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
