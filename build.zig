const std = @import("std");

/// Configures library modules, unit tests and the tree benchmark.
/// - `b` - build graph that receives all steps and artifacts.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("bit_tree", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    _ = mod;

    const test_step = b.step("test", "Run unit tests");
    for ([_][]const u8{
        "src/bitset.zig",
        "src/layer.zig",
        "src/bit_word.zig",
        "src/bit_tree.zig",
    }) |src| {
        const unit_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(src),
                .target = target,
                .optimize = optimize,
            }),
        });
        test_step.dependOn(&b.addRunArtifact(unit_tests).step);
    }

    const bench_bit_tree_exe = b.addExecutable(.{
        .name = "bench_bit_tree",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_bit_tree.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_bench_bit_tree = b.addRunArtifact(bench_bit_tree_exe);
    if (b.args) |args| run_bench_bit_tree.addArgs(args);

    const bench_bit_tree_step = b.step("bench_bit_tree", "Run tree benchmarks with history");
    bench_bit_tree_step.dependOn(&run_bench_bit_tree.step);
}
