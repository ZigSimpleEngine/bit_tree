const std = @import("std");

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
        "src/bit_set.zig",
        "src/layer.zig",
        "src/bit_word.zig",
        "src/new_bit_tree.zig",
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

    const bench_ntree_exe = b.addExecutable(.{
        .name = "bench_ntree",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_ntree.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_bench_ntree = b.addRunArtifact(bench_ntree_exe);
    if (b.args) |args| run_bench_ntree.addArgs(args);

    const bench_ntree_step = b.step("bench_ntree", "Run ntree benchmarks with history");
    bench_ntree_step.dependOn(&run_bench_ntree.step);
}
