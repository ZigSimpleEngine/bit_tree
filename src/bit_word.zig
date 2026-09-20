const std = @import("std");
const math = std.math;
const utilities = @import("utilities.zig");
const BitState = utilities.BitState;

/// Backing integer width shared by bitsets, layers and trees.
pub const WordType = enum {
    /// Eight-bit words, smallest memory footprint for tiny sets.
    u8,
    /// Sixteen-bit words, compact storage for small cardinalities.
    u16,
    /// Thirty-two-bit words, balanced width for mid-size sets.
    u32,
    /// Sixty-four-bit words, fastest scans on 64-bit targets.
    u64,

    /// Resolves the enum tag into its concrete unsigned integer type.
    /// - `wt` - word width tag to convert.
    ///
    /// Return - matching u8/u16/u32/u64 type.
    pub fn Type(comptime wt: WordType) type {
        return switch (wt) {
            .u8 => u8,
            .u16 => u16,
            .u32 => u32,
            .u64 => u64,
        };
    }
};

/// Compile-time bit arithmetic kit for one word width.
/// - `wt` - word width the generated helpers operate on.
///
/// Return - struct with shifts, masks and merge helpers.
pub fn BitWord(comptime wt: WordType) type {
    return struct {
        /// Echoes the configured width so generic code can branch on it.
        pub const word_type: WordType = wt;
        /// Concrete unsigned integer used for backing storage.
        pub const Word: type = wt.Type();
        /// All-ones word, base for building inverted masks.
        pub const max_value: Word = math.maxInt(Word);
        /// Bits per word, drives all id conversions.
        pub const word_type_bits: u32 = @bitSizeOf(Word);
        /// Shift that converts between bit and word ids.
        pub const shift_type_bits: u32 = math.log2_int(u32, word_type_bits);
        /// Narrow index type that always fits inside one word.
        pub const Shift = @Int(.unsigned, shift_type_bits);
        /// Largest valid intra-word index, used for remaining-bit math.
        pub const shift_max_value: Shift = word_type_bits - 1;
        /// Clears low word bits to round an id down to a word start.
        pub const word_mask_inverted: u32 = ~(word_type_bits - 1);

        /// Converts a global bit id into its containing word id.
        /// - `bit_id` - global bit index.
        ///
        /// Return - word index holding the bit.
        pub inline fn bitToWordId(bit_id: u32) u32 {
            return bit_id >> shift_type_bits;
        }

        /// Converts a word id into the global id of its first bit.
        /// - `word_id` - word index to expand.
        ///
        /// Return - global bit id of the word start.
        pub inline fn wordToBitId(word_id: u32) u32 {
            return word_id << shift_type_bits;
        }

        /// Rounds a bit id down to its word start for range scans.
        /// - `bit_id` - global bit index to align.
        ///
        /// Return - first bit id of the containing word.
        pub inline fn floorBitId(bit_id: u32) u32 {
            return bit_id & word_mask_inverted;
        }

        /// Extracts the intra-word offset of a global bit id.
        /// - `bit_id` - global bit index to split.
        ///
        /// Return - position of the bit inside its word.
        pub inline fn bitIdInWord(bit_id: u32) Shift {
            return @truncate(bit_id);
        }

        /// Counts bits remaining in the word from an offset upward.
        /// - `bit_id` - global bit index whose tail is measured.
        ///
        /// Return - number of higher bits left in the same word.
        pub inline fn remainBitsInWord(bit_id: u32) Shift {
            return shift_max_value - bitIdInWord(bit_id);
        }

        /// Computes how many words cover a bit count, rounding up.
        /// - `bits_count` - number of valid bits to store.
        ///
        /// Return - backing word count with padding included.
        pub inline fn bitsToWordsCount(bits_count: u32) u32 {
            const val = @intFromBool(bitIdInWord(bits_count) > 0);
            return bitToWordId(bits_count - val) + val;
        }

        /// Builds a mask with low bits set for partial-word bounds.
        /// - `bits_count` - number of low bits to enable.
        ///
        /// Return - mask with exactly the low bits set.
        pub inline fn maskStart(bits_count: Shift) Word {
            return ~(max_value << bits_count);
        }

        /// Builds a mask with high bits set for partial-word bounds.
        /// - `bits_count` - number of high bits to enable.
        ///
        /// Return - mask with exactly the high bits set.
        pub inline fn maskEnd(bits_count: Shift) Word {
            return ~(max_value >> bits_count);
        }

        /// Builds an edge mask that keeps both ends and clears the middle.
        /// - `start_bits` - number of low bits to enable.
        /// - `end_bits` - number of high bits to enable.
        ///
        /// Return - combined start and end mask.
        pub inline fn maskStartEnd(start_bits: Shift, end_bits: Shift) Word {
            return maskStart(start_bits) | maskEnd(end_bits);
        }

        /// Builds an inverted start mask that clears low bits.
        /// - `bits_count` - number of low bits to clear.
        ///
        /// Return - mask with low bits cleared.
        pub inline fn maskStartInverted(bits_count: Shift) Word {
            return max_value << bits_count;
        }

        /// Builds an inverted end mask that clears high bits.
        /// - `bits_count` - number of high bits to clear.
        ///
        /// Return - mask with high bits cleared.
        pub inline fn maskEndInverted(bits_count: Shift) Word {
            return max_value >> bits_count;
        }

        /// Builds a middle mask that keeps the center and clears edges.
        /// - `start_bits` - number of low bits to clear.
        /// - `end_bits` - number of high bits to clear.
        ///
        /// Return - mask with only the middle bits set.
        pub inline fn maskStartEndInverted(start_bits: Shift, end_bits: Shift) Word {
            return maskStart(start_bits) & maskEnd(end_bits);
        }

        /// Saturating inverted start mask that tolerates full-range counts.
        /// - `count` - number of low bits to clear, clamps at word size.
        ///
        /// Return - cleared-low mask, zero when the count covers the word.
        pub inline fn maskStartInvertedClamped(count: u32) Word {
            return if (count >= word_type_bits) 0 else maskStartInverted(@truncate(count));
        }

        /// Saturating inverted end mask that tolerates full-range counts.
        /// - `count` - number of high bits to clear, clamps at word size.
        ///
        /// Return - cleared-high mask, zero when the count covers the word.
        pub inline fn maskEndInvertedClamped(count: u32) Word {
            return if (count >= word_type_bits) 0 else maskEndInverted(@truncate(count));
        }

        /// Saturating middle mask with full-range start and end counts.
        /// - `start_count` - low bits to clear, clamps at word size.
        /// - `end_count` - high bits to clear, clamps at word size.
        ///
        /// Return - middle mask valid for any u32 counts.
        pub inline fn maskStartEndInvertedClamped(start_count: u32, end_count: u32) Word {
            return maskStartInvertedClamped(start_count) | maskEndInvertedClamped(end_count);
        }

        /// Saturating start mask that tolerates full-range counts.
        /// - `count` - number of low bits to set, clamps at word size.
        ///
        /// Return - low mask, all ones when the count covers the word.
        pub inline fn maskStartClamped(count: u32) Word {
            return if (count >= word_type_bits) max_value else maskStart(@truncate(count));
        }

        /// Saturating end mask that tolerates full-range counts.
        /// - `count` - number of high bits to set, clamps at word size.
        ///
        /// Return - high mask, all ones when the count covers the word.
        pub inline fn maskEndClamped(count: u32) Word {
            return if (count >= word_type_bits) max_value else maskEnd(@truncate(count));
        }

        /// Saturating edge mask with full-range start and end counts.
        /// - `start_count` - low bits to set, clamps at word size.
        /// - `end_count` - high bits to set, clamps at word size.
        ///
        /// Return - combined edge mask valid for any u32 counts.
        pub inline fn maskStartEndClamped(start_count: u32, end_count: u32) Word {
            return maskStartClamped(start_count) | maskEndClamped(end_count);
        }

        /// Selects bits from two words under a mask without branches.
        /// - `zero` - source for cleared mask bits.
        /// - `one` - source for set mask bits.
        /// - `mask` - selects which source each position takes.
        ///
        /// Return - merged word with masked lanes from one.
        pub inline fn merge(zero: Word, one: Word, mask: Word) Word {
            return zero ^ ((zero ^ one) & mask);
        }

        /// Reads one bit as a logical active/inactive state.
        /// - `word` - backing word to sample.
        /// - `bit_id_in_word` - intra-word position to read.
        ///
        /// Return - decoded bit state.
        pub inline fn readBitState(word: Word, bit_id_in_word: Shift) BitState {
            return @enumFromInt(@as(u1, @truncate(word >> bit_id_in_word)));
        }

        /// Reads one raw bit as u1 for arithmetic fast paths.
        /// - `word` - backing word to sample.
        /// - `bit_id_in_word` - intra-word position to read.
        ///
        /// Return - raw bit value.
        pub inline fn readBit(word: Word, bit_id_in_word: Shift) u1 {
            return @truncate(word >> bit_id_in_word);
        }
    };
}

const t = std.testing;
const print = std.debug.print;

test "BitWord u64" {
    const BitWord64 = BitWord(.u64);
    const in = [_]u32{ 0, 1, 33, 63, 64, 65, 99, 127, 128, 129, 10000 };
    const bit_to_word_id = [in.len]u32{ 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 156 };
    const word_to_bit_id = [in.len]u32{ 0, 64, 2112, 4032, 4096, 4160, 6336, 8128, 8192, 8256, 640000 };
    const floor_bit_id = [in.len]u32{ 0, 0, 0, 0, 64, 64, 64, 64, 128, 128, 9984 };
    const bit_id_in_word = [in.len]u32{ 0, 1, 33, 63, 0, 1, 35, 63, 0, 1, 16 };
    const remain_bits_in_word = [in.len]u32{ 63, 62, 30, 0, 63, 62, 28, 0, 63, 62, 47 };
    const bits_to_words_count = [in.len]u32{ 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 157 };

    for (in, 0..) |value, i| {
        try t.expectEqual(bit_to_word_id[i], BitWord64.bitToWordId(value));
        try t.expectEqual(word_to_bit_id[i], BitWord64.wordToBitId(value));
        try t.expectEqual(floor_bit_id[i], BitWord64.floorBitId(value));
        try t.expectEqual(bit_id_in_word[i], @as(BitWord64.Shift, @truncate(value)));
        try t.expectEqual(remain_bits_in_word[i], BitWord64.remainBitsInWord(value));
        try t.expectEqual(bits_to_words_count[i], BitWord64.bitsToWordsCount(value));
    }

    try t.expectEqual(0, BitState.inactive.toWordState(.u64));
    try t.expectEqual(BitWord64.max_value, BitState.active.toWordState(.u64));
}

test "WordType maps to int types" {
    try t.expect(u8 == WordType.u8.Type());
    try t.expect(u16 == WordType.u16.Type());
    try t.expect(u32 == WordType.u32.Type());
    try t.expect(u64 == WordType.u64.Type());
    try t.expectEqual(WordType.u8, BitWord(.u8).word_type);
    try t.expectEqual(WordType.u64, BitWord(.u64).word_type);
    try t.expect(u8 == BitWord(.u8).Word);
    try t.expect(u64 == BitWord(.u64).Word);
}

test "word helpers: clamped masks" {
    const BW8 = BitWord(.u8);
    const BW64 = BitWord(.u64);
    try t.expectEqual(@as(u32, 7), BW8.shift_max_value);
    try t.expectEqual(@as(u32, 63), BW64.shift_max_value);

    try t.expectEqual(@as(u8, 0), BW8.maskStartClamped(0));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskStartClamped(8));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskStartClamped(99));
    try t.expectEqual(@as(u8, 0x3F), BW8.maskStartClamped(6));
    try t.expectEqual(@as(u64, 0x3F), BW64.maskStartClamped(6));

    try t.expectEqual(@as(u8, 0), BW8.maskEndClamped(0));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskEndClamped(8));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskEndClamped(99));
    try t.expectEqual(@as(u8, 0xFC), BW8.maskEndClamped(6));
    try t.expectEqual(@as(u64, 0xFC00000000000000), BW64.maskEndClamped(6));

    try t.expectEqual(@as(u8, 0), BW8.maskStartEndClamped(0, 0));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskStartEndClamped(8, 0));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskStartEndClamped(0, 8));
    try t.expectEqual(@as(u8, 0xFF), BW8.maskStartEndClamped(5, 5));
    try t.expectEqual(@as(u8, 0x03 | 0xC0), BW8.maskStartEndClamped(2, 2));
}

test "masks and merges" {
    const BitWord8 = BitWord(.u8);
    const start = [_]u32{ 0, 1, 3, 7, 15, 31, 63, 127, 0, 1 };
    const end = [_]u32{ 0, 128, 192, 224, 240, 248, 252, 254, 0, 128 };
    const both = [_]u32{ 0, 129, 195, 231, 255, 255, 255, 255, 0, 129, 195, 231, 255, 255, 255, 255 };

    for (0..start.len) |i| {
        const value = start[i];
        const result = BitWord8.maskStart(@truncate(i));
        try t.expectEqual(value, result);
    }

    for (0..end.len) |i| {
        try t.expectEqual(end[i], BitWord8.maskEnd(@truncate(i)));
    }

    for (0..both.len) |i| {
        const j: BitWord8.Shift = @truncate(i);
        const result = BitWord8.maskStartEnd(j, j);
        try t.expectEqual(both[i], result);
    }
    try t.expectEqual(127, BitWord(.u8).maskStart(7));
    try t.expectEqual(32_767, BitWord(.u16).maskStart(15));
    try t.expectEqual(2_147_483_647, BitWord(.u32).maskStart(31));
    try t.expectEqual(9_223_372_036_854_775_807, BitWord(.u64).maskStart(63));

    try t.expectEqual(254, BitWord(.u8).maskEnd(7));
    try t.expectEqual(65_534, BitWord(.u16).maskEnd(15));
    try t.expectEqual(4_294_967_294, BitWord(.u32).maskEnd(31));
    try t.expectEqual(18_446_744_073_709_551_614, BitWord(.u64).maskEnd(63));

    const target_byte: u8 = 0b1000_1010;
    try t.expectEqual(BitState.inactive, BitWord8.readBitState(target_byte, 0));
    try t.expectEqual(0, BitWord8.readBit(target_byte, 0));
    try t.expectEqual(BitState.active, BitWord8.readBitState(target_byte, 1));
    try t.expectEqual(1, BitWord8.readBit(target_byte, 1));
    try t.expectEqual(BitState.inactive, BitWord8.readBitState(target_byte, 2));
    try t.expectEqual(BitState.active, BitWord8.readBitState(target_byte, 3));
    try t.expectEqual(BitState.inactive, BitWord8.readBitState(target_byte, 4));
    try t.expectEqual(BitState.inactive, BitWord8.readBitState(target_byte, 5));
    try t.expectEqual(BitState.inactive, BitWord8.readBitState(target_byte, 6));
    try t.expectEqual(BitState.active, BitWord8.readBitState(target_byte, 7));

    const merge_0 = [_]u8{ 0b1010_1010, 0b1111_0000, 0b1111_0000, 0b1010_1010, 0b1010_1010, 0b0000_0000 };
    const merge_1 = [_]u8{ 0b0101_0101, 0b0000_1111, 0b0000_1111, 0b0101_0101, 0b0101_0101, 0b1111_1111 };
    const merge_m = [_]u8{ 0b1100_1100, 0b1111_0000, 0b0000_1111, 0b1010_1010, 0b0101_0101, 0b1010_1010 };
    const merge_r = [_]u8{ 0b0110_0110, 0b0000_0000, 0b1111_1111, 0b0000_0000, 0b1111_1111, 0b1010_1010 };

    for (0..merge_0.len) |i| {
        try t.expectEqual(merge_r[i], BitWord8.merge(
            merge_0[i],
            merge_1[i],
            merge_m[i],
        ));

        try t.expectEqual(~merge_r[i], BitWord8.merge(
            merge_1[i],
            merge_0[i],
            merge_m[i],
        ));
    }
}
