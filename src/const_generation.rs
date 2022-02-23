// ANTI-CAPITALIST SOFTWARE LICENSE (v 1.4)
// 
// Copyright © 2022 Alex Blunt (alexmadeathing)
// 
// This is anti-capitalist software, released for free use by individuals and
// organizations that do not operate by capitalist principles.
// 
// Permission is hereby granted, free of charge, to any person or organization
// (the "User") obtaining a copy of this software and associated documentation
// files (the "Software"), to use, copy, modify, merge, distribute, and/or sell
// copies of the Software, subject to the following conditions: 
//
// 1. The above copyright notice and this permission notice shall be included in
// all copies or modified versions of the Software.
// 
// 2. The User is one of the following:
//   a. An individual person, laboring for themselves
//   b. A non-profit organization
//   c. An educational institution
//   d. An organization that seeks shared profit for all of its members, and
//      allows non-members to set the cost of their labor
// 
// 3. If the User is an organization with owners, then all owners are workers
// and all workers are owners with equal equity and/or equal vote.
// 
// 4. If the User is an organization, then the User is not law enforcement or
// military, or working for or under either.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY
// KIND, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

use std::mem::size_of;

// For most const operations, we will use the largest available integer
// This assumption avoids the necessity to implement multiple versions of each of the following const functions
// Instantiations of each const function can therefore safely blind cast to the appropriate type

// Int log is not yet stable, so we need our own version
// See: https://github.com/rust-lang/rust/issues/70887
// NOTE - Does not check for parameter validity because the use cases are limited and it's not exposed to the user
const fn ilog(base: usize, n: usize) -> usize {
    let mut r = 0;
    let mut b = base;
    while b <= n {
        b = b * base;
        r += 1;
    }
    r
}

// Builds a dilated mask with p repetitions of 1 bits separated by (q - 1) 0 bits
// See Notation 4.2 in citation [1]
#[inline]
pub(crate) const fn build_dilated_mask(p_repetitions: usize, q_width: usize) -> u128 {
    let mut r = p_repetitions;
    let mut v = 0;
    while r > 0 {
        v = (v << q_width) | 1;
        r -= 1;
    }
    v
}

// Calculates the maximum undilated value that fits into D-dilated T
#[inline]
pub(crate) const fn build_undilated_max<T, const D: usize>() -> u128 {
    let bits_available = size_of::<T>() * 8;
    let s_minus_1 = bits_available / D - 1;

    // By shifting in two phases like this, we can avoid overflow in case of D1
    // NOTE - We can't just use Wrapping because this is a const fn
    let partial_max = (1 << s_minus_1) - 1;
    (partial_max << 1) | 0x1
}

// Calculates the maximum D-dilation round (total number of D-dilation rounds minus one)
// See Algorithm 9 in citation [1]
#[inline]
pub(crate) const fn dilate_max_round<T, const D: usize>() -> usize {
    let s = (size_of::<T>() * 8) / D;
    ilog(D - 1, s)
}

// Calculates the D-dilation multiplier for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn dilate_mult<T, const D: usize>(round: usize) -> u128 {
    let round_inv = dilate_max_round::<T, D>() - round;
    build_dilated_mask(D - 1, (D - 1).pow(round_inv as u32 + 1))
}

// Calculates the D-dilation mask for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn dilate_mask<T, const D: usize>(round: usize) -> u128 {
    let round_inv = dilate_max_round::<T, D>() - round;
    let num_set_bits = (D - 1).pow(round_inv as u32);
    let num_blank_bits = num_set_bits * (D - 1);
    let sequence_length = num_set_bits + num_blank_bits;
    let mut repetitions = (size_of::<T>() * 8) / sequence_length + 1;

    let mut v = 0;
    while repetitions > 0 {
        v = (v << sequence_length) | ((1 << num_set_bits) - 1);
        repetitions -= 1;
    }
    v
}

// Calculates the maximum D-undilation round (total number of D-undilation rounds minus one)
// See Algorithm 10 in citation [1]
#[inline]
pub(crate) const fn undilate_max_round<T, const D: usize>() -> usize {
    let s = (size_of::<T>() * 8) / D;
    ilog(D, s)
}

// Calculates the D-undilation multiplier for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn undilate_mult<T, const D: usize>(round: usize) -> u128 {
    build_dilated_mask(D, D.pow(round as u32) * (D - 1))
}

// Calculates the D-undilation mask for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn undilate_mask<T, const D: usize>(round: usize) -> u128 {
    let s = (size_of::<T>() * 8) / D;
    let num_blank_bits = D.pow(round as u32 + 1) * (D - 1);
    let num_set_bits = D.pow(round as u32 + 1);

    let sequence_length = num_blank_bits + num_set_bits;
    let left_most_bit = D * (s - 1) + 1;

    let num_set_bits = if num_set_bits < left_most_bit { num_set_bits } else { left_most_bit };

    let initial_shift = left_most_bit - num_set_bits;

    let mut v = ((1 << num_set_bits) - 1) << initial_shift;
    let mut repetitions = left_most_bit / sequence_length;
    while repetitions > 0 {
        v = (v >> sequence_length) | (((1 << num_set_bits) - 1) << initial_shift);
        repetitions -= 1;
    }
    v
}

#[inline]
pub(crate) const fn undilate_shift<T, const D: usize>() -> usize {
    let s = (size_of::<T>() * 8) / D;
    (D * (s - 1) + 1) - s
}

#[cfg(test)]
mod tests {
    use paste::paste;
    use std::marker::PhantomData;

    use super::ilog;

    struct DilationTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_dilation_test_data {
        ($t:ty, $d:literal, $dil_mask:literal, $num_rounds:literal, $(($round:literal, $mult:literal, $mask:literal)),*) => {
            impl DilationTestData<$t, $d> {
                #[inline]
                fn dilated_mask() -> $t {
                    $dil_mask
                }
                
                #[inline]
                fn num_rounds() -> usize {
                    $num_rounds
                }

                #[inline]
                fn test_cases() -> Vec<(usize, $t, $t)> {
                    vec![$(($round, $mult, $mask)),*]
                }
            }
        };
    }

    // (Type, D, max_round, per round: (round, mult, mask)*)

    impl_dilation_test_data!(u8, 1, 0xff, 0,);
    impl_dilation_test_data!(u8, 2, 0x55, 0,);
    impl_dilation_test_data!(u8, 3, 0x09, 2, (0, 0x11, 0xC3), (1, 0x05, 0x49));
    impl_dilation_test_data!(u8, 4, 0x11, 1, (0, 0x49, 0x11));
    impl_dilation_test_data!(u8, 5, 0x01, 1, (0, 0x11, 0x21));
    impl_dilation_test_data!(u8, 6, 0x01, 1, (0, 0x21, 0x41));
    impl_dilation_test_data!(u8, 7, 0x01, 1, (0, 0x41, 0x81));
    impl_dilation_test_data!(u8, 8, 0x01, 1, (0, 0x81, 0x01));

    impl_dilation_test_data!(u16, 1, 0xffff, 0,);
    impl_dilation_test_data!(u16, 2, 0x5555, 0,);
    impl_dilation_test_data!(u16, 3, 0x1249, 3, (0, 0x0101, 0xF00F), (1, 0x0011, 0x30C3), (2, 0x0005, 0x9249));
    impl_dilation_test_data!(u16, 4, 0x1111, 2, (0, 0x0201, 0x7007), (1, 0x0049, 0x1111));
    impl_dilation_test_data!(u16, 5, 0x0421, 1, (0, 0x1111, 0x8421));
    impl_dilation_test_data!(u16, 6, 0x0041, 1, (0, 0x8421, 0x1041));
    impl_dilation_test_data!(u16, 7, 0x0081, 1, (0, 0x1041, 0x4081));
    impl_dilation_test_data!(u16, 8, 0x0101, 1, (0, 0x4081, 0x0101));

    impl_dilation_test_data!(u32, 1, 0xffffffff, 0,);
    impl_dilation_test_data!(u32, 2, 0x55555555, 0,);
    impl_dilation_test_data!(u32, 3, 0x09249249, 4, (0, 0x00010001, 0xFF0000FF), (1, 0x00000101, 0x0F00F00F), (2, 0x00000011, 0xC30C30C3), (3, 0x00000005, 0x49249249));
    impl_dilation_test_data!(u32, 4, 0x11111111, 2, (0, 0x00040201, 0x07007007), (1, 0x00000049, 0x11111111));
    impl_dilation_test_data!(u32, 5, 0x02108421, 2, (0, 0x00010001, 0x00f0000f), (1, 0x00001111, 0x42108421));
    impl_dilation_test_data!(u32, 6, 0x01041041, 2, (0, 0x02000001, 0xc000001f), (1, 0x00108421, 0x41041041));
    impl_dilation_test_data!(u32, 7, 0x00204081, 1, (0, 0x41041041, 0x10204081));
    impl_dilation_test_data!(u32, 8, 0x01010101, 1, (0, 0x10204081, 0x01010101));

    impl_dilation_test_data!(u64, 1, 0xffffffffffffffff, 0,);
    impl_dilation_test_data!(u64, 2, 0x5555555555555555, 0,);
    impl_dilation_test_data!(u64, 3, 0x1249249249249249, 5, (0, 0x0000000100000001, 0xFFFF00000000FFFF), (1, 0x0000000000010001, 0x00FF0000FF0000FF), (2, 0x0000000000000101, 0xF00F00F00F00F00F), (3, 0x0000000000000011, 0x30C30C30C30C30C3), (4, 0x0000000000000005, 0x9249249249249249));
    impl_dilation_test_data!(u64, 4, 0x1111111111111111, 3, (0, 0x0040000008000001, 0x00001ff0000001ff), (1, 0x0000000000040201, 0x7007007007007007), (2, 0x0000000000000049, 0x1111111111111111));
    impl_dilation_test_data!(u64, 5, 0x0084210842108421, 2, (0, 0x0001000100010001, 0xf0000f0000f0000f), (1, 0x0000000000001111, 0x1084210842108421));
    impl_dilation_test_data!(u64, 6, 0x0041041041041041, 2, (0, 0x0004000002000001, 0xf0000007c000001f), (1, 0x0000000000108421, 0x1041041041041041));
    impl_dilation_test_data!(u64, 7, 0x0102040810204081, 2, (0, 0x0000001000000001, 0x0000fc000000003f), (1, 0x0000000041041041, 0x8102040810204081));
    impl_dilation_test_data!(u64, 8, 0x0101010101010101, 2, (0, 0x0002000000000001, 0x7f0000000000007f), (1, 0x0000040810204081, 0x0101010101010101));

    impl_dilation_test_data!(u128, 1, 0xffffffffffffffffffffffffffffffff, 0,);
    impl_dilation_test_data!(u128, 2, 0x55555555555555555555555555555555, 0,);
    impl_dilation_test_data!(u128, 3, 0x09249249249249249249249249249249, 6, (0, 0x00000000000000010000000000000001, 0xffffffff0000000000000000ffffffff), (1, 0x00000000000000000000000100000001, 0x0000ffff00000000ffff00000000ffff), (2, 0x00000000000000000000000000010001, 0xff0000ff0000ff0000ff0000ff0000ff), (3, 0x00000000000000000000000000000101, 0x0f00f00f00f00f00f00f00f00f00f00f), (4, 0x00000000000000000000000000000011, 0xc30c30c30c30c30c30c30c30c30c30c3), (5, 0x00000000000000000000000000000005, 0x49249249249249249249249249249249));
    impl_dilation_test_data!(u128, 4, 0x11111111111111111111111111111111, 4, (0, 0x00000000000200000000000000000001, 0xfffff000000000000000000007ffffff), (1, 0x00000000000000000040000008000001, 0x001ff0000001ff0000001ff0000001ff), (2, 0x00000000000000000000000000040201, 0x07007007007007007007007007007007), (3, 0x00000000000000000000000000000049, 0x11111111111111111111111111111111));
    impl_dilation_test_data!(u128, 5, 0x01084210842108421084210842108421, 3, (0, 0x00000000000000010000000000000001, 0x00000000ffff0000000000000000ffff), (1, 0x00000000000000000001000100010001, 0x0f0000f0000f0000f0000f0000f0000f), (2, 0x00000000000000000000000000001111, 0x21084210842108421084210842108421));
    impl_dilation_test_data!(u128, 6, 0x01041041041041041041041041041041, 2, (0, 0x00000010000008000004000002000001, 0x1f0000007c000001f0000007c000001f), (1, 0x00000000000000000000000000108421, 0x41041041041041041041041041041041));
    impl_dilation_test_data!(u128, 7, 0x00810204081020408102040810204081, 2, (0, 0x00001000000001000000001000000001, 0xc000000003f000000000fc000000003f), (1, 0x00000000000000000000000041041041, 0x40810204081020408102040810204081));
    impl_dilation_test_data!(u128, 8, 0x01010101010101010101010101010101, 2, (0, 0x00000004000000000002000000000001, 0x007f0000000000007f0000000000007f), (1, 0x00000000000000000000040810204081, 0x01010101010101010101010101010101));

    struct UndilationTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_undilation_test_data {
        ($t:ty, $d:literal, $undil_max:literal, $undil_shift:literal, $num_rounds:literal, $(($round:literal, $mult:literal, $mask:literal)),*) => {
            impl UndilationTestData<$t, $d> {
                #[inline]
                fn undilated_max() -> $t {
                    $undil_max
                }

                #[inline]
                fn undilate_shift() -> usize {
                    $undil_shift
                }
                
                #[inline]
                fn num_rounds() -> usize {
                    $num_rounds
                }

                #[inline]
                fn test_cases() -> Vec<(usize, $t, $t)> {
                    vec![$(($round, $mult, $mask)),*]
                }
            }
        };
    }

    // (Type, D, max_round, per round: (round, mult, mask)*)

    impl_undilation_test_data!(u8, 1, 0xff, 0, 0,);
    impl_undilation_test_data!(u8, 2, 0x0f, 3, 3, (0, 0x03, 0x66), (1, 0x05, 0x78), (2, 0x11, 0x7F));
    impl_undilation_test_data!(u8, 3, 0x03, 2, 1, (0, 0x15, 0x0e));
    impl_undilation_test_data!(u8, 4, 0x03, 3, 1, (0, 0x49, 0x1e));
    impl_undilation_test_data!(u8, 5, 0x01, 0, 1, (0, 0x11, 0x01));
    impl_undilation_test_data!(u8, 6, 0x01, 0, 1, (0, 0x21, 0x01));
    impl_undilation_test_data!(u8, 7, 0x01, 0, 1, (0, 0x41, 0x01));
    impl_undilation_test_data!(u8, 8, 0x01, 0, 1, (0, 0x81, 0x01));

    impl_undilation_test_data!(u16, 1, 0xffff, 0, 0,);
    impl_undilation_test_data!(u16, 2, 0x00ff, 7, 4, (0, 0x0003, 0x6666), (1, 0x0005, 0x7878), (2, 0x0011, 0x7f80), (3, 0x0101, 0x7fff));
    impl_undilation_test_data!(u16, 3, 0x001f, 8, 2, (0, 0x0015, 0x1c0e), (1, 0x1041, 0x1ff0));
    impl_undilation_test_data!(u16, 4, 0x000f, 9, 2, (0, 0x0249, 0x1e00), (1, 0x1001, 0x1fff));
    impl_undilation_test_data!(u16, 5, 0x0007, 8, 1, (0, 0x1111, 0x07c0));
    impl_undilation_test_data!(u16, 6, 0x0003, 5, 1, (0, 0x8421, 0x007e));
    impl_undilation_test_data!(u16, 7, 0x0003, 6, 1, (0, 0x1041, 0x00fe));
    impl_undilation_test_data!(u16, 8, 0x0003, 7, 1, (0, 0x4081, 0x01fe));

    impl_undilation_test_data!(u32, 1, 0xffffffff, 0, 0,);
    impl_undilation_test_data!(u32, 2, 0x0000ffff, 15, 5, (0, 0x00000003, 0x66666666), (1, 0x00000005, 0x78787878), (2, 0x00000011, 0x7F807F80), (3, 0x00000101, 0x7FFF8000), (4, 0x00010001, 0x7fffffff));
    impl_undilation_test_data!(u32, 3, 0x000003ff, 18, 3, (0, 0x00000015, 0x0E070381), (1, 0x00001041, 0x0FF80001), (2, 0x00040001, 0x0FFFFFFE));
    impl_undilation_test_data!(u32, 4, 0x000000ff, 21, 2, (0, 0x00000249, 0x1e001e00), (1, 0x01001001, 0x1fffe000));
    impl_undilation_test_data!(u32, 5, 0x0000003f, 20, 2, (0, 0x00011111, 0x03e00001), (1, 0x00100001, 0x03fffffe));
    impl_undilation_test_data!(u32, 6, 0x0000001f, 20, 1, (0, 0x02108421, 0x01f80000));
    impl_undilation_test_data!(u32, 7, 0x0000000f, 18, 1, (0, 0x41041041, 0x003f8000));
    impl_undilation_test_data!(u32, 8, 0x0000000f, 21, 1, (0, 0x10204081, 0x01fe0000));

    impl_undilation_test_data!(u64, 1, 0xffffffffffffffff, 0, 0,);
    impl_undilation_test_data!(u64, 2, 0x00000000ffffffff, 31, 6, (0, 0x0000000000000003, 0x6666666666666666), (1, 0x0000000000000005, 0x7878787878787878), (2, 0x0000000000000011, 0x7F807F807F807F80), (3, 0x0000000000000101, 0x7FFF80007FFF8000), (4, 0x0000000000010001, 0x7FFFFFFF80000000), (5, 0x0000000100000001, 0x7fffffffffffffff));
    impl_undilation_test_data!(u64, 3, 0x00000000001fffff, 40, 3, (0, 0x0000000000000015, 0x1c0e070381c0e070), (1, 0x0000000000001041, 0x1ff00003fe00007f), (2, 0x0000001000040001, 0x1ffffffc00000000));
    impl_undilation_test_data!(u64, 4, 0x000000000000ffff, 45, 3, (0, 0x0000000000000249, 0x1e001e001e001e00), (1, 0x0000001001001001, 0x1fffe00000000000), (2, 0x0001000000000001, 0x1fffffffffffffff));
    impl_undilation_test_data!(u64, 5, 0x0000000000000fff, 44, 2, (0, 0x0000000000011111, 0x00f800007c00003e), (1, 0x1000010000100001, 0x00ffffff80000000));
    impl_undilation_test_data!(u64, 6, 0x00000000000003ff, 45, 2, (0, 0x0000000002108421, 0x007e00000007e000), (1, 0x1000000040000001, 0x007ffffffff80000));
    impl_undilation_test_data!(u64, 7, 0x00000000000001ff, 48, 2, (0, 0x0000001041041041, 0x01fc0000000000fe), (1, 0x0000040000000001, 0x01ffffffffffff00));
    impl_undilation_test_data!(u64, 8, 0x00000000000000ff, 49, 2, (0, 0x0002040810204081, 0x01fe000000000000), (1, 0x0100000000000001, 0x01ffffffffffffff));
    
    impl_undilation_test_data!(u128, 1, 0xffffffffffffffffffffffffffffffff, 0,   0,);
    impl_undilation_test_data!(u128, 2, 0x0000000000000000ffffffffffffffff, 63,  7, (0, 0x00000000000000000000000000000003, 0x66666666666666666666666666666666), (1, 0x00000000000000000000000000000005, 0x78787878787878787878787878787878), (2, 0x00000000000000000000000000000011, 0x7f807f807f807f807f807f807f807f80), (3, 0x00000000000000000000000000000101, 0x7fff80007fff80007fff80007fff8000), (4, 0x00000000000000000000000000010001, 0x7fffffff800000007fffffff80000000), (5, 0x00000000000000000000000100000001, 0x7fffffffffffffff8000000000000000), (6, 0x00000000000000010000000000000001, 0x7fffffffffffffffffffffffffffffff));
    impl_undilation_test_data!(u128, 3, 0x0000000000000000000003ffffffffff, 82,  4, (0, 0x00000000000000000000000000000015, 0x0e070381c0e070381c0e070381c0e070), (1, 0x00000000000000000000000000001041, 0x0ff80001ff00003fe00007fc0000ff80), (2, 0x00000000000000000000001000040001, 0x0ffffffe00000000000007ffffff0000), (3, 0x00001000000000000040000000000001, 0x0ffffffffffffffffffff80000000000));
    impl_undilation_test_data!(u128, 4, 0x000000000000000000000000ffffffff, 93,  3, (0, 0x00000000000000000000000000000249, 0x1e001e001e001e001e001e001e001e00), (1, 0x00000000000000000000001001001001, 0x1fffe000000000001fffe00000000000), (2, 0x00000001000000000001000000000001, 0x1fffffffffffffffe000000000000000));
    impl_undilation_test_data!(u128, 5, 0x00000000000000000000000001ffffff, 96,  3, (0, 0x00000000000000000000000000011111, 0x01f00000f800007c00003e00001f0000), (1, 0x00000000000100001000010000100001, 0x01ffffff000000000000000000000000), (2, 0x00000010000000000000000000000001, 0x01ffffffffffffffffffffffffffffff));
    impl_undilation_test_data!(u128, 6, 0x000000000000000000000000001fffff, 100, 2, (0, 0x00000000000000000000000002108421, 0x01f80000001f80000001f80000001f80), (1, 0x01000000040000001000000040000001, 0x01ffffffffe000000000000000000000));
    impl_undilation_test_data!(u128, 7, 0x0000000000000000000000000003ffff, 102, 2, (0, 0x00000000000000000000001041041041, 0x00fe00000000007f00000000003f8000), (1, 0x40000000001000000000040000000001, 0x00ffffffffffff800000000000000000));
    impl_undilation_test_data!(u128, 8, 0x0000000000000000000000000000ffff, 105, 2, (0, 0x00000000000000000002040810204081, 0x01fe00000000000001fe000000000000), (1, 0x00010000000000000100000000000001, 0x01fffffffffffffffe00000000000000));

    #[test]
    fn ilog_is_correct() {
        for d in 2..8 as usize {
            // Don't test too many values of i as we may bump up against floating point error
            for i in 1..64 as usize {
                let f_log = (i as f32).log(d as f32);

                // To address possible floating point error:
                //   - Detect the case where f_log is very close to, but slightly below, the actual target value
                //   - Bump it up by a small amount to force the cast to u128 to be accurate
                // The actual floating point value is not important, only its floored integer component (implicit in the cast)
                let f_log = if f_log.ceil() - f_log < std::f32::EPSILON {
                    f_log + 0.1
                } else {
                    f_log
                };
                assert_eq!(ilog(d, i), f_log as usize);
            }
        }
    }

    macro_rules! const_generation_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< $t _d $d >] {
                    use std::mem::size_of;
                    use super::{DilationTestData, UndilationTestData};
                    use super::super::{build_dilated_mask, build_undilated_max, dilate_max_round, dilate_mult, dilate_mask, undilate_max_round, undilate_mult, undilate_mask, undilate_shift};

                    #[test]
                    fn dilated_mask_is_correct() {
                        assert_eq!(build_dilated_mask((size_of::<$t>() * 8 / $d) as usize, $d as usize) as $t, DilationTestData::<$t, $d>::dilated_mask());
                    }

                    #[test]
                    fn dilate_max_round_is_correct() {
                        let expect_num_rounds = DilationTestData::<$t, $d>::num_rounds();
                        if expect_num_rounds > 0 {
                            assert_eq!(dilate_max_round::<$t, $d>(), expect_num_rounds - 1);
                        }
                    }

                    #[test]
                    fn dilate_mult_is_correct() {
                        for (round, mult, _) in DilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(dilate_mult::<$t, $d>(round) as $t, mult);
                        }
                    }

                    #[test]
                    fn dilate_mask_is_correct() {
                        for (round, _, mask) in DilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(dilate_mask::<$t, $d>(round) as $t, mask);
                        }
                    }

                    #[test]
                    fn undilated_max_is_correct() {
                        assert_eq!(build_undilated_max::<$t, $d>() as $t, UndilationTestData::<$t, $d>::undilated_max());
                    }

                    #[test]
                    fn undilate_max_round_is_correct() {
                        let expect_num_rounds = UndilationTestData::<$t, $d>::num_rounds();
                        if expect_num_rounds > 0 {
                            assert_eq!(undilate_max_round::<$t, $d>(), expect_num_rounds - 1);
                        }
                    }

                    #[test]
                    fn undilate_mult_is_correct() {
                        for (round, mult, _) in UndilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(undilate_mult::<$t, $d>(round) as $t, mult);
                        }
                    }

                    #[test]
                    fn undilate_mask_is_correct() {
                        for (round, _, mask) in UndilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(undilate_mask::<$t, $d>(round) as $t, mask);
                        }
                    }

                    #[test]
                    fn undilate_shift_is_correct() {
                        assert_eq!(undilate_shift::<$t, $d>(), UndilationTestData::<$t, $d>::undilate_shift());
                    }
                }
            }
        )+}
    }
    const_generation_tests!(u8, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u32, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u64, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u128, 1, 2, 3, 4, 5, 6, 7, 8);
}
