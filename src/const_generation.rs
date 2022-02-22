use core::num;
use std::mem::size_of;

// For most const operations, we will use the largest available integer
// This assumption avoids the necessity to implement multiple versions of each of the following const functions
// Instantiations of each const function can therefore safely blind cast to the appropriate type
#[cfg(not(has_i128))]
pub(crate) type UIntX = u64;
#[cfg(has_i128)]
pub(crate) type UIntX = u128;

// Int log is not yet stable, so we need our own version
// See: https://github.com/rust-lang/rust/issues/70887
// NOTE - Does not check for parameter validity because the use cases are limited and it's not exposed to the user
const fn ilog(base: UIntX, n: UIntX) -> UIntX {
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
pub(crate) const fn build_dilated_mask(p: UIntX, q: UIntX) -> UIntX {
    let mut r = p;
    let mut v = 0;
    while r > 0 {
        v = (v << q) | 1;
        r -= 1;
    }
    v
}

// Calculates the maximum undilated value that fits into D-dilated T
#[inline]
pub(crate) const fn build_undilated_max<T, const D: usize>() -> UIntX {
    let bits_available = size_of::<T>() * 8;
    let s_minus_1 = bits_available / D - 1;

    // By shifting in two phases like this, we can avoid overflow in case of D1
    let partial_max = (1 << s_minus_1) - 1;
    (partial_max << 1) | 0x1
}

// Calculates the maximum D-dilation round (total number of D-dilation rounds minus one)
// See Algorithm 9 in citation [1]
#[inline]
pub(crate) const fn dilate_max_round<T, const D: UIntX>() -> UIntX {
    let s = (size_of::<T>() * 8) as UIntX / D;
    ilog(D - 1, s)
}

// Calculates the D-dilation multiplier for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn dilate_mult<T, const D: UIntX>(round: UIntX) -> UIntX {
    let round_inv = dilate_max_round::<T, D>() - round;
    build_dilated_mask(D - 1, (D - 1).pow(round_inv as u32 + 1))
}

// Calculates the D-dilation mask for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn dilate_mask<T, const D: UIntX>(round: UIntX) -> UIntX {
    let round_inv = dilate_max_round::<T, D>() - round;
    let num_set_bits = (D - 1).pow(round_inv as u32);
    let num_blank_bits = (D - 1).pow(round_inv as u32 + 1);
    let sequence_length = num_set_bits + num_blank_bits;
    let mut repetitions = (size_of::<T>() * 8) as UIntX / sequence_length + 1;

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
pub(crate) const fn undilate_max_round<T, const D: UIntX>() -> UIntX {
    let s = (size_of::<T>() * 8) as UIntX / D;
    ilog(D, s)
}

// Calculates the D-undilation multiplier for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn undilate_mult<T, const D: UIntX>(round: UIntX) -> UIntX {
    build_dilated_mask(D, D.pow(round as u32) * (D - 1))
}

// Calculates the D-undilation mask for each round
// See section IV.D in citation [1]
#[inline]
pub(crate) const fn undilate_mask<T, const D: UIntX>(round: UIntX) -> UIntX {
    let num_blank_bits = D.pow(round as u32 + 1) * (D - 1);
    let num_set_bits = D.pow(round as u32 + 1);
    0
}

#[cfg(test)]
mod tests {
    use paste::paste;
    use std::marker::PhantomData;

    use super::{ilog, UIntX};

    struct TestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_test_data {
        ($t:ty, $d:literal, $dil_mask:expr, $undil_max:expr) => {
            impl TestData<$t, $d> {
                #[inline]
                fn dilated_mask() -> $t {
                    ($dil_mask) as $t
                }

                #[inline]
                fn undilated_max() -> $t {
                    ($undil_max) as $t
                }
            }
        };
    }
    // (Type, D, dil_mask, undil_max)

    impl_test_data!(u8, 1, 0xff, 0xff);
    impl_test_data!(u8, 2, 0x55, 0x0f);
    impl_test_data!(u8, 3, 0x09, 0x03);
    impl_test_data!(u8, 4, 0x11, 0x03);
    impl_test_data!(u8, 5, 0x01, 0x01);
    impl_test_data!(u8, 6, 0x01, 0x01);
    impl_test_data!(u8, 7, 0x01, 0x01);
    impl_test_data!(u8, 8, 0x01, 0x01);

    impl_test_data!(u16, 1, 0xffff, 0xffff);
    impl_test_data!(u16, 2, 0x5555, 0x00ff);
    impl_test_data!(u16, 3, 0x1249, 0x001f);
    impl_test_data!(u16, 4, 0x1111, 0x000f);
    impl_test_data!(u16, 5, 0x0421, 0x0007);
    impl_test_data!(u16, 6, 0x0041, 0x0003);
    impl_test_data!(u16, 7, 0x0081, 0x0003);
    impl_test_data!(u16, 8, 0x0101, 0x0003);

    impl_test_data!(u32, 1, 0xffffffff, 0xffffffff);
    impl_test_data!(u32, 2, 0x55555555, 0x0000ffff);
    impl_test_data!(u32, 3, 0x09249249, 0x000003ff);
    impl_test_data!(u32, 4, 0x11111111, 0x000000ff);
    impl_test_data!(u32, 5, 0x02108421, 0x0000003f);
    impl_test_data!(u32, 6, 0x01041041, 0x0000001f);
    impl_test_data!(u32, 7, 0x00204081, 0x0000000f);
    impl_test_data!(u32, 8, 0x01010101, 0x0000000f);

    impl_test_data!(u64, 1, 0xffffffffffffffff, 0xffffffffffffffff);
    impl_test_data!(u64, 2, 0x5555555555555555, 0x00000000ffffffff);
    impl_test_data!(u64, 3, 0x1249249249249249, 0x00000000001fffff);
    impl_test_data!(u64, 4, 0x1111111111111111, 0x000000000000ffff);
    impl_test_data!(u64, 5, 0x0084210842108421, 0x0000000000000fff);
    impl_test_data!(u64, 6, 0x0041041041041041, 0x00000000000003ff);
    impl_test_data!(u64, 7, 0x0102040810204081, 0x00000000000001ff);
    impl_test_data!(u64, 8, 0x0101010101010101, 0x00000000000000ff);

    struct DilationTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_dilation_test_data {
        ($t:ty, $d:literal, $max_round:expr, $(($round:literal, $mult:literal, $mask:literal)),*) => {
            impl DilationTestData<$t, $d> {
                #[inline]
                fn max_round() -> Option<$t> {
                    $max_round
                }

                #[inline]
                fn test_cases() -> Vec<(UIntX, $t, $t)> {
                    vec![$(($round, $mult as $t, $mask as $t)),*]
                }
            }
        };
    }

    // (Type, D, max_round, per round: (round, mult, mask)*)

    impl_dilation_test_data!(u8, 1, None,);
    impl_dilation_test_data!(u8, 2, None,);
    impl_dilation_test_data!(u8, 3, Some(1), (0, 0x11, 0xC3), (1, 0x05, 0x49));
    impl_dilation_test_data!(u8, 4, Some(0), (0, 0x49, 0x11));
    impl_dilation_test_data!(u8, 5, Some(0), (0, 0x11, 0x21));
    impl_dilation_test_data!(u8, 6, Some(0), (0, 0x21, 0x41));
    impl_dilation_test_data!(u8, 7, Some(0), (0, 0x41, 0x81));
    impl_dilation_test_data!(u8, 8, Some(0), (0, 0x81, 0x01));

    impl_dilation_test_data!(u16, 1, None,);
    impl_dilation_test_data!(u16, 2, None,);
    impl_dilation_test_data!(
        u16,
        3,
        Some(2),
        (0, 0x0101, 0xF00F),
        (1, 0x0011, 0x30C3),
        (2, 0x0005, 0x9249)
    );
    impl_dilation_test_data!(u16, 4, Some(1), (0, 0x0201, 0x7007), (1, 0x0049, 0x1111));
    impl_dilation_test_data!(u16, 5, Some(0), (0, 0x1111, 0x8421));
    impl_dilation_test_data!(u16, 6, Some(0), (0, 0x8421, 0x1041));
    impl_dilation_test_data!(u16, 7, Some(0), (0, 0x1041, 0x4081));
    impl_dilation_test_data!(u16, 8, Some(0), (0, 0x4081, 0x0101));

    impl_dilation_test_data!(u32, 1, None,);
    impl_dilation_test_data!(u32, 2, None,);
    impl_dilation_test_data!(
        u32,
        3,
        Some(3),
        (0, 0x00010001, 0xFF0000FF),
        (1, 0x00000101, 0x0F00F00F),
        (2, 0x00000011, 0xC30C30C3),
        (3, 0x00000005, 0x49249249)
    );
    impl_dilation_test_data!(
        u32,
        4,
        Some(1),
        (0, 0x00040201, 0x07007007),
        (1, 0x00000049, 0x11111111)
    );
    impl_dilation_test_data!(
        u32,
        5,
        Some(1),
        (0, 0x00010001, 0x00f0000f),
        (1, 0x00001111, 0x42108421)
    );
    impl_dilation_test_data!(
        u32,
        6,
        Some(1),
        (0, 0x02000001, 0xc000001f),
        (1, 0x00108421, 0x41041041)
    );
    impl_dilation_test_data!(u32, 7, Some(0), (0, 0x41041041, 0x10204081));
    impl_dilation_test_data!(u32, 8, Some(0), (0, 0x10204081, 0x01010101));

    impl_dilation_test_data!(u64, 1, None,);
    impl_dilation_test_data!(u64, 2, None,);
    impl_dilation_test_data!(
        u64,
        3,
        Some(4),
        (0, 0x0000000100000001, 0xFFFF00000000FFFF),
        (1, 0x0000000000010001, 0x00FF0000FF0000FF),
        (2, 0x0000000000000101, 0xF00F00F00F00F00F),
        (3, 0x0000000000000011, 0x30C30C30C30C30C3),
        (4, 0x0000000000000005, 0x9249249249249249)
    );
    impl_dilation_test_data!(
        u64,
        4,
        Some(2),
        (0, 0x0040000008000001, 0x00001ff0000001ff),
        (1, 0x0000000000040201, 0x7007007007007007),
        (2, 0x0000000000000049, 0x1111111111111111)
    );
    impl_dilation_test_data!(
        u64,
        5,
        Some(1),
        (0, 0x0001000100010001, 0xf0000f0000f0000f),
        (1, 0x0000000000001111, 0x1084210842108421)
    );
    impl_dilation_test_data!(
        u64,
        6,
        Some(1),
        (0, 0x0004000002000001, 0xf0000007c000001f),
        (1, 0x0000000000108421, 0x1041041041041041)
    );
    impl_dilation_test_data!(
        u64,
        7,
        Some(1),
        (0, 0x0000001000000001, 0x0000fc000000003f),
        (1, 0x0000000041041041, 0x8102040810204081)
    );
    impl_dilation_test_data!(
        u64,
        8,
        Some(1),
        (0, 0x0002000000000001, 0x7f0000000000007f),
        (1, 0x0000040810204081, 0x0101010101010101)
    );

    struct UndilationTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_undilation_test_data {
        ($t:ty, $d:literal, $max_round:expr, $(($round:literal, $mult:literal, $mask:literal)),*) => {
            impl UndilationTestData<$t, $d> {
                #[inline]
                fn max_round() -> Option<$t> {
                    $max_round
                }

                #[inline]
                fn test_cases() -> Vec<(UIntX, $t, $t)> {
                    vec![$(($round, $mult as $t, $mask as $t)),*]
                }
            }
        };
    }

    // (Type, D, max_round, per round: (round, mult, mask)*)

    impl_undilation_test_data!(u8, 1, None,);
    impl_undilation_test_data!(u8, 2, Some(2), (0, 0x03, 0), (1, 0x05, 0), (2, 0x11, 0));
    impl_undilation_test_data!(u8, 3, Some(0), (0, 0x15, 0));
    impl_undilation_test_data!(u8, 4, Some(0), (0, 0x49, 0));
    impl_undilation_test_data!(u8, 5, Some(0), (0, 0x11, 0));
    impl_undilation_test_data!(u8, 6, Some(0), (0, 0x21, 0));
    impl_undilation_test_data!(u8, 7, Some(0), (0, 0x41, 0));
    impl_undilation_test_data!(u8, 8, Some(0), (0, 0x81, 0));

    impl_undilation_test_data!(u16, 1, None,);
    impl_undilation_test_data!(u16, 2, Some(3), (0, 0x0003, 0), (1, 0x0005, 0), (2, 0x0011, 0), (3, 0x0101, 0));
    impl_undilation_test_data!(u16, 3, Some(1), (0, 0x0015, 0), (1, 0x1041, 0));
    impl_undilation_test_data!(u16, 4, Some(1), (0, 0x0249, 0), (1, 0x1001, 0));
    impl_undilation_test_data!(u16, 5, Some(0), (0, 0x1111, 0));
    impl_undilation_test_data!(u16, 6, Some(0), (0, 0x8421, 0));
    impl_undilation_test_data!(u16, 7, Some(0), (0, 0x1041, 0));
    impl_undilation_test_data!(u16, 8, Some(0), (0, 0x4081, 0));

    impl_undilation_test_data!(u32, 1, None,);
    impl_undilation_test_data!(u32, 2, Some(4), (0, 0x00000003, 0), (1, 0x00000005, 0), (2, 0x00000011, 0), (3, 0x00000101, 0), (4, 0x00010001, 0));
    impl_undilation_test_data!(u32, 3, Some(2), (0, 0x00000015, 0), (1, 0x00001041, 0), (2, 0x00040001, 0));
    impl_undilation_test_data!(u32, 4, Some(1), (0, 0x00000249, 0), (1, 0x1001001, 0));
    impl_undilation_test_data!(u32, 5, Some(1), (0, 0x00011111, 0), (1, 0x100001, 0));
    impl_undilation_test_data!(u32, 6, Some(0), (0, 0x02108421, 0));
    impl_undilation_test_data!(u32, 7, Some(0), (0, 0x41041041, 0));
    impl_undilation_test_data!(u32, 8, Some(0), (0, 0x10204081, 0));

    impl_undilation_test_data!(u64, 1, None,);
    impl_undilation_test_data!(u64, 2, Some(5), (0, 0x0000000000000003, 0), (1, 0x0000000000000005, 0), (2, 0x0000000000000011, 0), (3, 0x0000000000000101, 0), (4, 0x0000000000010001, 0), (5, 0x0000000100000001, 0));
    impl_undilation_test_data!(u64, 3, Some(2), (0, 0x0000000000000015, 0), (1, 0x0000000000001041, 0), (2, 0x0000001000040001, 0));
    impl_undilation_test_data!(u64, 4, Some(2), (0, 0x0000000000000249, 0), (1, 0x0000001001001001, 0), (2, 0x0001000000000001, 0));
    impl_undilation_test_data!(u64, 5, Some(1), (0, 0x0000000000011111, 0), (1, 0x1000010000100001, 0));
    impl_undilation_test_data!(u64, 6, Some(1), (0, 0x0000000002108421, 0), (1, 0x1000000040000001, 0));
    impl_undilation_test_data!(u64, 7, Some(1), (0, 0x0000001041041041, 0), (1, 0x0000040000000001, 0));
    impl_undilation_test_data!(u64, 8, Some(1), (0, 0x0002040810204081, 0), (1, 0x0100000000000001, 0));

    #[test]
    fn ilog_is_correct() {
        for d in 2..8 as UIntX {
            // Don't test too many values of i as we may bump up against floating point error
            for i in 1..64 as UIntX {
                let f_log = (i as f32).log(d as f32);

                // To address possible floating point error:
                //   - Detect the case where f_log is very close to, but slightly below, the actual target value
                //   - Bump it up by a small amount to force the cast to UIntX to be accurate
                // The actual floating point value is not important, only its floored integer component (implicit in the cast)
                let f_log = if f_log.ceil() - f_log < std::f32::EPSILON {
                    f_log + 0.1
                } else {
                    f_log
                };
                assert_eq!(ilog(d, i), f_log as UIntX);
            }
        }
    }

    macro_rules! const_generation_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< $t _d $d >] {
                    use std::mem::size_of;
                    use super::{TestData, DilationTestData, UndilationTestData};
                    use super::super::{UIntX, build_dilated_mask, build_undilated_max, dilate_max_round, dilate_mult, dilate_mask, undilate_max_round, undilate_mult};

                    #[test]
                    fn dilated_mask_correct() {
                        assert_eq!(build_dilated_mask((size_of::<$t>() * 8 / $d) as UIntX, $d as UIntX) as $t, TestData::<$t, $d>::dilated_mask());
                    }

                    #[test]
                    fn undilated_max_correct() {
                        assert_eq!(build_undilated_max::<$t, $d>() as $t, TestData::<$t, $d>::undilated_max());
                    }

                    #[test]
                    fn dilate_max_round_is_correct() {
                        if let Some(expect_max_round) = DilationTestData::<$t, $d>::max_round() {
                            assert_eq!(dilate_max_round::<$t, $d>() as $t, expect_max_round);
                        }
                    }

                    #[test]
                    fn dilate_mult_and_mask_are_correct() {
                        for (round, mult, mask) in DilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(dilate_mult::<$t, $d>(round as UIntX) as $t, mult);
                            assert_eq!(dilate_mask::<$t, $d>(round as UIntX) as $t, mask);
                        }
                    }

                    #[test]
                    fn undilate_max_round_is_correct() {
                        if let Some(expect_max_round) = UndilationTestData::<$t, $d>::max_round() {
                            assert_eq!(undilate_max_round::<$t, $d>() as $t, expect_max_round);
                        }
                    }

                    #[test]
                    fn undilate_mult_and_mask_are_correct() {
                        for (round, mult, mask) in UndilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(undilate_mult::<$t, $d>(round as UIntX) as $t, mult);
//                            assert_eq!(dilate_mask::<$t, $d>(round as UIntX) as $t, mask);
                        }
                    }
                }
            }
        )+}
    }

    const_generation_tests!(u8, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u32, 1, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u64, 1, 2, 3, 4, 5, 6, 7, 8);
}
