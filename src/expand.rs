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

use std::marker::PhantomData;

use crate::{internal, SupportedType, Adapter, DilatedInt};

/// Expand<T, D> is an adapter which describes the current dilation implementation - the inner type adapts to the combination of T and D, choosing the smallest type that would contain all bits of the outer type D-dilated. There are naturally a limited number of T and D combinations available as the the largest integer type supported is u128
#[derive(Debug, PartialEq, Eq)]
pub struct Expand<T, const D: usize>(PhantomData<T>) where T: SupportedType;

macro_rules! impl_expand {
    ($outer:ty, $(($d:literal, $inner:ty)),+) => {$(
        impl Adapter for Expand<$outer, $d> {
            type Outer = $outer;
            type Inner = $inner;
            const D: usize = $d;
            const UNDILATED_BITS: usize = <$outer>::BITS as usize;
            const UNDILATED_MAX: Self::Outer = <$outer>::MAX;
            const DILATED_BITS: usize = Self::UNDILATED_BITS * $d;
            const DILATED_MAX: Self::Inner = internal::build_dilated_mask(Self::UNDILATED_BITS, $d) as Self::Inner;

            #[inline]
            fn dilate(value: Self::Outer) -> Self::Inner {
                internal::dilate::<Self::Inner, $d>(value as Self::Inner)
            }

            #[inline]
            fn undilate(value: Self::Inner) -> Self::Outer {
                internal::undilate::<Self::Inner, $d>(value) as Self::Outer
            }
        }
    )+}
}

impl_expand!(u8, (1, u8), (2, u16), (3, u32), (4, u32), (5, u64), (6, u64), (7, u64), (8, u64), (9, u128), (10, u128), (11, u128), (12, u128), (13, u128), (14, u128), (15, u128), (16, u128));
impl_expand!(u16, (1, u16), (2, u32), (3, u64), (4, u64), (5, u128), (6, u128), (7, u128), (8, u128));
impl_expand!(u32, (1, u32), (2, u64), (3, u128), (4, u128));
impl_expand!(u64, (1, u64), (2, u128));
impl_expand!(u128, (1, u128));

#[cfg(target_pointer_width = "16")]
impl_expand!(usize, (1, u16), (2, u32), (3, u64), (4, u64), (5, u128), (6, u128), (7, u128), (8, u128));

#[cfg(target_pointer_width = "32")]
impl_expand!(usize, (1, u32), (2, u64), (3, u128), (4, u128));

#[cfg(target_pointer_width = "64")]
impl_expand!(usize, (1, u64), (2, u128));

pub trait DilateExpand: SupportedType {
    #[inline]
    fn dilate_expand<const D: usize>(self) -> DilatedInt<Expand<Self, D>> where Expand::<Self, D>: Adapter<Outer = Self> {
        DilatedInt::<Expand<Self, D>>(Expand::<Self, D>::dilate(self))
    }
}

impl<T> DilateExpand for T where T: SupportedType { }

#[cfg(test)]
mod tests {
    use paste::paste;

    use crate::{Adapter, shared_test_data::{TestData, impl_test_data}};
    use super::Expand;
   
    impl_test_data!(Expand<u8, 01>, 0xff, u8::MAX);
    impl_test_data!(Expand<u8, 02>, 0x5555, u8::MAX);
    impl_test_data!(Expand<u8, 03>, 0x00249249, u8::MAX);
    impl_test_data!(Expand<u8, 04>, 0x11111111, u8::MAX);
    impl_test_data!(Expand<u8, 05>, 0x0000000842108421, u8::MAX);
    impl_test_data!(Expand<u8, 06>, 0x0000041041041041, u8::MAX);
    impl_test_data!(Expand<u8, 07>, 0x0002040810204081, u8::MAX);
    impl_test_data!(Expand<u8, 08>, 0x0101010101010101, u8::MAX);
    // If testing up to 16 dimensions, these can be uncommented
//    impl_test_data!(Expand<u8, 09>, 0x00000000000000008040201008040201, u8::MAX);
//    impl_test_data!(Expand<u8, 10>, 0x00000000000000401004010040100401, u8::MAX);
//    impl_test_data!(Expand<u8, 11>, 0x00000000000020040080100200400801, u8::MAX);
//    impl_test_data!(Expand<u8, 12>, 0x00000000001001001001001001001001, u8::MAX);
//    impl_test_data!(Expand<u8, 13>, 0x00000000080040020010008004002001, u8::MAX);
//    impl_test_data!(Expand<u8, 14>, 0x00000004001000400100040010004001, u8::MAX);
//    impl_test_data!(Expand<u8, 15>, 0x00000200040008001000200040008001, u8::MAX);
//    impl_test_data!(Expand<u8, 16>, 0x00010001000100010001000100010001, u8::MAX);

    impl_test_data!(Expand<u16, 1>, 0xffff, u16::MAX);
    impl_test_data!(Expand<u16, 2>, 0x55555555, u16::MAX);
    impl_test_data!(Expand<u16, 3>, 0x0000249249249249, u16::MAX);
    impl_test_data!(Expand<u16, 4>, 0x1111111111111111, u16::MAX);
    impl_test_data!(Expand<u16, 5>, 0x00000000000008421084210842108421, u16::MAX);
    impl_test_data!(Expand<u16, 6>, 0x00000000041041041041041041041041, u16::MAX);
    impl_test_data!(Expand<u16, 7>, 0x00000204081020408102040810204081, u16::MAX);
    impl_test_data!(Expand<u16, 8>, 0x01010101010101010101010101010101, u16::MAX);

    impl_test_data!(Expand<u32, 1>, 0xffffffff, u32::MAX);
    impl_test_data!(Expand<u32, 2>, 0x5555555555555555, u32::MAX);
    impl_test_data!(Expand<u32, 3>, 0x00000000249249249249249249249249, u32::MAX);
    impl_test_data!(Expand<u32, 4>, 0x11111111111111111111111111111111, u32::MAX);

    impl_test_data!(Expand<u64, 1>, 0xffffffffffffffff, u64::MAX);
    impl_test_data!(Expand<u64, 2>, 0x55555555555555555555555555555555, u64::MAX);

    impl_test_data!(Expand<u128, 1>, 0xffffffffffffffffffffffffffffffff, u128::MAX);

    macro_rules! impl_expand_test_data_usize {
        ($emulated_t:ty, $($d:literal),+) => {$(
            impl_test_data!(Expand<usize, $d>, TestData::<Expand<$emulated_t, $d>>::dilated_max() as <Expand<usize, $d> as Adapter>::Inner, TestData::<Expand<$emulated_t, $d>>::undilated_max() as <Expand<usize, $d> as Adapter>::Outer);
        )+}
    }
    #[cfg(target_pointer_width = "16")]
    impl_expand_test_data_usize!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    impl_expand_test_data_usize!(u32, 1, 2, 3, 4);
    #[cfg(target_pointer_width = "64")]
    impl_expand_test_data_usize!(u64, 1, 2);

    macro_rules! impl_expand_dilated_int_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< expand_ $t _d $d >] {
                    use crate::shared_test_data::{TestData, VALUES, DILATION_TEST_CASES};
                    use crate::{Adapter, DilatedInt, Undilate};
                    use super::super::{Expand, DilateExpand};

                    #[test]
                    fn undilated_max_is_correct() {
                        assert_eq!(Expand::<$t, $d>::UNDILATED_MAX, TestData::<Expand<$t, $d>>::undilated_max());
                    }

                    #[test]
                    fn dilated_max_is_correct() {
                        assert_eq!(Expand::<$t, $d>::DILATED_MAX, TestData::<Expand<$t, $d>>::dilated_max());
                    }

                    #[test]
                    fn dilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t;
                                let dilated = (*dilated_a ^ *dilated_b) as <Expand<$t, $d> as Adapter>::Inner & TestData::<Expand<$t, $d>>::dilated_max();
                                assert_eq!(Expand::<$t, $d>::dilate(undilated), dilated);
                                assert_eq!(undilated.dilate_expand::<$d>(), DilatedInt::<Expand<$t, $d>>(dilated));
                            }
                        }
                    }

                    #[test]
                    fn undilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t;
                                let dilated = (*dilated_a ^ *dilated_b) as <Expand<$t, $d> as Adapter>::Inner & TestData::<Expand<$t, $d>>::dilated_max();
                                assert_eq!(Expand::<$t, $d>::undilate(dilated), undilated);
                                assert_eq!(DilatedInt::<Expand<$t, $d>>(dilated).undilate(), undilated);
                            }
                        }
                    }

                    #[test]
                    fn add_is_correct() {
                        let test_cases = [
                            (VALUES[$d][0], VALUES[$d][0], VALUES[$d][0]), // 0 + 0 = 0
                            (VALUES[$d][0], VALUES[$d][1], VALUES[$d][1]), // 0 + 1 = 1
                            (VALUES[$d][0], VALUES[$d][2], VALUES[$d][2]), // 0 + 2 = 2
                            (VALUES[$d][1], VALUES[$d][0], VALUES[$d][1]), // 1 + 0 = 1
                            (VALUES[$d][1], VALUES[$d][1], VALUES[$d][2]), // 1 + 1 = 2
                            (VALUES[$d][1], VALUES[$d][2], VALUES[$d][3]), // 1 + 2 = 3
                            (VALUES[$d][2], VALUES[$d][0], VALUES[$d][2]), // 2 + 0 = 2
                            (VALUES[$d][2], VALUES[$d][1], VALUES[$d][3]), // 2 + 1 = 3
                            (VALUES[$d][2], VALUES[$d][2], VALUES[$d][4]), // 2 + 2 = 4
                            (TestData::<Expand<$t, $d>>::dilated_max() as u128, VALUES[$d][1], VALUES[$d][0]), // max + 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Expand<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type InnerT = <Expand<$t, $d> as Adapter>::Inner;
                            assert_eq!(DilatedInt::<Expand<$t, $d>>(*a as InnerT) + DilatedInt::<Expand<$t, $d>>(*b as InnerT), DilatedInt::<Expand<$t, $d>>(*ans as InnerT));
                        }
                    }

                    #[test]
                    fn add_assign_is_correct() {
                        let test_cases = [
                            (VALUES[$d][0], VALUES[$d][0], VALUES[$d][0]), // 0 += 0 = 0
                            (VALUES[$d][0], VALUES[$d][1], VALUES[$d][1]), // 0 += 1 = 1
                            (VALUES[$d][0], VALUES[$d][2], VALUES[$d][2]), // 0 += 2 = 2
                            (VALUES[$d][1], VALUES[$d][0], VALUES[$d][1]), // 1 += 0 = 1
                            (VALUES[$d][1], VALUES[$d][1], VALUES[$d][2]), // 1 += 1 = 2
                            (VALUES[$d][1], VALUES[$d][2], VALUES[$d][3]), // 1 += 2 = 3
                            (VALUES[$d][2], VALUES[$d][0], VALUES[$d][2]), // 2 += 0 = 2
                            (VALUES[$d][2], VALUES[$d][1], VALUES[$d][3]), // 2 += 1 = 3
                            (VALUES[$d][2], VALUES[$d][2], VALUES[$d][4]), // 2 += 2 = 4
                            (TestData::<Expand<$t, $d>>::dilated_max() as u128, VALUES[$d][1], VALUES[$d][0]), // max += 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Expand<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type InnerT = <Expand<$t, $d> as Adapter>::Inner;
                            let mut assigned = DilatedInt::<Expand<$t, $d>>(*a as InnerT);
                            assigned += DilatedInt::<Expand<$t, $d>>(*b as InnerT);
                            assert_eq!(assigned, DilatedInt::<Expand<$t, $d>>(*ans as InnerT));
                        }
                    }

                    #[test]
                    fn sub_is_correct() {
                        let test_cases = [
                            (VALUES[$d][2], VALUES[$d][0], VALUES[$d][2]), // 2 - 0 = 2
                            (VALUES[$d][2], VALUES[$d][1], VALUES[$d][1]), // 2 - 1 = 1
                            (VALUES[$d][2], VALUES[$d][2], VALUES[$d][0]), // 2 - 2 = 0
                            (VALUES[$d][3], VALUES[$d][0], VALUES[$d][3]), // 3 - 0 = 3
                            (VALUES[$d][3], VALUES[$d][1], VALUES[$d][2]), // 3 - 1 = 2
                            (VALUES[$d][3], VALUES[$d][2], VALUES[$d][1]), // 3 - 2 = 1
                            (VALUES[$d][4], VALUES[$d][0], VALUES[$d][4]), // 4 - 0 = 4
                            (VALUES[$d][4], VALUES[$d][1], VALUES[$d][3]), // 4 - 1 = 3
                            (VALUES[$d][4], VALUES[$d][2], VALUES[$d][2]), // 4 - 2 = 2
                            (VALUES[$d][0], VALUES[$d][1], TestData::<Expand<$t, $d>>::dilated_max() as u128), // 0 - 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Expand<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type InnerT = <Expand<$t, $d> as Adapter>::Inner;
                            assert_eq!(DilatedInt::<Expand<$t, $d>>(*a as InnerT) - DilatedInt::<Expand<$t, $d>>(*b as InnerT), DilatedInt::<Expand<$t, $d>>(*ans as InnerT));
                        }
                    }

                    #[test]
                    fn sub_assign_is_correct() {
                        let test_cases = [
                            (VALUES[$d][2], VALUES[$d][0], VALUES[$d][2]), // 2 -= 0 = 2
                            (VALUES[$d][2], VALUES[$d][1], VALUES[$d][1]), // 2 -= 1 = 1
                            (VALUES[$d][2], VALUES[$d][2], VALUES[$d][0]), // 2 -= 2 = 0
                            (VALUES[$d][3], VALUES[$d][0], VALUES[$d][3]), // 3 -= 0 = 3
                            (VALUES[$d][3], VALUES[$d][1], VALUES[$d][2]), // 3 -= 1 = 2
                            (VALUES[$d][3], VALUES[$d][2], VALUES[$d][1]), // 3 -= 2 = 1
                            (VALUES[$d][4], VALUES[$d][0], VALUES[$d][4]), // 4 -= 0 = 4
                            (VALUES[$d][4], VALUES[$d][1], VALUES[$d][3]), // 4 -= 1 = 3
                            (VALUES[$d][4], VALUES[$d][2], VALUES[$d][2]), // 4 -= 2 = 2
                            (VALUES[$d][0], VALUES[$d][1], TestData::<Expand<$t, $d>>::dilated_max() as u128), // 0 -= 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Expand<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type InnerT = <Expand<$t, $d> as Adapter>::Inner;
                            let mut assigned = DilatedInt::<Expand<$t, $d>>(*a as InnerT);
                            assigned -= DilatedInt::<Expand<$t, $d>>(*b as InnerT);
                            assert_eq!(assigned, DilatedInt::<Expand<$t, $d>>(*ans as InnerT));
                        }
                    }
                }
            }
        )+}
    }
    // Technically, u8 can go up to 16 dimensions, but that would double the amount of inline test data
    impl_expand_dilated_int_tests!(u8, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_expand_dilated_int_tests!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_expand_dilated_int_tests!(u32, 1, 2, 3, 4);
    impl_expand_dilated_int_tests!(u64, 1, 2);
    impl_expand_dilated_int_tests!(u128, 1);

    #[cfg(target_pointer_width = "16")]
    impl_expand_dilated_int_tests!(usize, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    impl_expand_dilated_int_tests!(usize, 1, 2, 3, 4);
    #[cfg(target_pointer_width = "64")]
    impl_expand_dilated_int_tests!(usize, 1, 2);
}
