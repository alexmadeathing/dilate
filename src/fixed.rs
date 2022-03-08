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

/// Dilating using the Fixed adapter creates a dilated integer of the same type
/// as the input integer, giving the user control over the storage type. The
/// maximum number of dilatable input bits is defined by the adapter and depends
/// on the combination of T and D.
/// 
/// ```
/// use dilate::*;
/// 
/// assert_eq!(0b1101u16.dilate_fixed::<2>().0, 0b01010001);
/// 
/// assert_eq!(Fixed::<u16, 2>::dilate(0b1101).0, 0b01010001);
/// ```
/// *Two methods for dilating u16 into u16 using the Fixed adapter*
/// Fixed<T, D> is an adapter which describes the pre-0.4.0 dilation implementation - the dilated type is the same as the undilated type and a limited subset of bits in the undilated type can be dilated. This is useful when you want absolute control over the dilated type used and want to fit as many dilated bits into the dilated type as possible.
/// 
/// # Which Adapter to Choose
/// There are currently two types of Adapter. To help decide which is right for
/// your application, consider the following:
/// 
/// Use [Expand] when you want all bits of the source integer to be dilated and
/// you don't mind how the dilated integer is stored behind the scenes. This is
/// the most intuitive method of interacting with dilated integers.
/// 
/// Use [Fixed] when you want control over the storage type and want to
/// maximise the number of bits occupied within that storage type.
/// [Fixed] is potentially more memory efficient than [Expand].
/// 
/// Notice that the difference between the two is that of focus; [Expand]
/// focusses on maximising the usage of the source integer, whereas [Fixed]
/// focusses on maximising the usage of the dilated integer.
#[derive(Debug, PartialEq, Eq)]
pub struct Fixed<T, const D: usize>(PhantomData<T>) where T: SupportedType;

macro_rules! impl_fixed {
    ($t:ty, $($d:literal),+) => {$(
        impl Adapter for Fixed<$t, $d> {
            type Undilated = $t;
            type Dilated = $t;
            const UNDILATED_BITS: usize = <$t>::BITS as usize / $d;
            const UNDILATED_MAX: Self::Undilated = internal::build_fixed_undilated_max::<$t, $d>() as $t;
            const DILATED_BITS: usize = Self::UNDILATED_BITS * $d;
            const DILATED_MAX: Self::Dilated = internal::build_dilated_mask(Self::UNDILATED_BITS, $d) as Self::Dilated;

            #[inline]
            fn dilate(value: Self::Undilated) -> DilatedInt<Self> {
                DilatedInt::<Self>(internal::dilate::<Self::Dilated, $d>(value))
            }

            #[inline]
            fn undilate(value: DilatedInt<Self>) -> Self::Undilated {
                internal::undilate::<Self::Dilated, $d>(value.0)
            }
        }
    )+}
}

impl_fixed!(u8, 1, 2, 3, 4);
impl_fixed!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
impl_fixed!(u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_fixed!(u64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_fixed!(u128, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_fixed!(usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

pub trait DilateFixed: SupportedType {
    #[inline]
    fn dilate_fixed<const D: usize>(self) -> DilatedInt<Fixed<Self, D>> where Fixed::<Self, D>: Adapter<Undilated = Self> {
        Fixed::<Self, D>::dilate(self)
    }
}

impl<T> DilateFixed for T where T: SupportedType { }

#[cfg(test)]
mod tests {
    use paste::paste;

    use crate::{Adapter, shared_test_data::{TestData, impl_test_data}};
    use super::Fixed;
   
    impl_test_data!(Fixed<u8, 01>, 0xff, 0xff);
    impl_test_data!(Fixed<u8, 02>, 0x55, 0x0f);
    impl_test_data!(Fixed<u8, 03>, 0x09, 0x03);
    impl_test_data!(Fixed<u8, 04>, 0x11, 0x03);

    impl_test_data!(Fixed<u16, 1>, 0xffff, 0xffff);
    impl_test_data!(Fixed<u16, 2>, 0x5555, 0x00ff);
    impl_test_data!(Fixed<u16, 3>, 0x1249, 0x001f);
    impl_test_data!(Fixed<u16, 4>, 0x1111, 0x000f);
    impl_test_data!(Fixed<u16, 5>, 0x0421, 0x0007);
    impl_test_data!(Fixed<u16, 6>, 0x0041, 0x0003);
    impl_test_data!(Fixed<u16, 7>, 0x0081, 0x0003);
    impl_test_data!(Fixed<u16, 8>, 0x0101, 0x0003);

    impl_test_data!(Fixed<u32, 1>, 0xffffffff, 0xffffffff);
    impl_test_data!(Fixed<u32, 2>, 0x55555555, 0x0000ffff);
    impl_test_data!(Fixed<u32, 3>, 0x09249249, 0x000003ff);
    impl_test_data!(Fixed<u32, 4>, 0x11111111, 0x000000ff);
    impl_test_data!(Fixed<u32, 5>, 0x02108421, 0x0000003f);
    impl_test_data!(Fixed<u32, 6>, 0x01041041, 0x0000001f);
    impl_test_data!(Fixed<u32, 7>, 0x00204081, 0x0000000f);
    impl_test_data!(Fixed<u32, 8>, 0x01010101, 0x0000000f);

    impl_test_data!(Fixed<u64, 1>, 0xffffffffffffffff, 0xffffffffffffffff);
    impl_test_data!(Fixed<u64, 2>, 0x5555555555555555, 0x00000000ffffffff);
    impl_test_data!(Fixed<u64, 3>, 0x1249249249249249, 0x00000000001fffff);
    impl_test_data!(Fixed<u64, 4>, 0x1111111111111111, 0x000000000000ffff);
    impl_test_data!(Fixed<u64, 5>, 0x0084210842108421, 0x0000000000000fff);
    impl_test_data!(Fixed<u64, 6>, 0x0041041041041041, 0x00000000000003ff);
    impl_test_data!(Fixed<u64, 7>, 0x0102040810204081, 0x00000000000001ff);
    impl_test_data!(Fixed<u64, 8>, 0x0101010101010101, 0x00000000000000ff);

    impl_test_data!(Fixed<u128, 1>, 0xffffffffffffffffffffffffffffffff, 0xffffffffffffffffffffffffffffffff);
    impl_test_data!(Fixed<u128, 2>, 0x55555555555555555555555555555555, 0x0000000000000000ffffffffffffffff);
    impl_test_data!(Fixed<u128, 3>, 0x09249249249249249249249249249249, 0x0000000000000000000003ffffffffff);
    impl_test_data!(Fixed<u128, 4>, 0x11111111111111111111111111111111, 0x000000000000000000000000ffffffff);
    impl_test_data!(Fixed<u128, 5>, 0x01084210842108421084210842108421, 0x00000000000000000000000001ffffff);
    impl_test_data!(Fixed<u128, 6>, 0x01041041041041041041041041041041, 0x000000000000000000000000001fffff);
    impl_test_data!(Fixed<u128, 7>, 0x00810204081020408102040810204081, 0x0000000000000000000000000003ffff);
    impl_test_data!(Fixed<u128, 8>, 0x01010101010101010101010101010101, 0x0000000000000000000000000000ffff);

    macro_rules! impl_fixed_test_data_usize {
        ($emulated_t:ty, $($d:literal),+) => {$(
            impl_test_data!(Fixed<usize, $d>, TestData::<Fixed<$emulated_t, $d>>::dilated_max() as <Fixed<usize, $d> as Adapter>::Dilated, TestData::<Fixed<$emulated_t, $d>>::undilated_max() as <Fixed<usize, $d> as Adapter>::Dilated);
        )+}
    }
    #[cfg(target_pointer_width = "16")]
    impl_fixed_test_data_usize!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    impl_fixed_test_data_usize!(u32, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "64")]
    impl_fixed_test_data_usize!(u64, 1, 2, 3, 4, 5, 6, 7, 8);

    macro_rules! impl_fixed_dilated_int_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< fixed_ $t _d $d >] {
                    use crate::shared_test_data::{TestData, VALUES, DILATION_TEST_CASES};
                    use crate::{Adapter, DilatedInt, Undilate};
                    use super::super::{Fixed, DilateFixed};

                    #[test]
                    fn undilated_max_is_correct() {
                        assert_eq!(Fixed::<$t, $d>::UNDILATED_MAX, TestData::<Fixed<$t, $d>>::undilated_max());
                    }

                    #[test]
                    fn dilated_max_is_correct() {
                        assert_eq!(Fixed::<$t, $d>::DILATED_MAX, TestData::<Fixed<$t, $d>>::dilated_max());
                    }

                    #[test]
                    fn dilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<Fixed<$t, $d>>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as <Fixed<$t, $d> as Adapter>::Dilated & TestData::<Fixed<$t, $d>>::dilated_max();
                                assert_eq!(Fixed::<$t, $d>::dilate(undilated), DilatedInt::<Fixed<$t, $d>>(dilated));
                                assert_eq!(undilated.dilate_fixed::<$d>(), DilatedInt::<Fixed<$t, $d>>(dilated));
                            }
                        }
                    }

                    #[test]
                    fn undilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<Fixed<$t, $d>>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as <Fixed<$t, $d> as Adapter>::Dilated & TestData::<Fixed<$t, $d>>::dilated_max();
                                assert_eq!(Fixed::<$t, $d>::undilate(DilatedInt::<Fixed<$t, $d>>(dilated)), undilated);
                                assert_eq!(DilatedInt::<Fixed<$t, $d>>(dilated).undilate(), undilated);
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
                            (TestData::<Fixed<$t, $d>>::dilated_max() as u128, VALUES[$d][1], VALUES[$d][0]), // max + 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Fixed<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type DilatedT = <Fixed<$t, $d> as Adapter>::Dilated;
                            assert_eq!(DilatedInt::<Fixed<$t, $d>>(*a as DilatedT) + DilatedInt::<Fixed<$t, $d>>(*b as DilatedT), DilatedInt::<Fixed<$t, $d>>(*ans as DilatedT));
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
                            (TestData::<Fixed<$t, $d>>::dilated_max() as u128, VALUES[$d][1], VALUES[$d][0]), // max += 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Fixed<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type DilatedT = <Fixed<$t, $d> as Adapter>::Dilated;
                            let mut assigned = DilatedInt::<Fixed<$t, $d>>(*a as DilatedT);
                            assigned += DilatedInt::<Fixed<$t, $d>>(*b as DilatedT);
                            assert_eq!(assigned, DilatedInt::<Fixed<$t, $d>>(*ans as DilatedT));
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
                            (VALUES[$d][0], VALUES[$d][1], TestData::<Fixed<$t, $d>>::dilated_max() as u128), // 0 - 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Fixed<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type DilatedT = <Fixed<$t, $d> as Adapter>::Dilated;
                            assert_eq!(DilatedInt::<Fixed<$t, $d>>(*a as DilatedT) - DilatedInt::<Fixed<$t, $d>>(*b as DilatedT), DilatedInt::<Fixed<$t, $d>>(*ans as DilatedT));
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
                            (VALUES[$d][0], VALUES[$d][1], TestData::<Fixed<$t, $d>>::dilated_max() as u128), // 0 -= 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<Fixed<$t, $d>>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            type DilatedT = <Fixed<$t, $d> as Adapter>::Dilated;
                            let mut assigned = DilatedInt::<Fixed<$t, $d>>(*a as DilatedT);
                            assigned -= DilatedInt::<Fixed<$t, $d>>(*b as DilatedT);
                            assert_eq!(assigned, DilatedInt::<Fixed<$t, $d>>(*ans as DilatedT));
                        }
                    }
                }
            }
        )+}
    }
    impl_fixed_dilated_int_tests!(u8, 1, 2, 3, 4);
    impl_fixed_dilated_int_tests!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(u32, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(u64, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(u128, 1, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(usize, 1, 2, 3, 4, 5, 6, 7, 8);
}
