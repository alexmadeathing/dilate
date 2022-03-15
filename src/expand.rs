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

// # References and Acknowledgments
// Many thanks to the authors of the following white papers:
// * [1] Converting to and from Dilated Integers - Rajeev Raman and David S. Wise
// * [2] Integer Dilation and Contraction for Quadtrees and Octrees - Leo Stocco and Gunther Schrack
// * [3] Fast Additions on Masked Integers - Michael D Adams and David S Wise
//
// Permission has been explicitly granted to reproduce the agorithms within each paper.

use std::marker::PhantomData;

use crate::{internal, DilatableType, DilationMethod, DilatedInt};

/// A DilationMethod implementation which provides expanding dilation meta information
/// 
/// This trait implementation describes the types involved with an expanding
/// dilation as well as some useful constants and wrapper methods which
/// actually perform the dilations.
/// 
/// Although this trait implementation provides the functions for performing
/// dilations, users should generally prefer to dilate via the [DilateExpand]
/// trait and its [dilate_expand()](DilateExpand::dilate_expand()) method,
/// which is generally less verbose and therefore more user friendly.
///
/// # Examples
/// ```
/// use dilate::*;
///
/// assert_eq!(Expand::<u8, 2>::UNDILATED_MAX, 255);
/// assert_eq!(Expand::<u8, 2>::UNDILATED_BITS, 8);
/// 
/// assert_eq!(Expand::<u8, 2>::DILATED_MAX, 0b0101010101010101);
/// assert_eq!(Expand::<u8, 2>::DILATED_BITS, 16);
/// 
/// let original: u8 = 0b1101;
/// let dilated = Expand::<u8, 2>::dilate(original);
///
/// assert_eq!(dilated.value(), 0b01010001);
/// assert_eq!(dilated, DilatedInt::<Expand<u8, 2>>::new(0b01010001));
/// 
/// assert_eq!(Expand::<u8, 2>::undilate(dilated), original);
/// ```
/// 
/// For more detailed information, see [dilate_expand()](crate::expand::DilateExpand::dilate_expand())
#[derive(Debug, PartialEq, Eq)]
pub struct Expand<T, const D: usize>(PhantomData<T>) where T: DilatableType;

macro_rules! impl_expand {
    ($undilated:ty, $(($d:literal, $dilated:ty)),+) => {$(
        impl DilationMethod for Expand<$undilated, $d> {
            type Undilated = $undilated;
            type Dilated = $dilated;
            const D: usize = $d;
            const UNDILATED_BITS: usize = <$undilated>::BITS as usize;
            const UNDILATED_MAX: Self::Undilated = <$undilated>::MAX;
            const DILATED_BITS: usize = Self::UNDILATED_BITS * $d;
            const DILATED_MAX: Self::Dilated = internal::build_dilated_mask(Self::UNDILATED_BITS, $d) as Self::Dilated;
            const DILATED_MASK: Self::Dilated = Self::DILATED_MAX * ((1 << $d) - 1);
            const DILATED_ZERO: Self::Dilated = 0;
            const DILATED_ONE: Self::Dilated = 1;

            #[inline]
            fn dilate(value: Self::Undilated) -> DilatedInt<Self> {
                DilatedInt::<Self>(internal::dilate::<Self::Dilated, $d>(value as Self::Dilated))
            }

            #[inline]
            fn undilate(value: DilatedInt<Self>) -> Self::Undilated {
                internal::undilate::<Self::Dilated, $d>(value.0) as Self::Undilated
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

/// A convenience trait for dilating integers using the [Expand] [DilationMethod]
///
/// This trait is implemented by all supported integer types and provides a
/// convenient and human readable way to dilate integers. Simply call the
/// [DilateExpand::dilate_expand()] method to perform the dilation.
pub trait DilateExpand: DilatableType {
    /// Dilates all bits of the source integer into a larger integer type
    ///
    /// Dilating using the expand method creates a dilated integer large enough to
    /// hold all bits of the original integer. The exact type of the resultant
    /// integer is defined by the individual [Expand] implementations and depends
    /// on the number of bits in the source integer and how large the dilation is,
    /// denoted by D.
    ///
    /// # Examples
    /// ```
    /// use dilate::*;
    ///
    /// let value: u8 = 0b1101;
    ///
    /// assert_eq!(value.dilate_expand::<2>(), DilatedInt::<Expand<u8, 2>>::new(0b01010001));
    /// assert_eq!(value.dilate_expand::<2>().value(), 0b01010001);
    /// ```
    ///
    /// # The Storage Type
    /// The size required to store the dilated version of an integer is determined by
    /// the size in bits `S` of the integer multiplied by the dilation amount `D`.
    /// Thus in all `D > 1` cases, the dilated version of a value will be stored
    /// using a larger integer than T. The exact integer type chosen is determined by
    /// finding the smallest integer that contains `S * D` bits. Because the largest
    /// supported integer is u128, there is a fixed upper limit of `S * D <= 128`.
    /// For example: `Expand::<u8, 16>` is valid because `8 * 16 = 128`.
    /// `Expand::<u16, 10>` is not valid because `16 * 10 > 128`. There are currently
    /// no plans to support larger types, although the implementation is
    /// theoretically possible.
    ///
    /// # Supported Expand Dilations
    /// The following is a list of supported combinations of types `T` and dilation
    /// amounts `D` and their underlying expanded type. The maximum dilatable value
    /// for Expand dilations is always T::MAX.
    ///
    /// | T      | D   | Expands To | | T      | D   | Expands To |
    /// | ------ | --- | ---------- | | ------ | --- | ---------- |
    /// | `u8`   | 1   | `u8`       | | `u16`  | 1   | `u16`      |
    /// | `u8`   | 2   | `u16`      | | `u16`  | 2   | `u32`      |
    /// | `u8`   | 3   | `u32`      | | `u16`  | 3   | `u64`      |
    /// | `u8`   | 4   | `u32`      | | `u16`  | 4   | `u64`      |
    /// | `u8`   | 5   | `u64`      | | `u16`  | 5   | `u128`     |
    /// | `u8`   | 6   | `u64`      | | `u16`  | 6   | `u128`     |
    /// | `u8`   | 7   | `u64`      | | `u16`  | 7   | `u128`     |
    /// | `u8`   | 8   | `u64`      | | `u16`  | 8   | `u128`     |
    /// | `u8`   | 9   | `u128`     | | ...    | ... | ...        |
    /// | `u8`   | 10  | `u128`     | | `u32`  | 1   | `u32`      |
    /// | `u8`   | 11  | `u128`     | | `u32`  | 2   | `u64`      |
    /// | `u8`   | 12  | `u128`     | | `u32`  | 3   | `u128`     |
    /// | `u8`   | 13  | `u128`     | | `u32`  | 4   | `u128`     |
    /// | `u8`   | 14  | `u128`     | | ...    | ... | ...        |
    /// | `u8`   | 15  | `u128`     | | `u64`  | 1   | `u64`      |
    /// | `u8`   | 16  | `u128`     | | `u64`  | 2   | `u128`     |
    /// | ...    | ... | ...        | | ...    | ... | ...        |
    /// | ...    | ... | ...        | | `u128` | 1   | `u128`     |
    ///
    /// Please note that usize is also supported and its behaviour is the same as the
    /// relevant integer type for your platform. For example, on a 32 bit system,
    /// usize is interpreted as a u32 and will have the same expansion types as u32.
    ///
    /// See also [Expand<T, D>::dilate()](crate::DilationMethod::dilate())
    #[inline]
    fn dilate_expand<const D: usize>(self) -> DilatedInt<Expand<Self, D>> where Expand::<Self, D>: DilationMethod<Undilated = Self> {
        Expand::<Self, D>::dilate(self)
    }
}

impl<T> DilateExpand for T where T: DilatableType { }

#[cfg(test)]
mod tests {
    use paste::paste;

    use crate::{DilationMethod, shared_test_data::{TestData, impl_test_data}};
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
            impl_test_data!(Expand<usize, $d>, TestData::<Expand<$emulated_t, $d>>::dilated_max() as <Expand<usize, $d> as DilationMethod>::Dilated, TestData::<Expand<$emulated_t, $d>>::undilated_max() as <Expand<usize, $d> as DilationMethod>::Undilated);
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
                    use crate::{DilationMethod, DilatedInt, Undilate, AddOne, SubOne};
                    use super::super::{Expand, DilateExpand};

                    type DilationMethodT = Expand<$t, $d>;
                    type DilatedIntT = DilatedInt<DilationMethodT>;
                    type DilatedT = <DilationMethodT as DilationMethod>::Dilated;

                    #[test]
                    fn undilated_max_is_correct() {
                        assert_eq!(DilationMethodT::UNDILATED_MAX, TestData::<DilationMethodT>::undilated_max());
                    }

                    #[test]
                    fn dilated_max_is_correct() {
                        assert_eq!(DilationMethodT::DILATED_MAX, TestData::<DilationMethodT>::dilated_max());
                    }

                    #[test]
                    fn new_invalid_panics() {
                        for bit in 0..DilatedT::BITS {
                            if bit % $d != 0 {
                                let dilated = 1 << bit;
                                let result = std::panic::catch_unwind(|| DilatedIntT::new(dilated));
                                if !result.is_err() {
                                    panic!("Test did not panic as expected");
                                }
                            }
                        }
                    }

                    #[test]
                    fn new_valid_stores_correct_value() {
                        assert_eq!(DilatedIntT::new(VALUES[$d][0] as DilatedT).0, VALUES[$d][0] as DilatedT);
                        assert_eq!(DilatedIntT::new(VALUES[$d][1] as DilatedT).0, VALUES[$d][1] as DilatedT);
                        assert_eq!(DilatedIntT::new(VALUES[$d][2] as DilatedT).0, VALUES[$d][2] as DilatedT);
                        assert_eq!(DilatedIntT::new(VALUES[$d][3] as DilatedT).0, VALUES[$d][3] as DilatedT);
                        assert_eq!(DilatedIntT::new(DilationMethodT::DILATED_MAX).0, DilationMethodT::DILATED_MAX);
                    }

                    #[test]
                    fn value_returns_unmodified_value() {
                        assert_eq!(DilatedInt::<DilationMethodT>(VALUES[$d][0] as DilatedT).value(), VALUES[$d][0] as DilatedT);
                        assert_eq!(DilatedInt::<DilationMethodT>(VALUES[$d][1] as DilatedT).value(), VALUES[$d][1] as DilatedT);
                        assert_eq!(DilatedInt::<DilationMethodT>(VALUES[$d][2] as DilatedT).value(), VALUES[$d][2] as DilatedT);
                        assert_eq!(DilatedInt::<DilationMethodT>(VALUES[$d][3] as DilatedT).value(), VALUES[$d][3] as DilatedT);
                        assert_eq!(DilatedInt::<DilationMethodT>(DilationMethodT::DILATED_MAX).value(), DilationMethodT::DILATED_MAX);
                    }

                    #[test]
                    fn dilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t;
                                let dilated = (*dilated_a ^ *dilated_b) as DilatedT & TestData::<DilationMethodT>::dilated_max();
                                assert_eq!(DilationMethodT::dilate(undilated), DilatedInt::<DilationMethodT>(dilated));
                                assert_eq!(undilated.dilate_expand::<$d>(), DilatedInt::<DilationMethodT>(dilated));
                            }
                        }
                    }

                    #[test]
                    fn undilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t;
                                let dilated = (*dilated_a ^ *dilated_b) as DilatedT & TestData::<DilationMethodT>::dilated_max();
                                assert_eq!(DilationMethodT::undilate(DilatedInt::<DilationMethodT>(dilated)), undilated);
                                assert_eq!(DilatedInt::<DilationMethodT>(dilated).undilate(), undilated);
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
                            (TestData::<DilationMethodT>::dilated_max() as u128, VALUES[$d][1], VALUES[$d][0]), // max + 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<DilationMethodT>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            assert_eq!(DilatedInt::<DilationMethodT>(*a as DilatedT) + DilatedInt::<DilationMethodT>(*b as DilatedT), DilatedInt::<DilationMethodT>(*ans as DilatedT));
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
                            (TestData::<DilationMethodT>::dilated_max() as u128, VALUES[$d][1], VALUES[$d][0]), // max += 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<DilationMethodT>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            let mut assigned = DilatedInt::<DilationMethodT>(*a as DilatedT);
                            assigned += DilatedInt::<DilationMethodT>(*b as DilatedT);
                            assert_eq!(assigned, DilatedInt::<DilationMethodT>(*ans as DilatedT));
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
                            (VALUES[$d][0], VALUES[$d][1], TestData::<DilationMethodT>::dilated_max() as u128), // 0 - 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<DilationMethodT>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            assert_eq!(DilatedInt::<DilationMethodT>(*a as DilatedT) - DilatedInt::<DilationMethodT>(*b as DilatedT), DilatedInt::<DilationMethodT>(*ans as DilatedT));
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
                            (VALUES[$d][0], VALUES[$d][1], TestData::<DilationMethodT>::dilated_max() as u128), // 0 -= 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<DilationMethodT>::dilated_max() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            let mut assigned = DilatedInt::<DilationMethodT>(*a as DilatedT);
                            assigned -= DilatedInt::<DilationMethodT>(*b as DilatedT);
                            assert_eq!(assigned, DilatedInt::<DilationMethodT>(*ans as DilatedT));
                        }
                    }

                    #[test]
                    fn add_one_is_correct() {
                        for i in 0..10 {
                            let value = VALUES[$d][i] as DilatedT & DilationMethodT::DILATED_MAX;
                            let value_add_one = VALUES[$d][i + 1] as DilatedT & DilationMethodT::DILATED_MAX;
                            assert_eq!(DilatedInt::<DilationMethodT>(value).add_one().0, value_add_one);
                        }
                        assert_eq!(DilatedInt::<DilationMethodT>(DilationMethodT::DILATED_MAX).add_one().0, 0);
                    }

                    #[test]
                    fn sub_one_is_correct() {
                        for i in 10..0 {
                            let value = VALUES[$d][i] as DilatedT & DilationMethodT::DILATED_MAX;
                            let value_sub_one = VALUES[$d][i - 1] as DilatedT & DilationMethodT::DILATED_MAX;
                            assert_eq!(DilatedInt::<DilationMethodT>(value).sub_one().0, value_sub_one);
                        }
                        assert_eq!(DilatedInt::<DilationMethodT>(0).sub_one().0, DilationMethodT::DILATED_MAX);
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
