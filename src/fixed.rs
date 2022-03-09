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

use crate::{internal, SupportedType, DilationMethod, DilatedInt};

/// Dilates a subset of bits of the source integer into the same integer type
///
/// Dilating using the Fixed method dilates as many bits as possible from the
/// source integer into an integer of the same type, maximising the memory usage
/// of a single type. This is useful when you want absolute control over the
/// dilated type used and want to fit as many dilated bits into the dilated type
/// as possible.
///
/// The number of dilatable bits is known ahead of time and can be retrieved
/// using the [Fixed::UNDILATED_BITS](crate::DilationMethod::UNDILATED_BITS)
/// constant.
/// 
/// Note that when using the fixed method, attempting to dilate a value that
/// would not fit into the same type will yield a panic. You may use
/// [Fixed::<T, D>::UNDILATED_MAX](crate::DilationMethod::UNDILATED_MAX) (or
/// the table below) to determine whether your value will dilate successfully.
///
/// It is not necessary to use Fixed directly. Instead, there's a handy helper
/// trait implemented by all supported integers called DilateFixed. This trait
/// provides a more convenient method of interating with fixed dilations -
/// simply call the [dilate_fixed()](DilateFixed::dilate_fixed()) method
/// on your integer to dilate it.
///
/// # Examples
/// ```
/// use dilate::*;
///
/// let value: u16 = 0b1101;
///
/// assert_eq!(value.dilate_fixed::<2>(), DilatedInt::<Fixed<u16, 2>>(0b01010001));
/// assert_eq!(value.dilate_fixed::<2>().0, 0b01010001);
///
/// assert_eq!(Fixed::<u16, 2>::dilate(value), DilatedInt::<Fixed<u16, 2>>(0b01010001));
/// assert_eq!(Fixed::<u16, 2>::dilate(value).0, 0b01010001);
/// ```
/// *Two methods for dilating u16 into u16 using the Fixed method*
///
/// # Supported Dilations via Fixed
/// The following is a list of supported combinations of types `T`, dilation
/// amounts `D`, and the maximum dilatable value. The source integer and the
/// internal dilated integer types are the same for Fixed dilations.
///
/// | T      | D   | Max Value    | | T      | D   | Max Value                            |
/// | ------ | --- | ------------ | | ------ | --- | ------------------------------------ |
/// | `u8`   | 1   | `0xff`       | | `u64`  | 1   | `0xffffffffffffffff`                 |
/// | `u8`   | 2   | `0x0f`       | | `u64`  | 2   | `0x00000000ffffffff`                 |
/// | `u8`   | 3   | `0x03`       | | `u64`  | 3   | `0x00000000001fffff`                 |
/// | `u8`   | 4   | `0x03`       | | `u64`  | 4   | `0x000000000000ffff`                 |
/// | ...    | ... | ...          | | `u64`  | 5   | `0x0000000000000fff`                 |
/// | `u16`  | 1   | `0xffff`     | | `u64`  | 6   | `0x00000000000003ff`                 |
/// | `u16`  | 2   | `0x00ff`     | | `u64`  | 7   | `0x00000000000001ff`                 |
/// | `u16`  | 3   | `0x001f`     | | `u64`  | 8   | `0x00000000000000ff`                 |
/// | `u16`  | 4   | `0x000f`     | | `u64`  | 9   | `0x000000000000007f`                 |
/// | `u16`  | 5   | `0x0007`     | | `u64`  | 10  | `0x000000000000003f`                 |
/// | `u16`  | 6   | `0x0003`     | | `u64`  | 11  | `0x000000000000001f`                 |
/// | `u16`  | 7   | `0x0003`     | | `u64`  | 12  | `0x000000000000001f`                 |
/// | `u16`  | 8   | `0x0003`     | | `u64`  | 13  | `0x000000000000000f`                 |
/// | ...    | ... | ...          | | `u64`  | 14  | `0x000000000000000f`                 |
/// | `u32`  | 1   | `0xffffffff` | | `u64`  | 15  | `0x000000000000000f`                 |
/// | `u32`  | 2   | `0x0000ffff` | | `u64`  | 16  | `0x000000000000000f`                 |
/// | `u32`  | 3   | `0x000003ff` | | ...    | ... | ...                                  |
/// | `u32`  | 4   | `0x000000ff` | | `u128` | 1   | `0xffffffffffffffffffffffffffffffff` |
/// | `u32`  | 5   | `0x0000003f` | | `u128` | 2   | `0x0000000000000000ffffffffffffffff` |
/// | `u32`  | 6   | `0x0000001f` | | `u128` | 3   | `0x0000000000000000000003ffffffffff` |
/// | `u32`  | 7   | `0x0000000f` | | `u128` | 4   | `0x000000000000000000000000ffffffff` |
/// | `u32`  | 8   | `0x0000000f` | | `u128` | 5   | `0x00000000000000000000000001ffffff` |
/// | `u32`  | 9   | `0x00000007` | | `u128` | 6   | `0x000000000000000000000000001fffff` |
/// | `u32`  | 10  | `0x00000007` | | `u128` | 7   | `0x0000000000000000000000000003ffff` |
/// | `u32`  | 11  | `0x00000003` | | `u128` | 8   | `0x0000000000000000000000000000ffff` |
/// | `u32`  | 12  | `0x00000003` | | `u128` | 9   | `0x00000000000000000000000000003fff` |
/// | `u32`  | 13  | `0x00000003` | | `u128` | 10  | `0x00000000000000000000000000000fff` |
/// | `u32`  | 14  | `0x00000003` | | `u128` | 11  | `0x000000000000000000000000000007ff` |
/// | `u32`  | 15  | `0x00000003` | | `u128` | 12  | `0x000000000000000000000000000003ff` |
/// | `u32`  | 16  | `0x00000003` | | `u128` | 13  | `0x000000000000000000000000000001ff` |
/// | ...    | ... | ...          | | `u128` | 14  | `0x000000000000000000000000000001ff` |
/// | ...    | ... | ...          | | `u128` | 15  | `0x000000000000000000000000000000ff` |
/// | ...    | ... | ...          | | `u128` | 16  | `0x000000000000000000000000000000ff` |
///
/// Please note that usize is also supported and its behaviour is the same as the
/// relevant integer type for your platform. For example, on a 32 bit system,
/// usize is interpreted as a u32 and will have the same max dilatable value as u32.
///
/// # Which Dilation Method to Choose
/// There are currently two distinct ways to dilate integers; via the
/// [DilateExpand](crate::expand::DilateExpand) and
/// [DilateFixed](crate::fixed::DilateFixed) trait implementations. To help
/// decide which is right for your application, consider the following:
/// 
/// Use [dilate_expand()](crate::expand::DilateExpand::dilate_expand()) when
/// you want all bits of the source integer to be dilated and you don't mind
/// how the dilated integer is stored behind the scenes. This is the most
/// intuitive method of interacting with dilated integers.
/// 
/// Use [dilate_fixed()](crate::fixed::DilateFixed::dilate_fixed()) when you
/// want control over the storage type and want to maximise the number of bits
/// occupied within that storage type.
/// 
/// Notice that the difference between the two is that of focus;
/// [dilate_expand()](crate::expand::DilateExpand::dilate_expand()) focusses on
/// maximising the usage of the source integer, whereas
/// [dilate_fixed()](crate::fixed::DilateFixed::dilate_fixed()) focusses on
/// maximising the usage of the dilated integer.
/// 
/// You may also use the raw [Expand](crate::expand::Expand) and
/// [Fixed](crate::fixed::Fixed) [DilationMethod](crate::DilationMethod)
/// implementations directly, though they tend to be more verbose.
#[derive(Debug, PartialEq, Eq)]
pub struct Fixed<T, const D: usize>(PhantomData<T>) where T: SupportedType;

macro_rules! impl_fixed {
    ($t:ty, $($d:literal),+) => {$(
        impl DilationMethod for Fixed<$t, $d> {
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

/// A convenience trait for dilating integers using the [Fixed] method
///
/// This trait is implemented by all supported integer types and provides a
/// convenient and human readable way to dilate integers. Simply call the
/// [DilateFixed::dilate_fixed()] method to perform the dilation.
pub trait DilateFixed: SupportedType {
    /// This method carries out the fixed dilation process
    ///
    /// It converts a raw [Undilated](DilationMethod::Undilated) value to a
    /// [DilatedInt].
    ///
    /// This method is provided as a more convenient way to interact with the
    /// [Fixed] dilation method.
    /// 
    /// Note that when using the fixed method, attempting to dilate a value that
    /// would not fit into the same type will yield a panic. You may use
    /// [Fixed::<T, D>::UNDILATED_MAX](crate::DilationMethod::UNDILATED_MAX)
    /// to determine whether your value will dilate successfully.
    ///
    /// # Examples
    /// ```
    /// use dilate::*;
    ///
    /// let value: u16 = 0b1101;
    ///
    /// assert_eq!(value.dilate_fixed::<3>(), DilatedInt::<Fixed<u16, 3>>(0b001001000001));
    /// assert_eq!(value.dilate_fixed::<3>().0, 0b001001000001);
    /// ```
    ///
    /// See also [Fixed<T, D>::dilate()](crate::DilationMethod::dilate())
    #[inline]
    fn dilate_fixed<const D: usize>(self) -> DilatedInt<Fixed<Self, D>> where Fixed::<Self, D>: DilationMethod<Undilated = Self> {
        Fixed::<Self, D>::dilate(self)
    }
}

impl<T> DilateFixed for T where T: SupportedType { }

#[cfg(test)]
mod tests {
    use paste::paste;

    use crate::{DilationMethod, shared_test_data::{TestData, impl_test_data}};
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
            impl_test_data!(Fixed<usize, $d>, TestData::<Fixed<$emulated_t, $d>>::dilated_max() as <Fixed<usize, $d> as DilationMethod>::Dilated, TestData::<Fixed<$emulated_t, $d>>::undilated_max() as <Fixed<usize, $d> as DilationMethod>::Dilated);
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
                    use crate::{DilationMethod, DilatedInt, Undilate};
                    use super::super::{Fixed, DilateFixed};

                    #[test]
                    fn undilated_max_is_correct() {
                        assert_eq!(Fixed::<$t, $d>::UNDILATED_MAX, TestData::<Fixed<$t, $d>>::undilated_max());
                    }

                    #[test]
                    fn dilated_max_is_correct() {
                        assert_eq!(Fixed::<$t, $d>::DILATED_MAX, TestData::<Fixed<$t, $d>>::dilated_max());
                    }

                    // Unique to Fixed dilations
                    #[test]
                    #[should_panic(expected = "Attempting to dilate a value exceeds maximum (See DilationMethod::UNDILATED_MAX)")]
                    fn dilate_too_large_a_should_panic() {
                        if $d != 1 {
                            Fixed::<$t, $d>::dilate(TestData::<Fixed<$t, $d>>::undilated_max() + 1);
                        } else {
                            // D1 will never panic because the maximum dilatable value is equal to T::MAX
                            // So we'll hack a panic in here
                            panic!("Attempting to dilate a value exceeds maximum (See DilationMethod::UNDILATED_MAX)");
                        }
                    }

                    // Unique to Fixed dilations
                    #[test]
                    #[should_panic(expected = "Attempting to dilate a value exceeds maximum (See DilationMethod::UNDILATED_MAX)")]
                    fn dilate_too_large_b_should_panic() {
                        if $d != 1 {
                            (TestData::<Fixed<$t, $d>>::undilated_max() + 1).dilate_fixed::<$d>();
                        } else {
                            // D1 will never panic because the maximum dilatable value is equal to T::MAX
                            // So we'll hack a panic in here
                            panic!("Attempting to dilate a value exceeds maximum (See DilationMethod::UNDILATED_MAX)");
                        }
                    }

                    #[test]
                    fn dilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<Fixed<$t, $d>>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as <Fixed<$t, $d> as DilationMethod>::Dilated & TestData::<Fixed<$t, $d>>::dilated_max();
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
                                let dilated = (*dilated_a ^ *dilated_b) as <Fixed<$t, $d> as DilationMethod>::Dilated & TestData::<Fixed<$t, $d>>::dilated_max();
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
                            type DilatedT = <Fixed<$t, $d> as DilationMethod>::Dilated;
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
                            type DilatedT = <Fixed<$t, $d> as DilationMethod>::Dilated;
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
                            type DilatedT = <Fixed<$t, $d> as DilationMethod>::Dilated;
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
                            type DilatedT = <Fixed<$t, $d> as DilationMethod>::Dilated;
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
