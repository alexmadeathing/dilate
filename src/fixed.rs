// Copyright (c) 2024 Alex Blunt
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// # References and Acknowledgments
// Many thanks to the authors of the following white papers:
// * [1] Converting to and from Dilated Integers - Rajeev Raman and David S. Wise
// * [2] Integer Dilation and Contraction for Quadtrees and Octrees - Leo Stocco and Gunther Schrack
// * [3] Fast Additions on Masked Integers - Michael D Adams and David S Wise
//
// Permission has been explicitly granted to reproduce the agorithms within each paper.

use crate::{internal, DilatableType, DilatedInt, DilationMethod};

/// A DilationMethod implementation which provides fixed dilation meta information
///
/// This trait implementation describes the types involved with a fixed type
/// dilation as well as some useful constants and wrapper methods which
/// actually perform the dilations.
///
/// Although this trait implementation provides the functions for performing
/// dilations, users should generally prefer to dilate via the [DilateExpand](crate::expand::DilateExpand)
/// trait and its [dilate_fixed()](DilateFixed::dilate_fixed()) method,
/// which is generally less verbose and therefore more user friendly.
///
/// # Examples
/// ```rust
/// use dilate::*;
///
/// assert_eq!(Fixed::<u16, 2>::UNDILATED_MAX, 255);
/// assert_eq!(Fixed::<u16, 2>::UNDILATED_BITS, 8);
///
/// assert_eq!(Fixed::<u16, 2>::DILATED_MAX, 0b0101010101010101);
/// assert_eq!(Fixed::<u16, 2>::DILATED_BITS, 16);
///
/// let original: u16 = 0b1101;
/// let dilated = Fixed::<u16, 2>::dilate(original);
///
/// assert_eq!(dilated.value(), 0b01010001);
/// assert_eq!(dilated, DilatedInt::<Fixed<u16, 2>>::new(0b01010001));
///
/// assert_eq!(Fixed::<u16, 2>::undilate(dilated), original);
/// ```
///
/// For more detailed information, see [dilate_fixed()](crate::fixed::DilateFixed::dilate_fixed())
#[derive(Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct Fixed<T, const D: usize>(core::marker::PhantomData<T>)
where
    T: DilatableType;

macro_rules! impl_fixed {
    ($t:ty, $($d:literal),+) => {$(
        impl DilationMethod for Fixed<$t, $d> {
            type Undilated = $t;
            type Dilated = $t;
            const D: usize = $d;
            const UNDILATED_BITS: usize = <$t>::BITS as usize / $d;
            const UNDILATED_MAX: Self::Undilated = internal::build_fixed_undilated_max::<$t, $d>() as $t;
            const DILATED_BITS: usize = Self::UNDILATED_BITS * $d;
            const DILATED_MAX: Self::Dilated = internal::build_dilated_mask(Self::UNDILATED_BITS, $d) as Self::Dilated;
            const DILATED_MASK: Self::Dilated = Self::DILATED_MAX * ((1 << $d) - 1);

            #[inline]
            fn to_dilated(undilated: Self::Undilated) -> Self::Dilated {
                undilated
            }

            #[inline]
            fn to_undilated(dilated: Self::Dilated) -> Self::Undilated {
                dilated
            }
        
            #[inline]
            fn dilate(value: Self::Undilated) -> DilatedInt<Self> {
                DilatedInt::<Self>(internal::dilate_implicit::<Self::Dilated, $d>(value))
            }

            #[inline]
            fn undilate(value: DilatedInt<Self>) -> Self::Undilated {
                internal::undilate_implicit::<Self::Dilated, $d>(value.0)
            }
        }
    )+}
}

impl_fixed!(u8, 2, 3, 4);
impl_fixed!(u16, 2, 3, 4, 5, 6, 7, 8);
impl_fixed!(u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_fixed!(u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_fixed!(u128, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
impl_fixed!(usize, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

/// A convenience trait for dilating integers using the [Fixed] [DilationMethod]
///
/// This trait is implemented by all supported integer types and provides a
/// convenient and human readable way to dilate integers. Simply call the
/// [DilateFixed::dilate_fixed()] method to perform the dilation.
pub trait DilateFixed: DilatableType {
    /// Dilates a subset of bits of the source integer into the same integer type
    ///
    /// Dilating using the fixed method dilates a subset of bits from the
    /// source integer into an integer of the same type, maximising the memory usage
    /// of a single type. This is useful when you want absolute control over the
    /// dilated type used and want to fit as many dilated bits into the dilated type
    /// as possible.
    ///
    /// The number of dilatable bits is known ahead of time and can be retrieved
    /// using the [Fixed::UNDILATED_BITS](crate::DilationMethod::UNDILATED_BITS)
    /// constant.
    ///
    /// # Panics
    /// When using the fixed method, attempting to dilate a value that
    /// would not fit into the same type will yield a panic. You may use
    /// [Fixed::<T, D>::UNDILATED_MAX](crate::DilationMethod::UNDILATED_MAX) (or
    /// the table below) to determine whether your value will dilate successfully.
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// let value: u16 = 0b1101;
    ///
    /// assert_eq!(value.dilate_fixed::<2>(), DilatedInt::<Fixed<u16, 2>>::new(0b01010001));
    /// assert_eq!(value.dilate_fixed::<2>().value(), 0b01010001);
    ///
    /// // Panics with large values
    /// assert!(std::panic::catch_unwind(|| 0xffffu16.dilate_fixed::<2>()).is_err());
    /// ```
    ///
    /// # Supported Fixed Dilations
    /// The following is a list of supported combinations of types `T`, dilation
    /// amounts `D`, and the maximum dilatable value. The source integer and the
    /// internal dilated integer types are the same for Fixed dilations.
    ///
    /// | T      | D   | Max Value                            |
    /// | ------ | --- | ------------------------------------ |
    /// | `u8`   | 2   | `0x0f`                               |
    /// | `u8`   | 3   | `0x03`                               |
    /// | `u8`   | 4   | `0x03`                               |
    /// | ...    | ... | ...                                  |
    /// | `u16`  | 2   | `0x00ff`                             |
    /// | `u16`  | 3   | `0x001f`                             |
    /// | `u16`  | 4   | `0x000f`                             |
    /// | `u16`  | 5   | `0x0007`                             |
    /// | `u16`  | 6   | `0x0003`                             |
    /// | `u16`  | 7   | `0x0003`                             |
    /// | `u16`  | 8   | `0x0003`                             |
    /// | ...    | ... | ...                                  |
    /// | `u32`  | 2   | `0x0000ffff`                         |
    /// | `u32`  | 3   | `0x000003ff`                         |
    /// | `u32`  | 4   | `0x000000ff`                         |
    /// | `u32`  | 5   | `0x0000003f`                         |
    /// | `u32`  | 6   | `0x0000001f`                         |
    /// | `u32`  | 7   | `0x0000000f`                         |
    /// | `u32`  | 8   | `0x0000000f`                         |
    /// | `u32`  | 9   | `0x00000007`                         |
    /// | `u32`  | 10  | `0x00000007`                         |
    /// | `u32`  | 11  | `0x00000003`                         |
    /// | `u32`  | 12  | `0x00000003`                         |
    /// | `u32`  | 13  | `0x00000003`                         |
    /// | `u32`  | 14  | `0x00000003`                         |
    /// | `u32`  | 15  | `0x00000003`                         |
    /// | `u32`  | 16  | `0x00000003`                         |
    /// | ...    | ... | ...                                  |
    /// | `u64`  | 2   | `0x00000000ffffffff`                 |
    /// | `u64`  | 3   | `0x00000000001fffff`                 |
    /// | `u64`  | 4   | `0x000000000000ffff`                 |
    /// | `u64`  | 5   | `0x0000000000000fff`                 |
    /// | `u64`  | 6   | `0x00000000000003ff`                 |
    /// | `u64`  | 7   | `0x00000000000001ff`                 |
    /// | `u64`  | 8   | `0x00000000000000ff`                 |
    /// | `u64`  | 9   | `0x000000000000007f`                 |
    /// | `u64`  | 10  | `0x000000000000003f`                 |
    /// | `u64`  | 11  | `0x000000000000001f`                 |
    /// | `u64`  | 12  | `0x000000000000001f`                 |
    /// | `u64`  | 13  | `0x000000000000000f`                 |
    /// | `u64`  | 14  | `0x000000000000000f`                 |
    /// | `u64`  | 15  | `0x000000000000000f`                 |
    /// | `u64`  | 16  | `0x000000000000000f`                 |
    /// | ...    | ... | ...                                  |
    /// | `u128` | 2   | `0x0000000000000000ffffffffffffffff` |
    /// | `u128` | 3   | `0x0000000000000000000003ffffffffff` |
    /// | `u128` | 4   | `0x000000000000000000000000ffffffff` |
    /// | `u128` | 5   | `0x00000000000000000000000001ffffff` |
    /// | `u128` | 6   | `0x000000000000000000000000001fffff` |
    /// | `u128` | 7   | `0x0000000000000000000000000003ffff` |
    /// | `u128` | 8   | `0x0000000000000000000000000000ffff` |
    /// | `u128` | 9   | `0x00000000000000000000000000003fff` |
    /// | `u128` | 10  | `0x00000000000000000000000000000fff` |
    /// | `u128` | 11  | `0x000000000000000000000000000007ff` |
    /// | `u128` | 12  | `0x000000000000000000000000000003ff` |
    /// | `u128` | 13  | `0x000000000000000000000000000001ff` |
    /// | `u128` | 14  | `0x000000000000000000000000000001ff` |
    /// | `u128` | 15  | `0x000000000000000000000000000000ff` |
    /// | `u128` | 16  | `0x000000000000000000000000000000ff` |
    ///
    /// Please note that usize is also supported and its behaviour is the same as the
    /// relevant integer type for your platform. For example, on a 32 bit system,
    /// usize is interpreted as a u32 and will have the same max dilatable value as u32.
    ///
    /// See also [Fixed<T, D>::dilate()](crate::DilationMethod::dilate())
    #[inline]
    fn dilate_fixed<const D: usize>(self) -> DilatedInt<Fixed<Self, D>>
    where
        Fixed<Self, D>: DilationMethod<Undilated = Self>,
    {
        Fixed::<Self, D>::dilate(self)
    }
}

impl<T> DilateFixed for T where T: DilatableType {}

#[cfg(test)]
mod tests {
    use paste::paste;

    use super::Fixed;
    use crate::{
        shared_test_data::{impl_test_data, TestData},
        DilationMethod,
    };

    // TODO move to a file
    impl_test_data!(Fixed<u8, 02>, 0x55, 0x0f);
    impl_test_data!(Fixed<u8, 03>, 0x09, 0x03);
    impl_test_data!(Fixed<u8, 04>, 0x11, 0x03);

    impl_test_data!(Fixed<u16, 2>, 0x5555, 0x00ff);
    impl_test_data!(Fixed<u16, 3>, 0x1249, 0x001f);
    impl_test_data!(Fixed<u16, 4>, 0x1111, 0x000f);
    impl_test_data!(Fixed<u16, 5>, 0x0421, 0x0007);
    impl_test_data!(Fixed<u16, 6>, 0x0041, 0x0003);
    impl_test_data!(Fixed<u16, 7>, 0x0081, 0x0003);
    impl_test_data!(Fixed<u16, 8>, 0x0101, 0x0003);

    impl_test_data!(Fixed<u32, 2>, 0x55555555, 0x0000ffff);
    impl_test_data!(Fixed<u32, 3>, 0x09249249, 0x000003ff);
    impl_test_data!(Fixed<u32, 4>, 0x11111111, 0x000000ff);
    impl_test_data!(Fixed<u32, 5>, 0x02108421, 0x0000003f);
    impl_test_data!(Fixed<u32, 6>, 0x01041041, 0x0000001f);
    impl_test_data!(Fixed<u32, 7>, 0x00204081, 0x0000000f);
    impl_test_data!(Fixed<u32, 8>, 0x01010101, 0x0000000f);

    impl_test_data!(Fixed<u64, 2>, 0x5555555555555555, 0x00000000ffffffff);
    impl_test_data!(Fixed<u64, 3>, 0x1249249249249249, 0x00000000001fffff);
    impl_test_data!(Fixed<u64, 4>, 0x1111111111111111, 0x000000000000ffff);
    impl_test_data!(Fixed<u64, 5>, 0x0084210842108421, 0x0000000000000fff);
    impl_test_data!(Fixed<u64, 6>, 0x0041041041041041, 0x00000000000003ff);
    impl_test_data!(Fixed<u64, 7>, 0x0102040810204081, 0x00000000000001ff);
    impl_test_data!(Fixed<u64, 8>, 0x0101010101010101, 0x00000000000000ff);

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
    impl_fixed_test_data_usize!(u16, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    impl_fixed_test_data_usize!(u32, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "64")]
    impl_fixed_test_data_usize!(u64, 2, 3, 4, 5, 6, 7, 8);

    macro_rules! impl_fixed_dilated_int_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< fixed_ $t _d $d >] {
                    extern crate std;

                    use crate::shared_test_data::{TestData, VALUES, dilation_test_cases};
                    use crate::{DilationMethod, DilatedInt};
                    use super::super::{Fixed, DilateFixed};

                    type DilationMethodT = Fixed<$t, $d>;
                    type DilatedIntT = DilatedInt<DilationMethodT>;
                    type DilatedT = <DilationMethodT as DilationMethod>::Dilated;
                    type UndilatedT = <DilationMethodT as DilationMethod>::Undilated;

                    #[test]
                    fn undilated_max_is_correct() {
                        assert_eq!(DilationMethodT::UNDILATED_MAX, TestData::<DilationMethodT>::undilated_max());
                    }

                    #[test]
                    fn dilated_max_is_correct() {
                        assert_eq!(DilationMethodT::DILATED_MAX, TestData::<DilationMethodT>::dilated_max());
                    }

                    #[test]
                    fn to_dilated_is_correct() {
                        assert_eq!(DilationMethodT::to_dilated(0), 0);
                        assert_eq!(DilationMethodT::to_dilated(1), 1);
                        assert_eq!(DilationMethodT::to_dilated(2), 2);
                        assert_eq!(DilationMethodT::to_dilated(3), 3);

                        // When using Fixed, moving from undilated to dilated is never lossy
                        assert_eq!(DilationMethodT::to_dilated(UndilatedT::MAX), DilatedT::MAX);
                    }

                    #[test]
                    fn to_undilated_is_correct() {
                        assert_eq!(DilationMethodT::to_undilated(0), 0);
                        assert_eq!(DilationMethodT::to_undilated(1), 1);
                        assert_eq!(DilationMethodT::to_undilated(2), 2);
                        assert_eq!(DilationMethodT::to_undilated(3), 3);

                        // When using Fixed, moving from dilated to undilated is never lossy
                        assert_eq!(DilationMethodT::to_undilated(DilatedT::MAX), UndilatedT::MAX);
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

                    // Unique to Fixed dilations
                    #[test]
                    #[should_panic(expected = "Attempting to dilate a value which exceeds maximum (See DilationMethod::UNDILATED_MAX)")]
                    fn dilate_too_large_a_should_panic() {
                        if $d != 1 {
                            DilationMethodT::dilate(TestData::<DilationMethodT>::undilated_max() + 1);
                        } else {
                            // D1 will never panic because the maximum dilatable value is equal to T::MAX
                            // So we'll hack a panic in here
                            panic!("Attempting to dilate a value which exceeds maximum (See DilationMethod::UNDILATED_MAX)");
                        }
                    }

                    // Unique to Fixed dilations
                    #[test]
                    #[should_panic(expected = "Attempting to dilate a value which exceeds maximum (See DilationMethod::UNDILATED_MAX)")]
                    fn dilate_too_large_b_should_panic() {
                        if $d != 1 {
                            (TestData::<DilationMethodT>::undilated_max() + 1).dilate_fixed::<$d>();
                        } else {
                            // D1 will never panic because the maximum dilatable value is equal to T::MAX
                            // So we'll hack a panic in here
                            panic!("Attempting to dilate a value which exceeds maximum (See DilationMethod::UNDILATED_MAX)");
                        }
                    }

                    #[test]
                    fn dilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in dilation_test_cases($d).iter() {
                            for (undilated_b, dilated_b) in dilation_test_cases($d).iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<DilationMethodT>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as DilatedT & TestData::<DilationMethodT>::dilated_max();
                                assert_eq!(DilationMethod::dilate(undilated), DilatedInt::<DilationMethodT>(dilated));
                                assert_eq!(undilated.dilate_fixed::<$d>(), DilatedInt::<DilationMethodT>(dilated));
                            }
                        }
                    }

                    #[test]
                    fn undilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in dilation_test_cases($d).iter() {
                            for (undilated_b, dilated_b) in dilation_test_cases($d).iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<DilationMethodT>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as DilatedT & TestData::<DilationMethodT>::dilated_max();
                                assert_eq!(DilationMethod::undilate(DilatedInt::<DilationMethodT>(dilated)), undilated);
                                assert_eq!(DilatedInt::<DilationMethodT>(dilated).undilate(), undilated);
                            }
                        }
                    }

                    #[test]
                    fn dilate_and_undilate() {
                        let max_undilated = (DilationMethodT::UNDILATED_MAX as usize).min(100000);
                        for i in 0..max_undilated {
                            assert_eq!(DilationMethodT::dilate(i as UndilatedT).undilate(), i as UndilatedT);
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
                            assert_eq!(DilatedInt::<DilationMethodT>(*a as DilatedT).add(DilatedInt::<DilationMethodT>(*b as DilatedT)), DilatedInt::<DilationMethodT>(*ans as DilatedT));
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
                            assigned.add_assign(DilatedInt::<DilationMethodT>(*b as DilatedT));
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
                            assert_eq!(DilatedInt::<DilationMethodT>(*a as DilatedT).sub(DilatedInt::<DilationMethodT>(*b as DilatedT)), DilatedInt::<DilationMethodT>(*ans as DilatedT));
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
                            assigned.sub_assign(DilatedInt::<DilationMethodT>(*b as DilatedT));
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
    impl_fixed_dilated_int_tests!(u8, 2, 3, 4);
    impl_fixed_dilated_int_tests!(u16, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(u32, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(u64, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(u128, 2, 3, 4, 5, 6, 7, 8);
    impl_fixed_dilated_int_tests!(usize, 2, 3, 4, 5, 6, 7, 8);
}
