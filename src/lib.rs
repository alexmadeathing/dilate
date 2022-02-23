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

// A Rust implementation of:

// [1] Converting to and from Dilated Integers
// By: Rajeev Raman and David S. Wise
// Permission has been granted to reproduce the agorithms within this paper

// [2] Integer Dilation and Contraction for Quadtrees and Octrees
// By: Leo Stocco and Gunther Schrack
// Permission has been granted to reproduce the agorithms within this paper

// [3] Fast Additions on Masked Integers
// By: Michael D Adams and David S Wise
// Permission has been granted to reproduce the agorithms within this paper

use std::{mem::size_of, num::Wrapping};

// Math ops
use std::ops::{Add, AddAssign, Sub, SubAssign, BitAnd, Not};

mod const_generation;
use const_generation::{
    build_dilated_mask, build_undilated_max, dilate_mask, dilate_max_round, dilate_mult,
    undilate_mask, undilate_max_round, undilate_mult, undilate_shift,
};

// NOTE Until we have stable specialization, D is limited to 1-8
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DilatedInt<T, const D: usize>(pub T);

// Dilated mask trait
pub trait DilatedMask<T> {
    fn dilated_mask() -> T;
}

macro_rules! dilated_int_mask_impls {
    ($($t:ty),+) => {$(
        impl<const D: usize> DilatedMask<$t> for DilatedInt<$t, D> {
            #[inline]
            fn dilated_mask() -> $t {
                build_dilated_mask(size_of::<$t>() * 8 / D, D) as $t
            }
        }
    )+}
}
dilated_int_mask_impls!(u8, u16, u32, u64, u128, usize);

// Undilated maximum value trait
pub trait UndilatedMax<T> {
    fn undilated_max() -> T;
}

macro_rules! dilated_int_undilated_max_impls {
    ($($t:ty),+) => {$(
        impl<const D: usize> UndilatedMax<$t> for DilatedInt<$t, D> {
            #[inline]
            fn undilated_max() -> $t {
                build_undilated_max::<$t, D>() as $t
            }
        }
    )+}
}
dilated_int_undilated_max_impls!(u8, u16, u32, u64, u128, usize);

// ============================================================================
// Implement From for D 1 dilated integers (no dilation - provided for easier
// compatibility with user systems that allow D 1 operations)
macro_rules! dilated_int_d1_from_impls {
    ($($t:ty),+) => {$(
        // From $t to dilated $t
        impl From<$t> for DilatedInt<$t, 1> {
            #[inline]
            fn from(value: $t) -> Self {
                Self(value)
            }
        }

        // From dilated $t to $t
        impl From<DilatedInt<$t, 1>> for $t {
            #[inline]
            fn from(dilated: DilatedInt<$t, 1>) -> Self {
                dilated.0
            }
        }
    )+}
}
dilated_int_d1_from_impls!(u8, u16, u32, u64, u128);

// ============================================================================
// Implement From for D 2 dilated integers
impl From<u8> for DilatedInt<u8, 2> {
    #[inline]
    fn from(value: u8) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [2]
        let mut v = value;
        v = (v | (v << 2)) & 0x33;
        v = (v | (v << 1)) & 0x55;
        Self(v)
    }
}

impl From<u16> for DilatedInt<u16, 2> {
    #[inline]
    fn from(value: u16) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [2]
        let mut v = value;
        v = (v | (v << 4)) & 0x0F0F;
        v = (v | (v << 2)) & 0x3333;
        v = (v | (v << 1)) & 0x5555;
        Self(v)
    }
}

impl From<u32> for DilatedInt<u32, 2> {
    #[inline]
    fn from(value: u32) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [2]
        let mut v = value;
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        Self(v)
    }
}

impl From<u64> for DilatedInt<u64, 2> {
    #[inline]
    fn from(value: u64) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [2]
        let mut v = value;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F;
        v = (v | (v << 2)) & 0x3333333333333333;
        v = (v | (v << 1)) & 0x5555555555555555;
        Self(v)
    }
}

impl From<u128> for DilatedInt<u128, 2> {
    #[inline]
    fn from(value: u128) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [2]
        let mut v = value;
        v = (v | (v << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333333333333333333333333333;
        v = (v | (v << 1)) & 0x55555555555555555555555555555555;
        Self(v)
    }
}

impl From<DilatedInt<u8, 2>> for u8 {
    #[inline]
    fn from(dilated: DilatedInt<u8, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x03) & Wrapping(0x66);
        v = v * Wrapping(0x05) & Wrapping(0x78);
        v.0 >> 3
    }
}

impl From<DilatedInt<u16, 2>> for u16 {
    #[inline]
    fn from(dilated: DilatedInt<u16, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x003) & Wrapping(0x6666);
        v = v * Wrapping(0x005) & Wrapping(0x7878);
        v = v * Wrapping(0x011) & Wrapping(0x7f80);
        v.0 >> 7
    }
}

impl From<DilatedInt<u32, 2>> for u32 {
    #[inline]
    fn from(dilated: DilatedInt<u32, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x00000003) & Wrapping(0x66666666);
        v = v * Wrapping(0x00000005) & Wrapping(0x78787878);
        v = v * Wrapping(0x00000011) & Wrapping(0x7F807F80);
        v = v * Wrapping(0x00000101) & Wrapping(0x7FFF8000);
        v.0 >> 15
    }
}

impl From<DilatedInt<u64, 2>> for u64 {
    #[inline]
    fn from(dilated: DilatedInt<u64, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x00003) & Wrapping(0x6666666666666666);
        v = v * Wrapping(0x00005) & Wrapping(0x7878787878787878);
        v = v * Wrapping(0x00011) & Wrapping(0x7F807F807F807F80);
        v = v * Wrapping(0x00101) & Wrapping(0x7FFF80007FFF8000);
        v = v * Wrapping(0x10001) & Wrapping(0x7FFFFFFF80000000);
        v.0 >> 31
    }
}

impl From<DilatedInt<u128, 2>> for u128 {
    #[inline]
    fn from(dilated: DilatedInt<u128, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x000000003) & Wrapping(0x66666666666666666666666666666666);
        v = v * Wrapping(0x000000005) & Wrapping(0x78787878787878787878787878787878);
        v = v * Wrapping(0x000000011) & Wrapping(0x7f807f807f807f807f807f807f807f80);
        v = v * Wrapping(0x000000101) & Wrapping(0x7fff80007fff80007fff80007fff8000);
        v = v * Wrapping(0x000010001) & Wrapping(0x7fffffff800000007fffffff80000000);
        v = v * Wrapping(0x100000001) & Wrapping(0x7fffffffffffffff8000000000000000);
        v.0 >> 63
    }
}

// ============================================================================
// Implement From for D 3 dilated integers
impl From<u8> for DilatedInt<u8, 3> {
    #[inline]
    fn from(value: u8) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [1]
        let mut v = Wrapping(value);
        v = (v * Wrapping(0x011)) & Wrapping(0xC3);
        v = (v * Wrapping(0x005)) & Wrapping(0x49);
        Self(v.0)
    }
}

impl From<u16> for DilatedInt<u16, 3> {
    #[inline]
    fn from(value: u16) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [1]
        let mut v = Wrapping(value);
        v = (v * Wrapping(0x101)) & Wrapping(0xF00F);
        v = (v * Wrapping(0x011)) & Wrapping(0x30C3);
        v = (v * Wrapping(0x005)) & Wrapping(0x9249);
        Self(v.0)
    }
}

impl From<u32> for DilatedInt<u32, 3> {
    #[inline]
    fn from(value: u32) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [1]
        let mut v = Wrapping(value);
        v = (v * Wrapping(0x10001)) & Wrapping(0xFF0000FF);
        v = (v * Wrapping(0x00101)) & Wrapping(0x0F00F00F);
        v = (v * Wrapping(0x00011)) & Wrapping(0xC30C30C3);
        v = (v * Wrapping(0x00005)) & Wrapping(0x49249249);
        Self(v.0)
    }
}

impl From<u64> for DilatedInt<u64, 3> {
    #[inline]
    fn from(value: u64) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [1]
        let mut v = Wrapping(value);
        v = (v * Wrapping(0x100000001)) & Wrapping(0xFFFF00000000FFFF);
        v = (v * Wrapping(0x000010001)) & Wrapping(0x00FF0000FF0000FF);
        v = (v * Wrapping(0x000000101)) & Wrapping(0xF00F00F00F00F00F);
        v = (v * Wrapping(0x000000011)) & Wrapping(0x30C30C30C30C30C3);
        v = (v * Wrapping(0x000000005)) & Wrapping(0x9249249249249249);
        Self(v.0)
    }
}

impl From<u128> for DilatedInt<u128, 3> {
    #[inline]
    fn from(value: u128) -> Self {
        // It does not make sense to dilate to a larger integer type as the
        // output integer size would depend on the number of dimensions
        debug_assert!(
            value <= Self::undilated_max(),
            "Paremeter 'value' exceeds maximum"
        );

        // See citation [1]
        let mut v = Wrapping(value);
        v = (v * Wrapping(0x10000000000000001)) & Wrapping(0xFFFFFFFF0000000000000000FFFFFFFF);
        v = (v * Wrapping(0x00000000100000001)) & Wrapping(0x0000FFFF00000000FFFF00000000FFFF);
        v = (v * Wrapping(0x00000000000010001)) & Wrapping(0xFF0000FF0000FF0000FF0000FF0000FF);
        v = (v * Wrapping(0x00000000000000101)) & Wrapping(0x0F00F00F00F00F00F00F00F00F00F00F);
        v = (v * Wrapping(0x00000000000000011)) & Wrapping(0xC30C30C30C30C30C30C30C30C30C30C3);
        v = (v * Wrapping(0x00000000000000005)) & Wrapping(0x49249249249249249249249249249249);
        Self(v.0)
    }
}

impl From<DilatedInt<u8, 3>> for u8 {
    #[inline]
    fn from(dilated: DilatedInt<u8, 3>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x15)) & Wrapping(0x0e);
        v.0 >> 2
    }
}

impl From<DilatedInt<u16, 3>> for u16 {
    #[inline]
    fn from(dilated: DilatedInt<u16, 3>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x0015)) & Wrapping(0x1c0e);
        v = (v * Wrapping(0x1041)) & Wrapping(0x1ff0);
        v.0 >> 8
    }
}

impl From<DilatedInt<u32, 3>> for u32 {
    #[inline]
    fn from(dilated: DilatedInt<u32, 3>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x00015)) & Wrapping(0x0E070381);
        v = (v * Wrapping(0x01041)) & Wrapping(0x0FF80001);
        v = (v * Wrapping(0x40001)) & Wrapping(0x0FFFFFFE);
        v.0 >> 18
    }
}

impl From<DilatedInt<u64, 3>> for u64 {
    fn from(dilated: DilatedInt<u64, 3>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x0000000000000015)) & Wrapping(0x1c0e070381c0e070);
        v = (v * Wrapping(0x0000000000001041)) & Wrapping(0x1ff00003fe00007f);
        v = (v * Wrapping(0x0000001000040001)) & Wrapping(0x1ffffffc00000000);
        v.0 >> 40
    }
}

impl From<DilatedInt<u128, 3>> for u128 {
    fn from(dilated: DilatedInt<u128, 3>) -> Self {
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x00000000000000000000000000000015))
            & Wrapping(0x0e070381c0e070381c0e070381c0e070);
        v = (v * Wrapping(0x00000000000000000000000000001041))
            & Wrapping(0x0ff80001ff00003fe00007fc0000ff80);
        v = (v * Wrapping(0x00000000000000000000001000040001))
            & Wrapping(0x0ffffffe00000000000007ffffff0000);
        v = (v * Wrapping(0x00001000000000000040000000000001))
            & Wrapping(0x0ffffffffffffffffffff80000000000);
        v.0 >> 82
    }
}

// ============================================================================
// Implement From for DN dilated integers
// Until we have stable specialization, this must be implemented manually
macro_rules! dilated_int_dn_from_impls {
    ($t:ty, $($d:literal),+) => {$(
        impl From<$t> for DilatedInt<$t, $d> {
            #[inline]
            fn from(value: $t) -> Self {
                // It does not make sense to dilate to a larger integer type as the
                // output integer size would depend on the number of dimensions
                debug_assert!(value <= Self::undilated_max(), "Paremeter 'value' exceeds maximum");

                let mut v = Wrapping(value);
                let mut i = 0;
                while i <= dilate_max_round::<$t, $d>() {
                    v = (v * Wrapping(dilate_mult::<$t, $d>(i) as $t)) & Wrapping(dilate_mask::<$t, $d>(i) as $t);
                    i += 1;
                }

                Self(v.0)
            }
        }

        impl From<DilatedInt<$t, $d>> for $t {
            #[inline]
            fn from(dilated: DilatedInt<$t, $d>) -> $t {
                let mut v = Wrapping(dilated.0);
                let mut i = 0;
                while i <= undilate_max_round::<$t, $d>() {
                    v = (v * Wrapping(undilate_mult::<$t, $d>(i) as $t)) & Wrapping(undilate_mask::<$t, $d>(i) as $t);
                    i += 1;
                }

                v.0 >> undilate_shift::<$t, $d>() as $t
            }
        }
    )+}
}

// D 1, 2, 3 cases handled separately
dilated_int_dn_from_impls!(u8, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u16, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u32, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u64, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u128, 4, 5, 6, 7, 8);

// ============================================================================

macro_rules! dilated_int_usize_from_impls {
    ($t:ty, $($d:literal),+) => {$(
        impl From<usize> for DilatedInt<usize, $d> {
            #[inline]
            fn from(value: usize) -> Self {
                Self(DilatedInt::<$t, $d>::from(value as $t).0 as usize)
            }
        }

        impl From<DilatedInt<usize, $d>> for usize {
            fn from(dilated: DilatedInt<usize, $d>) -> Self {
                <$t>::from(DilatedInt::<$t, $d>(dilated.0 as $t)) as usize
            }
        }
    )+}
}

// Bootstrap usize (16 bit) for any number of dimensions
#[cfg(target_pointer_width = "16")]
dilated_int_usize_from_impls!(u16, 1, 2, 3, 4, 5, 6, 7, 8);

// Bootstrap usize (32 bit) for any number of dimensions
#[cfg(target_pointer_width = "32")]
dilated_int_usize_from_impls!(u32, 1, 2, 3, 4, 5, 6, 7, 8);

// Bootstrap usize (64 bit) for any number of dimensions
#[cfg(target_pointer_width = "64")]
dilated_int_usize_from_impls!(u64, 1, 2, 3, 4, 5, 6, 7, 8);

// Bootstrap usize (64 bit) for any number of dimensions
#[cfg(target_pointer_width = "128")]
dilated_int_usize_from_impls!(u128, 1, 2, 3, 4, 5, 6, 7, 8);

// ============================================================================
// Arithmetic trait impls

impl<T, const D: usize> Add for DilatedInt<T, D>
where
    Self: DilatedMask<T>,
    T: Not<Output = T> + BitAnd<Output = T>,
    Wrapping<T>: Add<Output = Wrapping<T>>
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self((Wrapping(self.0) + Wrapping(!Self::dilated_mask()) + Wrapping(rhs.0)).0 & Self::dilated_mask())
    }
}

impl<T, const D: usize> AddAssign for DilatedInt<T, D>
where
    Self: DilatedMask<T>,
    T: Copy + Not<Output = T> + BitAnd<Output = T>,
    Wrapping<T>: Add<Output = Wrapping<T>>
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = (Wrapping(self.0) + Wrapping(!Self::dilated_mask()) + Wrapping(rhs.0)).0 & Self::dilated_mask();
    }
}

impl<T, const D: usize> Sub for DilatedInt<T, D>
where
    Self: DilatedMask<T>,
    T: BitAnd<Output = T>,
    Wrapping<T>: Sub<Output = Wrapping<T>>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self((Wrapping(self.0) - Wrapping(rhs.0)).0 & Self::dilated_mask())
    }
}

impl<T, const D: usize> SubAssign for DilatedInt<T, D>
where
    Self: DilatedMask<T>,
    T: Copy + BitAnd<Output = T>,
    Wrapping<T>: Sub<Output = Wrapping<T>>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = (Wrapping(self.0) - Wrapping(rhs.0)).0 & Self::dilated_mask();
    }
}

// ============================================================================
// Inner helper functions

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use lazy_static::lazy_static;
    use paste::paste;

    struct TestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_test_data {
        ($t:ty, $d:literal, $dilated_mask:expr, $undilated_max:expr) => {
            impl TestData<$t, $d> {
                #[inline]
                fn dilated_mask() -> $t {
                    $dilated_mask
                }

                #[inline]
                fn undilated_max() -> $t {
                    $undilated_max
                }
            }
        };
    }
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

    impl_test_data!(
        u128,
        1,
        0xffffffffffffffffffffffffffffffff,
        0xffffffffffffffffffffffffffffffff
    );
    impl_test_data!(
        u128,
        2,
        0x55555555555555555555555555555555,
        0x0000000000000000ffffffffffffffff
    );
    impl_test_data!(
        u128,
        3,
        0x09249249249249249249249249249249,
        0x0000000000000000000003ffffffffff
    );
    impl_test_data!(
        u128,
        4,
        0x11111111111111111111111111111111,
        0x000000000000000000000000ffffffff
    );
    impl_test_data!(
        u128,
        5,
        0x01084210842108421084210842108421,
        0x00000000000000000000000001ffffff
    );
    impl_test_data!(
        u128,
        6,
        0x01041041041041041041041041041041,
        0x000000000000000000000000001fffff
    );
    impl_test_data!(
        u128,
        7,
        0x00810204081020408102040810204081,
        0x0000000000000000000000000003ffff
    );
    impl_test_data!(
        u128,
        8,
        0x01010101010101010101010101010101,
        0x0000000000000000000000000000ffff
    );

    macro_rules! impl_dil_mask_usize {
        ($innert:ty, $($d:literal),+) => {$(
            impl_test_data!(usize, $d, TestData::<$innert, $d>::dilated_mask() as usize, TestData::<$innert, $d>::undilated_max() as usize);
        )+}
    }
    #[cfg(target_pointer_width = "16")]
    impl_dil_mask_usize!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    impl_dil_mask_usize!(u32, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "64")]
    impl_dil_mask_usize!(u64, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "128")]
    impl_dil_mask_usize!(u128, 1, 2, 3, 4, 5, 6, 7, 8);

    // NOTE - The following test cases are shared between all types
    //        The values are first cast to the target type, then masked with either:
    //            undilated_max() for undilated values
    //            dilated_mask() for dilated values
    //        This procedure ensures that the test data is 100% valid in all cases
    //        Furthermore, every test case is xor'd with every other test case to
    //        perform more tests with fewer hand written values
    lazy_static! {
        static ref DILATION_UNDILATION_TEST_CASES: [Vec<(u128, u128)>; 9] = [
            // D0 (not used)
            Vec::new(),

            // D1 (data should pass through unchanged)
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0xffffffffffffffffffffffffffffffff),
                (0x0000000000000000ffffffffffffffff, 0x0000000000000000ffffffffffffffff),
                (0x00000000ffffffff00000000ffffffff, 0x00000000ffffffff00000000ffffffff),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x0000ffff0000ffff0000ffff0000ffff),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00ff00ff00ff00ff00ff00ff00ff00ff),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f),
                (0x33333333333333333333333333333333, 0x33333333333333333333333333333333),
                (0x55555555555555555555555555555555, 0x55555555555555555555555555555555),
            ],

            // D2
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x55555555555555555555555555555555),
                (0x0000000000000000ffffffffffffffff, 0x55555555555555555555555555555555),
                (0x00000000ffffffff00000000ffffffff, 0x00000000000000005555555555555555),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x00000000555555550000000055555555),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00005555000055550000555500005555),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x00550055005500550055005500550055),
                (0x33333333333333333333333333333333, 0x05050505050505050505050505050505),
                (0x55555555555555555555555555555555, 0x11111111111111111111111111111111),
            ],

            // D3
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x09249249249249249249249249249249),
                (0x0000000000000000ffffffffffffffff, 0x09249249249249249249249249249249),
                (0x00000000ffffffff00000000ffffffff, 0x00000000249249249249249249249249),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x09249249000000000000249249249249),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00249249000000249249000000249249),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x09000249000249000249000249000249),
                (0x33333333333333333333333333333333, 0x09009009009009009009009009009009),
                (0x55555555555555555555555555555555, 0x01041041041041041041041041041041),
            ],

            // D4
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x11111111111111111111111111111111),
                (0x0000000000000000ffffffffffffffff, 0x11111111111111111111111111111111),
                (0x00000000ffffffff00000000ffffffff, 0x11111111111111111111111111111111),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x00000000000000001111111111111111),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00000000111111110000000011111111),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x00001111000011110000111100001111),
                (0x33333333333333333333333333333333, 0x00110011001100110011001100110011),
                (0x55555555555555555555555555555555, 0x01010101010101010101010101010101),
            ],

            // D5
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x01084210842108421084210842108421),
                (0x0000000000000000ffffffffffffffff, 0x01084210842108421084210842108421),
                (0x00000000ffffffff00000000ffffffff, 0x01084210842108421084210842108421),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x00000000000008421084210842108421),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00084210842100000000000842108421),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x01000000842100000084210000008421),
                (0x33333333333333333333333333333333, 0x01000210002100021000210002100021),
                (0x55555555555555555555555555555555, 0x01004010040100401004010040100401),
            ],

            // D6
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x01041041041041041041041041041041),
                (0x0000000000000000ffffffffffffffff, 0x01041041041041041041041041041041),
                (0x00000000ffffffff00000000ffffffff, 0x01041041041041041041041041041041),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x00000000041041041041041041041041),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x01041041000000000000041041041041),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x00041041000000041041000000041041),
                (0x33333333333333333333333333333333, 0x01000041000041000041000041000041),
                (0x55555555555555555555555555555555, 0x01001001001001001001001001001001),
            ],

            // D7
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x00810204081020408102040810204081),
                (0x0000000000000000ffffffffffffffff, 0x00810204081020408102040810204081),
                (0x00000000ffffffff00000000ffffffff, 0x00810204081020408102040810204081),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x00000204081020408102040810204081),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00810000000000000002040810204081),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x00810000000020408100000000204081),
                (0x33333333333333333333333333333333, 0x00810000081000008100000810000081),
                (0x55555555555555555555555555555555, 0x00010004001000400100040010004001),
            ],

            // D8
            vec![
                (0x00000000000000000000000000000000, 0x00000000000000000000000000000000),
                (0xffffffffffffffffffffffffffffffff, 0x01010101010101010101010101010101),
                (0x0000000000000000ffffffffffffffff, 0x01010101010101010101010101010101),
                (0x00000000ffffffff00000000ffffffff, 0x01010101010101010101010101010101),
                (0x0000ffff0000ffff0000ffff0000ffff, 0x01010101010101010101010101010101),
                (0x00ff00ff00ff00ff00ff00ff00ff00ff, 0x00000000000000000101010101010101),
                (0x0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f, 0x00000000010101010000000001010101),
                (0x33333333333333333333333333333333, 0x00000101000001010000010100000101),
                (0x55555555555555555555555555555555, 0x00010001000100010001000100010001),
            ],
        ];
    }

    // The first 32 values in each dimension
    // Used for testing arithmetic
    const VALUES: [[u128; 32]; 9] = [
        // D0 (not used)
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
        // D1
        [
            0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
            0x1f,
        ],
        // D2
        [
            0x0, 0x1, 0x4, 0x5, 0x10, 0x11, 0x14, 0x15, 0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54,
            0x55, 0x100, 0x101, 0x104, 0x105, 0x110, 0x111, 0x114, 0x115, 0x140, 0x141, 0x144,
            0x145, 0x150, 0x151, 0x154, 0x155,
        ],
        // D3
        [
            0x0, 0x1, 0x8, 0x9, 0x40, 0x41, 0x48, 0x49, 0x200, 0x201, 0x208, 0x209, 0x240, 0x241,
            0x248, 0x249, 0x1000, 0x1001, 0x1008, 0x1009, 0x1040, 0x1041, 0x1048, 0x1049, 0x1200,
            0x1201, 0x1208, 0x1209, 0x1240, 0x1241, 0x1248, 0x1249,
        ],
        // D4
        [
            0x0, 0x1, 0x10, 0x11, 0x100, 0x101, 0x110, 0x111, 0x1000, 0x1001, 0x1010, 0x1011,
            0x1100, 0x1101, 0x1110, 0x1111, 0x10000, 0x10001, 0x10010, 0x10011, 0x10100, 0x10101,
            0x10110, 0x10111, 0x11000, 0x11001, 0x11010, 0x11011, 0x11100, 0x11101, 0x11110,
            0x11111,
        ],
        // D5
        [
            0x0, 0x1, 0x20, 0x21, 0x400, 0x401, 0x420, 0x421, 0x8000, 0x8001, 0x8020, 0x8021,
            0x8400, 0x8401, 0x8420, 0x8421, 0x100000, 0x100001, 0x100020, 0x100021, 0x100400,
            0x100401, 0x100420, 0x100421, 0x108000, 0x108001, 0x108020, 0x108021, 0x108400,
            0x108401, 0x108420, 0x108421,
        ],
        // D6
        [
            0x0, 0x1, 0x40, 0x41, 0x1000, 0x1001, 0x1040, 0x1041, 0x40000, 0x40001, 0x40040,
            0x40041, 0x41000, 0x41001, 0x41040, 0x41041, 0x1000000, 0x1000001, 0x1000040,
            0x1000041, 0x1001000, 0x1001001, 0x1001040, 0x1001041, 0x1040000, 0x1040001, 0x1040040,
            0x1040041, 0x1041000, 0x1041001, 0x1041040, 0x1041041,
        ],
        // D7
        [
            0x0, 0x1, 0x80, 0x81, 0x4000, 0x4001, 0x4080, 0x4081, 0x200000, 0x200001, 0x200080,
            0x200081, 0x204000, 0x204001, 0x204080, 0x204081, 0x10000000, 0x10000001, 0x10000080,
            0x10000081, 0x10004000, 0x10004001, 0x10004080, 0x10004081, 0x10200000, 0x10200001,
            0x10200080, 0x10200081, 0x10204000, 0x10204001, 0x10204080, 0x10204081,
        ],
        // D8
        [
            0x0,
            0x1,
            0x100,
            0x101,
            0x10000,
            0x10001,
            0x10100,
            0x10101,
            0x1000000,
            0x1000001,
            0x1000100,
            0x1000101,
            0x1010000,
            0x1010001,
            0x1010100,
            0x1010101,
            0x100000000,
            0x100000001,
            0x100000100,
            0x100000101,
            0x100010000,
            0x100010001,
            0x100010100,
            0x100010101,
            0x101000000,
            0x101000001,
            0x101000100,
            0x101000101,
            0x101010000,
            0x101010001,
            0x101010100,
            0x101010101,
        ],
    ];

    macro_rules! integer_dilation_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< $t _d $d >] {
                    use super::{TestData, DILATION_UNDILATION_TEST_CASES, VALUES};
                    use super::super::{DilatedInt, DilatedMask, UndilatedMax};

                    #[test]
                    fn dilated_mask_correct() {
                        assert_eq!(DilatedInt::<$t, $d>::dilated_mask(), TestData::<$t, $d>::dilated_mask());
                    }

                    #[test]
                    fn undilated_max_correct() {
                        assert_eq!(DilatedInt::<$t, $d>::undilated_max(), TestData::<$t, $d>::undilated_max());
                    }

                    #[test]
                    #[should_panic(expected = "Paremeter 'value' exceeds maximum")]
                    fn from_int_too_large_panics() {
                        // D1 dilated ints have no max value
                        // This is a hack, but it means we can run the same tests for all D values
                        if $d != 1 {
                            let _ = DilatedInt::<$t, $d>::from(TestData::<$t, $d>::undilated_max() + 1);
                        } else {
                            panic!("Paremeter 'value' exceeds maximum");
                        }
                    }

                    #[test]
                    fn from_raw_int_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<$t, $d>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as $t & TestData::<$t, $d>::dilated_mask();

                                assert_eq!(DilatedInt::<$t, $d>::from(undilated).0, dilated);
                            }
                        }
                    }

                    #[test]
                    fn to_raw_int_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<$t, $d>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as $t & TestData::<$t, $d>::dilated_mask();

                                assert_eq!($t::from(DilatedInt::<$t, $d>(dilated)), undilated);
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
                            (TestData::<$t, $d>::dilated_mask() as u128, VALUES[$d][1], VALUES[$d][0]), // max + 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<$t, $d>::dilated_mask() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            assert_eq!(DilatedInt::<$t, $d>(*a as $t) + DilatedInt::<$t, $d>(*b as $t), DilatedInt::<$t, $d>(*ans as $t));
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
                            (TestData::<$t, $d>::dilated_mask() as u128, VALUES[$d][1], VALUES[$d][0]), // max += 1 = 0
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<$t, $d>::dilated_mask() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            let mut assigned = DilatedInt::<$t, $d>(*a as $t);
                            assigned += DilatedInt::<$t, $d>(*b as $t);
                            assert_eq!(assigned, DilatedInt::<$t, $d>(*ans as $t));
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
                            (VALUES[$d][0], VALUES[$d][1], TestData::<$t, $d>::dilated_mask() as u128), // 0 - 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<$t, $d>::dilated_mask() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            assert_eq!(DilatedInt::<$t, $d>(*a as $t) - DilatedInt::<$t, $d>(*b as $t), DilatedInt::<$t, $d>(*ans as $t));
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
                            (VALUES[$d][0], VALUES[$d][1], TestData::<$t, $d>::dilated_mask() as u128), // 0 -= 1 = max
                        ];

                        // Some formats won't support arithmetic (for example u8 D8)
                        // So we have to filter to ensure they support all numbers involved with a particular test case
                        let mask_u128 = TestData::<$t, $d>::dilated_mask() as u128;
                        for (a, b, ans) in test_cases.iter().filter(|(a, b, ans)| *a <= mask_u128 && *b <= mask_u128 && *ans <= mask_u128) {
                            let mut assigned = DilatedInt::<$t, $d>(*a as $t);
                            assigned -= DilatedInt::<$t, $d>(*b as $t);
                            assert_eq!(assigned, DilatedInt::<$t, $d>(*ans as $t));
                        }
                    }
                }
            }
        )+}
    }
    integer_dilation_tests!(u8, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(u32, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(u64, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(u128, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(usize, 1, 2, 3, 4, 5, 6, 7, 8);
}
