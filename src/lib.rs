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

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![deny(rustdoc::invalid_rust_codeblocks)]

//! A compact, high performance integer dilation library for Rust.
//! 
//! This library provides efficient casting to and from dilated representation
//! as well as various efficient mathematical operations between dilated integers.
//! 
//! Integer dilation is the process of converting cartesian indices (eg.
//! coordinates) into a format suitable for use in D-dimensional [Morton
//! Order](https://en.wikipedia.org/wiki/Z-order_curve) bit sequences. The
//! dilation process takes an integer's bit sequence and inserts a number of 0
//! bits (`D - 1`) between each original bit successively. Thus, the original bit
//! sequence becomes evenly padded. For example:
//! * `0b1101` D2-dilated becomes `0b1010001` (values chosen arbitrarily)
//! * `0b1011` D3-dilated becomes `0b1000001001`
//! 
//! The process of undilation, or 'contraction', does the opposite:
//! * `0b1010001` D2-contracted becomes `0b1101`
//! * `0b1000001001` D3-contracted becomes `0b1011`
//! 
//! The size required to store the dilated version of an integer is determined
//! by the size in bits `S` of the integer multiplied by the dilation amount `D`.
//! Thus in all `D > 1` cases, the dilated version of a value will be stored using
//! a larger integer than T. Because the largest supported integer is u128, there
//! is a fixed upper limit of `S * D <= 128`. For example: `DilatedInt::<u32, 4>`
//! is valid because `32 * 4 = 128`. `DilatedInt::<u16, 10>` is not valid because
//! `16 * 10 > 128`. There are currently no plans to support larger types,
//! although the implementation is theoretically possible.
//! 
//! # Examples
//! 
//! Example 2-Dilation Usage:
//! ```
//! use dilate::DilatedInt;
//! 
//! let dilated = DilatedInt::<u32, 2>::from(0b1101);
//! assert_eq!(dilated.0, 0b1010001);
//! 
//! assert_eq!(u32::from(dilated), 0b1101);
//! ```
//! 
//! Example 3-Dilation Usage:
//! ```
//! use dilate::DilatedInt;
//! 
//! let dilated = DilatedInt::<u32, 3>::from(0b1011);
//! assert_eq!(dilated.0, 0b1000001001);
//! 
//! assert_eq!(u32::from(dilated), 0b1011);
//! ```

use std::fmt;
use std::{mem::size_of, num::Wrapping};

use std::ops::{Add, AddAssign, Sub, SubAssign, BitAnd, Not};

mod const_generation;
use const_generation::{
    build_dilated_mask, dilate_mask, dilate_max_round, dilate_mult,
    undilate_mask, undilate_max_round, undilate_mult, undilate_shift,
};

/// Determines inner type required to store D-dilated T values
pub trait Inner<const D: usize> {
    /// Inner type required to store D-dilated T values
    type Type: Copy + Eq + Ord + std::hash::Hash + Default + std::fmt::Debug;
}

macro_rules! impl_inner {
    ($outer:ty, $(($d:literal, $inner:ty)),+) => {$(
        impl Inner<$d> for $outer {
            type Type = $inner;
        }
    )+}
}

impl_inner!(u8, (1, u8), (2, u16), (3, u32), (4, u32), (5, u64), (6, u64), (7, u64), (8, u64), (9, u128), (10, u128), (11, u128), (12, u128), (13, u128), (14, u128), (15, u128), (16, u128));
impl_inner!(u16, (1, u16), (2, u32), (3, u64), (4, u64), (5, u128), (6, u128), (7, u128), (8, u128));
impl_inner!(u32, (1, u32), (2, u64), (3, u128), (4, u128));
impl_inner!(u64, (1, u64), (2, u128));
impl_inner!(u128, (1, u128));

#[cfg(target_pointer_width = "16")]
impl_inner!(usize, (1, u16), (2, u32), (3, u64), (4, u64), (5, u128), (6, u128), (7, u128), (8, u128));

#[cfg(target_pointer_width = "32")]
impl_inner!(usize, (1, u32), (2, u64), (3, u128), (4, u128));

#[cfg(target_pointer_width = "64")]
impl_inner!(usize, (1, u64), (2, u128));

#[cfg(target_pointer_width = "128")]
impl_inner!(usize, (1, u128));

/// The primary interface for dilating and undilating integers.
/// 
/// DilatedInt performs the dilation and undilation process via the standard Rust
/// From trait. To dilate an integer, you use DilatedInt's implementation of
/// the `from()` method. The resulting tuple struct contains the dilated integer,
/// the value which may be obtained via the tuple member `.0`. To convert back to
/// a regular integer, you use the integer's implementation of the `from()`
/// method, passing the DilatedInt as a parameter.
/// 
/// # Examples
/// 
/// Example 2-Dilation Usage:
/// ```
/// use dilate::DilatedInt;
/// 
/// let dilated = DilatedInt::<u32, 2>::from(0b1101);
/// assert_eq!(dilated.0, 0b1010001);
/// 
/// assert_eq!(u32::from(dilated), 0b1101);
/// ```
/// 
/// Example 3-Dilation Usage:
/// ```
/// use dilate::DilatedInt;
/// 
/// let dilated = DilatedInt::<u32, 3>::from(0b1011);
/// assert_eq!(dilated.0, 0b1000001001);
/// 
/// assert_eq!(u32::from(dilated), 0b1011);
/// ```
// 
// NOTE - Not exposing this to docs yet as example is quite involved
// Whilst the application of dilated integers are not limited to [Morton
// Order](https://en.wikipedia.org/wiki/Z-order_curve) bit sequences, they
// are an ideal candidate.
// To dilate a set of cartesian indices and produce a Morton encoded integer,
// you may use bit shift and or operators to combine multiple dilations:
// ```
// let x_dilated = DilatedInt::<u32, 3>::from(123);
// let y_dilated = DilatedInt::<u32, 3>::from(456);
// let z_dilated = DilatedInt::<u32, 3>::from(789);
// 
// let morton_encoded = (x_dilated.0 << 0) | (y_dilated.0 << 1) | (z_dilated.0 << 2);
// 
// assert_eq!(u32::from(DilatedInt::<u32, 3>((morton_encoded >> 0) & DilatedInt::<u32, 3>::dilated_mask())), 123);
// assert_eq!(u32::from(DilatedInt::<u32, 3>((morton_encoded >> 1) & DilatedInt::<u32, 3>::dilated_mask())), 456);
// assert_eq!(u32::from(DilatedInt::<u32, 3>((morton_encoded >> 2) & DilatedInt::<u32, 3>::dilated_mask())), 789);
// ```
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DilatedInt<T, const D: usize>(pub <T as Inner<D>>::Type) where T: Inner<D>;

impl<T, const D: usize> fmt::Display for DilatedInt<T, D> where T: Inner<D>, <T as Inner<D>>::Type: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Provides access to the mask of dilated bits
/// 
/// The DilatedMask trait is implemented for DilatedInt for all supported types
/// and all values of D.
pub trait DilatedMask<T, const D: usize> where T: Inner<D> {
    /// The `dilated_mask()` function returns a set of dilated N 1 bits, each
    /// separated by D-1 0 bits, where N is equal to T::BITS. It is the dilated
    /// equivalent of T::MAX.
    /// This function is zero cost and will most likely be reduced by the compiler to
    /// a single constant value.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use dilate::{DilatedInt, DilatedMask};
    /// 
    /// assert_eq!(DilatedInt::<u8, 2>::dilated_mask(), 0b0101010101010101);
    /// assert_eq!(DilatedInt::<u16, 3>::dilated_mask(), 0b001001001001001001001001001001001001001001001001);
    /// ```
    fn dilated_mask() -> <T as Inner<D>>::Type;
}

macro_rules! dilated_int_mask_impls {
    ($t:ty, $($d:literal),+) => {$(
        impl DilatedMask<$t, $d> for DilatedInt<$t, $d> {
            #[inline]
            fn dilated_mask() -> <$t as Inner<$d>>::Type {
                build_dilated_mask(size_of::<$t>() * 8, $d) as <$t as Inner<$d>>::Type
            }
        }
    )+}
}
dilated_int_mask_impls!(u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
dilated_int_mask_impls!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
dilated_int_mask_impls!(u32, 1, 2, 3, 4);
dilated_int_mask_impls!(u64, 1, 2);
dilated_int_mask_impls!(u128, 1);
#[cfg(target_pointer_width = "16")]
dilated_int_mask_impls!(usize, 1, 2, 3, 4, 5, 6, 7, 8);
#[cfg(target_pointer_width = "32")]
dilated_int_mask_impls!(usize, 1, 2, 3, 4);
#[cfg(target_pointer_width = "64")]
dilated_int_mask_impls!(usize, 1, 2);
#[cfg(target_pointer_width = "128")]
dilated_int_mask_impls!(usize, 1);

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
        // See citation [2]
        let mut v = value as u16;
        v = (v | (v << 4)) & 0x0F0F;
        v = (v | (v << 2)) & 0x3333;
        v = (v | (v << 1)) & 0x5555;
        Self(v)
    }
}

impl From<u16> for DilatedInt<u16, 2> {
    #[inline]
    fn from(value: u16) -> Self {
        // See citation [2]
        let mut v = value as u32;
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        Self(v)
    }
}

impl From<u32> for DilatedInt<u32, 2> {
    #[inline]
    fn from(value: u32) -> Self {
        // See citation [2]
        let mut v = value as u64;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF;
        v = (v | (v << 08)) & 0x00FF00FF00FF00FF;
        v = (v | (v << 04)) & 0x0F0F0F0F0F0F0F0F;
        v = (v | (v << 02)) & 0x3333333333333333;
        v = (v | (v << 01)) & 0x5555555555555555;
        Self(v)
    }
}

impl From<u64> for DilatedInt<u64, 2> {
    #[inline]
    fn from(value: u64) -> Self {
        // See citation [2]
        let mut v = value as u128;
        v = (v | (v << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
        v = (v | (v << 08)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
        v = (v | (v << 04)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
        v = (v | (v << 02)) & 0x33333333333333333333333333333333;
        v = (v | (v << 01)) & 0x55555555555555555555555555555555;
        Self(v)
    }
}

impl From<DilatedInt<u8, 2>> for u8 {
    #[inline]
    fn from(dilated: DilatedInt<u8, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x003) & Wrapping(0x6666);
        v = v * Wrapping(0x005) & Wrapping(0x7878);
        v = v * Wrapping(0x011) & Wrapping(0x7f80);
        (v.0 >> 7) as u8
    }
}

impl From<DilatedInt<u16, 2>> for u16 {
    #[inline]
    fn from(dilated: DilatedInt<u16, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x00000003) & Wrapping(0x66666666);
        v = v * Wrapping(0x00000005) & Wrapping(0x78787878);
        v = v * Wrapping(0x00000011) & Wrapping(0x7F807F80);
        v = v * Wrapping(0x00000101) & Wrapping(0x7FFF8000);
        (v.0 >> 15) as u16
    }
}

impl From<DilatedInt<u32, 2>> for u32 {
    #[inline]
    fn from(dilated: DilatedInt<u32, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x00003) & Wrapping(0x6666666666666666);
        v = v * Wrapping(0x00005) & Wrapping(0x7878787878787878);
        v = v * Wrapping(0x00011) & Wrapping(0x7F807F807F807F80);
        v = v * Wrapping(0x00101) & Wrapping(0x7FFF80007FFF8000);
        v = v * Wrapping(0x10001) & Wrapping(0x7FFFFFFF80000000);
        (v.0 >> 31) as u32
    }
}

impl From<DilatedInt<u64, 2>> for u64 {
    #[inline]
    fn from(dilated: DilatedInt<u64, 2>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = v * Wrapping(0x000000003) & Wrapping(0x66666666666666666666666666666666);
        v = v * Wrapping(0x000000005) & Wrapping(0x78787878787878787878787878787878);
        v = v * Wrapping(0x000000011) & Wrapping(0x7f807f807f807f807f807f807f807f80);
        v = v * Wrapping(0x000000101) & Wrapping(0x7fff80007fff80007fff80007fff8000);
        v = v * Wrapping(0x000010001) & Wrapping(0x7fffffff800000007fffffff80000000);
        v = v * Wrapping(0x100000001) & Wrapping(0x7fffffffffffffff8000000000000000);
        (v.0 >> 63) as u64
    }
}

// ============================================================================
// Implement From for D 3 dilated integers
impl From<u8> for DilatedInt<u8, 3> {
    #[inline]
    fn from(value: u8) -> Self {
        // See citation [1]
        let mut v = Wrapping(value as u32);
        v = (v * Wrapping(0x10001)) & Wrapping(0xFF0000FF);
        v = (v * Wrapping(0x00101)) & Wrapping(0x0F00F00F);
        v = (v * Wrapping(0x00011)) & Wrapping(0xC30C30C3);
        v = (v * Wrapping(0x00005)) & Wrapping(0x49249249);
        Self(v.0)
    }
}

impl From<u16> for DilatedInt<u16, 3> {
    #[inline]
    fn from(value: u16) -> Self {
        // See citation [1]
        let mut v = Wrapping(value as u64);
        v = (v * Wrapping(0x100000001)) & Wrapping(0xFFFF00000000FFFF);
        v = (v * Wrapping(0x000010001)) & Wrapping(0x00FF0000FF0000FF);
        v = (v * Wrapping(0x000000101)) & Wrapping(0xF00F00F00F00F00F);
        v = (v * Wrapping(0x000000011)) & Wrapping(0x30C30C30C30C30C3);
        v = (v * Wrapping(0x000000005)) & Wrapping(0x9249249249249249);
        Self(v.0)
    }
}

impl From<u32> for DilatedInt<u32, 3> {
    #[inline]
    fn from(value: u32) -> Self {
        // See citation [1]
        let mut v = Wrapping(value as u128);
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
        v = (v * Wrapping(0x00015)) & Wrapping(0x0E070381);
        v = (v * Wrapping(0x01041)) & Wrapping(0x0FF80001);
        v = (v * Wrapping(0x40001)) & Wrapping(0x0FFFFFFE);
        (v.0 >> 18) as u8
    }
}

impl From<DilatedInt<u16, 3>> for u16 {
    #[inline]
    fn from(dilated: DilatedInt<u16, 3>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x0000000000000015)) & Wrapping(0x1c0e070381c0e070);
        v = (v * Wrapping(0x0000000000001041)) & Wrapping(0x1ff00003fe00007f);
        v = (v * Wrapping(0x0000001000040001)) & Wrapping(0x1ffffffc00000000);
        (v.0 >> 40) as u16
    }
}

impl From<DilatedInt<u32, 3>> for u32 {
    #[inline]
    fn from(dilated: DilatedInt<u32, 3>) -> Self {
        // See citation [1]
        let mut v = Wrapping(dilated.0);
        v = (v * Wrapping(0x00000000000000000000000000000015))
            & Wrapping(0x0e070381c0e070381c0e070381c0e070);
        v = (v * Wrapping(0x00000000000000000000000000001041))
            & Wrapping(0x0ff80001ff00003fe00007fc0000ff80);
        v = (v * Wrapping(0x00000000000000000000001000040001))
            & Wrapping(0x0ffffffe00000000000007ffffff0000);
        v = (v * Wrapping(0x00001000000000000040000000000001))
            & Wrapping(0x0ffffffffffffffffffff80000000000);
        (v.0 >> 82) as u32
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
                type InnerT = <$t as Inner<$d>>::Type;
                let mut v = Wrapping(value as InnerT);
                let mut i = 0;
                while i <= dilate_max_round::<InnerT, $d>() {
                    v = (v * Wrapping(dilate_mult::<InnerT, $d>(i) as InnerT)) & Wrapping(dilate_mask::<InnerT, $d>(i) as InnerT);
                    i += 1;
                }

                Self(v.0)
            }
        }

        impl From<DilatedInt<$t, $d>> for $t {
            #[inline]
            fn from(dilated: DilatedInt<$t, $d>) -> $t {
                type InnerT = <$t as Inner<$d>>::Type;
                let mut v = Wrapping(dilated.0);
                let mut i = 0;
                while i <= undilate_max_round::<InnerT, $d>() {
                    v = (v * Wrapping(undilate_mult::<InnerT, $d>(i) as InnerT)) & Wrapping(undilate_mask::<InnerT, $d>(i) as InnerT);
                    i += 1;
                }

                (v.0 >> undilate_shift::<InnerT, $d>() as InnerT) as $t
            }
        }
    )+}
}

// D 1, 2, 3 cases handled separately
dilated_int_dn_from_impls!(u8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
dilated_int_dn_from_impls!(u16, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u32, 4);

// ============================================================================

macro_rules! dilated_int_usize_from_impls {
    ($t:ty, $($d:literal),+) => {$(
        impl From<usize> for DilatedInt<usize, $d> {
            #[inline]
            fn from(value: usize) -> Self {
                Self(DilatedInt::<$t, $d>::from(value as $t).0 as <usize as Inner<$d>>::Type)
            }
        }

        impl From<DilatedInt<usize, $d>> for usize {
            fn from(dilated: DilatedInt<usize, $d>) -> Self {
                <$t>::from(DilatedInt::<$t, $d>(dilated.0 as <$t as Inner<$d>>::Type)) as usize
            }
        }
    )+}
}

// Bootstrap usize (16 bit)
#[cfg(target_pointer_width = "16")]
dilated_int_usize_from_impls!(u16, 1, 2, 3, 4, 5, 6, 7, 8);

// Bootstrap usize (32 bit)
#[cfg(target_pointer_width = "32")]
dilated_int_usize_from_impls!(u32, 1, 2, 3, 4);

// Bootstrap usize (64 bit)
#[cfg(target_pointer_width = "64")]
dilated_int_usize_from_impls!(u64, 1, 2);

// Bootstrap usize (128 bit) - Is this even worth it?
#[cfg(target_pointer_width = "128")]
dilated_int_usize_from_impls!(u128, 1);

// ============================================================================
// Arithmetic trait impls

impl<T, const D: usize> Add for DilatedInt<T, D>
where
    Self: DilatedMask<T, D>,
    T: Inner<D>,
    <T as Inner<D>>::Type: Copy + Default + Not<Output = <T as Inner<D>>::Type> + BitAnd<Output = <T as Inner<D>>::Type>,
    Wrapping<<T as Inner<D>>::Type>: Add<Output = Wrapping<<T as Inner<D>>::Type>>
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self((Wrapping(self.0) + Wrapping(!Self::dilated_mask()) + Wrapping(rhs.0)).0 & Self::dilated_mask())
    }
}

impl<T, const D: usize> AddAssign for DilatedInt<T, D>
where
    Self: DilatedMask<T, D>,
    T: Inner<D>,
    <T as Inner<D>>::Type: Copy + Default + Not<Output = <T as Inner<D>>::Type> + BitAnd<Output = <T as Inner<D>>::Type>,
    Wrapping<<T as Inner<D>>::Type>: Add<Output = Wrapping<<T as Inner<D>>::Type>>
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = (Wrapping(self.0) + Wrapping(!Self::dilated_mask()) + Wrapping(rhs.0)).0 & Self::dilated_mask();
    }
}

impl<T, const D: usize> Sub for DilatedInt<T, D>
where
    Self: DilatedMask<T, D>,
    T: Inner<D>,
    <T as Inner<D>>::Type: Copy + Default + BitAnd<Output = <T as Inner<D>>::Type>,
    Wrapping<<T as Inner<D>>::Type>: Sub<Output = Wrapping<<T as Inner<D>>::Type>>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self((Wrapping(self.0) - Wrapping(rhs.0)).0 & Self::dilated_mask())
    }
}

impl<T, const D: usize> SubAssign for DilatedInt<T, D>
where
    Self: DilatedMask<T, D>,
    T: Inner<D>,
    <T as Inner<D>>::Type: Copy + Default + BitAnd<Output = <T as Inner<D>>::Type>,
    Wrapping<<T as Inner<D>>::Type>: Sub<Output = Wrapping<<T as Inner<D>>::Type>>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = (Wrapping(self.0) - Wrapping(rhs.0)).0 & Self::dilated_mask();
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use lazy_static::lazy_static;
    use paste::paste;

    use super::Inner;

    struct TestData<T, const D: usize> {
        marker: PhantomData<T>,
    }
    
    macro_rules! impl_test_data {
        ($t:ty, $d:literal, $dilated_mask:expr) => {
            impl TestData<$t, $d> {
                #[inline]
                fn dilated_mask() -> <$t as Inner<$d>>::Type {
                    $dilated_mask
                }
            }
        };
    }
    impl_test_data!(u8, 01, 0xff);
    impl_test_data!(u8, 02, 0x5555);
    impl_test_data!(u8, 03, 0x00249249);
    impl_test_data!(u8, 04, 0x11111111);
    impl_test_data!(u8, 05, 0x0000000842108421);
    impl_test_data!(u8, 06, 0x0000041041041041);
    impl_test_data!(u8, 07, 0x0002040810204081);
    impl_test_data!(u8, 08, 0x0101010101010101);
    // If testing up to 16 dimensions, these can be uncommented
//    impl_test_data!(u8, 09, 0x00000000000000008040201008040201);
//    impl_test_data!(u8, 10, 0x00000000000000401004010040100401);
//    impl_test_data!(u8, 11, 0x00000000000020040080100200400801);
//    impl_test_data!(u8, 12, 0x00000000001001001001001001001001);
//    impl_test_data!(u8, 13, 0x00000000080040020010008004002001);
//    impl_test_data!(u8, 14, 0x00000004001000400100040010004001);
//    impl_test_data!(u8, 15, 0x00000200040008001000200040008001);
//    impl_test_data!(u8, 16, 0x00010001000100010001000100010001);

    impl_test_data!(u16, 1, 0xffff);
    impl_test_data!(u16, 2, 0x55555555);
    impl_test_data!(u16, 3, 0x0000249249249249);
    impl_test_data!(u16, 4, 0x1111111111111111);
    impl_test_data!(u16, 5, 0x00000000000008421084210842108421);
    impl_test_data!(u16, 6, 0x00000000041041041041041041041041);
    impl_test_data!(u16, 7, 0x00000204081020408102040810204081);
    impl_test_data!(u16, 8, 0x01010101010101010101010101010101);

    impl_test_data!(u32, 1, 0xffffffff);
    impl_test_data!(u32, 2, 0x5555555555555555);
    impl_test_data!(u32, 3, 0x00000000249249249249249249249249);
    impl_test_data!(u32, 4, 0x11111111111111111111111111111111);

    impl_test_data!(u64, 1, 0xffffffffffffffff);
    impl_test_data!(u64, 2, 0x55555555555555555555555555555555);

    impl_test_data!(u128, 1, 0xffffffffffffffffffffffffffffffff);

    macro_rules! impl_test_data_usize {
        ($innert:ty, $($d:literal),+) => {$(
            impl_test_data!(usize, $d, TestData::<$innert, $d>::dilated_mask() as <usize as Inner<$d>>::Type);
        )+}
    }
    #[cfg(target_pointer_width = "16")]
    impl_test_data_usize!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    impl_test_data_usize!(u32, 1, 2, 3, 4);
    #[cfg(target_pointer_width = "64")]
    impl_test_data_usize!(u64, 1, 2);
    #[cfg(target_pointer_width = "128")]
    impl_test_data_usize!(u128, 1);

    // NOTE - The following test cases are shared between all types (up to D8)
    //        For undilated values, we simply cast to the target type
    //        For dilated values, we cast to the target inner type and mask with dilated_mask()
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

    // The first 32 values in each dimension (up to D8)
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
                    use super::super::{Inner, DilatedInt, DilatedMask};

                    #[test]
                    fn dilated_mask_correct() {
                        assert_eq!(DilatedInt::<$t, $d>::dilated_mask(), TestData::<$t, $d>::dilated_mask());
                    }

                    #[test]
                    fn from_raw_int_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t;
                                let dilated = (*dilated_a ^ *dilated_b) as <$t as Inner<$d>>::Type & TestData::<$t, $d>::dilated_mask();

                                assert_eq!(DilatedInt::<$t, $d>::from(undilated).0, dilated);
                            }
                        }
                    }

                    #[test]
                    fn to_raw_int_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t;
                                let dilated = (*dilated_a ^ *dilated_b) as <$t as Inner<$d>>::Type & TestData::<$t, $d>::dilated_mask();

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
                            type InnerT = <$t as Inner<$d>>::Type;
                            assert_eq!(DilatedInt::<$t, $d>(*a as InnerT) + DilatedInt::<$t, $d>(*b as InnerT), DilatedInt::<$t, $d>(*ans as InnerT));
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
                            type InnerT = <$t as Inner<$d>>::Type;
                            let mut assigned = DilatedInt::<$t, $d>(*a as InnerT);
                            assigned += DilatedInt::<$t, $d>(*b as InnerT);
                            assert_eq!(assigned, DilatedInt::<$t, $d>(*ans as InnerT));
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
                            type InnerT = <$t as Inner<$d>>::Type;
                            assert_eq!(DilatedInt::<$t, $d>(*a as InnerT) - DilatedInt::<$t, $d>(*b as InnerT), DilatedInt::<$t, $d>(*ans as InnerT));
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
                            type InnerT = <$t as Inner<$d>>::Type;
                            let mut assigned = DilatedInt::<$t, $d>(*a as InnerT);
                            assigned -= DilatedInt::<$t, $d>(*b as InnerT);
                            assert_eq!(assigned, DilatedInt::<$t, $d>(*ans as InnerT));
                        }
                    }
                }
            }
        )+}
    }
    // Technically, u8 can go up to 16 dimensions, but that would double the amount of inline test data
    integer_dilation_tests!(u8, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(u16, 1, 2, 3, 4, 5, 6, 7, 8);
    integer_dilation_tests!(u32, 1, 2, 3, 4);
    integer_dilation_tests!(u64, 1, 2);
    integer_dilation_tests!(u128, 1);

    #[cfg(target_pointer_width = "16")]
    integer_dilation_tests!(usize, 1, 2, 3, 4, 5, 6, 7, 8);
    #[cfg(target_pointer_width = "32")]
    integer_dilation_tests!(usize, 1, 2, 3, 4);
    #[cfg(target_pointer_width = "64")]
    integer_dilation_tests!(usize, 1, 2);
    #[cfg(target_pointer_width = "128")]
    integer_dilation_tests!(usize, 1);
}
