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
//! in various forms as well as several efficient mathematical operations
//! between dilated integers.
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
//! The process of undilation, or 'undilateion', does the opposite:
//! * `0b1010001` D2-undilated becomes `0b1101`
//! * `0b1000001001` D3-undilated becomes `0b1011`
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
//! use dilate::{Adapter, DilateExpand, Undilate};
//! 
//! let original: u8 = 0b1101;
//! 
//! let dilated = original.dilate_expand::<2>();
//! assert_eq!(dilated.0, 0b1010001);
//! 
//! assert_eq!(dilated.undilate(), original);
//! ```
//! 
//! Example 3-Dilation Usage:
//! ```
//! use dilate::{Adapter, DilateExpand, Undilate};
//! 
//! let original: u8 = 0b1011;
//! 
//! let dilated = original.dilate_expand::<3>();
//! assert_eq!(dilated.0, 0b1000001001);
//! 
//! assert_eq!(dilated.undilate(), original);
//! ```

mod consts;
mod internal;

use std::{fmt, marker::PhantomData, ops::{Add, Not, BitAnd, AddAssign, Sub, SubAssign}, num::Wrapping};
use consts::{build_fixed_undilated_max, build_dilated_mask};

pub trait SupportedType: internal::DilateExplicit + internal::UndilateExplicit { }
impl SupportedType for u8 { }
impl SupportedType for u16 { }
impl SupportedType for u32 { }
impl SupportedType for u64 { }
impl SupportedType for u128 { }
impl SupportedType for usize { }

pub trait Adapter {
    type Outer: SupportedType;
    type Inner: SupportedType;
    const D: usize;
    const UNDILATED_BITS: usize;
    const UNDILATED_MAX: Self::Outer;
    const DILATED_BITS: usize;

    /// The `DILATED_MAX` constant holds a set of N dilated 1 bits, each
    /// separated by D-1 0 bits, where N is equal to DILATED_BITS.
    /// 
    /// When using Expand<T, D>, DILATED_MAX is the dilated equivalent
    /// of T::MAX.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use dilate::{Adapter, Expand, Fixed};
    /// 
    /// assert_eq!(Expand::<u8, 2>::DILATED_MAX, 0b0101010101010101);
    /// assert_eq!(Expand::<u16, 3>::DILATED_MAX, 0b001001001001001001001001001001001001001001001001);
    /// 
    /// assert_eq!(Fixed::<u8, 2>::DILATED_MAX, 0b01010101);
    /// assert_eq!(Fixed::<u16, 3>::DILATED_MAX, 0b0001001001001001);
    /// ```
    const DILATED_MAX: Self::Inner;

    fn dilate(value: Self::Outer) -> Self::Inner;
    fn undilate(value: Self::Inner) -> Self::Outer;
}

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
            const DILATED_MAX: Self::Inner = build_dilated_mask(Self::UNDILATED_BITS, $d) as Self::Inner;

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

#[derive(Debug, PartialEq, Eq)]
pub struct Fixed<T, const D: usize>(PhantomData<T>) where T: SupportedType;

macro_rules! impl_fixed {
    ($t:ty, $($d:literal),+) => {$(
        impl Adapter for Fixed<$t, $d> {
            type Outer = $t;
            type Inner = $t;
            const D: usize = $d;
            const UNDILATED_BITS: usize = <$t>::BITS as usize / $d;
            const UNDILATED_MAX: Self::Outer = build_fixed_undilated_max::<$t, $d>() as $t;
            const DILATED_BITS: usize = Self::UNDILATED_BITS * $d;
            const DILATED_MAX: Self::Inner = build_dilated_mask(Self::UNDILATED_BITS, $d) as Self::Inner;

            #[inline]
            fn dilate(value: Self::Outer) -> Self::Inner {
                internal::dilate::<Self::Inner, $d>(value)
            }

            #[inline]
            fn undilate(value: Self::Inner) -> Self::Outer {
                internal::undilate::<Self::Inner, $d>(value)
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
/// use dilate::{Adapter, DilateExpand, Undilate};
/// 
/// let original: u8 = 0b1101;
/// 
/// let dilated = original.dilate_expand::<2>();
/// assert_eq!(dilated.0, 0b1010001);
/// 
/// assert_eq!(dilated.undilate(), original);
/// ```
/// 
/// Example 3-Dilation Usage:
/// ```
/// use dilate::{Adapter, DilateExpand, Undilate};
/// 
/// let original: u8 = 0b1011;
/// 
/// let dilated = original.dilate_expand::<3>();
/// assert_eq!(dilated.0, 0b1000001001);
/// 
/// assert_eq!(dilated.undilate(), original);
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
pub struct DilatedInt<A>(pub A::Inner) where A: Adapter;

impl<A> fmt::Display for DilatedInt<A> where A: Adapter, A::Inner: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait DilateExpand: SupportedType {
    #[inline]
    fn dilate_expand<const D: usize>(self) -> DilatedInt<Expand<Self, D>> where Expand::<Self, D>: Adapter<Outer = Self> {
        DilatedInt::<Expand<Self, D>>(Expand::<Self, D>::dilate(self))
    }
}

impl<T> DilateExpand for T where T: SupportedType { }

pub trait DilateFixed: SupportedType {
    #[inline]
    fn dilate_fixed<const D: usize>(self) -> DilatedInt<Fixed<Self, D>> where Fixed::<Self, D>: Adapter<Outer = Self> {
        DilatedInt::<Fixed<Self, D>>(Fixed::<Self, D>::dilate(self))
    }
}

impl<T> DilateFixed for T where T: SupportedType { }

pub trait Undilate {
    type Output;

    fn undilate(self) -> Self::Output;
}

impl<A> Undilate for DilatedInt<A> where A: Adapter {
    type Output = A::Outer;

    fn undilate(self) -> Self::Output {
        A::undilate(self.0)
    }
}

// ============================================================================
// Arithmetic trait impls

impl<A> Add for DilatedInt<A>
where
    A: Adapter,
    A::Inner: Copy + Default + Not<Output = A::Inner> + BitAnd<Output = A::Inner>,
    Wrapping<A::Inner>: Add<Output = Wrapping<A::Inner>>
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self((Wrapping(self.0) + Wrapping(!A::DILATED_MAX) + Wrapping(rhs.0)).0 & A::DILATED_MAX)
    }
}

impl<A> AddAssign for DilatedInt<A>
where
    A: Adapter,
    A::Inner: Copy + Default + Not<Output = A::Inner> + BitAnd<Output = A::Inner>,
    Wrapping<A::Inner>: Add<Output = Wrapping<A::Inner>>
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = (Wrapping(self.0) + Wrapping(!A::DILATED_MAX) + Wrapping(rhs.0)).0 & A::DILATED_MAX;
    }
}

impl<A> Sub for DilatedInt<A>
where
    A: Adapter,
    A::Inner: Copy + Default + BitAnd<Output = A::Inner>,
    Wrapping<A::Inner>: Sub<Output = Wrapping<A::Inner>>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self((Wrapping(self.0) - Wrapping(rhs.0)).0 & A::DILATED_MAX)
    }
}

impl<A> SubAssign for DilatedInt<A>
where
    A: Adapter,
    A::Inner: Copy + Default + BitAnd<Output = A::Inner>,
    Wrapping<A::Inner>: Sub<Output = Wrapping<A::Inner>>
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = (Wrapping(self.0) - Wrapping(rhs.0)).0 & A::DILATED_MAX;
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use lazy_static::lazy_static;
    use paste::paste;

    use super::{Adapter, Expand, Fixed};

    // ========================================================================
    // SHARED DILATION TEST DATA
    
    struct TestData<T> where T: Adapter {
        marker: PhantomData<T>,
    }
    
    macro_rules! impl_test_data {
        ($adapter_t:ty, $dil_max:expr, $con_max:expr) => {
            impl TestData<$adapter_t> {
                #[inline]
                fn dilated_max() -> <$adapter_t as Adapter>::Inner {
                    $dil_max
                }

                #[inline]
                fn undilated_max() -> <$adapter_t as Adapter>::Outer {
                    $con_max
                }
            }
        };
    }
    
    // NOTE - The following test cases are shared between all types (up to D8)
    //        For undilated values, we simply cast to the target type (and mask with undilated_max() for Fixed adapters)
    //        For dilated values, we cast to the target inner type and mask with dilated_max()
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

    // ========================================================================
    // EXPAND DILATION TESTS
    
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
                    use super::{TestData, DILATION_UNDILATION_TEST_CASES, VALUES};
                    use super::super::{Adapter, Expand, DilatedInt, DilateExpand, Undilate};

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
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
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
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
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

    // ========================================================================
    // FIXED DILATION TESTS

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
            impl_test_data!(Fixed<usize, $d>, TestData::<Fixed<$emulated_t, $d>>::dilated_max() as <Fixed<usize, $d> as Adapter>::Inner, TestData::<Fixed<$emulated_t, $d>>::undilated_max() as <Fixed<usize, $d> as Adapter>::Inner);
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
                    use super::{TestData, DILATION_UNDILATION_TEST_CASES, VALUES};
                    use super::super::{Adapter, Fixed, DilatedInt, DilateFixed, Undilate};

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
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<Fixed<$t, $d>>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as <Fixed<$t, $d> as Adapter>::Inner & TestData::<Fixed<$t, $d>>::dilated_max();
                                assert_eq!(Fixed::<$t, $d>::dilate(undilated), dilated);
                                assert_eq!(undilated.dilate_fixed::<$d>(), DilatedInt::<Fixed<$t, $d>>(dilated));
                            }
                        }
                    }

                    #[test]
                    fn undilate_is_correct() {
                        // To create many more valid test cases, we doubly iterate all of them and xor the values
                        for (undilated_a, dilated_a) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                            for (undilated_b, dilated_b) in DILATION_UNDILATION_TEST_CASES[$d].iter() {
                                let undilated = (*undilated_a ^ *undilated_b) as $t & TestData::<Fixed<$t, $d>>::undilated_max();
                                let dilated = (*dilated_a ^ *dilated_b) as <Fixed<$t, $d> as Adapter>::Inner & TestData::<Fixed<$t, $d>>::dilated_max();
                                assert_eq!(Fixed::<$t, $d>::undilate(dilated), undilated);
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
                            type InnerT = <Fixed<$t, $d> as Adapter>::Inner;
                            assert_eq!(DilatedInt::<Fixed<$t, $d>>(*a as InnerT) + DilatedInt::<Fixed<$t, $d>>(*b as InnerT), DilatedInt::<Fixed<$t, $d>>(*ans as InnerT));
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
                            type InnerT = <Fixed<$t, $d> as Adapter>::Inner;
                            let mut assigned = DilatedInt::<Fixed<$t, $d>>(*a as InnerT);
                            assigned += DilatedInt::<Fixed<$t, $d>>(*b as InnerT);
                            assert_eq!(assigned, DilatedInt::<Fixed<$t, $d>>(*ans as InnerT));
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
                            type InnerT = <Fixed<$t, $d> as Adapter>::Inner;
                            assert_eq!(DilatedInt::<Fixed<$t, $d>>(*a as InnerT) - DilatedInt::<Fixed<$t, $d>>(*b as InnerT), DilatedInt::<Fixed<$t, $d>>(*ans as InnerT));
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
                            type InnerT = <Fixed<$t, $d> as Adapter>::Inner;
                            let mut assigned = DilatedInt::<Fixed<$t, $d>>(*a as InnerT);
                            assigned -= DilatedInt::<Fixed<$t, $d>>(*b as InnerT);
                            assert_eq!(assigned, DilatedInt::<Fixed<$t, $d>>(*ans as InnerT));
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
