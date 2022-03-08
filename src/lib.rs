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
//! use dilate::prelude::*;
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
//! use dilate::prelude::*;
//! 
//! let original: u8 = 0b1011;
//! 
//! let dilated = original.dilate_expand::<3>();
//! assert_eq!(dilated.0, 0b1000001001);
//! 
//! assert_eq!(dilated.undilate(), original);
//! ```

mod internal;
pub mod fixed;
pub mod expand;

pub mod prelude {
    pub use super::{Adapter, DilatedInt, Undilate};
    pub use crate::expand::{Expand, DilateExpand};
    pub use crate::fixed::{Fixed, DilateFixed};
}

use std::{fmt, marker::PhantomData, ops::{Add, Not, BitAnd, AddAssign, Sub, SubAssign}, num::Wrapping};
use internal::{build_fixed_undilated_max, build_dilated_mask};

/// Denotes an integer type supported by dilation and undilation methods
pub trait SupportedType: internal::DilateExplicit + internal::UndilateExplicit { }
impl SupportedType for u8 { }
impl SupportedType for u16 { }
impl SupportedType for u32 { }
impl SupportedType for u64 { }
impl SupportedType for u128 { }
impl SupportedType for usize { }

/// Dilation adapters allow for custom decoupled dilation behaviours
/// 
/// An adapter describes the method of dilation, including the inner and outer types involved, wrapper methods to forward to the appropriate dilation functions, and some useful constants.
/// 
/// There are currently two types of adapter impls: Expand and Fixed
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
    /// use dilate::prelude::*;
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
/// use dilate::prelude::*;
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
/// use dilate::prelude::*;
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
pub(crate) mod shared_test_data {
    use std::marker::PhantomData;

    use lazy_static::lazy_static;

    use super::Adapter;

    pub struct TestData<T> where T: Adapter {
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
    pub(crate) use impl_test_data;
    
    // NOTE - The following test cases are shared between all types (up to D8)
    //        For undilated values, we simply cast to the target type (and mask with undilated_max() for Fixed adapters)
    //        For dilated values, we cast to the target inner type and mask with dilated_max()
    //        This procedure ensures that the test data is 100% valid in all cases
    //        Furthermore, every test case is xor'd with every other test case to
    //        perform more tests with fewer hand written values
    lazy_static! {
        pub static ref DILATION_TEST_CASES: [Vec<(u128, u128)>; 9] = [
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
    pub const VALUES: [[u128; 32]; 9] = [
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
}
