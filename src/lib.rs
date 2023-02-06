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

#![no_std]
#![warn(missing_docs)]
#![deny(rustdoc::invalid_rust_codeblocks)]

//! A compact, high performance integer dilation library for Rust.
//!
//! Integer dilation is the process of converting cartesian indices (eg.
//! coordinates) into a format suitable for use in D-dimensional algorithms
//! such [Morton Order](https://en.wikipedia.org/wiki/Z-order_curve) curves.
//! The dilation process takes an integer's bit sequence and inserts a number
//! of 0 bits (`D - 1`) between each original bit successively. Thus, the
//! original bit sequence becomes evenly padded. For example:
//! * `0b1101` D2-dilated becomes `0b1010001` (values chosen arbitrarily)
//! * `0b1011` D3-dilated becomes `0b1000001001`
//!
//! The process of undilation, or 'contraction', does the opposite:
//! * `0b1010001` D2-undilated becomes `0b1101`
//! * `0b1000001001` D3-undilated becomes `0b1011`
//!
//! This libary also supports a limited subset of arthimetic operations on
//! dilated integers via the standard rust operater traits. Whilst slightly
//! more involved than regular integer arithmetic, these operations are still
//! highly performant.
//!
//! # Which Dilation Method to Choose
//! There are currently two distinct ways to dilate integers; Expanding or
//! Fixed. To help decide which is right for your application, consider the
//! following:
//!
//! Use [dilate_expand()](crate::expand::DilateExpand::dilate_expand()) when
//! you want all bits of the source integer to be dilated and you don't mind
//! how the dilated integer is stored behind the scenes. This is the most
//! intuitive method of interacting with dilated integers.
//!
//! Use [dilate_fixed()](crate::fixed::DilateFixed::dilate_fixed()) when you
//! want control over the storage type and want to maximise the number of bits
//! occupied within that storage type.
//!
//! Notice that the difference between the two is that of focus;
//! [dilate_expand()](crate::expand::DilateExpand::dilate_expand()) focusses on
//! maximising the usage of the source integer, whereas
//! [dilate_fixed()](crate::fixed::DilateFixed::dilate_fixed()) focusses on
//! maximising the usage of the dilated integer.
//!
//! You may also use the raw [Expand](crate::expand::Expand) and
//! [Fixed](crate::fixed::Fixed) [DilationMethod](crate::DilationMethod)
//! implementations directly, though they tend to be more verbose.
//!
//! # Supported Dilations
//! For more information on the supported dilations and possible type
//! combinations, please see
//! [Supported Expand Dilations](crate::expand::DilateExpand#supported-expand-dilations)
//! and
//! [Supported Fixed Dilations](crate::fixed::DilateFixed#supported-fixed-dilations).
//!
//! # Examples
//! ```rust
//! use dilate::*;
//!
//! let original: u8 = 0b1101;
//!
//! // Dilating
//! let dilated = original.dilate_expand::<2>();
//! assert_eq!(dilated.value(), 0b1010001);
//!
//! // This is the actual dilated type
//! assert_eq!(dilated, DilatedInt::<Expand<u8, 2>>::new(0b1010001));
//!
//! // Undilating
//! assert_eq!(dilated.undilate(), original);
//! ```
//! *Example 2-dilation and undilation usage*
//!
//! ```rust
//! use dilate::*;
//!
//! let original: u8 = 0b1011;
//!
//! // Dilating
//! let dilated = original.dilate_expand::<3>();
//! assert_eq!(dilated.value(), 0b1000001001);
//!
//! // This is the actual dilated type
//! assert_eq!(dilated, DilatedInt::<Expand<u8, 3>>::new(0b1000001001));
//!
//! // Undilating
//! assert_eq!(dilated.undilate(), original);
//! ```
//! *Example 3-dilation and undilation usage*

use core::{hash::Hash, fmt::Debug};

mod internal;

/// Contains the Expand dilation method and all supporting items
pub mod expand;
use internal::NumTraits;

pub use crate::expand::{DilateExpand, Expand};

/// Contains the Fixed dilation method and all supporting items
pub mod fixed;
pub use crate::fixed::{DilateFixed, Fixed};

/// Denotes an integer type supported by the various dilation and undilation methods
pub trait DilatableType:
    Default +
    Copy +
    Ord +
    Hash +
    Debug +
    internal::DilateExplicit +
    internal::UndilateExplicit
{
}
impl DilatableType for u8 {}
impl DilatableType for u16 {}
impl DilatableType for u32 {}
impl DilatableType for u64 {}
impl DilatableType for u128 {}
impl DilatableType for usize {}

/// Allows for custom decoupled dilation behaviours
///
/// An implementation of DilationMethod describes the manner in which dilation
/// occurs, including the dilated and undilated types involved, wrapper methods
/// to forward to the appropriate dilation functions, and some useful constants.
///
/// It is possible to construct your own dilation methods by implementing the
/// [DilationMethod] trait and optionally constructing your own value relative
/// dilation trait similar to [DilateExpand](expand::DilateExpand).
pub trait DilationMethod: Default + Copy + Ord + Hash + Debug {
    /// The external undilated integer type
    type Undilated: DilatableType;

    /// The internal dilated integer type
    type Dilated: DilatableType;

    /// The dilation amount
    const D: usize;

    /// The number of bits in the [DilationMethod::Undilated] type which
    /// may be dilated into [DilationMethod::Dilated]
    ///
    /// It may be smaller than the number of bits in
    /// [DilationMethod::Undilated] depending on the dilation method used.
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(Expand::<u8, 2>::UNDILATED_BITS, 8);
    /// assert_eq!(Fixed::<u16, 2>::UNDILATED_BITS, 8);
    /// ```
    const UNDILATED_BITS: usize;

    /// The maximum undilated value which may be dilated by this dilation method
    ///
    /// This value holds a set of N consecutive 1 bits, where N is equal to
    /// [DilationMethod::UNDILATED_BITS].
    ///
    /// It may be smaller than the maximum value of
    /// [DilationMethod::Undilated] depending on the dilation method used.
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(Expand::<u8, 2>::UNDILATED_MAX, 255);
    /// assert_eq!(Fixed::<u16, 2>::UNDILATED_MAX, 255);
    /// ```
    const UNDILATED_MAX: Self::Undilated;

    /// The number of maximally dilated bits occupied in [DilationMethod::Dilated]
    ///
    /// This constant describes how many bits of [DilationMethod::Dilated] are
    /// utilised, including the padding 0 bits.
    ///
    /// It may be smaller than the maximum number of bits available in
    /// [DilationMethod::Dilated] depending on the dilation method used.
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(Expand::<u8, 2>::DILATED_BITS, 16);
    /// assert_eq!(Fixed::<u16, 3>::DILATED_BITS, 15);
    /// ```
    const DILATED_BITS: usize;

    /// The maximum dilated value that can be stored in [DilationMethod::Dilated]
    ///
    /// This constant holds a set of N dilated 1 bits, each separated by a
    /// number of padding 0 bits, where N is equal to [DilationMethod::UNDILATED_BITS]
    /// and the number of padding 0 bits depends on the dilation method used.
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(Expand::<u8, 2>::DILATED_MAX, 0b0101010101010101);
    /// assert_eq!(Fixed::<u16, 3>::DILATED_MAX, 0b0001001001001001);
    /// ```
    const DILATED_MAX: Self::Dilated;

    /// A mask of all dilated bits, including 0 bits
    ///
    /// This constant holds a set of consecutive N 1 bits, where N is equal to
    /// [DilationMethod::DILATED_BITS].
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(Expand::<u8, 3>::DILATED_MASK, 0xffffff);
    /// assert_eq!(Fixed::<u16, 3>::DILATED_MASK, 0x7FFF);
    /// ```
    const DILATED_MASK: Self::Dilated;

    /// This function casts an undilated integer to a dilated integer
    /// 
    /// No dilation process is carried out, this is simply a cast to the
    /// dilated integer type.
    fn to_dilated(undilated: Self::Undilated) -> Self::Dilated;

    /// This function converts a dilated integer to an undilated integer
    /// 
    /// No undilation process is carried out, this is simply a cast to the
    /// undilated integer type.
    /// 
    /// Additionally, this function is potentially lossy and should be used
    /// only when you are certain that the conversion is valid or you don't
    /// mind losing some bits. For more secure number casts, users may
    /// prefer NumCast from the
    /// [NumTraits](https://crates.io/crates/num-traits) crate.
    fn to_undilated(dilated: Self::Dilated) -> Self::Undilated;

    /// This function carries out the dilation process, converting the
    /// [DilationMethod::Undilated] value to a [DilatedInt].
    ///
    /// This function is exposed as a secondary interface and you may prefer
    /// the more human friendly trait methods: [DilateExpand::dilate_expand()]
    /// and [DilateFixed::dilate_fixed()].
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(Expand::<u8, 2>::dilate(0b1101), DilatedInt::<Expand<u8, 2>>::new(0b01010001));
    /// assert_eq!(Expand::<u8, 2>::dilate(0b1101).value(), 0b01010001);
    ///
    /// assert_eq!(Fixed::<u16, 3>::dilate(0b1101), DilatedInt::<Fixed<u16, 3>>::new(0b001001000001));
    /// assert_eq!(Fixed::<u16, 3>::dilate(0b1101).value(), 0b001001000001);
    /// ```
    ///
    /// See also [DilateExpand::dilate_expand()], [DilateFixed::dilate_fixed()]
    fn dilate(value: Self::Undilated) -> DilatedInt<Self>;

    /// This function carries out the undilation process, converting a
    /// [DilatedInt] back to an [DilationMethod::Undilated] value.
    ///
    /// This function is exposed as a secondary interface and you may prefer
    /// the more human friendly method: [DilatedInt::undilate()].
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// let dilated = Expand::<u8, 2>::dilate(0b1101);
    /// assert_eq!(Expand::<u8, 2>::undilate(dilated), 0b1101);
    ///
    /// let dilated = Fixed::<u16, 3>::dilate(0b1101);
    /// assert_eq!(Fixed::<u16, 3>::undilate(dilated), 0b1101);
    /// ```
    ///
    /// See also [DilatedInt::undilate()]
    fn undilate(value: DilatedInt<Self>) -> Self::Undilated;
}

/// A storage type holding and identifying dilated integers
///
/// DilatedInt holds a dilated integer and allows for specialised dilated
/// arithmetic methods.
///
/// To dilate a regular integer and yield a DilatedInt, use the
/// [DilateExpand::dilate_expand()] or [DilateFixed::dilate_fixed()]
/// trait methods. These traits are implemented for all [DilatableType]
/// integers.
///
/// The stored dilated value may be obtained using the [DilatedInt::value()] method.
///
/// To undilate from a DilatedInt and yield a regular integer, use the
/// [DilatedInt::undilate()] method. This trait is implemented for
/// all DilatedInt types.
///
/// # Examples
///
/// Example 2-Dilation Usage:
/// ```rust
/// use dilate::*;
///
/// let original: u8 = 0b1101;
///
/// let dilated = original.dilate_expand::<2>();
/// assert_eq!(dilated, DilatedInt::<Expand<u8, 2>>::new(0b1010001));
/// assert_eq!(dilated.value(), 0b1010001);
///
/// assert_eq!(dilated.undilate(), original);
/// ```
///
/// Example 3-Dilation Usage:
/// ```rust
/// use dilate::*;
///
/// let original: u8 = 0b1011;
///
/// let dilated = original.dilate_expand::<3>();
/// assert_eq!(dilated, DilatedInt::<Expand<u8, 3>>::new(0b1000001001));
/// assert_eq!(dilated.value(), 0b1000001001);
///
/// assert_eq!(dilated.undilate(), original);
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DilatedInt<DM>(DM::Dilated)
where
    DM: DilationMethod;

impl<DM> DilatedInt<DM>
where
    DM: DilationMethod,
{
    /// Create new dilated int from known dilated value
    ///
    /// This function creates a new [DilatedInt]. The parameter 'dilated'
    /// should be a valid pre-dilated integer containing bits set only in
    /// positions indicated by [DilationMethod::DILATED_MAX]. This ensures
    /// validity for the various encoding and arithmetic methods.
    ///
    /// # Panics
    /// * Panics if parameter 'dilated' contains bits outside expected positions indicated by [DilationMethod::DILATED_MASK]
    #[inline]
    pub fn new(dilated: DM::Dilated) -> Self {
        debug_assert!(dilated.bit_and(DM::DILATED_MAX.bit_not()) == DM::Dilated::zero(), "Parameter 'dilated' contains bits outside expected positions indicated by DilationMethod::DILATED_MAX");
        Self(dilated)
    }

    /// Access dilated value
    #[inline]
    #[must_use]
    pub fn value(&self) -> DM::Dilated {
        self.0
    }

    /// This method carries out the undilation process, converting a
    /// [DilatedInt] back to a regular undilated integer value.
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// let dilated = 0b1101u8.dilate_expand::<2>();
    /// assert_eq!(dilated.undilate(), 0b1101);
    ///
    /// let dilated = 0b1101u16.dilate_expand::<3>();
    /// assert_eq!(dilated.undilate(), 0b1101);
    /// ```
    ///
    /// See also [DilationMethod::undilate()]
    #[inline]
    #[must_use]
    pub fn undilate(self) -> DM::Undilated {
        DM::undilate(self)
    }

    /// Adds the value 1 to the dilated int and returns the result
    ///
    /// This method is slightly more optimal than adding 1 via the '+' operator
    ///
    /// This method wraps the output value between 0 and [DILATED_MAX](DilationMethod::DILATED_MAX) (inclusive)
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(0u8.dilate_expand::<3>().add_one().undilate(), 1);
    /// assert_eq!(127u8.dilate_expand::<3>().add_one().undilate(), 128);
    /// assert_eq!(255u8.dilate_expand::<3>().add_one().undilate(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub fn add_one(self) -> Self {
        Self(
            self.0
                .sub_wrapping(DM::DILATED_MAX)
                .bit_and(DM::DILATED_MAX),
        )
    }

    /// Subtracts the value 1 from the dilated int and returns the result
    ///
    /// This method wraps the output value between 0 and [DILATED_MAX](DilationMethod::DILATED_MAX) (inclusive)
    ///
    /// # Examples
    /// ```rust
    /// use dilate::*;
    ///
    /// assert_eq!(0u8.dilate_expand::<3>().sub_one().undilate(), 255);
    /// assert_eq!(127u8.dilate_expand::<3>().sub_one().undilate(), 126);
    /// assert_eq!(255u8.dilate_expand::<3>().sub_one().undilate(), 254);
    /// ```
    #[inline]
    #[must_use]
    pub fn sub_one(self) -> Self {
        Self(
            self.0
                .sub_wrapping(DM::Dilated::one())
                .bit_and(DM::DILATED_MAX),
        )
    }

    /// Adds two dilated integers and returns the resultant dilated integer
    /// 
    /// # Wrapping
    /// The arithmetic is implemented in a way which enforces wrapping, so if
    /// the result of the operation would overflow, the resultant value is
    /// wrapped at the limits instead - this is somewhat contrary to the Rust
    /// default of panicking on overflow.
    /// 
    /// # no_std
    /// By default, dilate is `no_std` compliant. When the feature `std` is
    /// provided, you may also use the `+` operator to perform this operation.
    /// 
    /// # Examples
    /// ```rust
    /// use dilate::*;
    /// 
    /// let value_a: u16 = 123;
    /// let value_b: u16 = 456;
    /// 
    /// let dilated_a = value_a.dilate_expand::<3>();
    /// let dilated_b = value_b.dilate_expand::<3>();
    /// 
    /// assert_eq!(dilated_a.add(dilated_b).undilate(), value_a + value_b);
    /// ```
    #[inline]
    #[must_use]
    pub fn add(self, rhs: Self) -> Self {
        Self(
            self.0
                .add_wrapping(DM::DILATED_MAX.bit_not())
                .add_wrapping(rhs.0)
                .bit_and(DM::DILATED_MAX),
        )
    }

    /// Adds two dilated integers and assigns the result to the left operand
    /// 
    /// # Wrapping
    /// The arithmetic is implemented in a way which enforces wrapping, so if
    /// the result of the operation would overflow, the resultant value is
    /// wrapped at the limits instead - this is somewhat contrary to the Rust
    /// default of panicking on overflow.
    /// 
    /// # no_std
    /// By default, dilate is `no_std` compliant. When the feature `std` is
    /// provided, you may also use the `+=` operator to perform this operation.
    /// 
    /// # Examples
    /// ```rust
    /// use dilate::*;
    /// 
    /// let value_a: u16 = 123;
    /// let value_b: u16 = 456;
    /// 
    /// let mut dilated_a = value_a.dilate_expand::<3>();
    /// let dilated_b = value_b.dilate_expand::<3>();
    /// 
    /// dilated_a.add_assign(dilated_b);
    /// 
    /// assert_eq!(dilated_a.undilate(), value_a + value_b);
    /// ```
    #[inline]
    pub fn add_assign(&mut self, rhs: Self) {
        *self = self.add(rhs);
    }

    /// Subtracts two dilated integers and returns the resultant dilated integer
    /// 
    /// # Wrapping
    /// The arithmetic is implemented in a way which enforces wrapping, so if
    /// the result of the operation would overflow, the resultant value is
    /// wrapped at the limits instead - this is somewhat contrary to the Rust
    /// default of panicking on overflow.
    /// 
    /// # no_std
    /// By default, dilate is `no_std` compliant. When the feature `std` is
    /// provided, you may also use the `-` operator to perform this operation.
    /// 
    /// # Examples
    /// ```rust
    /// use dilate::*;
    /// 
    /// let value_a: u16 = 456;
    /// let value_b: u16 = 123;
    /// 
    /// let dilated_a = value_a.dilate_expand::<3>();
    /// let dilated_b = value_b.dilate_expand::<3>();
    /// 
    /// assert_eq!(dilated_a.sub(dilated_b).undilate(), value_a - value_b);
    /// ```
    #[inline]
    #[must_use]
    pub fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub_wrapping(rhs.0).bit_and(DM::DILATED_MAX))
    }

    /// Subtracts two dilated integers and assigns the result to the left operand
    /// 
    /// # Wrapping
    /// The arithmetic is implemented in a way which enforces wrapping, so if
    /// the result of the operation would overflow, the resultant value is
    /// wrapped at the limits instead - this is somewhat contrary to the Rust
    /// default of panicking on overflow.
    /// 
    /// # no_std
    /// By default, dilate is `no_std` compliant. When the feature `std` is
    /// provided, you may also use the `-=` operator to perform this operation.
    /// 
    /// # Examples
    /// ```rust
    /// use dilate::*;
    /// 
    /// let value_a: u16 = 456;
    /// let value_b: u16 = 123;
    /// 
    /// let mut dilated_a = value_a.dilate_expand::<3>();
    /// let dilated_b = value_b.dilate_expand::<3>();
    /// 
    /// dilated_a.sub_assign(dilated_b);
    /// 
    /// assert_eq!(dilated_a.undilate(), value_a - value_b);
    /// ```
    #[inline]
    pub fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub(rhs);
    }
}

#[cfg(feature = "std")]
pub use std_impls::*;

#[cfg(feature = "std")]
mod std_impls {
    extern crate std;

    use super::{DilatedInt, DilationMethod};
    use std::{
        fmt,
        ops::{Add, AddAssign, Sub, SubAssign},
    };

    impl<DM> fmt::Display for DilatedInt<DM>
    where
        DM: DilationMethod,
        DM::Dilated: fmt::Display,
    {
        #[inline]
        #[must_use]
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", self.value())
        }
    }

    impl<DM> Add for DilatedInt<DM>
    where
        DM: DilationMethod,
    {
        type Output = Self;

        #[inline]
        #[must_use]
        fn add(self, rhs: Self) -> Self::Output {
            self.add(rhs)
        }
    }

    impl<DM> AddAssign for DilatedInt<DM>
    where
        DM: DilationMethod,
    {
        #[inline]
        fn add_assign(&mut self, rhs: Self) {
            self.add_assign(rhs);
        }
    }

    impl<DM> Sub for DilatedInt<DM>
    where
        DM: DilationMethod,
    {
        type Output = Self;

        #[inline]
        #[must_use]
        fn sub(self, rhs: Self) -> Self::Output {
            self.sub(rhs)
        }
    }

    impl<DM> SubAssign for DilatedInt<DM>
    where
        DM: DilationMethod,
    {
        #[inline]
        fn sub_assign(&mut self, rhs: Self) {
            self.sub_assign(rhs);
        }
    }
}

#[cfg(test)]
pub(crate) mod shared_test_data {
    extern crate std;
    use std::marker::PhantomData;

    use lazy_static::lazy_static;

    use super::DilationMethod;

    pub struct TestData<T>
    where
        T: DilationMethod,
    {
        marker: PhantomData<T>,
    }

    macro_rules! impl_test_data {
        ($method_t:ty, $dil_max:expr, $con_max:expr) => {
            impl TestData<$method_t> {
                #[inline]
                fn dilated_max() -> <$method_t as DilationMethod>::Dilated {
                    $dil_max
                }

                #[inline]
                fn undilated_max() -> <$method_t as DilationMethod>::Undilated {
                    $con_max
                }
            }
        };
    }
    pub(crate) use impl_test_data;

    // TODO move to a file
    // NOTE - The following test cases are shared between all types (up to D8)
    //        For undilated values, we simply cast to the target type (and mask with undilated_max() for the Fixed method)
    //        For dilated values, we cast to the target inner type and mask with dilated_max()
    //        This procedure ensures that the test data is 100% valid in all cases
    //        Furthermore, every test case is xor'd with every other test case to
    //        perform more tests with fewer hand written values
    lazy_static! {
        pub static ref DILATION_TEST_CASES: [std::vec::Vec<(u128, u128)>; 9] = [
            // D0 (not used)
            std::vec::Vec::new(),

            // D1 (not used)
            std::vec::Vec::new(),

            // D2
            std::vec![
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
            std::vec![
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
            std::vec![
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
            std::vec![
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
            std::vec![
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
            std::vec![
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
            std::vec![
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

    // TODO move to a file
    // The first 32 values in each dimension (up to D8)
    // Used for testing arithmetic
    pub const VALUES: [[u128; 32]; 9] = [
        // D0 (not used)
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ],
        // D1 (not used)
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
