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

use std::{num::Wrapping, ops::{Mul, BitAnd, Shr}, marker::PhantomData};

use crate::consts::{dilate_max_round, dilate_mult, dilate_mask, undilate_max_round, undilate_mult, undilate_mask, undilate_shift};

macro_rules! impl_dilate_dn {
    () => {
        #[inline]
        fn dilate_explicit_dn<const D: usize>(self) -> Self {
            debug_assert!(D > 2, "Generic parameter 'D' must be greater than 2");
            let mut v = Wrapping(self);
            let mut i = 0;
            while i <= dilate_max_round::<Self, D>() {
                v = (v * Wrapping(dilate_mult::<Self, D>(i) as Self)) & Wrapping(dilate_mask::<Self, D>(i) as Self);
                i += 1;
            }
            v.0
        }
    };
}

macro_rules! impl_undilate_dn {
    () => {
        #[inline]
        fn undilate_explicit_dn<const D: usize>(self) -> Self {
            debug_assert!(D > 1, "Generic parameter 'D' must be greater than 1");
            let mut v = Wrapping(self);
            let mut i = 0;
            while i <= undilate_max_round::<Self, D>() {
                v = (v * Wrapping(undilate_mult::<Self, D>(i) as Self)) & Wrapping(undilate_mask::<Self, D>(i) as Self);
                i += 1;
            }
            v.0 >> undilate_shift::<Self, D>() as Self
        }
    };
}

pub trait DilateExplicit: Sized {
    fn dilate_explicit_d2(self) -> Self;
    fn dilate_explicit_d3(self) -> Self;
    fn dilate_explicit_dn<const D: usize>(self) -> Self;
}

impl DilateExplicit for u8 {
    #[inline]
    fn dilate_explicit_d2(self) -> Self {
        // See citation [2]
        let mut v = self;
        v = (v | (v << 2)) & 0x33;
        v = (v | (v << 1)) & 0x55;
        v
    }

    #[inline]
    fn dilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x11)) & Wrapping(0xC3);
        v = (v * Wrapping(0x05)) & Wrapping(0x49);
        v.0
    }

    impl_dilate_dn!();
}

impl DilateExplicit for u16 {
    #[inline]
    fn dilate_explicit_d2(self) -> Self {
        // See citation [2]
        let mut v = self;
        v = (v | (v << 4)) & 0x0F0F;
        v = (v | (v << 2)) & 0x3333;
        v = (v | (v << 1)) & 0x5555;
        v
    }

    #[inline]
    fn dilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x101)) & Wrapping(0xF00F);
        v = (v * Wrapping(0x011)) & Wrapping(0x30C3);
        v = (v * Wrapping(0x005)) & Wrapping(0x9249);
        v.0
    }

    impl_dilate_dn!();
}

impl DilateExplicit for u32 {
    #[inline]
    fn dilate_explicit_d2(self) -> Self {
        // See citation [2]
        let mut v = self;
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        v
    }

    #[inline]
    fn dilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x10001)) & Wrapping(0xFF0000FF);
        v = (v * Wrapping(0x00101)) & Wrapping(0x0F00F00F);
        v = (v * Wrapping(0x00011)) & Wrapping(0xC30C30C3);
        v = (v * Wrapping(0x00005)) & Wrapping(0x49249249);
        v.0
    }

    impl_dilate_dn!();
}

impl DilateExplicit for u64 {
    #[inline]
    fn dilate_explicit_d2(self) -> Self {
        // See citation [2]
        let mut v = self;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF;
        v = (v | (v << 08)) & 0x00FF00FF00FF00FF;
        v = (v | (v << 04)) & 0x0F0F0F0F0F0F0F0F;
        v = (v | (v << 02)) & 0x3333333333333333;
        v = (v | (v << 01)) & 0x5555555555555555;
        v
    }

    #[inline]
    fn dilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x100000001)) & Wrapping(0xFFFF00000000FFFF);
        v = (v * Wrapping(0x000010001)) & Wrapping(0x00FF0000FF0000FF);
        v = (v * Wrapping(0x000000101)) & Wrapping(0xF00F00F00F00F00F);
        v = (v * Wrapping(0x000000011)) & Wrapping(0x30C30C30C30C30C3);
        v = (v * Wrapping(0x000000005)) & Wrapping(0x9249249249249249);
        v.0
    }

    impl_dilate_dn!();
}

impl DilateExplicit for u128 {
    #[inline]
    fn dilate_explicit_d2(self) -> Self {
        // See citation [2]
        let mut v = self;
        v = (v | (v << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
        v = (v | (v << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
        v = (v | (v << 08)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
        v = (v | (v << 04)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
        v = (v | (v << 02)) & 0x33333333333333333333333333333333;
        v = (v | (v << 01)) & 0x55555555555555555555555555555555;
        v
    }

    #[inline]
    fn dilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x10000000000000001)) & Wrapping(0xFFFFFFFF0000000000000000FFFFFFFF);
        v = (v * Wrapping(0x00000000100000001)) & Wrapping(0x0000FFFF00000000FFFF00000000FFFF);
        v = (v * Wrapping(0x00000000000010001)) & Wrapping(0xFF0000FF0000FF0000FF0000FF0000FF);
        v = (v * Wrapping(0x00000000000000101)) & Wrapping(0x0F00F00F00F00F00F00F00F00F00F00F);
        v = (v * Wrapping(0x00000000000000011)) & Wrapping(0xC30C30C30C30C30C30C30C30C30C30C3);
        v = (v * Wrapping(0x00000000000000005)) & Wrapping(0x49249249249249249249249249249249);
        v.0
    }

    impl_dilate_dn!();
}

impl DilateExplicit for usize {
    #[inline]
    fn dilate_explicit_d2(self) -> Self {
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16).dilate_explicit_d2();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32).dilate_explicit_d2();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64).dilate_explicit_d2();
        r as usize
    }

    #[inline]
    fn dilate_explicit_d3(self) -> Self {
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16).dilate_explicit_d3();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32).dilate_explicit_d3();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64).dilate_explicit_d3();
        r as usize
    }

    impl_dilate_dn!();
}

pub trait UndilateExplicit: Sized {
    fn undilate_explicit_d2(self) -> Self;
    fn undilate_explicit_d3(self) -> Self;
    fn undilate_explicit_dn<const D: usize>(self) -> Self;
}

impl UndilateExplicit for u8 {
    #[inline]
    fn undilate_explicit_d2(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = v * Wrapping(0x3) & Wrapping(0x66);
        v = v * Wrapping(0x5) & Wrapping(0x78);
        v.0 >> 3
    }

    #[inline]
    fn undilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x15)) & Wrapping(0x0e);
        v.0 >> 2
    }

    impl_undilate_dn!();
}

impl UndilateExplicit for u16 {
    #[inline]
    fn undilate_explicit_d2(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = v * Wrapping(0x003) & Wrapping(0x6666);
        v = v * Wrapping(0x005) & Wrapping(0x7878);
        v = v * Wrapping(0x011) & Wrapping(0x7f80);
        v.0 >> 7
    }

    #[inline]
    fn undilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x0015)) & Wrapping(0x1c0e);
        v = (v * Wrapping(0x1041)) & Wrapping(0x1ff0);
        v.0 >> 8
    }

    impl_undilate_dn!();
}

impl UndilateExplicit for u32 {
    #[inline]
    fn undilate_explicit_d2(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = v * Wrapping(0x00000003) & Wrapping(0x66666666);
        v = v * Wrapping(0x00000005) & Wrapping(0x78787878);
        v = v * Wrapping(0x00000011) & Wrapping(0x7F807F80);
        v = v * Wrapping(0x00000101) & Wrapping(0x7FFF8000);
        v.0 >> 15
    }

    #[inline]
    fn undilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x00015)) & Wrapping(0x0E070381);
        v = (v * Wrapping(0x01041)) & Wrapping(0x0FF80001);
        v = (v * Wrapping(0x40001)) & Wrapping(0x0FFFFFFE);
        v.0 >> 18
    }

    impl_undilate_dn!();
}

impl UndilateExplicit for u64 {
    #[inline]
    fn undilate_explicit_d2(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = v * Wrapping(0x00003) & Wrapping(0x6666666666666666);
        v = v * Wrapping(0x00005) & Wrapping(0x7878787878787878);
        v = v * Wrapping(0x00011) & Wrapping(0x7F807F807F807F80);
        v = v * Wrapping(0x00101) & Wrapping(0x7FFF80007FFF8000);
        v = v * Wrapping(0x10001) & Wrapping(0x7FFFFFFF80000000);
        v.0 >> 31
    }

    #[inline]
    fn undilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x0000000000000015)) & Wrapping(0x1c0e070381c0e070);
        v = (v * Wrapping(0x0000000000001041)) & Wrapping(0x1ff00003fe00007f);
        v = (v * Wrapping(0x0000001000040001)) & Wrapping(0x1ffffffc00000000);
        v.0 >> 40
    }

    impl_undilate_dn!();
}

impl UndilateExplicit for u128 {
    #[inline]
    fn undilate_explicit_d2(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = v * Wrapping(0x000000003) & Wrapping(0x66666666666666666666666666666666);
        v = v * Wrapping(0x000000005) & Wrapping(0x78787878787878787878787878787878);
        v = v * Wrapping(0x000000011) & Wrapping(0x7f807f807f807f807f807f807f807f80);
        v = v * Wrapping(0x000000101) & Wrapping(0x7fff80007fff80007fff80007fff8000);
        v = v * Wrapping(0x000010001) & Wrapping(0x7fffffff800000007fffffff80000000);
        v = v * Wrapping(0x100000001) & Wrapping(0x7fffffffffffffff8000000000000000);
        v.0 >> 63
    }

    #[inline]
    fn undilate_explicit_d3(self) -> Self {
        // See citation [1]
        let mut v = Wrapping(self);
        v = (v * Wrapping(0x00000000000000000000000000000015)) & Wrapping(0x0e070381c0e070381c0e070381c0e070);
        v = (v * Wrapping(0x00000000000000000000000000001041)) & Wrapping(0x0ff80001ff00003fe00007fc0000ff80);
        v = (v * Wrapping(0x00000000000000000000001000040001)) & Wrapping(0x0ffffffe00000000000007ffffff0000);
        v = (v * Wrapping(0x00001000000000000040000000000001)) & Wrapping(0x0ffffffffffffffffffff80000000000);
        v.0 >> 82
    }

    impl_undilate_dn!();
}

impl UndilateExplicit for usize {
    #[inline]
    fn undilate_explicit_d2(self) -> Self {
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16).undilate_explicit_d2();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32).undilate_explicit_d2();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64).undilate_explicit_d2();
        r as usize
    }

    #[inline]
    fn undilate_explicit_d3(self) -> Self {
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16).undilate_explicit_d3();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32).undilate_explicit_d3();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64).undilate_explicit_d3();
        r as usize
    }

    impl_undilate_dn!();
}

#[inline]
pub fn dilate<T, const D: usize>(value: T) -> T where T: DilateExplicit {
    match D {
        1 => value,
        2 => value.dilate_explicit_d2(),
        3 => value.dilate_explicit_d3(),
        _ => value.dilate_explicit_dn::<D>(),
    }
}

#[inline]
pub fn undilate<T, const D: usize>(value: T) -> T where T: UndilateExplicit {
    match D {
        1 => value,
        2 => value.undilate_explicit_d2(),
        3 => value.undilate_explicit_d3(),
        _ => value.undilate_explicit_dn::<D>(),
    }
}
