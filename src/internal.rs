// # References and Acknowledgments
// Many thanks to the authors of the following white papers:
// * [1] Converting to and from Dilated Integers - Rajeev Raman and David S. Wise
// * [2] Integer Dilation and Contraction for Quadtrees and Octrees - Leo Stocco and Gunther Schrack
// * [3] Fast Additions on Masked Integers - Michael D Adams and David S Wise
//
// Permission has been explicitly granted to reproduce the agorithms within each paper.

use core::fmt::Debug;
use core::hash::Hash;

pub(crate) trait Sealed: Sized + Clone + Copy + Default + Debug + PartialEq + Eq + PartialOrd + Ord + Hash {
    // Main entrypoints
    fn is_valid_dilated_value(self, dilated_max: Self) -> bool;
    fn dilate<const D: usize>(self) -> Self;
    fn undilate<const D: usize>(self) -> Self;
    fn add_one(self, dilated_max: Self) -> Self;
    fn sub_one(self, dilated_max: Self) -> Self;
    fn add(self, rhs: Self, dilated_max: Self) -> Self;
    fn sub(self, rhs: Self, dilated_max: Self) -> Self;

    // Utility methods
    fn _dilate_d2(self) -> Self;
    fn _dilate_d3(self) -> Self;
    fn _dilate_dn<const D: usize>(self) -> Self;
    fn _undilate_d2(self) -> Self;
    fn _undilate_d3(self) -> Self;
    fn _undilate_dn<const D: usize>(self) -> Self;
}

macro_rules! impl_dilatable_type_common {
    () => {
        #[inline(always)]
        fn is_valid_dilated_value(self, dilated_max: Self) -> bool {
            self & !dilated_max == 0
        }

        #[inline(always)]
        fn dilate<const D: usize>(self) -> Self {
            debug_assert!(
                self <= build_fixed_undilated_max::<Self, D>() as Self,
                "Attempting to dilate a value which exceeds maximum (See DilationMethod::UNDILATED_MAX)"
            );
        
            match D {
                2 => self._dilate_d2(),
                3 => self._dilate_d3(),
                _ => self._dilate_dn::<D>(),
            }
        }
        
        #[inline(always)]
        fn undilate<const D: usize>(self) -> Self {
            match D {
                2 => self._undilate_d2(),
                3 => self._undilate_d3(),
                _ => self._undilate_dn::<D>(),
            }
        }

        #[inline(always)]
        fn add_one(self, dilated_max: Self) -> Self {
            self.wrapping_sub(dilated_max) & dilated_max
        }

        #[inline(always)]
        fn sub_one(self, dilated_max: Self) -> Self {
            self.wrapping_sub(1) & dilated_max
        }

        #[inline(always)]
        fn add(self, rhs: Self, dilated_max: Self) -> Self {
            self.wrapping_add(!dilated_max).wrapping_add(rhs) & dilated_max
        }

        #[inline(always)]
        fn sub(self, rhs: Self, dilated_max: Self) -> Self {
            self.wrapping_sub(rhs) & dilated_max
        }

        #[inline(always)]
        fn _dilate_dn<const D: usize>(mut self) -> Self {
            debug_assert!(D > 2, "Generic parameter 'D' must be greater than 2");
            let mut i = 0;
            while i <= dilate_max_round::<Self, D>() {
                self = self.wrapping_mul(dilate_mult::<Self, D>(i) as Self)
                    & dilate_mask::<Self, D>(i) as Self;
                i += 1;
            }
            self
        }

        #[inline(always)]
        fn _undilate_dn<const D: usize>(mut self) -> Self {
            debug_assert!(D > 1, "Generic parameter 'D' must be greater than 1");
            let mut i = 0;
            while i <= undilate_max_round::<Self, D>() {
                self = self.wrapping_mul(undilate_mult::<Self, D>(i) as Self)
                    & undilate_mask::<Self, D>(i) as Self;
                i += 1;
            }
            self >> undilate_shift::<Self, D>() as Self
        }
    };
}

impl Sealed for u8 {
    impl_dilatable_type_common!();

    // See citation [2]
    #[inline(always)]
    fn _dilate_d2(mut self) -> Self {
        self = (self | (self << 2)) & 0x33;
        self = (self | (self << 1)) & 0x55;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _dilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x11) & 0xC3;
        self = self.wrapping_mul(0x05) & 0x49;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d2(mut self) -> Self {
        self = self.wrapping_mul(0x3) & 0x66;
        self = self.wrapping_mul(0x5) & 0x78;
        self >> 3
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x15) & 0x0e;
        self >> 2
    }
}

impl Sealed for u16 {
    impl_dilatable_type_common!();
    
    // See citation [2]
    #[inline(always)]
    fn _dilate_d2(mut self) -> Self {
        self = (self | (self << 4)) & 0x0F0F;
        self = (self | (self << 2)) & 0x3333;
        self = (self | (self << 1)) & 0x5555;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _dilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x101) & 0xF00F;
        self = self.wrapping_mul(0x011) & 0x30C3;
        self = self.wrapping_mul(0x005) & 0x9249;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d2(mut self) -> Self {
        self = self.wrapping_mul(0x003) & 0x6666;
        self = self.wrapping_mul(0x005) & 0x7878;
        self = self.wrapping_mul(0x011) & 0x7f80;
        self >> 7
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x0015) & 0x1c0e;
        self = self.wrapping_mul(0x1041) & 0x1ff0;
        self >> 8
    }
}

impl Sealed for u32 {
    impl_dilatable_type_common!();
    
    // See citation [2]
    #[inline(always)]
    fn _dilate_d2(mut self) -> Self {
        self = (self | (self << 8)) & 0x00FF00FF;
        self = (self | (self << 4)) & 0x0F0F0F0F;
        self = (self | (self << 2)) & 0x33333333;
        self = (self | (self << 1)) & 0x55555555;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _dilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x10001) & 0xFF0000FF;
        self = self.wrapping_mul(0x00101) & 0x0F00F00F;
        self = self.wrapping_mul(0x00011) & 0xC30C30C3;
        self = self.wrapping_mul(0x00005) & 0x49249249;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d2(mut self) -> Self {
        self = self.wrapping_mul(0x00000003) & 0x66666666;
        self = self.wrapping_mul(0x00000005) & 0x78787878;
        self = self.wrapping_mul(0x00000011) & 0x7F807F80;
        self = self.wrapping_mul(0x00000101) & 0x7FFF8000;
        self >> 15
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x00015) & 0x0E070381;
        self = self.wrapping_mul(0x01041) & 0x0FF80001;
        self = self.wrapping_mul(0x40001) & 0x0FFFFFFE;
        self >> 18
    }
}

impl Sealed for u64 {
    impl_dilatable_type_common!();
    
    // See citation [2]
    #[inline(always)]
    fn _dilate_d2(mut self) -> Self {
        self = (self | (self << 16)) & 0x0000FFFF0000FFFF;
        self = (self | (self << 08)) & 0x00FF00FF00FF00FF;
        self = (self | (self << 04)) & 0x0F0F0F0F0F0F0F0F;
        self = (self | (self << 02)) & 0x3333333333333333;
        self = (self | (self << 01)) & 0x5555555555555555;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _dilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x100000001) & 0xFFFF00000000FFFF;
        self = self.wrapping_mul(0x000010001) & 0x00FF0000FF0000FF;
        self = self.wrapping_mul(0x000000101) & 0xF00F00F00F00F00F;
        self = self.wrapping_mul(0x000000011) & 0x30C30C30C30C30C3;
        self = self.wrapping_mul(0x000000005) & 0x9249249249249249;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d2(mut self) -> Self {
        self = self.wrapping_mul(0x00003) & 0x6666666666666666;
        self = self.wrapping_mul(0x00005) & 0x7878787878787878;
        self = self.wrapping_mul(0x00011) & 0x7F807F807F807F80;
        self = self.wrapping_mul(0x00101) & 0x7FFF80007FFF8000;
        self = self.wrapping_mul(0x10001) & 0x7FFFFFFF80000000;
        self >> 31
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x0000000000000015) & 0x1c0e070381c0e070;
        self = self.wrapping_mul(0x0000000000001041) & 0x1ff00003fe00007f;
        self = self.wrapping_mul(0x0000001000040001) & 0x1ffffffc00000000;
        self >> 40
    }
}

impl Sealed for u128 {
    impl_dilatable_type_common!();
    
    // See citation [2]
    #[inline(always)]
    fn _dilate_d2(mut self) -> Self {
        self = (self | (self << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
        self = (self | (self << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
        self = (self | (self << 08)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
        self = (self | (self << 04)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
        self = (self | (self << 02)) & 0x33333333333333333333333333333333;
        self = (self | (self << 01)) & 0x55555555555555555555555555555555;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _dilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x10000000000000001) & 0xFFFFFFFF0000000000000000FFFFFFFF;
        self = self.wrapping_mul(0x00000000100000001) & 0x0000FFFF00000000FFFF00000000FFFF;
        self = self.wrapping_mul(0x00000000000010001) & 0xFF0000FF0000FF0000FF0000FF0000FF;
        self = self.wrapping_mul(0x00000000000000101) & 0x0F00F00F00F00F00F00F00F00F00F00F;
        self = self.wrapping_mul(0x00000000000000011) & 0xC30C30C30C30C30C30C30C30C30C30C3;
        self = self.wrapping_mul(0x00000000000000005) & 0x49249249249249249249249249249249;
        self
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d2(mut self) -> Self {
        self = self.wrapping_mul(0x000000003) & 0x66666666666666666666666666666666;
        self = self.wrapping_mul(0x000000005) & 0x78787878787878787878787878787878;
        self = self.wrapping_mul(0x000000011) & 0x7f807f807f807f807f807f807f807f80;
        self = self.wrapping_mul(0x000000101) & 0x7fff80007fff80007fff80007fff8000;
        self = self.wrapping_mul(0x000010001) & 0x7fffffff800000007fffffff80000000;
        self = self.wrapping_mul(0x100000001) & 0x7fffffffffffffff8000000000000000;
        self >> 63
    }

    // See citation [1]
    #[inline(always)]
    fn _undilate_d3(mut self) -> Self {
        self = self.wrapping_mul(0x00000000000000000000000000000015) & 0x0e070381c0e070381c0e070381c0e070;
        self = self.wrapping_mul(0x00000000000000000000000000001041) & 0x0ff80001ff00003fe00007fc0000ff80;
        self = self.wrapping_mul(0x00000000000000000000001000040001) & 0x0ffffffe00000000000007ffffff0000;
        self = self.wrapping_mul(0x00001000000000000040000000000001) & 0x0ffffffffffffffffffff80000000000;
        self >> 82
    }
}

impl Sealed for usize {
    impl_dilatable_type_common!();
    
    #[inline(always)]
    fn _dilate_d2(self) -> Self {
        #[cfg(target_pointer_width = "8")]
        let r = (self as u8)._dilate_d2();
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16)._dilate_d2();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32)._dilate_d2();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64)._dilate_d2();
        r as usize
    }

    #[inline(always)]
    fn _dilate_d3(self) -> Self {
        #[cfg(target_pointer_width = "8")]
        let r = (self as u8)._dilate_d3();
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16)._dilate_d3();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32)._dilate_d3();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64)._dilate_d3();
        r as usize
    }

    #[inline(always)]
    fn _undilate_d2(self) -> Self {
        #[cfg(target_pointer_width = "8")]
        let r = (self as u8)._undilate_d2();
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16)._undilate_d2();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32)._undilate_d2();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64)._undilate_d2();
        r as usize
    }

    #[inline(always)]
    fn _undilate_d3(self) -> Self {
        #[cfg(target_pointer_width = "8")]
        let r = (self as u8)._undilate_d3();
        #[cfg(target_pointer_width = "16")]
        let r = (self as u16)._undilate_d3();
        #[cfg(target_pointer_width = "32")]
        let r = (self as u32)._undilate_d3();
        #[cfg(target_pointer_width = "64")]
        let r = (self as u64)._undilate_d3();
        r as usize
    }
}

// For most const operations, we will use the largest available integer
// This assumption avoids the necessity to implement multiple versions of each of the following const functions
// Instantiations of each const function can therefore safely blind cast to the appropriate type

// Calculates the maximum undilated value that fits into D-dilated T (fixed)
#[inline(always)]
pub(crate) const fn build_fixed_undilated_max<T, const D: usize>() -> u128 {
    let bits_available = core::mem::size_of::<T>() * 8;
    let s_minus_1 = bits_available / D - 1;

    // By shifting in two phases like this, we can avoid overflow in case of D1
    // NOTE - We can't just use Wrapping because this is a const fn
    let partial_max = (1 << s_minus_1) - 1;
    (partial_max << 1) | 0x1
}

// Builds a dilated mask with p repetitions of 1 bits separated by (q - 1) 0 bits
// See Notation 4.2 in citation [1]
#[inline(always)]
pub(crate) const fn build_dilated_mask(p_repetitions: usize, q_width: usize) -> u128 {
    let mut r = p_repetitions;
    let mut v = 0;
    while r > 0 {
        v = (v << q_width) | 1;
        r -= 1;
    }
    v
}

// Calculates the maximum D-dilation round (total number of D-dilation rounds minus one)
// See Algorithm 9 in citation [1]
#[inline(always)]
pub(crate) const fn dilate_max_round<T, const D: usize>() -> usize {
    let s = (core::mem::size_of::<T>() * 8) / D;
    s.ilog(D - 1) as usize
}

// Calculates the D-dilation multiplier for each round
// See section IV.D in citation [1]
#[inline(always)]
pub(crate) const fn dilate_mult<T, const D: usize>(round: usize) -> u128 {
    let round_inv = dilate_max_round::<T, D>() - round;
    build_dilated_mask(D - 1, (D - 1).pow(round_inv as u32 + 1))
}

// Calculates the D-dilation mask for each round
// See section IV.D in citation [1]
#[inline(always)]
pub(crate) const fn dilate_mask<T, const D: usize>(round: usize) -> u128 {
    let round_inv = dilate_max_round::<T, D>() - round;
    let num_set_bits = (D - 1).pow(round_inv as u32);
    let num_blank_bits = num_set_bits * (D - 1);
    let sequence_length = num_set_bits + num_blank_bits;
    let mut repetitions = (core::mem::size_of::<T>() * 8) / sequence_length + 1;

    let mut v = 0;
    while repetitions > 0 {
        v = (v << sequence_length) | ((1 << num_set_bits) - 1);
        repetitions -= 1;
    }
    v
}

// Calculates the maximum D-undilation round (total number of D-undilation rounds minus one)
// See Algorithm 10 in citation [1]
#[inline(always)]
pub(crate) const fn undilate_max_round<T, const D: usize>() -> usize {
    let s = (core::mem::size_of::<T>() * 8) / D;
    s.ilog(D) as usize
}

// Calculates the D-undilation multiplier for each round
// See section IV.D in citation [1]
#[inline(always)]
pub(crate) const fn undilate_mult<T, const D: usize>(round: usize) -> u128 {
    build_dilated_mask(D, D.pow(round as u32) * (D - 1))
}

// Calculates the D-undilation mask for each round
// See section IV.D in citation [1]
#[inline(always)]
pub(crate) const fn undilate_mask<T, const D: usize>(round: usize) -> u128 {
    let s = (core::mem::size_of::<T>() * 8) / D;
    let num_blank_bits = D.pow(round as u32 + 1) * (D - 1);
    let num_set_bits = D.pow(round as u32 + 1);

    let sequence_length = num_blank_bits + num_set_bits;
    let left_most_bit = D * (s - 1) + 1;

    let num_set_bits = if num_set_bits < left_most_bit {
        num_set_bits
    } else {
        left_most_bit
    };

    let initial_shift = left_most_bit - num_set_bits;

    let mut v = ((1 << num_set_bits) - 1) << initial_shift;
    let mut repetitions = left_most_bit / sequence_length;
    while repetitions > 0 {
        v = (v >> sequence_length) | (((1 << num_set_bits) - 1) << initial_shift);
        repetitions -= 1;
    }
    v
}

#[inline(always)]
pub(crate) const fn undilate_shift<T, const D: usize>() -> usize {
    let s = (core::mem::size_of::<T>() * 8) / D;
    (D * (s - 1) + 1) - s
}

// TEST NOTE - Although not strictly necessary, we test the constant generation
// functions here. We could leave it all up to the user facing tests, but const
// generation is an extremely fiddly process, so it makes sense to test them.
#[cfg(test)]
mod tests {
    extern crate std;

    use paste::paste;
    use std::marker::PhantomData;

    struct DilationTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_dilation_test_data {
        ($t:ty, $d:literal, $dil_max:literal, $num_rounds:literal, $(($round:literal, $mult:literal, $mask:literal)),*) => {
            impl DilationTestData<$t, $d> {
                #[inline(always)]
                fn dilated_max() -> $t {
                    $dil_max
                }

                #[inline(always)]
                fn num_rounds() -> usize {
                    $num_rounds
                }

                #[inline(always)]
                fn test_cases() -> std::vec::Vec<(usize, $t, $t)> {
                    std::vec![$(($round, $mult, $mask)),*]
                }
            }
        };
    }

    // This is kinda insane - TODO move to a file
    impl_dilation_test_data!(u8, 2, 0x55, 0,);
    impl_dilation_test_data!(u8, 3, 0x09, 2, (0, 0x11, 0xC3), (1, 0x05, 0x49));
    impl_dilation_test_data!(u8, 4, 0x11, 1, (0, 0x49, 0x11));
    impl_dilation_test_data!(u8, 5, 0x01, 1, (0, 0x11, 0x21));
    impl_dilation_test_data!(u8, 6, 0x01, 1, (0, 0x21, 0x41));
    impl_dilation_test_data!(u8, 7, 0x01, 1, (0, 0x41, 0x81));
    impl_dilation_test_data!(u8, 8, 0x01, 1, (0, 0x81, 0x01));

    impl_dilation_test_data!(u16, 2, 0x5555, 0,);
    impl_dilation_test_data!(u16, 3, 0x1249, 3, (0, 0x0101, 0xF00F), (1, 0x0011, 0x30C3), (2, 0x0005, 0x9249));
    impl_dilation_test_data!(u16, 4, 0x1111, 2, (0, 0x0201, 0x7007), (1, 0x0049, 0x1111));
    impl_dilation_test_data!(u16, 5, 0x0421, 1, (0, 0x1111, 0x8421));
    impl_dilation_test_data!(u16, 6, 0x0041, 1, (0, 0x8421, 0x1041));
    impl_dilation_test_data!(u16, 7, 0x0081, 1, (0, 0x1041, 0x4081));
    impl_dilation_test_data!(u16, 8, 0x0101, 1, (0, 0x4081, 0x0101));

    impl_dilation_test_data!(u32, 2, 0x55555555, 0,);
    impl_dilation_test_data!(u32, 3, 0x09249249, 4, (0, 0x00010001, 0xFF0000FF), (1, 0x00000101, 0x0F00F00F), (2, 0x00000011, 0xC30C30C3), (3, 0x00000005, 0x49249249));
    impl_dilation_test_data!(u32, 4, 0x11111111, 2, (0, 0x00040201, 0x07007007), (1, 0x00000049, 0x11111111));
    impl_dilation_test_data!(u32, 5, 0x02108421, 2, (0, 0x00010001, 0x00f0000f), (1, 0x00001111, 0x42108421));
    impl_dilation_test_data!(u32, 6, 0x01041041, 2, (0, 0x02000001, 0xc000001f), (1, 0x00108421, 0x41041041));
    impl_dilation_test_data!(u32, 7, 0x00204081, 1, (0, 0x41041041, 0x10204081));
    impl_dilation_test_data!(u32, 8, 0x01010101, 1, (0, 0x10204081, 0x01010101));

    impl_dilation_test_data!(u64, 2, 0x5555555555555555, 0,);
    impl_dilation_test_data!(u64, 3, 0x1249249249249249, 5, (0, 0x0000000100000001, 0xFFFF00000000FFFF), (1, 0x0000000000010001, 0x00FF0000FF0000FF), (2, 0x0000000000000101, 0xF00F00F00F00F00F), (3, 0x0000000000000011, 0x30C30C30C30C30C3), (4, 0x0000000000000005, 0x9249249249249249));
    impl_dilation_test_data!(u64, 4, 0x1111111111111111, 3, (0, 0x0040000008000001, 0x00001ff0000001ff), (1, 0x0000000000040201, 0x7007007007007007), (2, 0x0000000000000049, 0x1111111111111111));
    impl_dilation_test_data!(u64, 5, 0x0084210842108421, 2, (0, 0x0001000100010001, 0xf0000f0000f0000f), (1, 0x0000000000001111, 0x1084210842108421));
    impl_dilation_test_data!(u64, 6, 0x0041041041041041, 2, (0, 0x0004000002000001, 0xf0000007c000001f), (1, 0x0000000000108421, 0x1041041041041041));
    impl_dilation_test_data!(u64, 7, 0x0102040810204081, 2, (0, 0x0000001000000001, 0x0000fc000000003f), (1, 0x0000000041041041, 0x8102040810204081));
    impl_dilation_test_data!(u64, 8, 0x0101010101010101, 2, (0, 0x0002000000000001, 0x7f0000000000007f), (1, 0x0000040810204081, 0x0101010101010101));

    impl_dilation_test_data!(u128, 2, 0x55555555555555555555555555555555, 0,);
    impl_dilation_test_data!(u128, 3, 0x09249249249249249249249249249249, 6, (0, 0x00000000000000010000000000000001, 0xffffffff0000000000000000ffffffff), (1, 0x00000000000000000000000100000001, 0x0000ffff00000000ffff00000000ffff), (2, 0x00000000000000000000000000010001, 0xff0000ff0000ff0000ff0000ff0000ff), (3, 0x00000000000000000000000000000101, 0x0f00f00f00f00f00f00f00f00f00f00f), (4, 0x00000000000000000000000000000011, 0xc30c30c30c30c30c30c30c30c30c30c3), (5, 0x00000000000000000000000000000005, 0x49249249249249249249249249249249));
    impl_dilation_test_data!(u128, 4, 0x11111111111111111111111111111111, 4, (0, 0x00000000000200000000000000000001, 0xfffff000000000000000000007ffffff), (1, 0x00000000000000000040000008000001, 0x001ff0000001ff0000001ff0000001ff), (2, 0x00000000000000000000000000040201, 0x07007007007007007007007007007007), (3, 0x00000000000000000000000000000049, 0x11111111111111111111111111111111));
    impl_dilation_test_data!(u128, 5, 0x01084210842108421084210842108421, 3, (0, 0x00000000000000010000000000000001, 0x00000000ffff0000000000000000ffff), (1, 0x00000000000000000001000100010001, 0x0f0000f0000f0000f0000f0000f0000f), (2, 0x00000000000000000000000000001111, 0x21084210842108421084210842108421));
    impl_dilation_test_data!(u128, 6, 0x01041041041041041041041041041041, 2, (0, 0x00000010000008000004000002000001, 0x1f0000007c000001f0000007c000001f), (1, 0x00000000000000000000000000108421, 0x41041041041041041041041041041041));
    impl_dilation_test_data!(u128, 7, 0x00810204081020408102040810204081, 2, (0, 0x00001000000001000000001000000001, 0xc000000003f000000000fc000000003f), (1, 0x00000000000000000000000041041041, 0x40810204081020408102040810204081));
    impl_dilation_test_data!(u128, 8, 0x01010101010101010101010101010101, 2, (0, 0x00000004000000000002000000000001, 0x007f0000000000007f0000000000007f), (1, 0x00000000000000000000040810204081, 0x01010101010101010101010101010101));

    struct UndilationTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_undilation_test_data {
        ($t:ty, $d:literal, $undil_max:literal, $undil_shift:literal, $num_rounds:literal, $(($round:literal, $mult:literal, $mask:literal)),*) => {
            impl UndilationTestData<$t, $d> {
                #[inline(always)]
                fn fixed_undilated_max() -> $t {
                    $undil_max
                }

                #[inline(always)]
                fn undilate_shift() -> usize {
                    $undil_shift
                }

                #[inline(always)]
                fn num_rounds() -> usize {
                    $num_rounds
                }

                #[inline(always)]
                fn test_cases() -> std::vec::Vec<(usize, $t, $t)> {
                    std::vec![$(($round, $mult, $mask)),*]
                }
            }
        };
    }

    // This is kinda insane - TODO move to a file
    impl_undilation_test_data!(u8, 2, 0x0f, 3, 3, (0, 0x03, 0x66), (1, 0x05, 0x78), (2, 0x11, 0x7F));
    impl_undilation_test_data!(u8, 3, 0x03, 2, 1, (0, 0x15, 0x0e));
    impl_undilation_test_data!(u8, 4, 0x03, 3, 1, (0, 0x49, 0x1e));
    impl_undilation_test_data!(u8, 5, 0x01, 0, 1, (0, 0x11, 0x01));
    impl_undilation_test_data!(u8, 6, 0x01, 0, 1, (0, 0x21, 0x01));
    impl_undilation_test_data!(u8, 7, 0x01, 0, 1, (0, 0x41, 0x01));
    impl_undilation_test_data!(u8, 8, 0x01, 0, 1, (0, 0x81, 0x01));

    impl_undilation_test_data!(u16, 2, 0x00ff, 7, 4, (0, 0x0003, 0x6666), (1, 0x0005, 0x7878), (2, 0x0011, 0x7f80), (3, 0x0101, 0x7fff));
    impl_undilation_test_data!(u16, 3, 0x001f, 8, 2, (0, 0x0015, 0x1c0e), (1, 0x1041, 0x1ff0));
    impl_undilation_test_data!(u16, 4, 0x000f, 9, 2, (0, 0x0249, 0x1e00), (1, 0x1001, 0x1fff));
    impl_undilation_test_data!(u16, 5, 0x0007, 8, 1, (0, 0x1111, 0x07c0));
    impl_undilation_test_data!(u16, 6, 0x0003, 5, 1, (0, 0x8421, 0x007e));
    impl_undilation_test_data!(u16, 7, 0x0003, 6, 1, (0, 0x1041, 0x00fe));
    impl_undilation_test_data!(u16, 8, 0x0003, 7, 1, (0, 0x4081, 0x01fe));

    impl_undilation_test_data!(u32, 2, 0x0000ffff, 15, 5, (0, 0x00000003, 0x66666666), (1, 0x00000005, 0x78787878), (2, 0x00000011, 0x7F807F80), (3, 0x00000101, 0x7FFF8000), (4, 0x00010001, 0x7fffffff));
    impl_undilation_test_data!(u32, 3, 0x000003ff, 18, 3, (0, 0x00000015, 0x0E070381), (1, 0x00001041, 0x0FF80001), (2, 0x00040001, 0x0FFFFFFE));
    impl_undilation_test_data!(u32, 4, 0x000000ff, 21, 2, (0, 0x00000249, 0x1e001e00), (1, 0x01001001, 0x1fffe000));
    impl_undilation_test_data!(u32, 5, 0x0000003f, 20, 2, (0, 0x00011111, 0x03e00001), (1, 0x00100001, 0x03fffffe));
    impl_undilation_test_data!(u32, 6, 0x0000001f, 20, 1, (0, 0x02108421, 0x01f80000));
    impl_undilation_test_data!(u32, 7, 0x0000000f, 18, 1, (0, 0x41041041, 0x003f8000));
    impl_undilation_test_data!(u32, 8, 0x0000000f, 21, 1, (0, 0x10204081, 0x01fe0000));

    impl_undilation_test_data!(u64, 2, 0x00000000ffffffff, 31, 6, (0, 0x0000000000000003, 0x6666666666666666), (1, 0x0000000000000005, 0x7878787878787878), (2, 0x0000000000000011, 0x7F807F807F807F80), (3, 0x0000000000000101, 0x7FFF80007FFF8000), (4, 0x0000000000010001, 0x7FFFFFFF80000000), (5, 0x0000000100000001, 0x7fffffffffffffff));
    impl_undilation_test_data!(u64, 3, 0x00000000001fffff, 40, 3, (0, 0x0000000000000015, 0x1c0e070381c0e070), (1, 0x0000000000001041, 0x1ff00003fe00007f), (2, 0x0000001000040001, 0x1ffffffc00000000));
    impl_undilation_test_data!(u64, 4, 0x000000000000ffff, 45, 3, (0, 0x0000000000000249, 0x1e001e001e001e00), (1, 0x0000001001001001, 0x1fffe00000000000), (2, 0x0001000000000001, 0x1fffffffffffffff));
    impl_undilation_test_data!(u64, 5, 0x0000000000000fff, 44, 2, (0, 0x0000000000011111, 0x00f800007c00003e), (1, 0x1000010000100001, 0x00ffffff80000000));
    impl_undilation_test_data!(u64, 6, 0x00000000000003ff, 45, 2, (0, 0x0000000002108421, 0x007e00000007e000), (1, 0x1000000040000001, 0x007ffffffff80000));
    impl_undilation_test_data!(u64, 7, 0x00000000000001ff, 48, 2, (0, 0x0000001041041041, 0x01fc0000000000fe), (1, 0x0000040000000001, 0x01ffffffffffff00));
    impl_undilation_test_data!(u64, 8, 0x00000000000000ff, 49, 2, (0, 0x0002040810204081, 0x01fe000000000000), (1, 0x0100000000000001, 0x01ffffffffffffff));

    impl_undilation_test_data!(u128, 2, 0x0000000000000000ffffffffffffffff, 063, 7, (0, 0x00000000000000000000000000000003, 0x66666666666666666666666666666666), (1, 0x00000000000000000000000000000005, 0x78787878787878787878787878787878), (2, 0x00000000000000000000000000000011, 0x7f807f807f807f807f807f807f807f80), (3, 0x00000000000000000000000000000101, 0x7fff80007fff80007fff80007fff8000), (4, 0x00000000000000000000000000010001, 0x7fffffff800000007fffffff80000000), (5, 0x00000000000000000000000100000001, 0x7fffffffffffffff8000000000000000), (6, 0x00000000000000010000000000000001, 0x7fffffffffffffffffffffffffffffff));
    impl_undilation_test_data!(u128, 3, 0x0000000000000000000003ffffffffff, 082, 4, (0, 0x00000000000000000000000000000015, 0x0e070381c0e070381c0e070381c0e070), (1, 0x00000000000000000000000000001041, 0x0ff80001ff00003fe00007fc0000ff80), (2, 0x00000000000000000000001000040001, 0x0ffffffe00000000000007ffffff0000), (3, 0x00001000000000000040000000000001, 0x0ffffffffffffffffffff80000000000));
    impl_undilation_test_data!(u128, 4, 0x000000000000000000000000ffffffff, 093, 3, (0, 0x00000000000000000000000000000249, 0x1e001e001e001e001e001e001e001e00), (1, 0x00000000000000000000001001001001, 0x1fffe000000000001fffe00000000000), (2, 0x00000001000000000001000000000001, 0x1fffffffffffffffe000000000000000));
    impl_undilation_test_data!(u128, 5, 0x00000000000000000000000001ffffff, 096, 3, (0, 0x00000000000000000000000000011111, 0x01f00000f800007c00003e00001f0000), (1, 0x00000000000100001000010000100001, 0x01ffffff000000000000000000000000), (2, 0x00000010000000000000000000000001, 0x01ffffffffffffffffffffffffffffff));
    impl_undilation_test_data!(u128, 6, 0x000000000000000000000000001fffff, 100, 2, (0, 0x00000000000000000000000002108421, 0x01f80000001f80000001f80000001f80), (1, 0x01000000040000001000000040000001, 0x01ffffffffe000000000000000000000));
    impl_undilation_test_data!(u128, 7, 0x0000000000000000000000000003ffff, 102, 2, (0, 0x00000000000000000000001041041041, 0x00fe00000000007f00000000003f8000), (1, 0x40000000001000000000040000000001, 0x00ffffffffffff800000000000000000));
    impl_undilation_test_data!(u128, 8, 0x0000000000000000000000000000ffff, 105, 2, (0, 0x00000000000000000002040810204081, 0x01fe00000000000001fe000000000000), (1, 0x00010000000000000100000000000001, 0x01fffffffffffffffe00000000000000));

    macro_rules! const_generation_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< $t _d $d >] {
                    use super::{DilationTestData, UndilationTestData};
                    use super::super::{build_fixed_undilated_max, build_dilated_mask, dilate_max_round, dilate_mult, dilate_mask, undilate_max_round, undilate_mult, undilate_mask, undilate_shift};

                    #[test]
                    fn fixed_undilated_max_is_correct() {
                        assert_eq!(build_fixed_undilated_max::<$t, $d>() as $t, UndilationTestData::<$t, $d>::fixed_undilated_max());
                    }

                    #[test]
                    fn dilated_max_is_correct() {
                        assert_eq!(build_dilated_mask((core::mem::size_of::<$t>() * 8 / $d) as usize, $d as usize) as $t, DilationTestData::<$t, $d>::dilated_max());
                    }

                    #[test]
                    fn dilate_max_round_is_correct() {
                        let expect_num_rounds = DilationTestData::<$t, $d>::num_rounds();
                        if expect_num_rounds > 0 {
                            assert_eq!(dilate_max_round::<$t, $d>(), expect_num_rounds - 1);
                        }
                    }

                    #[test]
                    fn dilate_mult_is_correct() {
                        for (round, mult, _) in DilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(dilate_mult::<$t, $d>(round) as $t, mult);
                        }
                    }

                    #[test]
                    fn dilate_mask_is_correct() {
                        for (round, _, mask) in DilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(dilate_mask::<$t, $d>(round) as $t, mask);
                        }
                    }

                    #[test]
                    fn undilate_max_round_is_correct() {
                        let expect_num_rounds = UndilationTestData::<$t, $d>::num_rounds();
                        if expect_num_rounds > 0 {
                            assert_eq!(undilate_max_round::<$t, $d>(), expect_num_rounds - 1);
                        }
                    }

                    #[test]
                    fn undilate_mult_is_correct() {
                        for (round, mult, _) in UndilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(undilate_mult::<$t, $d>(round) as $t, mult);
                        }
                    }

                    #[test]
                    fn undilate_mask_is_correct() {
                        for (round, _, mask) in UndilationTestData::<$t, $d>::test_cases() {
                            assert_eq!(undilate_mask::<$t, $d>(round) as $t, mask);
                        }
                    }

                    #[test]
                    fn undilate_shift_is_correct() {
                        assert_eq!(undilate_shift::<$t, $d>(), UndilationTestData::<$t, $d>::undilate_shift());
                    }
                }
            }
        )+}
    }
    const_generation_tests!(u8, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u16, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u32, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u64, 2, 3, 4, 5, 6, 7, 8);
    const_generation_tests!(u128, 2, 3, 4, 5, 6, 7, 8);
}
