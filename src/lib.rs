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

use std::mem::size_of;

// NOTE Until we have stable specialization, D is limited to 1-8
#[derive(Clone, Copy, Default, Debug)]
pub struct DilatedInt<T, const D: usize>(T);

// Dilated mask trait
// Returns the equivalent of DilatedInt::<T, D>::from(0xffffffffffffffff)
pub trait DilatedMask<T> {
    fn dilated_mask() -> T;
}

#[cfg(not(has_i128))]
const fn build_dilated_mask<const D: usize>() -> u64 {
    let mut s = size_of::<u64>() * 8 / D;
    let mut v = 0u64;
    while s > 0 {
        v = (v << D) | 1;
        s -= 1;
    }
    v
}

#[cfg(has_i128)]
const fn build_dilated_mask<const D: usize>() -> u128 {
    let mut s = size_of::<u128>() * 8 / D;
    let mut v = 0u128;
    while s > 0 {
        v = (v << D) | 1;
        s -= 1;
    }
    v
}

macro_rules! dilated_int_mask_impls {
    ($($t:ty),+) => {$(
        impl<const D: usize> DilatedMask<$t> for DilatedInt<$t, D> {
            #[inline]
            fn dilated_mask() -> $t {
                build_dilated_mask::<D>() as $t
            }
        }
    )+}
}
dilated_int_mask_impls!(u8, u16, u32, u64, usize);
#[cfg(has_i128)]
dilated_int_mask_impls!(u128);

// Undilated maximum value trait
pub trait UndilatedMax<T> {
    fn undilated_max() -> T;
}

#[cfg(not(has_i128))]
const fn build_undilated_max<T, const D: usize>() -> u64 {
    let bits_available = size_of::<T>() * 8;
    let mut s = bits_available / D;
    let mut v = 0u64;
    while s > 0 {
        v = (v << 1) | 1;
        s -= 1;
    }
    v
}

#[cfg(has_i128)]
const fn build_undilated_max<T, const D: usize>() -> u128 {
    let bits_available = size_of::<T>() * 8;
    let mut s = bits_available / D;
    let mut v = 0u128;
    while s > 0 {
        v = (v << 1) | 1;
        s -= 1;
    }
    v
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
dilated_int_undilated_max_impls!(u8, u16, u32, u64, usize);
#[cfg(has_i128)]
dilated_int_undilated_max_impls!(u128);

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
dilated_int_d1_from_impls!(u8, u16, u32, u64, usize);
#[cfg(has_i128)]
dilated_int_d1_from_impls!(u128);

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
        let mut r = value;
        r = (r | (r << 2)) & 0x33;
        r = (r | (r << 1)) & 0x55;
        Self(r)
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
        let mut r = value;
        r = (r | (r << 4)) & 0x0F0F;
        r = (r | (r << 2)) & 0x3333;
        r = (r | (r << 1)) & 0x5555;
        Self(r)
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
        let mut r = value;
        r = (r | (r << 8)) & 0x00FF00FF;
        r = (r | (r << 4)) & 0x0F0F0F0F;
        r = (r | (r << 2)) & 0x33333333;
        r = (r | (r << 1)) & 0x55555555;
        Self(r)
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
        let mut r = value;
        r = (r | (r << 16)) & 0x0000FFFF0000FFFF;
        r = (r | (r << 8)) & 0x00FF00FF00FF00FF;
        r = (r | (r << 4)) & 0x0F0F0F0F0F0F0F0F;
        r = (r | (r << 2)) & 0x3333333333333333;
        r = (r | (r << 1)) & 0x5555555555555555;
        Self(r)
    }
}

#[cfg(has_i128)]
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
        let mut r = value;
        r = (r | (r << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
        r = (r | (r << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
        r = (r | (r << 8)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
        r = (r | (r << 4)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
        r = (r | (r << 2)) & 0x33333333333333333333333333333333;
        r = (r | (r << 1)) & 0x55555555555555555555555555555555;
        Self(r)
    }
}

impl From<DilatedInt<u8, 2>> for u8 {
    #[inline]
    fn from(dilated: DilatedInt<u8, 2>) -> Self {
        // See citation [1]
        let mut t = dilated.0;
        t = t * 0x3 & (0xCC >> 1);
        t = t * 0x5 & (0xF0 >> 1);
        t >> 3
    }
}

impl From<DilatedInt<u16, 2>> for u16 {
    #[inline]
    fn from(dilated: DilatedInt<u16, 2>) -> Self {
        // See citation [1]
        let mut t = dilated.0;
        t = t * 0x003 & (0xCCCC >> 1);
        t = t * 0x005 & (0xF0F0 >> 1);
        t = t * 0x011 & (0xFF00 >> 1);
        t >> 7
    }
}

impl From<DilatedInt<u32, 2>> for u32 {
    #[inline]
    fn from(dilated: DilatedInt<u32, 2>) -> Self {
        // See citation [1]
        let mut t = dilated.0;
        t = t * 0x003 & (0xCCCCCCCC >> 1);
        t = t * 0x005 & (0xF0F0F0F0 >> 1);
        t = t * 0x011 & (0xFF00FF00 >> 1);
        t = t * 0x101 & (0xFFFF0000 >> 1);
        t >> 15
    }
}

impl From<DilatedInt<u64, 2>> for u64 {
    #[inline]
    fn from(dilated: DilatedInt<u64, 2>) -> Self {
        // See citation [1]
        let mut t = dilated.0;
        t = t * 0x00003 & (0xCCCCCCCCCCCCCCCC >> 1);
        t = t * 0x00005 & (0xF0F0F0F0F0F0F0F0 >> 1);
        t = t * 0x00011 & (0xFF00FF00FF00FF00 >> 1);
        t = t * 0x00101 & (0xFFFF0000FFFF0000 >> 1);
        t = t * 0x10001 & (0xFFFFFFFF00000000 >> 1);
        t >> 31
    }
}

#[cfg(has_i128)]
impl From<DilatedInt<u128, 2>> for u128 {
    #[inline]
    fn from(dilated: DilatedInt<u128, 2>) -> Self {
        // See citation [1]
        let mut t = dilated.0;
        t = t * 0x000000003 & (0xCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC >> 1);
        t = t * 0x000000005 & (0xF0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0 >> 1);
        t = t * 0x000000011 & (0xFF00FF00FF00FF00FF00FF00FF00FF00 >> 1);
        t = t * 0x000000101 & (0xFFFF0000FFFF0000FFFF0000FFFF0000 >> 1);
        t = t * 0x000010001 & (0xFFFFFFFF00000000FFFFFFFF00000000 >> 1);
        t = t * 0x100000001 & (0xFFFFFFFFFFFFFFFF0000000000000000 >> 1);
        t >> 63
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
        let mut r = value;
        r = (r * 0x011) & 0xC3;
        r = (r * 0x005) & 0x49;
        Self(r)
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
        let mut r = value;
        r = (r * 0x101) & 0xF00F;
        r = (r * 0x011) & 0x30C3;
        r = (r * 0x005) & 0x9249;
        Self(r)
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
        let mut r = value;
        r = (r * 0x10001) & 0xFF0000FF;
        r = (r * 0x00101) & 0x0F00F00F;
        r = (r * 0x00011) & 0xC30C30C3;
        r = (r * 0x00005) & 0x49249249;
        Self(r)
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
        let mut r = value;
        r = (r * 0x100000001) & 0xFFFF00000000FFFF;
        r = (r * 0x000010001) & 0x00FF0000FF0000FF;
        r = (r * 0x000000101) & 0xF00F00F00F00F00F;
        r = (r * 0x000000011) & 0x30C30C30C30C30C3;
        r = (r * 0x000000005) & 0x9249249249249249;
        Self(r)
    }
}

#[cfg(has_i128)]
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
        let mut r = value;
        r = (r * 0x10000000000000001) & 0xFFFFFFFF0000000000000000FFFFFFFF;
        r = (r * 0x00000000100000001) & 0x0000FFFF00000000FFFF00000000FFFF;
        r = (r * 0x00000000000010001) & 0xFF0000FF0000FF0000FF0000FF0000FF;
        r = (r * 0x00000000000000101) & 0x0F00F00F00F00F00F00F00F00F00F00F;
        r = (r * 0x00000000000000011) & 0xC30C30C30C30C30C30C30C30C30C30C3;
        r = (r * 0x00000000000000005) & 0x49249249249249249249249249249249;
        Self(r)
    }
}

impl From<DilatedInt<u8, 3>> for u8 {
    #[inline]
    fn from(dilated: DilatedInt<u8, 3>) -> Self {
        let mut t = dilated.0;
        t
    }
}

impl From<DilatedInt<u16, 3>> for u16 {
    #[inline]
    fn from(dilated: DilatedInt<u16, 3>) -> Self {
        let mut t = dilated.0;
        t
    }
}

impl From<DilatedInt<u32, 3>> for u32 {
    #[inline]
    fn from(dilated: DilatedInt<u32, 3>) -> Self {
        // See citation [1]
        let mut t = dilated.0;
        t = (t * 0x00015) & 0x0E070381;
        t = (t * 0x01041) & 0x0FF80001;
        t = (t * 0x40001) & 0x0FFC0000;
        t >> 18
    }
}

impl From<DilatedInt<u64, 3>> for u64 {
    fn from(dilated: DilatedInt<u64, 3>) -> Self {
        let mut t = dilated.0;
        t
    }
}

#[cfg(has_i128)]
impl From<DilatedInt<u128, 3>> for u128 {
    fn from(dilated: DilatedInt<u128, 3>) -> Self {
        let mut t = dilated.0;
        t
    }
}

/*
const fn x<T>(p: T, q: T) -> T {
    let mut p = p;
    let mut v = 0;
    while p > 0 {
        v += 1 << (p * q);
        p -= 1;
    }
    v
}*/

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

                let mut v = value;
//                let s = size_of::<T>() * 8 / ($d - 1);
//                for i in 1..s;

//                for i in (s - 1)..=0 {
//                    v | (v << (1 << i))
//                }
                Self(v)
            }
        }

        impl From<DilatedInt<$t, $d>> for $t {
            #[inline]
            fn from(dilated: DilatedInt<$t, $d>) -> $t {
                dilated.0
            }
        }
    )+}
}

// D 1, 2, 3 cases handled separately
dilated_int_dn_from_impls!(u8, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u16, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u32, 4, 5, 6, 7, 8);
dilated_int_dn_from_impls!(u64, 4, 5, 6, 7, 8);
#[cfg(has_i128)]
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

// D 1 case handled separately
// Bootstrap usize (16 bit) for any number of dimensions
#[cfg(target_pointer_width = "16")]
dilated_int_usize_from_impls!(u16, 2, 3, 4, 5, 6, 7, 8);

// Bootstrap usize (32 bit) for any number of dimensions
#[cfg(target_pointer_width = "32")]
dilated_int_usize_from_impls!(u32, 2, 3, 4, 5, 6, 7, 8);

// Bootstrap usize (64 bit) for any number of dimensions
#[cfg(target_pointer_width = "64")]
dilated_int_usize_from_impls!(u64, 2, 3, 4, 5, 6, 7, 8);

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use lazy_static::lazy_static;
    use paste::paste;

    struct DilMaskTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_dil_mask {
        ($t:ty, $d:literal, $v:expr) => {
            impl DilMaskTestData<$t, $d> {
                #[inline]
                fn dilated_mask() -> $t {
                    $v as $t
                }
            }
        };
    }
    impl_dil_mask!(u8, 1, 0xff);
    impl_dil_mask!(u8, 2, 0x55);
    impl_dil_mask!(u8, 3, 0x49);
    impl_dil_mask!(u8, 4, 0x11);
    impl_dil_mask!(u8, 5, 0x21);
    impl_dil_mask!(u8, 6, 0x41);
    impl_dil_mask!(u8, 7, 0x81);
    impl_dil_mask!(u8, 8, 0x01);

    impl_dil_mask!(u16, 1, 0xffff);
    impl_dil_mask!(u16, 2, 0x5555);
    impl_dil_mask!(u16, 3, 0x9249);
    impl_dil_mask!(u16, 4, 0x1111);
    impl_dil_mask!(u16, 5, 0x8421);
    impl_dil_mask!(u16, 6, 0x1041);
    impl_dil_mask!(u16, 7, 0x4081);
    impl_dil_mask!(u16, 8, 0x0101);

    impl_dil_mask!(u32, 1, 0xffffffff);
    impl_dil_mask!(u32, 2, 0x55555555);
    impl_dil_mask!(u32, 3, 0x49249249);
    impl_dil_mask!(u32, 4, 0x11111111);
    impl_dil_mask!(u32, 5, 0x42108421);
    impl_dil_mask!(u32, 6, 0x41041041);
    impl_dil_mask!(u32, 7, 0x10204081);
    impl_dil_mask!(u32, 8, 0x01010101);

    impl_dil_mask!(u64, 1, 0xffffffffffffffff);
    impl_dil_mask!(u64, 2, 0x5555555555555555);
    impl_dil_mask!(u64, 3, 0x1249249249249249);
    impl_dil_mask!(u64, 4, 0x1111111111111111);
    impl_dil_mask!(u64, 5, 0x0084210842108421);
    impl_dil_mask!(u64, 6, 0x0041041041041041);
    impl_dil_mask!(u64, 7, 0x0102040810204081);
    impl_dil_mask!(u64, 8, 0x0101010101010101);

    macro_rules! impl_dil_mask_usize {
        ($innert:ty) => {
            impl_dil_mask!(usize, 1, DilMaskTestData::<$innert, 1>::dilated_mask());
            impl_dil_mask!(usize, 2, DilMaskTestData::<$innert, 2>::dilated_mask());
            impl_dil_mask!(usize, 3, DilMaskTestData::<$innert, 3>::dilated_mask());
            impl_dil_mask!(usize, 4, DilMaskTestData::<$innert, 4>::dilated_mask());
            impl_dil_mask!(usize, 5, DilMaskTestData::<$innert, 5>::dilated_mask());
            impl_dil_mask!(usize, 6, DilMaskTestData::<$innert, 6>::dilated_mask());
            impl_dil_mask!(usize, 7, DilMaskTestData::<$innert, 7>::dilated_mask());
            impl_dil_mask!(usize, 8, DilMaskTestData::<$innert, 8>::dilated_mask());
        };
    }
    #[cfg(target_pointer_width = "16")]
    impl_dil_mask_usize!(u16);
    #[cfg(target_pointer_width = "32")]
    impl_dil_mask_usize!(u32);
    #[cfg(target_pointer_width = "64")]
    impl_dil_mask_usize!(u64);

    struct UndilMaxTestData<T, const D: usize> {
        marker: PhantomData<T>,
    }

    macro_rules! impl_undil_max {
        ($t:ty, $d:literal, $v:expr) => {
            impl UndilMaxTestData<$t, $d> {
                #[inline]
                fn undilated_max() -> $t {
                    $v as $t
                }
            }
        };
    }
    impl_undil_max!(u8, 1, 0xff);
    impl_undil_max!(u8, 2, 0x0f);
    impl_undil_max!(u8, 3, 0x03);
    impl_undil_max!(u8, 4, 0x03);
    impl_undil_max!(u8, 5, 0x01);
    impl_undil_max!(u8, 6, 0x01);
    impl_undil_max!(u8, 7, 0x01);
    impl_undil_max!(u8, 8, 0x01);

    impl_undil_max!(u16, 1, 0xffff);
    impl_undil_max!(u16, 2, 0x00ff);
    impl_undil_max!(u16, 3, 0x001f);
    impl_undil_max!(u16, 4, 0x000f);
    impl_undil_max!(u16, 5, 0x0007);
    impl_undil_max!(u16, 6, 0x0003);
    impl_undil_max!(u16, 7, 0x0003);
    impl_undil_max!(u16, 8, 0x0003);

    impl_undil_max!(u32, 1, 0xffffffff);
    impl_undil_max!(u32, 2, 0x0000ffff);
    impl_undil_max!(u32, 3, 0x000003ff);
    impl_undil_max!(u32, 4, 0x000000ff);
    impl_undil_max!(u32, 5, 0x0000003f);
    impl_undil_max!(u32, 6, 0x0000001f);
    impl_undil_max!(u32, 7, 0x0000000f);
    impl_undil_max!(u32, 8, 0x0000000f);

    impl_undil_max!(u64, 1, 0xffffffffffffffff);
    impl_undil_max!(u64, 2, 0x00000000ffffffff);
    impl_undil_max!(u64, 3, 0x00000000001fffff);
    impl_undil_max!(u64, 4, 0x000000000000ffff);
    impl_undil_max!(u64, 5, 0x0000000000000fff);
    impl_undil_max!(u64, 6, 0x00000000000003ff);
    impl_undil_max!(u64, 7, 0x00000000000001ff);
    impl_undil_max!(u64, 8, 0x00000000000000ff);

    macro_rules! impl_undil_max_usize {
        ($innert:ty) => {
            impl_undil_max!(usize, 1, UndilMaxTestData::<$innert, 1>::undilated_max());
            impl_undil_max!(usize, 2, UndilMaxTestData::<$innert, 2>::undilated_max());
            impl_undil_max!(usize, 3, UndilMaxTestData::<$innert, 3>::undilated_max());
            impl_undil_max!(usize, 4, UndilMaxTestData::<$innert, 4>::undilated_max());
            impl_undil_max!(usize, 5, UndilMaxTestData::<$innert, 5>::undilated_max());
            impl_undil_max!(usize, 6, UndilMaxTestData::<$innert, 6>::undilated_max());
            impl_undil_max!(usize, 7, UndilMaxTestData::<$innert, 7>::undilated_max());
            impl_undil_max!(usize, 8, UndilMaxTestData::<$innert, 8>::undilated_max());
        };
    }
    #[cfg(target_pointer_width = "16")]
    impl_undil_max_usize!(u16);
    #[cfg(target_pointer_width = "32")]
    impl_undil_max_usize!(u32);
    #[cfg(target_pointer_width = "64")]
    impl_undil_max_usize!(u64);

    /*    static TEST_DATA_FILENAME: &'static str = "dilated_int_test_data.json";
    fn import_test_data()
    lazy_static! {
        static ref TEST_DATA: super::LocationTestData<$t> = super::LocationTestData::<$t>::import(String::from(LOCATION_TEST_DATA_PATH));
    }*/
    /*
    lazy_static! {
        #[cfg(not(has_i128))]
        static ref UNDILATED_TEST_CASES: Vec<u64> = vec![
            0x0000000000000000,
            0xFFFFFFFF00000000,
            0xFFFF0000FFFF0000,
            0xFF00FF00FF00FF00,
            0xF0F0F0F0F0F0F0F0,
            0xcccccccccccccccc,
            0xaaaaaaaaaaaaaaaa,
            0xffffffffffffffff,
            0x5555555555555555,
            0x3333333333333333,
            0x0F0F0F0F0F0F0F0F,
            0x00FF00FF00FF00FF,
            0x0000FFFF0000FFFF,
            0x00000000FFFFFFFF,
        ];

        #[cfg(has_i128)]
        static ref UNDILATED_TEST_CASES: Vec<u64> = vec![
            0x00000000000000000000000000000000,
            0xFFFFFFFFFFFFFFFF0000000000000000,
            0xFFFFFFFF00000000FFFFFFFF00000000,
            0xFFFF0000FFFF0000FFFF0000FFFF0000,
            0xFF00FF00FF00FF00FF00FF00FF00FF00,
            0xF0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0,
            0xcccccccccccccccccccccccccccccccc,
            0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
            0xffffffffffffffffffffffffffffffff,
            0x55555555555555555555555555555555,
            0x33333333333333333333333333333333,
            0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F,
            0x00FF00FF00FF00FF00FF00FF00FF00FF,
            0x0000FFFF0000FFFF0000FFFF0000FFFF,
            0x00000000FFFFFFFF00000000FFFFFFFF,
            0x0000000000000000FFFFFFFFFFFFFFFF,
        ];

        static ref DILATED_TEST_CASES_D1: Vec<u64> = vec![
            0x0000000000000000,
            0xFFFFFFFF00000000,
            0xFFFF0000FFFF0000,
            0xFF00FF00FF00FF00,
            0xF0F0F0F0F0F0F0F0,
            0xcccccccccccccccc,
            0xaaaaaaaaaaaaaaaa,
            0xffffffffffffffff,
            0x5555555555555555,
            0x3333333333333333,
            0x0F0F0F0F0F0F0F0F,
            0x00FF00FF00FF00FF,
            0x0000FFFF0000FFFF,
            0x00000000FFFFFFFF,
        ];

        static ref DILATED_TEST_CASES_D2: Vec<u64> = vec![
            0x00000000000000000000000000000000,
            0xAAAAAAAAAAAAAAAA0000000000000000,
            0xFFFF0000FFFF0000,
            0xFF00FF00FF00FF00,
            0xF0F0F0F0F0F0F0F0,
            0xcccccccccccccccc,
            0xaaaaaaaaaaaaaaaa,
            0xffffffffffffffff,
            0x5555555555555555,
            0x3333333333333333,
            0x0F0F0F0F0F0F0F0F,
            0x00FF00FF00FF00FF,
            0x0000FFFF0000FFFF,
            0x00000000FFFFFFFF,
        ];

        static ref TEST_CASES: [Vec<(u64, u64)>; 3] = [
            // D0 test cases (not used)
            Vec::default(),

            // D1 test cases (undilated, dilated)
            vec![
                (0x00000000, 0x00000000),
                (0xFFFF0000, 0xFFFF0000),
                (0xFF00FF00, 0xFF00FF00),
                (0xF0F0F0F0, 0xF0F0F0F0),
                (0xcccccccc, 0xcccccccc),
                (0xaaaaaaaa, 0xaaaaaaaa),
                (0xffffffff, 0xffffffff),
                (0x55555555, 0x55555555),
                (0x33333333, 0x33333333),
                (0x0F0F0F0F, 0x0F0F0F0F),
                (0x00FF00FF, 0x00FF00FF),
                (0x0000FFFF, 0x0000FFFF),
            ],

            // D2 test cases
            vec![
                (0x00000000, 0x0000000000000000),
                (0xFFFF0000, 0xAAAAAAAA00000000),
                (0xFF00FF00, 0xAAAA0000AAAA0000),
                (0xF0F0F0F0, 0xAA00AA00AA00AA00),
                (0xcccccccc, 0xA0A0A0A0A0A0A0A0),
                (0xaaaaaaaa, 0xaaaaaaaa),
                (0xffffffff, 0xffffffff),
                (0x55555555, 0x55555555),
                (0x33333333, 0x33333333),
                (0x0F0F0F0F, 0x0F0F0F0F),
                (0x00FF00FF, 0x00FF00FF),
                (0x0000FFFF, 0x0000FFFF),
            ],
        ];
    }*/

    macro_rules! integer_dilation_tests {
        ($t:ty, $($d:literal),+) => {$(
            paste! {
                mod [< integer_dilation_ $t _ $d d >] {
                    use super::{DilMaskTestData, UndilMaxTestData};
                    use super::super::{DilatedInt, DilatedMask, UndilatedMax};

                    #[test]
                    fn dilated_mask_correct() {
                        assert_eq!(DilatedInt::<$t, $d>::dilated_mask(), DilMaskTestData::<$t, $d>::dilated_mask());
                    }

                    #[test]
                    fn undilated_max_correct() {
                        assert_eq!(DilatedInt::<$t, $d>::undilated_max(), UndilMaxTestData::<$t, $d>::undilated_max());
                    }

                    #[test]
                    #[should_panic(expected = "Paremeter 'value' exceeds maximum")]
                    fn from_int_too_large_panics() {
                        // D1 dilated ints have no max value
                        // This is a hack, but it means we can run the same tests for all D values
                        if $d != 1 {
                            let _ = DilatedInt::<$t, $d>::from(UndilMaxTestData::<$t, $d>::undilated_max() + 1);
                        } else {
                            panic!("Paremeter 'value' exceeds maximum");
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
    integer_dilation_tests!(usize, 1, 2, 3, 4, 5, 6, 7, 8);
}
