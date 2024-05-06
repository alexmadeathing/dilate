use std::sync::OnceLock;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use dilate::{DilateFixed, DilatedInt, DilationMethod, Fixed};

pub trait BenchNumTraits {
    fn from_usize(value: usize) -> Self;
    fn to_usize(self) -> usize;
}

macro_rules! impl_num_traits {
    ($($t:ty),+) => {$(
        impl BenchNumTraits for $t {
            #[inline(always)]
            fn from_usize(value: usize) -> Self {
                value as Self
            }

            #[inline(always)]
            fn to_usize(self) -> usize {
                self as usize
            }
        }
    )+};
}

impl_num_traits!(u8, u16, u32, u64, u128);

const UNDILATED_MAX: usize = 255;

fn dilated_values(dimension: usize) -> &'static std::vec::Vec<usize> {
    static DILATED_VALUES: OnceLock<[std::vec::Vec<usize>; 9]> = OnceLock::new();
    &DILATED_VALUES.get_or_init(|| {
        [
            // D0 (not used)
            std::vec::Vec::new(),

            // D1 (not used)
            std::vec::Vec::new(),

            // D2 - D8
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<2>().value()).collect(),
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<3>().value()).collect(),
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<4>().value()).collect(),
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<5>().value()).collect(),
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<6>().value()).collect(),
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<7>().value()).collect(),
            (0..=UNDILATED_MAX).into_iter().map(|v| v.dilate_fixed::<8>().value()).collect(),
        ]
    })[dimension]
}

fn benchmark_dilate_undilate<FDil: Fn(usize) -> usize, FUndil: Fn(usize) -> usize>(
    c: &mut Criterion,
    d: usize,
    crate_name: &str,
    dilate: FDil,
    undilate: FUndil,
) {
    let dilated_values = dilated_values(d);

    c.bench_function(
        format!("{}d dilate: {}", d, crate_name).as_str(),
        |b| {
            b.iter(|| {
                let mut i = 0;
                while i <= UNDILATED_MAX {
                    dilate(i);
                    i += 1;
                }
            })
        },
    );

    c.bench_function(
        format!("{}d undilate: {}", d, crate_name).as_str(),
        |b| {
            b.iter(|| {
                let mut i = 0;
                while i <= UNDILATED_MAX {
                    undilate(dilated_values[i]);
                    i += 1;
                }
            })
        },
    );
}

fn benchmark<const D: usize>(c: &mut Criterion)
where
    Fixed<u16, D>: DilationMethod,
    Fixed<u32, D>: DilationMethod,
    Fixed<u64, D>: DilationMethod,
    Fixed<u128, D>: DilationMethod,
    <Fixed<u16, D> as DilationMethod>::Dilated: BenchNumTraits,
    <Fixed<u16, D> as DilationMethod>::Undilated: BenchNumTraits,
    <Fixed<u32, D> as DilationMethod>::Dilated: BenchNumTraits,
    <Fixed<u32, D> as DilationMethod>::Undilated: BenchNumTraits,
    <Fixed<u64, D> as DilationMethod>::Dilated: BenchNumTraits,
    <Fixed<u64, D> as DilationMethod>::Undilated: BenchNumTraits,
    <Fixed<u128, D> as DilationMethod>::Dilated: BenchNumTraits,
    <Fixed<u128, D> as DilationMethod>::Undilated: BenchNumTraits,
{
    benchmark_dilate_undilate(
        c,
        D,
        "u16",
        |i| black_box(Fixed::<u16, D>::dilate(black_box(<Fixed<u16, D> as DilationMethod>::Undilated::from_usize(i))).value().to_usize()),
        |i| {
            black_box(Fixed::<u16, D>::undilate(black_box(DilatedInt::<Fixed<u16, D>>::new(<Fixed<u16, D> as DilationMethod>::Dilated::from_usize(i)))).to_usize())
        },
    );
    benchmark_dilate_undilate(
        c,
        D,
        "u32",
        |i| black_box(Fixed::<u32, D>::dilate(black_box(<Fixed<u32, D> as DilationMethod>::Undilated::from_usize(i))).value().to_usize()),
        |i| {
            black_box(Fixed::<u32, D>::undilate(black_box(DilatedInt::<Fixed<u32, D>>::new(<Fixed<u32, D> as DilationMethod>::Dilated::from_usize(i)))).to_usize())
        },
    );
    benchmark_dilate_undilate(
        c,
        D,
        "u64",
        |i| black_box(Fixed::<u64, D>::dilate(black_box(<Fixed<u64, D> as DilationMethod>::Undilated::from_usize(i))).value().to_usize()),
        |i| {
            black_box(Fixed::<u64, D>::undilate(black_box(DilatedInt::<Fixed<u64, D>>::new(<Fixed<u64, D> as DilationMethod>::Dilated::from_usize(i)))).to_usize())
        },
    );
    benchmark_dilate_undilate(
        c,
        D,
        "u128",
        |i| black_box(Fixed::<u128, D>::dilate(black_box(<Fixed<u128, D> as DilationMethod>::Undilated::from_usize(i))).value().to_usize()),
        |i| {
            black_box(Fixed::<u128, D>::undilate(black_box(DilatedInt::<Fixed<u128, D>>::new(<Fixed<u128, D> as DilationMethod>::Dilated::from_usize(i)))).to_usize())
        },
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    benchmark::<2>(c);
    benchmark::<3>(c);
    benchmark::<4>(c);

    benchmark::<8>(c); // Will force use of Dn methods
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(1000);
    targets = criterion_benchmark
);
criterion_main!(benches);
