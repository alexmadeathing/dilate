[![Crates.io](https://img.shields.io/crates/d/dilate.svg)](https://crates.io/crates/dilate)
[![alexmadeathing](https://circleci.com/gh/alexmadeathing/dilate.svg?style=shield)](https://app.circleci.com/pipelines/github/alexmadeathing/dilate?filter=all)
[![MIT License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE-MIT)
[![APACHE 2.0 License](https://img.shields.io/badge/license-APACHE%202.0-brightgreen)](LICENSE-APACHE)

# WARNING
This library is in an alpha stage of development. It is feature complete at a basic level, its interface may be subject to change.

For migration notes, please see: https://github.com/alexmadeathing/dilate/releases

# dilate
A compact, high performance integer dilation library for Rust.

Integer dilation is the process of converting cartesian indices (eg.
coordinates) into a format suitable for use in D-dimensional algorithms
such [Morton Order](https://en.wikipedia.org/wiki/Z-order_curve) curves.
The dilation process takes an integer's bit sequence and inserts a number
of 0 bits (`D - 1`) between each original bit successively. Thus, the
original bit sequence becomes evenly padded. For example:
* `0b1101` D2-dilated becomes `0b1010001`
* `0b1011` D3-dilated becomes `0b1000001001`

The process of undilation, or 'contraction', does the opposite:
* `0b1010001` D2-undilated becomes `0b1101`
* `0b1000001001` D3-undilated becomes `0b1011`

This libary also supports a limited subset of arthimetic operations on dilated
integers via the standard rust operater traits. Whilst slightly more involved
than regular integer arithmetic, these operations are still highly performant.

# Supported Dilations
For more information on the supported dilations and possible type
combinations, please see
[Supported Dilations via Expand](https://docs.rs/dilate/latest/dilate/expand/trait.DilateExpand.html#supported-expand-dilations)
and
[Supported Dilations via Fixed](https://docs.rs/dilate/latest/dilate/fixed/trait.DilateFixed.html#supported-fixed-dilations).

# Features
* High performance - Ready to use in performance sensitive contexts
* N-dimensional - Suitable for multi-dimensional applications (up to 16 dimensions under certain conditions)
* Type safe - Multiple input types with known output types (supports `u8`, `u16`, `u32`, `u64`, `u128`, `usize`)
* `no_std` - Suitable for embedded devices (additional standard library features can be enabled via the `std` feature)
* Extensible - Flexible trait based implementation
* No dependencies - Keeps your dependency tree clean

# Getting Started
First, link dilate into your project's cargo.toml.

Check for the latest version at [crates.io](https://crates.io/crates/dilate):
```toml
[dependencies]
dilate = "0.6.3"
# dilate = { version = "0.6.3", features = ["std"] } <- For std features like Add, Sub and Display
```

Next, import dilate into your project and try out some of the features:

```rust
use dilate::*;

let original: u8 = 0b1101;

// Dilating
let dilated = original.dilate_expand::<2>();
assert_eq!(dilated.value(), 0b1010001);

// This is the actual dilated type
assert_eq!(dilated, DilatedInt::<Expand<u8, 2>>(0b1010001));

// Undilating
assert_eq!(dilated.undilate(), original);
```
*Example 2-dilation and undilation usage*

```rust
use dilate::*;

let original: u8 = 0b1011;

// Dilating
let dilated = original.dilate_expand::<3>();
assert_eq!(dilated.value(), 0b1000001001);

// This is the actual dilated type
assert_eq!(dilated, DilatedInt::<Expand<u8, 3>>(0b1000001001));

// Undilating
assert_eq!(dilated.undilate(), original);
```
*Example 3-dilation and undilation usage*

For more detailed info, please see the [code reference](https://docs.rs/dilate/latest/dilate/).

# Roadmap
Please refer to the [Roadmap to V1.0](https://github.com/alexmadeathing/dilate/discussions/2) discussion.

# Contributing
Contributions are most welcome.

For bugs reports, please [submit a bug report](https://github.com/alexmadeathing/dilate/issues/new?assignees=&labels=bug&template=bug_report.md&title=).

For feature requests, please [submit a feature request](https://github.com/alexmadeathing/dilate/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=).

If you have ideas and want to contribute directly, pull requests are most welcome. If it's a small change just submit a pull request. If you're planning a large or breaking change, please create an [Idea discussion](https://github.com/alexmadeathing/dilate/discussions/new) in the discussions area so that we can reach a concensus.

# References and Acknowledgments
Many thanks to the authors of the following white papers:
* \[1\] Converting to and from Dilated Integers - Rajeev Raman and David S. Wise
* \[2\] Integer Dilation and Contraction for Quadtrees and Octrees - Leo Stocco and Gunther Schrack
* \[3\] Fast Additions on Masked Integers - Michael D Adams and David S Wise

Permission has been explicitly granted to reproduce the agorithms within each paper.

# License
Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
