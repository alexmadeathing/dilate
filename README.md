# WARNING
This library is work in progress. We currently support dilation and undilation of types u8, u16, u32, u64, u128, usize. We do not yet support arithmetic on dilated integers.

# dilate
A compact, high performance integer dilation library for Rust.

Integer dilation is the process of converting cartesian indices (eg. coordinates) into a format suitable for use in D-dimensional [Morton Order](https://en.wikipedia.org/wiki/Z-order_curve) bit sequences. The dilation process takes an integer's bit sequence and inserts a number of 0 bits (D - 1) between each original bit successively. Thus, the original bit sequence becomes evenly padded. For example:
* `0b1101` D2-dilated becomes `0b1010001` (values chosen arbitrarily)
* `0b1011` D3-dilated becomes `0b1000001001`

The process of undilation, or 'contraction', does the opposite:
* `0b1010001` D2-contracted becomes `0b1101`
* `0b1000001001` D3-contracted becomes `0b1011`

This library provides efficient casting to and from ordinary integer representation as well as various efficient mathematical operations between dilated integers.

# Goals
* High performance - Ready to use in performance sensitive contexts
* Multiple types - Supports u8, u16, u32, u64, u128, usize (signed versions not yet planned)
* N-dimensional - Suitable for multi-dimensional applications (up to 8 dimensions)
* Trait based implementation - Conforms to standard Rust implementation patterns
* No dependencies - Depends on Rust standard library only

# Docs
TODO

# Roadmap
Please refer to the [Roadmap to V1.0](https://github.com/alexmadeathing/dilate/discussions/2) discussion.

# Contributing
Contributions are most welcome.

For bugs reports, please [submit a bug report](https://github.com/alexmadeathing/dilate/issues/new?assignees=&labels=bug&template=bug_report.md&title=).

For feature requests, please [submit a feature request](https://github.com/alexmadeathing/dilate/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=).

If you have ideas and want to contribute directly, please start by creating an [Idea discussion](https://github.com/alexmadeathing/dilate/discussions/new) in the discussions area. Allow others to comment prior to committing to doing the work. When all parties agree on the design, the work may begin. When your code is ready to be published, please submit a pull request referring back to your Idea discussion. We are unlikely to accept a pull request that has not gone through this process, unless it is for a very small change.

# References and Acknowledgments
Many thanks to the authors of the following white papers:
* [1] Converting to and from Dilated Integers - Rajeev Raman and David S. Wise
* [2] Integer Dilation and Contraction for Quadtrees and Octrees - Leo Stocco and Gunther Schrack
* [3] Fast Additions on Masked Integers - Michael D Adams and David S Wise

Permission has been explicitly granted to reproduce the agorithms within each paper.

# License

dilate is licensed under the [Anti-Capitalist Software License (v 1.4)](https://github.com/alexmadeathing/dilate/blob/main/LICENSE.md). This means it is free and open source for use by individuals and organizations that do not operate by capitalist principles.
