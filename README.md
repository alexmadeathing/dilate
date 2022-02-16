# WARNING
This library is extremely work in progress - use only if you're willing to do a bit of leg work or help finish it.

# dilate
A compact, high performance integer dilation library for Rust.

Integer dilation is the process of converting cartesian indices (eg. coordinates) into a D-dimensional [Morton Order](https://en.wikipedia.org/wiki/Z-order_curve) sequence of bits. The dilation process takes an integer's bit sequence and inserts a number of 0 bits (D - 1) between each original bit successively. Thus, the original bit sequence becomes evenly padded. For example:
* `0b1101` 2-dilated becomes `0b1010001` (values chosen arbitrarily)
* `0b1011` 3-dilated becomes `0b1000001001`

The process of undilation, or 'contraction', does the opposite:
* `0b1010001` 2-contracted becomes `0b1101`
* `0b1000001001` 3-contracted becomes `0b1011`

This library provides efficient casting to and from ordinary integer representation as well as various efficient mathematical operations between dilated integers.

# Goals
* High performance - Ready to use in performance sensitive contexts
* N-dimensional - Suitable for multi-dimensional applications (up to 16 dimensions)
* Trait based implementation - Conforms to standard Rust implementation patterns
* No dependencies - Depends on Rust standard library only

# References and Acknowledgments
Many thanks to the authors of the following white papers:
* [1] Converting to and from Dilated Integers - Rajeev Raman and David S. Wise
* [2] Integer Dilation and Contraction for Quadtrees and Octrees - Leo Stocco and Gunther Schrack
* [3] Fast Additions on Masked Integers - Michael D Adams and David S Wise

Permission has been explicitly granted to reproduce the agorithms within each paper.
