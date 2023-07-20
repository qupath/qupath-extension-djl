## Version 0.2.0-SNAPSHOT

* Improve conversion of `NDArray` to more data types
  * Add `DjlTools.getXXX()` methods to get ints, floats, doubles, longs and booleans
* Estimate output size in `DjlDnnModel` if shape doesn't match NDLayout
  * This relaxes the assumption that the output layout should match the input


## Version 0.1.0

* Initial release