## Version 0.2.0

* Improve conversion of `NDArray` to more data types
  * Add `DjlTools.getXXX()` methods to get ints, floats, doubles, longs and booleans
* Estimate output size in `DjlDnnModel` if shape doesn't match NDLayout
  * This relaxes the assumption that the output layout should match the input
* New `DjlTools.get/setOverrideDevice()` methods to override DJL's default device selection
  * Primarily intended to explore `Device.fromName('mps')` on Apple Silicon (which sometimes works, sometimes doesn't...)


## Version 0.1.0

* Initial release