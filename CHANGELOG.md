## Version 0.4.0-SNAPSHOT

*Work-in-progress for QuPath v0.6.0*

* Compatibility with QuPath v0.6.0


## Version 0.3.0

* Compatibility with QuPath v0.5.0
  * Improved `DnnModel` implementation with better memory management
* New command to generate a launch script, optionally using a conda environment
  * This makes it easier to manage CUDA/cuDNN via conda
* Show more useful information in the engine download dialog
  * CUDA version if available
  * GPU compute capability
  * DJL and Engine versions
* Externalize strings

## Version 0.2.0

* Improve conversion of `NDArray` to more data types
  * Add `DjlTools.getXXX()` methods to get ints, floats, doubles, longs and booleans
* Estimate output size in `DjlDnnModel` if shape doesn't match NDLayout
  * This relaxes the assumption that the output layout should match the input
* New `DjlTools.get/setOverrideDevice()` methods to override DJL's default device selection
  * Primarily intended to explore `Device.fromName('mps')` on Apple Silicon (which sometimes works, sometimes doesn't...)

## Version 0.1.0

* Initial release