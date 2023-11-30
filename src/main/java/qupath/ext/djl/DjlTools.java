/*-
 * Copyright 2022 QuPath developers, University of Edinburgh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.djl;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.djl.Device;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacpp.indexer.BooleanIndexer;
import org.bytedeco.javacpp.indexer.ByteIndexer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.HalfIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacpp.indexer.UShortIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.dnn.DnnShape;
import qupath.opencv.tools.OpenCVTools;

/**
 * Tools to help work with Deep Java Library within QuPath.
 * 
 * @author Pete Bankhead
 */
public class DjlTools {
	
	private static final Logger logger = LoggerFactory.getLogger(DjlTools.class);
	
	/**
	 * PyTorch engine name
	 */
	public static String ENGINE_PYTORCH      = "PyTorch";
	
	/**
	 * TensorFlow engine name
	 */
	public static String ENGINE_TENSORFLOW   = "TensorFlow";
	
	/**
	 * MXNet engine name
	 */
	public static String ENGINE_MXNET        = "MXNet";
	
	/**
	 * TensorFlow Lite engine name
	 */
	public static String ENGINE_TFLITE       = "TFLite";
	
	/**
	 * ONNX Runtime engine name
	 */
	public static String ENGINE_ONNX_RUNTIME = "OnnxRuntime";
	
	/**
	 * XGBoost engine name
	 */
	public static String ENGINE_XGBOOST      = "XGBoost";
	
	/**
	 * LightGBM engine name
	 */
	public static String ENGINE_LIGHTGBM     = "LightGBM";
	
	/**
	 * Neo DLR engine name
	 */
	public static String ENGINE_DLR          = "DLR";
	
	/**
	 * TensorRT engine name
	 */
	public static String ENGINE_TENSORRT     = "TensorRT";
	
	/**
	 * PaddlePaddle engine name
	 */
	public static String ENGINE_PADDLEPADDLE = "PaddlePaddle";
	
	/**
	 * Maintain a set of all engines that are known to have been loaded.
	 * This enables us to check for available (and downloaded) engines without 
	 * having to try to instantiate a new one.
	 */
	public static Set<String> loadedEngines = new HashSet<>();

	/**
	 * Default devices for each engine.
	 * This can be used to override the default used by DJL.
	 */
	private static Map<String, Device> defaultDevices = new HashMap<>();
	
	static Set<String> ALL_ENGINES = Set.of(
			ENGINE_DLR, ENGINE_LIGHTGBM, ENGINE_MXNET, ENGINE_ONNX_RUNTIME, ENGINE_PADDLEPADDLE,
			ENGINE_PYTORCH, ENGINE_TENSORFLOW, ENGINE_TENSORRT, ENGINE_TFLITE, ENGINE_XGBOOST
			);
	
	
	/**
	 * Create a {@link DnnModel} wrapping a model from Deep Java Library.
	 * @param uri URI of the model file
	 * @param ndLayout a layout string, e.g. NCHW
	 * @param inputShape expected input shape, according to ndLayout
	 * @return
	 */
	public static DnnModel createDnnModel(URI uri, String ndLayout, int[] inputShape) {
		DnnShape shape = null;
		if (inputShape != null)
			shape = DnnShape.of(Arrays.stream(inputShape).mapToLong(i -> i).toArray());
		return createDnnModel(null, uri, ndLayout, Map.of(DnnModel.DEFAULT_INPUT_NAME, shape), null);
	}
	
	/**
	 * Create a {@link DnnModel} wrapping a model from Deep Java Library.
	 * @param engine name of the engine to use, or null to try to determine this from the URIs
	 * @param uri URI of the model file
	 * @param ndLayout a layout string, e.g. NCHW
	 * @param inputs input shapes, if known; if these are null, an attempt will be made to get them from DJL (but this does not always work)
	 * @param outputs outputs shapes, if known; if these are null, an attempt will be made to get them from DJL (but this does not always work)
	 * @return
	 */
	public static DnnModel createDnnModel(String engine, URI uri, String ndLayout, Map<String, DnnShape> inputs, Map<String, DnnShape> outputs) {
		return createDnnModel(engine, Collections.singletonList(uri), ndLayout, inputs, outputs);
	}
	
	/**
	 * Create a {@link DnnModel} wrapping a model from Deep Java Library, based on reading from multiple URIs.
	 * @param engine name of the engine to use, or null to try to determine this from the URIs
	 * @param uris one or more URIs needed to load the model file
	 * @param ndLayout a layout string, e.g. NCHW
	 * @param inputs input shapes, if known; if these are null, an attempt will be made to get them from DJL (but this does not always work)
	 * @param outputs outputs shapes, if known; if these are null, an attempt will be made to get them from DJL (but this does not always work)
	 * @return
	 */
	private static DnnModel createDnnModel(String engine, Collection<URI> uris, String ndLayout, Map<String, DnnShape> inputs, Map<String, DnnShape> outputs) {
		return new DjlDnnModel(engine, uris, ndLayout, inputs, outputs, false); // Eagerly initialize (so we know if it doesn't work sooner)
	}
	
	/**
	 * Check if an Deep Java Library engine is potentially available.
	 * Note that this does not necessarily mean that it has been downloaded or is supported 
	 * on this platform, but only that the necessary engine jar is on the classpath.
	 * @param name
	 * @return true if the engine jars are on the classpath, false otherwise
	 * @see #getEngine(String, boolean)
	 * @see #isEngineAvailable(String)
	 */
	public static boolean hasEngine(String name) {
		return Engine.hasEngine(name);
	}
	
	/**
	 * Check if an engine is available and ready for use without any additional downloading.
	 * This first checks whether an engine with the given name has already been seen 
	 * by this class; if so, the method returns true without attempting to instantiate 
	 * the engine.
	 * Otherwise, the method returns true only if {@code hasEngine(name)} is false or if 
	 * {@code getEngine(name, false)} is not null.
	 * @param name
	 * @return
	 * @see #hasEngine(String)
	 * @see #getEngine(String, boolean)
	 */
	public static boolean isEngineAvailable(String name) {
		if (loadedEngines.contains(name))
			return true;
		if (!hasEngine(name))
			return false;
		logger.debug("Need to try to get engine to test availability");
		return getEngine(name, false) != null;
	}
	
	private static final Object lock = new Object();
	
	
	/**
	 * Get an {@link Engine} by name.
	 * <p>
	 * This is similar to {@link Engine#getEngine(String)} except provides control over 
	 * whether the native libraries for the engine are downloaded if required.
	 * This avoids unexpectedly long blocking calls for an engine request if the native 
	 * libraries need to be downloaded.
	 * 
	 * @param name the name of the engine
	 * @param downloadIfNeeded if true, download the necessary native libraries if needed.
	 *                         If false, return null if the engine cannot be loaded.
	 * @return
	 * @throws IllegalArgumentException if the engine is not available, which means that {@link #hasEngine(String)} returns false
	 * @see #hasEngine(String)
	 * @see #isEngineAvailable(String)
	 */
	public static Engine getEngine(String name, boolean downloadIfNeeded) throws IllegalArgumentException {
		if (!hasEngine(name)) {
			throw new IllegalArgumentException("Requested engine " + name + " is not available!");
		}
				
		synchronized (lock) {
			var offlineStatus = System.getProperty("offline");
			try {
				if (downloadIfNeeded)
					System.setProperty("offline", "false");
				else
					System.setProperty("offline", "true");
				
				var engine = Engine.getEngine(name);
				if (engine != null)
					loadedEngines.add(name);
				return engine;
			} catch (Exception e) {
				if (downloadIfNeeded)
					logger.error("Unable to get engine " + name + ": " + e.getMessage(), e);
				else {
					var msg = e.getLocalizedMessage();
					if (msg == null)
						logger.warn("Unable to get engine {}", name);
					else
						logger.warn("Unable to get engine {} ({})", name, e.getMessage());
				}
				return null;
			} finally {
				System.setProperty("offline", offlineStatus);
			}
		}
	}
	
	
	
	static DnnShape convertShape(Shape shape) {
		return DnnShape.of(shape.getShape());
	}

	static ZooModel<NDList, NDList> loadModel(String engineName, URI... uris) throws ModelNotFoundException, MalformedModelException, IOException {
		return loadModel(engineName, NDList.class, NDList.class, null, uris);
	}

	static <P, Q> ZooModel<P, Q> loadModel(String engineName, Class<P> inputClass, Class<Q> outputClass, Translator<P, Q> translator, URI... uris) throws ModelNotFoundException, MalformedModelException, IOException {
		var sb = new StringBuilder();
		boolean isFirst = true;
		for (var uri : uris) {
			if (isFirst)
				isFirst = false;
			else
				sb.append(",");
			// TODO: Handle unzipping zipped models to a temporary directory, if needed
			var s = uri.toString();
			if (s.toLowerCase().startsWith("jar:file:") || s.toLowerCase().endsWith(".zip")) {
				logger.warn("Model URI is zipped - please unzip the model and recreate it");
			}
			sb.append(uri.toString());
		}
		return loadModel(engineName, inputClass, outputClass, translator, sb.toString());
	}

	private static <P, Q> ZooModel<P, Q> loadModel(String engineName, Class<P> inputClass, Class<Q> outputClass, Translator<P, Q> translator, String urls) throws ModelNotFoundException, MalformedModelException, IOException {
		var builder = Criteria.builder()
				.setTypes(inputClass, outputClass)
				.optModelUrls(urls)
				.optTranslator(translator)
				.optProgress(new ProgressBar());
		
		String selectedEngine = null;
		if (engineName != null) {
			if (Engine.getAllEngines().contains(engineName)) {
				selectedEngine = engineName;
			}
		}
		
		// Try to figure out the engine name
		if (selectedEngine == null) {
			var urlString = urls.toString().toLowerCase();
			if (urlString.endsWith(".onnx") && Engine.hasEngine("OnnxRuntime"))
				selectedEngine = "OnnxRuntime";
			else if ((urlString.endsWith("pytorch") || urlString.endsWith(".pt")) && Engine.hasEngine("PyTorch"))
				selectedEngine = "PyTorch";
			else if (urlString.endsWith(".tflite") && Engine.hasEngine("TFLite"))
				selectedEngine = "TFLite";
			else if ((urlString.endsWith(".pb") || urlString.endsWith("tf_savedmodel.zip") || urlString.endsWith("tf_savedmodel")) && Engine.hasEngine("TensorFlow"))
				selectedEngine = "TensorFlow";
		}

		if (selectedEngine != null) {
			builder.optEngine(selectedEngine);
			var device = defaultDevices.getOrDefault(selectedEngine, null);
			if (device != null) {
				builder.optDevice(device);
				builder.optOption("mapLocation", "true");
			}
		}
		
		var criteria = builder.build();
		return ModelZoo.loadModel(criteria);		
	}


	/**
	 * Set the default device for the specified engine.
	 * This will be used only whenever the model is build using this class, overriding
	 * DJL's default.
	 * <p>
	 * Note that the default device chosen automatically by DJL is usually fine,
	 * and so it is generally not necessary to set this.
	 * However it can be useful for exploring, or if DJL does not use the device you want.
	 * @param engineName
	 * @param device
	 */
	public static void setOverrideDevice(String engineName, Device device) {
		if (device == null)
			defaultDevices.remove(engineName);
		else
			defaultDevices.put(engineName, device);
	}

	/**
	 * Get the default device for the specified engine, which overrides DJL's default device for
	 * the specified engine.
	 * @param engineName
	 * @return the default device, or null if not set
	 */
	public static Device getOverrideDevice(String engineName) {
		return defaultDevices.getOrDefault(engineName, null);
	}

//	static ZooModel<Mat, Mat> loadModelCV(URI uri, String ndLayout) throws ModelNotFoundException, MalformedModelException, IOException {
//		var criteria = Criteria.builder()
//				.setTypes(Mat.class, Mat.class)
//				.optModelUrls(uri.toString())
//				.optProgress(new ProgressBar())
//				.optTranslator(new MatTranslator(ndLayout, ndLayout))
//				.build();
//		return ModelZoo.loadModel(criteria);		
//	}

	
	static Mat predict(Model model, Mat mat) throws TranslateException {
		try (var predictor = model.newPredictor(new MatTranslator("CHW", "CHW"))) {
			return predictor.batchPredict(Collections.singletonList(mat)).get(0);
		}
	}
	
	
	/**
	 * Convert an Opencv {@link Mat} to a Deep Java Library {@link NDArray}.
	 * Note that this ass
	 * @param manager an {@link NDManager}, required to create the NDArray
	 * @param mat the mat to convert
	 * @param ndLayout a layout string for the NDArray, e.g. "CHW"; currently, HW must appear together (in that order)
	 * @return an NDArray containing the values in the Mat, with the specified layout
	 */
	public static NDArray matToNDArray(NDManager manager, Mat mat, String ndLayout) {
		var dataType = getDataType(mat);
		if (dataType == DataType.UNKNOWN)
			throw new IllegalArgumentException("Unsupported data type for " + mat);
		var shape = getShape(mat, ndLayout);
		int indC = ndLayout.indexOf("C");
		int indHW = ndLayout.indexOf("HW");
		if (indHW < 0)
			throw new IllegalArgumentException("Expected layout contains HW, but provided layout is " + ndLayout);
		// Copy all at once is using the same storage order as OpenCV
		// TODO: Check what this order is!!!
		NDArray array = null;
		if (indC > indHW || shape.get(indC) == 1) {
			var buffer = mat.createBuffer();
			array = manager.create(buffer, shape, dataType);			
		} else {
			var shapeDims = shape.getShape();
			shapeDims[indC] = 1;
			var shapeChannel = new Shape(shapeDims, shape.getLayout());

			for (var mat2 : OpenCVTools.splitChannels(mat)) {
				var buffer = mat2.createBuffer();
				var arrayTemp = manager.create(buffer, shapeChannel, dataType);
				if (array == null)
					array = arrayTemp;
				else {
					var arrayTemp2 = array.concat(arrayTemp, indC);
					array.close();
					arrayTemp.close();
					array = arrayTemp2;
				}
			}
		}
		return array;
	}
	
	/**
	 * Convert an {@link NDArray} into an OpenCV {@link Mat}, automatically squeezing singleton dimensions.
	 * The {@link Mat} should have no more than 3 dimensions (height, width and channels).
	 * 
	 * @param array the NDArray to convert
	 * @param ndLayout a layout string, e.g. NCHW. If this is null, an attempt will be made to request the layout 
	 *                 from the array - however this can often fail if the dimensions are unknown.
	 * @return a {@link Mat} with pixels corresponding to the {@link NDArray}.
	 */
	public static Mat ndArrayToMat(NDArray array, String ndLayout) {
		return ndArrayToMat(array, ndLayout, true);
	}
	
	/**
	 * Convert an {@link NDArray} into an OpenCV {@link Mat}, optionally squeezing singleton dimensions.
	 * The {@link Mat} should have no more than 3 dimensions (height, width and channels).
	 * 
	 * @param array the NDArray to convert
	 * @param ndLayout a layout string, e.g. NCHW. If this is null, an attempt will be made to request the layout 
	 *                 from the array - however this can often fail if the dimensions are unknown.
	 * @param doSqueeze if true, squeeze singleton dimensions
	 * @return a {@link Mat} with pixels corresponding to the {@link NDArray}.
	 */
	// TODO: Check this for corner cases, and arrays with unexpected layouts
	public static Mat ndArrayToMat(NDArray array, String ndLayout, boolean doSqueeze) {
		var dataType = array.getDataType();
		var shape = array.getShape();
		if (ndLayout == null) {
			if (shape.isLayoutKnown())
				ndLayout = LayoutType.toString(shape.getLayout());
			else
				throw new IllegalArgumentException("Can't convert ndArray to Mat - layout is unknown");
		}
		
		// Get dimensions, trimming any leading/trailing ones
		int nDim = shape.dimension();
		int nLeading = doSqueeze ? shape.getLeadingOnes() : 0;
		int nTrailing = doSqueeze ? shape.getTrailingOnes() : 0;
		int[] dims = new int[nDim - nLeading - nTrailing];
		for (int i = 0; i < dims.length; i++) {
			dims[i] = (int)shape.get(i + nLeading);
		}
		if (doSqueeze) {
			array = array.squeeze();
			ndLayout = ndLayout.substring(nLeading, ndLayout.length()-nTrailing);
		}
		
		// If we have multiple channels, we might need to handle them separately
		int indH = ndLayout.indexOf("H");
		int indW = ndLayout.indexOf("W");
		int indC = ndLayout.indexOf("C");
		int height    = indH >= 0 && indH < dims.length ? dims[indH] : 1;
		int width     = indW >= 0 && indW < dims.length ? dims[indW] : 1;
		int nChannels = indC >= 0 && indC < dims.length ? dims[indC] : 1;
		if (nChannels > 1 && indC >= 0 && indC < indH) {
			Mat mat = new Mat();
			try (var scope = new PointerScope()) {
				List<Mat> channels = new ArrayList<>();
				try (var list = array.split(nChannels, indC)) {
					for (var ndChannel : list) {
						channels.add(ndArrayToMat(ndChannel, ndLayout, false));
						ndChannel.close();
					}
				}
				OpenCVTools.mergeChannels(channels, mat);
			}
			return mat;
		}
		
		Mat mat;
		var cvDepth = getMatDepth(dataType);
		if (dims.length <= 3 && width * height * nChannels == array.size()) {
			mat = new Mat(height, width, opencv_core.CV_MAKETYPE(cvDepth, nChannels));
		} else {
			mat = new Mat(dims, cvDepth);
		}
		try (var indexer = mat.createIndexer()) {
			if (indexer instanceof ByteIndexer) {
				((ByteIndexer) indexer).put(0L, array.toByteArray());
			} else if (indexer instanceof UByteIndexer) {
				((UByteIndexer) indexer).put(0L, getInts(array));
			} else if (indexer instanceof UShortIndexer) {
				((UShortIndexer) indexer).put(0L, getInts(array));
			} else if (indexer instanceof IntIndexer) {
				((IntIndexer) indexer).put(0L, getInts(array));
			} else if (indexer instanceof FloatIndexer) {
				((FloatIndexer) indexer).put(0L, getFloats(array));
			} else if (indexer instanceof HalfIndexer) {
				((HalfIndexer) indexer).put(0L,getFloats(array));
			} else if (indexer instanceof DoubleIndexer) {
				((DoubleIndexer) indexer).put(0L, getDoubles(array));
			} else if (indexer instanceof LongIndexer) {
				((LongIndexer) indexer).put(0L, getLongs(array));
			} else if (indexer instanceof BooleanIndexer) {
				((BooleanIndexer) indexer).put(0L, getBooleans(array));
			} else
				throw new IllegalArgumentException("Unable to convert array " + array + " to Mat");
		}
		return mat;
	}

	/**
	 * Extract array values as longs, converting if necessary.
	 * @param array
	 * @return
	 */
	public static long[] getLongs(NDArray array) {
		if (array.getDataType() == DataType.INT64) {
			try {
				return array.toLongArray();
			} catch (Exception e) {
				logger.error("Exception requesting longs from NDArray");
			}
		}
		return array.toType(DataType.INT64, true).toLongArray();
	}

	/**
	 * Extract array values as booleans, converting if necessary.
	 * @param array
	 * @return
	 */
	private static boolean[] getBooleans(NDArray array) {
		if (array.getDataType() == DataType.BOOLEAN) {
			try {
				return array.toBooleanArray();
			} catch (Exception e) {
				logger.error("Exception requesting ints from NDArray");
			}
		}
		return array.toType(DataType.BOOLEAN, true).toBooleanArray();
	}

	/**
	 * Extract array values as ints, converting if necessary.
	 * @param array
	 * @return
	 */
	private static int[] getInts(NDArray array) {
		if (array.getDataType() == DataType.INT32) {
			try {
				return array.toIntArray();
			} catch (Exception e) {
				logger.error("Exception requesting ints from NDArray");
			}
		} else if (array.getDataType() == DataType.UINT8) {
			try {
				return array.toUint8Array();
			} catch (Exception e) {
				logger.error("Exception requesting ints from NDArray");
			}
		}
		return array.toType(DataType.INT32, true).toIntArray();
	}

	/**
	 * Extract array values as doubles, converting if necessary.
	 * @param array
	 * @return
	 */
	private static double[] getDoubles(NDArray array) {
		if (array.getDataType() == DataType.FLOAT64) {
			try {
				return array.toDoubleArray();
			} catch (Exception e) {
				logger.error("Exception requesting doubles from NDArray");
			}
		}
		return array.toType(DataType.FLOAT64, true).toDoubleArray();
	}

	/**
	 * Extract array values as floats, converting if necessary.
	 * @param array
	 * @return
	 */
	private static float[] getFloats(NDArray array) {
		if (array.getDataType() == DataType.FLOAT32 || array.getDataType() == DataType.FLOAT16) {
			try {
				return array.toFloatArray();
			} catch (Exception e) {
				logger.error("Exception requesting floats from NDArray", e);
			}
		}
		return array.toType(DataType.FLOAT32, true).toFloatArray();
	}
	
	
	static class MatTranslator implements Translator<Mat, Mat> {
		
		private String inputLayoutNd, outputLayoutNd;
		
		MatTranslator(String inputLayoutNd, String outputLayoutNd) {
			this.inputLayoutNd = inputLayoutNd;
			this.outputLayoutNd = outputLayoutNd;
		}

		/**
		 * Convert Mat to NDArray and add to an NDList.
		 * Note that not all OpenCV types are supported.
		 * Specifically, 16-bit types should be avoided.
		 */
		@Override
		public NDList processInput(TranslatorContext ctx, Mat input) throws Exception {
			var ndarray = matToNDArray(ctx.getNDManager(), input, inputLayoutNd);
			return new NDList(ndarray);
		}

		@Override
		public Mat processOutput(TranslatorContext ctx, NDList list) throws Exception {
			var array = list.get(0);
			return ndArrayToMat(array, outputLayoutNd);
		}
		
	}
	
	
	static Shape getShape(Mat mat, String ndLayout) {
		List<Pair<Long, LayoutType>> pairs = new ArrayList<>();
		for (var layout : LayoutType.fromValue(ndLayout)) {
			switch (layout) {
			case CHANNEL:
				pairs.add(new Pair<>((long)mat.arrayChannels(), layout));
				break;
			case HEIGHT:
				pairs.add(new Pair<>((long)mat.arrayHeight(), layout));
				break;
			case WIDTH:
				pairs.add(new Pair<>((long)mat.arrayWidth(), layout));
				break;
			case BATCH:
			case DEPTH:
			case TIME:
			case UNKNOWN:
			default:
				pairs.add(new Pair<>(1L, layout));
			}
		}
		
		
		var shape = new Shape(new PairList<>(pairs));
		
		return shape;
	}
	
	static DataType getDataType(Mat mat) {
		switch (mat.depth()) {
		case opencv_core.CV_8U: return DataType.UINT8;
		case opencv_core.CV_8S: return DataType.INT8;
		case opencv_core.CV_32S: return DataType.INT32;
		case opencv_core.CV_32F: return DataType.FLOAT32;
		case opencv_core.CV_64F: return DataType.FLOAT64;
		case opencv_core.CV_16F: return DataType.FLOAT16; // TODO: Check this! 16-bit float support in OpenCV is limited
		case opencv_core.CV_16U: // Not supported
		case opencv_core.CV_16S:
		default: return DataType.UNKNOWN;
		}
	}
	
	static int getMatDepth(DataType dt) {
		switch (dt) {
		case BOOLEAN:
			return opencv_core.CV_8U;
		case FLOAT16:
			return opencv_core.CV_16F;
		case FLOAT32:
			return opencv_core.CV_32F;
		case FLOAT64:
			return opencv_core.CV_64F;
		case INT32:
			return opencv_core.CV_32S;
		case INT64:
			return opencv_core.CV_64F;
		case INT8:
			return opencv_core.CV_8S;
		case UINT8:
			return opencv_core.CV_8U;
		case STRING:
		case UNKNOWN:
		default:
			throw new UnsupportedOperationException("Cannot convert data type " + dt + " to Mat");
		}
		
	}
	

}
