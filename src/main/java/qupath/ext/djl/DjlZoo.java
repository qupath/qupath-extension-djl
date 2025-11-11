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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import ai.djl.modality.cv.output.Landmark;
import ai.djl.modality.cv.output.Mask;
import ai.djl.modality.cv.translator.BigGANTranslator;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.ClassLoaderUtils;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import org.locationtech.jts.geom.util.AffineTransformation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.analysis.images.SimpleImage;
import qupath.lib.geom.Point2;
import qupath.lib.images.ImageData;
import qupath.lib.images.servers.AbstractTileableImageServer;
import qupath.lib.images.servers.GeneratingImageServer;
import qupath.lib.images.servers.ImageServer;
import qupath.lib.images.servers.ImageServerBuilder.ServerBuilder;
import qupath.lib.images.servers.ImageServerMetadata;
import qupath.lib.images.servers.TileRequest;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjectTools;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.objects.hierarchy.PathObjectHierarchy;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;

import java.awt.image.BandedSampleModel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferFloat;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.lang.reflect.Type;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Helper class for working with DeepJavaLibrary model zoos.
 * 
 * @author Pete Bankhead
 *
 */
public class DjlZoo {
	
	private final static Logger logger = LoggerFactory.getLogger(DjlZoo.class);
	
	/**
	 * Print all available zoo models to the log.
	 */
	public static void logAvailableModels() {
		for (var entry : ModelZoo.listModels().entrySet()) {
			logger.info("Application: {}", entry.getKey());
			for (var artifact : entry.getValue()) {
				logger.info("  {}", artifact.toString());
			}
		}
	}
	
	/**
	 * Ordered list of the inputs we prefer (because we know how to handle them)
	 */
	private static final List<Class<?>> preferredInputs = Arrays.asList(
			Image.class, 
			NDList.class);

	/**
	 * Ordered list of the outputs we prefer (because we know how to handle them)
	 */
	private static final List<Class<?>> preferredOutputs = Arrays.asList(
			CategoryMask.class, 
			DetectedObjects.class,
			Joints.class, 
			Classifications.class, 
			Image.class, 
			NDList.class);


	/**
	 * List all zoo models matching a specific criteria.
	 * @param criteria
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listModels(Criteria<?, ?> criteria) throws ModelNotFoundException, IOException {
		var list = new ArrayList<MRL>();
		for (var entry : ModelZoo.listModels(criteria).entrySet()) {
			list.addAll(entry.getValue());
		}
		return list;
	}
	
	/**
	 * List all zoo models for a specific application.
	 * @param application
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listModels(Application application) throws ModelNotFoundException, IOException {
		var criteria = Criteria.builder()
				.optApplication(application)
				.build();
		return listModels(criteria);
	}

	/**
	 * List all zoo models.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listModels() throws ModelNotFoundException, IOException {
		return listModels(Criteria.builder().build());
	}
	
	/**
	 * List all zoo models for image classification.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listImageClassificationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.IMAGE_CLASSIFICATION);
	}
	
	/**
	 * List all zoo models for semantic segmentation.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listSemanticSegmentationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.SEMANTIC_SEGMENTATION);
	}
	
	/**
	 * List all zoo models for object detection.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listObjectDetectionModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.OBJECT_DETECTION);
	}

	/**
	 * List all zoo models for instance segmentation.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<MRL> listInstanceSegmentationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.INSTANCE_SEGMENTATION);
	}
	
	/**
	 * Try to load a zoo model for a given artifact.
	 * @param artifact
	 * @param allowDownload optionally allow the model to be downloaded if it isn't currently available
	 * @return
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 */
	public static ZooModel<?, ?> loadModel(Artifact artifact, boolean allowDownload) throws ModelNotFoundException, MalformedModelException, IOException {
		var criteria = buildCriteria(artifact, allowDownload);
        return criteria.loadModel();
	}
	
	/**
	 * Build {@link Criteria} from an existing artifact.
	 * This can be used to then load a model.
	 * @param artifact
	 * @param allowDownload optionally allow the model to be downloaded if it isn't currently available
	 * @return
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 */
	public static Criteria<?, ?> buildCriteria(Artifact artifact, boolean allowDownload) throws ModelNotFoundException, MalformedModelException, IOException {
		var before = System.getProperty("ai.djl.offline");
		try {
			if (allowDownload)
				System.setProperty("ai.djl.offline", "false");
			
			
			var application = artifact.getMetadata().getApplication();
			var builder = Criteria.builder()
					.optApplication(application)
					.optArtifactId(artifact.getMetadata().getArtifactId())
					.optProgress(new ProgressBar())
					.optArguments(artifact.getArguments())
					.optGroupId(artifact.getMetadata().getGroupId())
					.optFilters(artifact.getProperties())
					;
			
			var factoryClass = artifact.getArguments().getOrDefault("translatorFactory", null);
			if (factoryClass instanceof String) {
				var factory = getTranslatorFactory((String)factoryClass);
				var supportedTypes = factory.getSupportedTypes();
				var preferredTypes = supportedTypes.stream()
					.filter(p -> preferredInputs.contains(p.getKey()) && preferredOutputs.contains(p.getValue()))
					.sorted(Comparator.comparingInt((Pair<Type, Type> p) -> preferredInputs.indexOf(p.getKey()))
							.thenComparingInt(p -> preferredOutputs.indexOf(p.getValue())))
					.findFirst()
					.orElse(null);
				if (preferredTypes == null) {
					if (supportedTypes.size() == 1)
						preferredTypes = supportedTypes.iterator().next();
					logger.warn("No supported types found in " + factoryClass + " -\n"
							+ "Please call .builder().setTypes(inputClass, outputClass).build() to specify these directly");
				}
				if (preferredTypes != null)
					builder = builder.setTypes((Class<?>)preferredTypes.getKey(), (Class<?>)preferredTypes.getValue());
			} else {
				logger.warn("No translatorFactory specified - will try to choose suitable input/output class based on the application.\n" 
							+ "If this fails, please call .builder().setTypes(inputClass, outputClass).build() to specify these directly");
				if (application == Application.CV.IMAGE_CLASSIFICATION) {
					builder = builder.setTypes(Image.class, Classifications.class);
				} else if (application == Application.CV.SEMANTIC_SEGMENTATION) {
					builder = builder.setTypes(Image.class, CategoryMask.class);
				} else if (application == Application.CV.IMAGE_GENERATION) {
					builder = builder.setTypes(Image.class, Image.class);				
				} else if (application == Application.CV.OBJECT_DETECTION) {
					builder = builder.setTypes(Image.class, DetectedObjects.class);				
				} else if (application == Application.CV.INSTANCE_SEGMENTATION) {
					builder = builder.setTypes(Image.class, DetectedObjects.class);
				} else if (application == Application.CV.WORD_RECOGNITION) {
					builder = builder.setTypes(Image.class, DetectedObjects.class);
				} else if (application == Application.CV.POSE_ESTIMATION) {
					builder = builder.setTypes(Image.class, Joints.class);
				} else
					builder = builder.setTypes(NDList.class, NDList.class);
			}
			return builder.build();
		} finally {
			System.setProperty("ai.djl.offline", before);
		}
	}
	
	
	private static TranslatorFactory getTranslatorFactory(String factoryClass) {
		ClassLoader cl = ClassLoaderUtils.getContextClassLoader();
        return ClassLoaderUtils.initClass(cl, TranslatorFactory.class, factoryClass);
	}
	
	
	/**
	 * Create a ROI from a detected object, rescaling according to the {@link ImageRegion}.
	 * Note that this adapts depending upon the object has a mask, landmark or bounding box.
	 * Any mask if thresholded at 0.5; if more control is needed, use {@link #createROI(Mask, ImageRegion, double)}.
	 * @param obj
	 * @param region
	 * @return
	 */
	public static ROI createROI(DetectedObject obj, ImageRegion region) {
		BoundingBox box = obj.getBoundingBox();
		if (box instanceof Mask mask) {
			return createROI(mask, region, 0.5);
		} else if (box instanceof Landmark landmark) {
			return createROI(landmark, region);
		} else
			return createROI(box, region);
	}
	
	/**
	 * Create a rectangular ROI from a bounding box, rescaling according to the {@link ImageRegion}.
	 * @param box
	 * @param region
	 * @return
	 */
	public static ROI createROI(BoundingBox box, ImageRegion region) {
		var bounds = box.getBounds();
		double xo = 0.0;
		double yo = 0.0;
		double xScale = 1.0;
		double yScale = 1.0;
		var plane = ImagePlane.getDefaultPlane();
		if (region != null) {
			plane = region.getImagePlane();
			xo = region.getMinX();
			yo = region.getMinY();
			xScale = region.getWidth();
			yScale = region.getHeight();
		}
		return ROIs.createRectangleROI(
				xo + bounds.getX() * xScale,
				yo + bounds.getY() * yScale,
				bounds.getWidth()  * xScale,
				bounds.getHeight() * yScale,
				plane);
	}
	
	/**
	 * Create a ROI from a detected mask, rescaling according to the {@link ImageRegion}.
	 * @param mask
	 * @param region
	 * @param threshold
	 * @return
	 */
	public static ROI createROI(Mask mask, ImageRegion region, double threshold) {
		float[][] probs = mask.getProbDist();
		int h = probs.length;
		int w = probs[0].length;
		var buffer = new DataBufferFloat(w * h, 1);
		var sampleModel = new BandedSampleModel(buffer.getDataType(), w, h, 1);
		var raster = WritableRaster.createWritableRaster(sampleModel, buffer, null);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                raster.setSample(x, y, 0, probs[y][x]);
            }
        }
		if (region == null)
			region = ImageRegion.createInstance(0, 0, w, h, 0, 0);
		var geometry = ContourTracing.createTracedGeometry(raster, threshold, Double.POSITIVE_INFINITY, 0, null);

		var bounds = mask.getBounds();

		var transform = new AffineTransformation();
		transform.scale(1.0/raster.getWidth(), 1.0/raster.getHeight());
        if(!mask.isFullImageMask()) {
            transform.scale(bounds.getWidth(), bounds.getHeight());
            transform.translate(bounds.getX(), bounds.getY());
        }
		transform.scale(region.getWidth(), region.getHeight());
		transform.translate(region.getX(), region.getY());

		if (!transform.isIdentity())
			geometry = transform.transform(geometry);
		return GeometryTools.geometryToROI(geometry, region.getImagePlane());
	}

	/**
	 * Create a points ROI from a detected mask, rescaling according to the {@link ImageRegion}.
	 * @param landmark
	 * @param region
	 * @return
	 */
	public static ROI createROI(Landmark landmark, ImageRegion region) {
		double xo = 0.0;
		double yo = 0.0;
		var plane = ImagePlane.getDefaultPlane();
		if (region != null) {
			plane = region.getImagePlane();
			xo = region.getMinX();
			yo = region.getMinY();
		}
		var points = new ArrayList<Point2>();
		for (var p : landmark.getPath()) {
			points.add(new Point2(
					xo + p.getX() * region.getWidth(),
					yo + p.getY() * region.getHeight())
					);
		}
		return ROIs.createPointsROI(points, plane);
	}
	
	
	/**
	 * Run object detection for an entire image.
	 * @param model the model
	 * @param imageData the image within which to detect objects
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 * @throws TranslateException
	 */
	public static Optional<List<PathObject>> detect(ZooModel<Image, DetectedObjects> model, ImageData<BufferedImage> imageData) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
		return detect(model, imageData, Collections.singleton(imageData.getHierarchy().getRootObject()));
	}
	
	/**
	 * Run object detection within specified objects in an image.
	 * @param model the model
	 * @param imageData the image within which to detect objects
	 * @param parentObjects the parent objects, which become parents of what is detected; if null or the root object, the entire image is used for detection.
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 * @throws TranslateException
	 */
	public static Optional<List<PathObject>> detect(ZooModel<Image, DetectedObjects> model, ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {

		if (parentObjects == null)
			parentObjects = Collections.singleton(imageData.getHierarchy().getRootObject());

		// Check if we have a specified input size
		// Check if we have a specified input size
		var inputHeightWidth = getInputHeightWidth(model);
		long inputWidth = inputHeightWidth.get(0);
		long inputHeight = inputHeightWidth.get(1);

		// Set the threshold, reading from properties if possible
		double defaultThreshold = 0.5;
		var threshold = tryToParseDoubleProperty(model, "threshold", defaultThreshold);
		if (threshold != defaultThreshold) {
			logger.debug("Setting threshold to {} from model properties", threshold);
		}

		var server = imageData.getServer();
		double downsampleBase = server.getDownsampleForResolution(0);
		
		// Retain a map of parents to child objects that should be added
		// This means we can delay making modifications until process is complete - 
		// to avoid partial processing, and permit returning early if the thread is interrupted
		var map = new ConcurrentHashMap<PathObject, List<PathObject>>();
		// Maintain a list of all objects created
		var list = new ArrayList<PathObject>();
		
		try (var manager = model.getNDManager()) {
			var predictor = model.newPredictor();
						
			for (var parent : parentObjects) {
				
				parent.clearChildObjects();
				
				// We need a list, because the entire image might be a z-stack of time series
				List<RegionRequest> requests;
				var roi = parent.getROI();
				if (roi == null) {
					requests = getAllRequests(RegionRequest.createInstance(server, downsampleBase), server.nZSlices(), server.nTimepoints());
				} else {
					requests = Collections.singletonList(
							RegionRequest.createInstance(server.getPath(), downsampleBase, parent.getROI())
							);
				}
				
				var childObjects = map.computeIfAbsent(parent, p -> new ArrayList<>());
				for (var request : requests) {
					
					if (Thread.currentThread().isInterrupted()) {
						logger.warn("Detection interrupted! Discarding {} detection(s)", list.size());
						return Optional.empty();
					}
				
					// Make sure we request at a sensible resolution, to give a sensible input
					if (inputWidth > 0 || inputHeight > 0)
						request = updateDownsampleForInput(request, inputWidth, inputHeight);
					
					var img = server.readRegion(request);
					
					var detections = detect(predictor, img);
					for (var item : detections.items()) {
						var detected = (DetectedObject)item;
						if (detected.getProbability() < threshold)
							continue;
						
						var detectedROI = createROI(detected, request);
						
						// Trim ROI to fit inside the region, if needed
						if (roi != null && (detected.getBoundingBox() instanceof Mask || detectedROI.isPoint()))
							detectedROI = RoiTools.intersection(detectedROI, roi);
						
						if (detectedROI.isEmpty()) {
							logger.debug("ROI detected, but empty (class={}, prob={})", detected.getProbability(), item.getClassName());
							continue;
						}
						
						var newObject = PathObjects.createAnnotationObject(
								detectedROI,
								PathClass.fromString(item.getClassName())
								);
						try (var ml = newObject.getMeasurementList()) {
							ml.put("Class probability", item.getProbability());
						}
						list.add(newObject);
						childObjects.add(newObject);
					}
				}
			}
			
		}
		
		updateObjectsAndHierarchy(imageData.getHierarchy(), map, model);
		imageData.getHierarchy().fireHierarchyChangedEvent(DjlZoo.class);
		return Optional.of(list);
	}
	
//	/**
//	 * Apply an image-to-image model to an input image.
//	 * @param model
//	 * @param img
//	 * @return
//	 * @throws TranslateException
//	 */
//	public static <S, T> BufferedImage segment(ZooModel<S, T> model, BufferedImage img) throws TranslateException {
//		try (var manager = model.getNDManager()) {
//			var predictor = model.newPredictor();
//			var input = BufferedImageFactory.getInstance().fromImage(img).toNDArray(manager);
//			var output = predictor.predict((S)new NDList(input));
//			
//			
//			var outputImage = (Image)output;
//			return (BufferedImage)outputImage.getWrappedImage();
//		}
//	}
	
	/**
	 * Get region requests for all z-slices and timepoints.
	 * @param request
	 * @param nZSlices
	 * @param nTimepoints
	 * @return
	 */
	private static List<RegionRequest> getAllRequests(RegionRequest request, int nZSlices, int nTimepoints) {
		var list = new ArrayList<RegionRequest>();
		for (int t = 0; t < nTimepoints; t++) {
			request = request.updateT(t);
			for (int z = 0; z < nZSlices; z++) {
				request = request.updateZ(z);
				list.add(request);
			}			
		}
		return list;
	}

	/**
	 * Apply a segmentation model to segment objects within an image, and represent these as annotations.
	 * @param model the segmentation model
	 * @param imageData the image data
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted	
	 * @throws TranslateException
	 * @throws IOException
	 * @see #segmentObjects(ZooModel, ImageData, Collection, Function, boolean)
	 * @see #segmentAnnotations(ZooModel, ImageData, Collection)
	 */
	public static Optional<List<PathObject>> segmentAnnotations(ZooModel<Image, CategoryMask> model, ImageData<BufferedImage> imageData) throws TranslateException, IOException {
		return segmentAnnotations(model, imageData, Collections.singletonList(imageData.getHierarchy().getRootObject()));
	}
	
	/**
	 * Apply a segmentation model to segment objects within specified objects in an image, and represent these as annotations.
	 * @param model the segmentation model
	 * @param imageData the image data
	 * @param parentObjects parent objects within which the segmentation will be applied; if null or the root object, the full image will be used
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted	
	 * @throws TranslateException
	 * @throws IOException
	 * @see #segmentObjects(ZooModel, ImageData, Collection, Function, boolean)
	 * @see #segmentDetections(ZooModel, ImageData, Collection)
	 */
	public static Optional<List<PathObject>> segmentAnnotations(ZooModel<Image, CategoryMask> model, ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects) throws TranslateException, IOException {
		return segmentObjects(model, imageData, parentObjects, r -> PathObjects.createAnnotationObject(r), true);
	}
	
	/**
	 * Apply a segmentation model to segment objects within an image, and represent these as detections.
	 * @param model the segmentation model
	 * @param imageData the image data
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted	
	 * @throws TranslateException
	 * @throws IOException
	 * @see #segmentObjects(ZooModel, ImageData, Collection, Function, boolean)
	 * @see #segmentDetections(ZooModel, ImageData, Collection)
	 */
	public static Optional<List<PathObject>> segmentDetections(ZooModel<Image, CategoryMask> model, ImageData<BufferedImage> imageData) throws TranslateException, IOException {
		return segmentDetections(model, imageData, Collections.singletonList(imageData.getHierarchy().getRootObject()));
	}

	/**
	 * Apply a segmentation model to segment objects within specified objects in an image, and represent these as detections.
	 * @param model the segmentation model
	 * @param imageData the image data
	 * @param parentObjects parent objects within which the segmentation will be applied; if null or the root object, the full image will be used
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted	
	 * @throws TranslateException
	 * @throws IOException
	 * @see #segmentObjects(ZooModel, ImageData, Collection, Function, boolean)
	 * @see #segmentAnnotations(ZooModel, ImageData, Collection)
	 */
	public static Optional<List<PathObject>> segmentDetections(ZooModel<Image, CategoryMask> model, ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects) throws TranslateException, IOException {
		return segmentObjects(model, imageData, parentObjects, r -> PathObjects.createDetectionObject(r), true);
	}
	
	private static Shape getInputHeightWidth(Model model) {
		// Check if we have a specified input size
		long inputWidth = -1;
		long inputHeight = -1;
		var block = model.getBlock();
		if (block.isInitialized()) {
			var inputShape = getSingleInputShape(block.describeInput());
			inputWidth = getDim(inputShape, LayoutType.WIDTH);
			inputHeight = getDim(inputShape, LayoutType.HEIGHT);			
		} else {
			var w = (long)tryToParseDoubleProperty(model, "width", inputWidth);
			var h = (long)tryToParseDoubleProperty(model, "height", inputHeight);
			if (w != inputWidth || h != inputHeight) {
				logger.debug("Setting input size to {} x {}", inputWidth, inputHeight);
				inputWidth = w;
				inputHeight = h;
			}
		}
		return new Shape(inputHeight, inputWidth);
	}

	
	/**
	 * Apply a segmentation model to segment objects within an image.
	 * @param model the segmentation model
	 * @param imageData the image data
	 * @param parentObjects parent objects within which the segmentation will be applied; if null or the root object, the full image will be used
	 * @param creator function to create a new object; generally used to create annotations or detections
	 * @param skipBackground if true, no object will be created for the background (label 0)
	 * @return an optional containing a list of all objects added, or null if the detection was interrupted	
	 * @throws TranslateException
	 * @throws IOException
	 */
	public static Optional<List<PathObject>> segmentObjects(ZooModel<Image, CategoryMask> model, ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects, Function<ROI, PathObject> creator, boolean skipBackground) throws TranslateException, IOException {
		
		if (parentObjects == null)
			parentObjects = Collections.singleton(imageData.getHierarchy().getRootObject());
		
		// Check if we have a specified input size
		var inputHeightWidth = getInputHeightWidth(model);
		long inputWidth = inputHeightWidth.get(0);
		long inputHeight = inputHeightWidth.get(1);

		// Retain a map of parents to child objects that should be added
		// This means we can delay making modifications until process is complete - 
		// to avoid partial processing, and permit returning early if the thread is interrupted
		var map = new ConcurrentHashMap<PathObject, List<PathObject>>();
		// Maintain a list of all objects created
		var list = new ArrayList<PathObject>();

		// TODO: Support parallelization
		// See https://docs.djl.ai/docs/development/inference_performance_optimization.html
		var server = imageData.getServer();
		try (var predictor = model.newPredictor()) {
			for (var parent : parentObjects) {
				
				if (Thread.interrupted()) {
					logger.warn("Processing interrupted - {} object(s) will be discarded", list.size());
					return Optional.empty();
				}
			
				// Get all the requests we need - usually just one, but may be more for an entire z-stack/time series
				List<RegionRequest> requests;
				var roi = parent.getROI();
				if (roi != null) {
					var request = RegionRequest.createInstance(imageData.getServer().getPath(), server.getDownsampleForResolution(0), roi);
					request = updateDownsampleForInput(request, inputWidth, inputHeight);
					requests = Collections.singletonList(request);
				} else {
					var request = RegionRequest.createInstance(imageData.getServer());
					request = updateDownsampleForInput(request, inputWidth, inputHeight);
					requests = getAllRequests(request, server.nZSlices(), server.nTimepoints());
				}
				var childList = map.computeIfAbsent(parent, p -> new ArrayList<>());
				
				// Apply prediction across all requests
				for (var request : requests) {
					var img = imageData.getServer().readRegion(request);
					var segmented = segmentObjects(predictor, img, request, roi, creator, skipBackground);
					if (!segmented.isEmpty()) {
						childList.addAll(segmented);
						parent.addChildObjects(segmented);
						list.addAll(segmented);
					}
				}
			}
		}
		// If we get this far, make the updates
		updateObjectsAndHierarchy(imageData.getHierarchy(), map, model);
		return Optional.of(list);
	}
	
	
	private static double tryToParseDoubleProperty(Model model, String key, double defaultValue) {
		var value = model.getProperty(key);
		if (value == null || value.isBlank())
			return defaultValue;
		try {
			return (Double.parseDouble(value));
		} catch (NumberFormatException e) {
			logger.warn("Unable to parse property {} as double: {}", key, e.getLocalizedMessage());
			return defaultValue;
		}
	}
	
	
	private static void updateObjectsAndHierarchy(PathObjectHierarchy hierarchy, Map<PathObject, List<PathObject>> map, Object changeSource) {
		// If we get this far, make the updates
		boolean changes = false;
		for (var entry : map.entrySet()) {
			var parent = entry.getKey();
			var childObjects = entry.getValue();
			parent.clearChildObjects();
			parent.addChildObjects(childObjects);
			if (!childObjects.isEmpty())
				parent.setLocked(true); // Lock if we added anything
			changes = true;
		}

		if (changes) {
			hierarchy.fireHierarchyChangedEvent(changeSource);
			deselectDeletedObjects(hierarchy);
		}
	}
	
	
	
	/**
	 * Deselect any objects that were removed from the hierarchy.
	 * @param hierarchy
	 */
	private static void deselectDeletedObjects(PathObjectHierarchy hierarchy) {
		var toDeselect = hierarchy.getSelectionModel().getSelectedObjects()
				.stream()
				.filter(p -> !PathObjectTools.hierarchyContainsObject(hierarchy, p))
				.collect(Collectors.toList());
		if (!toDeselect.isEmpty())
			hierarchy.getSelectionModel().deselectObjects(toDeselect);
	}
	
	
	/**
	 * Update the downsample for a region request according to the specified input width and height.
	 * This is used to try to provide a sensibly-resized input to the model, rather than require 
	 * the resizing to be done on the DJL size (on a possibly far-too-large image).
	 * @param request
	 * @param inputWidth
	 * @param inputHeight
	 * @return
	 */
	private static RegionRequest updateDownsampleForInput(RegionRequest request, long inputWidth, long inputHeight) {
		if (inputWidth <= 0 && inputHeight <= 0)
			return request;
		double targetDownsampleWidth = inputWidth <= 0 ? request.getDownsample() : Math.round(request.getWidth() / (double)inputWidth);
		double targetDownsampleHeight = inputHeight <= 0 ? request.getDownsample() : Math.round(request.getHeight() / (double)inputHeight);
		double targetDownsample = Math.min(targetDownsampleWidth, targetDownsampleHeight);
		if (targetDownsample > request.getDownsample())
			return request.updateDownsample(targetDownsample);
		else
			return request;
	}
	
	
	private static long getDim(Shape shape, LayoutType layoutType) {
		if (shape != null) {
			for (int i = 0; i < shape.dimension(); i++) {
				if (layoutType.equals(shape.getLayoutType(i)))
					return shape.get(i);
			}
		}
		return -1;
	}
	
	private static Shape getSingleInputShape(PairList<String, Shape> input) {
		if (input == null)
			return null;
		if (input.size() != 1) {
			throw new IllegalArgumentException("Only single inputs are supported! Model requires " + input.size() + " inputs");
		}
		return input.get(0).getValue();
	}
	
	/**
	 * Apply a segmentation model to a single image, creating detections from the result.
	 * @param predictor the segmentation model predictor
	 * @param img the input image
	 * @param request a region request; if not null, this is used to scale and translate the segmented ROIs
	 * @return a list of all objects that were segmented
	 * @throws TranslateException
	 * @see #segmentObjects(Predictor, BufferedImage, RegionRequest, ROI, Function, boolean)
	 */
	public static List<PathObject> segmentDetections(Predictor<Image, CategoryMask> predictor, BufferedImage img, RegionRequest request) throws TranslateException {
		return segmentObjects(predictor, img, request, null, r -> PathObjects.createDetectionObject(r), true);
	}

	
	/**
	 * Apply a segmentation model to a single image, creating annotations from the result.
	 * @param predictor the segmentation model predictor
	 * @param img the input image
	 * @param request a region request; if not null, this is used to scale and translate the segmented ROIs
	 * @return a list of all objects that were segmented
	 * @throws TranslateException
	 * @see #segmentObjects(Predictor, BufferedImage, RegionRequest, ROI, Function, boolean)
	 */
	public static List<PathObject> segmentAnnotations(Predictor<Image, CategoryMask> predictor, BufferedImage img, RegionRequest request) throws TranslateException {
		return segmentObjects(predictor, img, request, null, r -> PathObjects.createAnnotationObject(r), true);
	}

	/**
	 * Apply a segmentation model to a single image.
	 * @param predictor the segmentation model predictor
	 * @param img the input image
	 * @param request a region request; if not null, this is used to scale and translate the segmented ROIs
	 * @param roiMask optional ROI to constrain the output; if not null, segmented ROIs will be intersected with the mask
	 * @param creator function to convert a segmented ROI to an object (usually an annotation or detection)
	 * @param skipBackground if true, no object will be created for the background (label 0)
	 * @return a list of all objects that were segmented
	 * @throws TranslateException
	 */
	public static List<PathObject> segmentObjects(Predictor<Image, CategoryMask> predictor, BufferedImage img, RegionRequest request, ROI roiMask, Function<ROI, PathObject> creator, boolean skipBackground) throws TranslateException {
		var map = segmentROIs(predictor, img, request, roiMask, skipBackground);
		return map.entrySet().stream().map(e -> createPathObject(creator, e.getValue(), e.getKey())).collect(Collectors.toList());
	}
	
	
	private static PathObject createPathObject(Function<ROI, PathObject> creator, ROI roi, String classification) {
		var pathObject = creator.apply(roi);
		if (classification != null)
			pathObject.setPathClass(PathClass.getInstance(classification));
		return pathObject;
	}
	
	
	private static Map<String, ROI> segmentROIs(Predictor<Image, CategoryMask> predictor, BufferedImage img, RegionRequest request, ROI roiMask, boolean skipBackground) throws TranslateException {
		var input = BufferedImageFactory.getInstance().fromImage(img);
		var output = predictor.predict(input);

		var classes = output.getClasses();
		int[][] maskOrig = output.getMask();
		var mask = new SimpleMaskImage(maskOrig);

		// Check which classes are represented
		// If we have many unrepresented classes, this avoids needing to check 
		// each individually
		int nClasses = classes.size();
		int[] hist = new int[nClasses];
		for (var row : maskOrig) {
			for (int val : row) {
				if (val >= 0 && val < nClasses)
					hist[val]++;
			}
		}

		// Loop through classes - skipping 0 if we want to skip background
		int startInd = skipBackground ? 1 : 0;
		var map = new LinkedHashMap<String, ROI>();
		for (int i = startInd; i < nClasses; i++) {
			// Check if we have any pixels
			if (hist[i] == 0)
				continue;
			var classification = classes.get(i);
			var roi = createROI(mask, request, i);
			// Add the ROI
			if (roi != null && !roi.isEmpty()) {
				// We may need to rescale if the output dimensions differ from the input
				double scaleX = img.getWidth() / (double)mask.getWidth();
				double scaleY = img.getHeight() / (double)mask.getHeight();
				if (scaleX != 1 || scaleY != 0) {
					if (request == null)
						roi = roi.scale(scaleX, scaleY);
					else
						roi = roi.scale(scaleX, scaleY, request.getX(), request.getY());
				}
				// Apply a ROI mask, if we have one
				if (roiMask != null)
					roi = RoiTools.intersection(roi, roiMask);
				// Create an object
				map.put(classification, roi);
			}
		}
		return map;
	}
	
	
	private static ROI createROI(SimpleImage mask, RegionRequest request, int val) {
		return ContourTracing.createTracedROI(mask, val, val, request);
	}
	
	
	
	/**
	 * Wrapper for the int[][] arrays that DJL uses as masks.
	 */
	private static class SimpleMaskImage implements SimpleImage {
		
		private int[][] values;
		private int width;
		private int height;
		
		private SimpleMaskImage(int[][] values) {
			this.values = values;
			this.width = values[0].length;
	        this.height = values.length;
		}

		@Override
		public float getValue(int x, int y) {
			return values[y][x];
		}

		@Override
		public int getWidth() {
			return width;
		}

		@Override
		public int getHeight() {
			return height;
		}
		
	}
	
	
	/**
	 * Apply a detection model predictor to an input image.
	 * @param predictor
	 * @param img
	 * @return
	 * @throws TranslateException
	 */
	public static DetectedObjects detect(Predictor<Image, DetectedObjects> predictor, BufferedImage img) throws TranslateException {
		var image = ImageFactory.getInstance().fromImage(img);
		return predictor.predict(image);
	}
	
	/**
	 * Apply a classification model predictor to an input image.
	 * @param predictor
	 * @param img
	 * @return
	 * @throws TranslateException
	 */
	public static Classifications classify(Predictor<Image, Classifications> predictor, BufferedImage img) throws TranslateException {
		var image = ImageFactory.getInstance().fromImage(img);
		return predictor.predict(image);
	}

	
	/**
	 * Generate images using a {@link BigGANTranslator}.
	 * <p>
	 * See <a href="https://docs.djl.ai/examples/docs/biggan.html">https://docs.djl.ai/examples/docs/biggan.html</a>
	 * @param model the model
	 * @param indices the indices defining what should be in the image
	 * @return the images generated from the input indices
	 * @throws TranslateException
	 */
	static List<BufferedImage> bigGanGenerate(ZooModel<int[], Image[]> model, int... indices) throws TranslateException {
		if (!(model.getTranslator() instanceof BigGANTranslator)) {
			// Log a warning - we can still try, but with lower expectations
			logger.warn("Model translater is not an instance of BigGANTranslator");
		}
		try (var predictor = model.newPredictor()) {
			var output = (Image[])predictor.predict(indices);
			return Arrays.stream(output).map(i -> toBufferedImage(i)).collect(Collectors.toList());
		}
	}
	
	
	/**
	 * Apply a predictor that takes a {@link BufferedImage} as input and provides another {@link BufferedImage} as output.
	 * @param predictor
	 * @param img
	 * @return
	 * @throws TranslateException
	 * @imple
	 */
	public static BufferedImage imageToImage(Predictor<Image, Image> predictor, BufferedImage img) throws TranslateException {
		var image = BufferedImageFactory.getInstance().fromImage(img);
		var output = predictor.predict(image);
		return toBufferedImage(output);
	}
	
	/**
	 * Try to convert a DJL {@link Image} to a Java {@link BufferedImage}.
	 * @param image
	 * @return
	 * @throws IllegalArgumentException if the image is of an unknown type that cannot be converted
	 */
	public static BufferedImage toBufferedImage(Image image) throws IllegalArgumentException {
		var wrapped = image.getWrappedImage();
		if (wrapped instanceof BufferedImage)
			return (BufferedImage)wrapped;
		throw new IllegalArgumentException("Need a java.awt.image.BufferedImage, but found " + wrapped);
	}
	
		
	
	/**
	 * Experimental (read: probably-not-very-useful) code to wrap an {@link ImageServer} to apply an image-to-image 
	 * prediction model to the tiles.
	 * @param model
	 * @param server
	 * @return
	 * @implNote The {@link ImageServer} created here cannot be serialized to JSON; it can only be used temporarily 
	 *           within a single QuPath session.
	 */
	static ImageServer<BufferedImage> wrapImageToImage(ZooModel<Image, Image> model, ImageServer<BufferedImage> server) {
		return new DjlPredictionImageServer(server, model);
	}
	
	
	static class DjlPredictionImageServer extends AbstractTileableImageServer implements GeneratingImageServer<BufferedImage> {
		
		private ImageServer<BufferedImage> server;
		private ZooModel<Image, Image> model;
		
		private ThreadLocal<Predictor<Image, Image>> predictors;
		
		DjlPredictionImageServer(ImageServer<BufferedImage> server, ZooModel<Image, Image> model) {
			this.server = server;
			this.model = model;
			this.predictors = ThreadLocal.withInitial(() -> model.newPredictor());
			var imageHeightWidth = getInputHeightWidth(model);
			long tileWidth = imageHeightWidth.size(1) <= 0 ? 512 : imageHeightWidth.size(1);
			long tileHeight = imageHeightWidth.size(0) <= 0 ? tileWidth : imageHeightWidth.size(0);
			setMetadata(
					new ImageServerMetadata.Builder(server.getMetadata())
					.preferredTileSize((int)tileWidth, (int)tileHeight)
					.build()
					);
		}

		@Override
		public Collection<URI> getURIs() {
			return server.getURIs();
		}

		@Override
		public String getServerType() {
			return "Deep Java Library prediction server";
		}

		@Override
		public ImageServerMetadata getOriginalMetadata() {
			return server.getOriginalMetadata();
		}

		@Override
		protected BufferedImage readTile(TileRequest tileRequest) throws IOException {
			if (server.isEmptyRegion(tileRequest.getRegionRequest()))
				return getEmptyTile(tileRequest.getTileWidth(), tileRequest.getTileHeight());
			var img = server.readRegion(tileRequest.getRegionRequest());
			try {
				return imageToImage(predictors.get(), img);
			} catch (TranslateException e) {
				throw new IOException(e);
			}
		}

		@Override
		protected ServerBuilder<BufferedImage> createServerBuilder() {
			throw new UnsupportedOperationException("DjlPredictionImageServer cannot currently be serialized");
		}

		@Override
		protected String createID() {
			return UUID.randomUUID().toString();
		}

		@Override
		public void close() throws Exception {
			super.close();
			model.close();
		}
		
		
	}
	

}
