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

import java.awt.image.BandedSampleModel;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferFloat;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.locationtech.jts.geom.util.AffineTransformation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Joints;
import ai.djl.ndarray.NDList;
import ai.djl.modality.cv.output.Landmark;
import ai.djl.modality.cv.output.Mask;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import qupath.lib.analysis.images.ContourTracing;
import qupath.lib.geom.Point2;
import qupath.lib.images.ImageData;
import qupath.lib.objects.PathObject;
import qupath.lib.objects.PathObjects;
import qupath.lib.objects.classes.PathClass;
import qupath.lib.regions.ImagePlane;
import qupath.lib.regions.ImageRegion;
import qupath.lib.regions.RegionRequest;
import qupath.lib.roi.GeometryTools;
import qupath.lib.roi.ROIs;
import qupath.lib.roi.RoiTools;
import qupath.lib.roi.interfaces.ROI;

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
		try {
			for (var entry : ModelZoo.listModels().entrySet()) {
				logger.info("Application: {}", entry.getKey());
				for (var artifact : entry.getValue()) {
					logger.info("  {}: {}", artifact.getName(), artifact.toString());
				}
			}
		} catch (ModelNotFoundException | IOException e) {
			logger.error("Exception requesting ModelZoo models: " + e.getLocalizedMessage(), e);
		}
	}

	/**
	 * List all zoo models matching a specific criteria.
	 * @param criteria
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listModels(Criteria<?, ?> criteria) throws ModelNotFoundException, IOException {
		var list = new ArrayList<Artifact>();
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
	public static List<Artifact> listModels(Application application) throws ModelNotFoundException, IOException {
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
	public static List<Artifact> listModels() throws ModelNotFoundException, IOException {
		return listModels(Criteria.builder().build());
	}
	
	/**
	 * List all zoo models for image classification.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listImageClassificationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.IMAGE_CLASSIFICATION);
	}
	
	/**
	 * List all zoo models for image enhancement.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listImageEnhancementModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.IMAGE_ENHANCEMENT);
	}
	
	/**
	 * List all zoo models for image generation.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listImageGenerationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.IMAGE_GENERATION);
	}
	
	/**
	 * List all zoo models for semantic segmentation.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listSemanticSegmentationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.SEMANTIC_SEGMENTATION);
	}
	
	/**
	 * List all zoo models for object detection.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listObjectDetectionModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.OBJECT_DETECTION);
	}

	/**
	 * List all zoo models for instance segmentation.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listInstanceSegmentationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.INSTANCE_SEGMENTATION);
	}
	
	/**
	 * List all zoo models for word recognition.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listWordRecognitionnModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.WORD_RECOGNITION);
	}
	
	/**
	 * List all zoo models for pose estimation.
	 * @return
	 * @throws ModelNotFoundException
	 * @throws IOException
	 */
	public static List<Artifact> listPoseEstimationModels() throws ModelNotFoundException, IOException {
		return listModels(Application.CV.POSE_ESTIMATION);
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
		var before = System.getProperty("offline");
		try {
			if (allowDownload)
				System.setProperty("offline", "false");
			
			var application = artifact.getMetadata().getApplication();
			var builder = Criteria.builder()
					.optApplication(application)
					.optArtifactId(artifact.getMetadata().getArtifactId())
					.optProgress(new ProgressBar())
					.optArguments(artifact.getArguments())
					.optGroupId(artifact.getMetadata().getGroupId())
					.optFilters(artifact.getProperties())
					;
			
			if (application == Application.CV.IMAGE_CLASSIFICATION) {
				builder = builder.setTypes(Image.class, Classifications.class);
			} else if (application == Application.CV.SEMANTIC_SEGMENTATION) {
				builder = builder.setTypes(Image.class, Image.class);
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
			
			return builder.build();
		} finally {
			System.setProperty("offline", before);
		}
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
		var box = obj.getBoundingBox();
		if (box instanceof Mask) {
			return createROI((Mask)box, region, 0.5);
		} else if (box instanceof Landmark) {
			return createROI((Landmark)box, region);
		} else
			return createROI((BoundingBox)box, region);
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
		var plane = ImagePlane.getDefaultPlane();
		if (region != null) {
			plane = region.getImagePlane();
			xo = region.getMinX();
			yo = region.getMinY();
		}
		return ROIs.createRectangleROI(
				xo + bounds.getX() * region.getWidth(),
				yo + bounds.getY() * region.getHeight(),
				bounds.getWidth()  * region.getWidth(),
				bounds.getHeight() * region.getHeight(),
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
		int w = probs.length;
		int h = probs[0].length;
		var buffer = new DataBufferFloat(w * h, 1);
		var sampleModel = new BandedSampleModel(buffer.getDataType(), w, h, 1);
		var raster = WritableRaster.createWritableRaster(sampleModel, buffer, null);
		for (int x = 0; x < w; x++) {
			float[] col = probs[x];
			for (int y = 0; y < h; y++) {
				raster.setSample(x, y, 0, col[y]);
			}
		}
			if (region == null)
				region = ImageRegion.createInstance(0, 0, w, h, 0, 0);
			var geometry = ContourTracing.createTracedGeometry(raster, threshold, Double.POSITIVE_INFINITY, 0, null);
			
			var bounds = mask.getBounds();
			
			var transform = new AffineTransformation();
			transform.scale(1.0/raster.getWidth(), 1.0/raster.getHeight());
			transform.scale(bounds.getWidth(), bounds.getHeight());
			transform.translate(bounds.getX(), bounds.getY());
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
	 * Build a model and run object detection for an entire image.
	 * @param criteria the criteria to build a model
	 * @param imageData the image within which to detect objects
	 * @return the total number of objects that were detected
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 * @throws TranslateException
	 */	
	public static int detect(Criteria<Image, DetectedObjects> criteria, ImageData<BufferedImage> imageData) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
		try (var model = criteria.loadModel()) {
			return detect(model, imageData);
		}
	}

	/**
	 * Build a model and run object detection within specified objects in an image.
	 * @param criteria the criteria to build a model
	 * @param imageData the image within which to detect objects
	 * @param parentObjects the parent objects, which become parents of what is detected; if the root object, the entire image is used for detection.
	 * @return the total number of objects that were detected
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 * @throws TranslateException
	 */	
	public static int detect(Criteria<Image, DetectedObjects> criteria, ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
		try (var model = criteria.loadModel()) {
			return detect(model, imageData, parentObjects);
		}
	}
	
	/**
	 * Run object detection for an entire image.
	 * @param model the model
	 * @param imageData the image within which to detect objects
	 * @return the total number of objects that were detected
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 * @throws TranslateException
	 */
	public static int detect(ZooModel<Image, DetectedObjects> model, ImageData<BufferedImage> imageData) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
		return detect(model, imageData, Collections.singleton(imageData.getHierarchy().getRootObject()));
	}
	
	/**
	 * Run object detection within specified objects in an image.
	 * @param model the model
	 * @param imageData the image within which to detect objects
	 * @param parentObjects the parent objects, which become parents of what is detected; if the root object, the entire image is used for detection.
	 * @return the total number of objects that were detected
	 * @throws ModelNotFoundException
	 * @throws MalformedModelException
	 * @throws IOException
	 * @throws TranslateException
	 */
	public static int detect(ZooModel<Image, DetectedObjects> model, ImageData<BufferedImage> imageData, Collection<? extends PathObject> parentObjects) throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
		var server = imageData.getServer();
		
		double downsample = server.getDownsampleForResolution(0);
		
		int nDetected = 0;
		
		try (var manager = model.getNDManager()) {
			var predictor = model.newPredictor();
			var factory = ImageFactory.getInstance();
						
			for (var parent : parentObjects) {
				
				parent.clearChildObjects();
				
				RegionRequest request;
				var roi = parent.getROI();
				if (roi != null)
					request = RegionRequest.createInstance(server.getPath(), downsample, parent.getROI());
				else
					request = RegionRequest.createInstance(server, downsample);
				
				var img = server.readRegion(request);
				
				var image = factory.fromImage(img);
				var detections = predictor.predict(image);
				for (var item : detections.items()) {
					var detected = (DetectedObject)item;
					if (detected.getProbability() < 0.5)
						continue;
					
					var detectedROI = createROI(detected, request);
					
					// Trim ROI to fit inside the region, if needed
					if (roi != null && (detected.getBoundingBox() instanceof Mask || detectedROI.isPoint()))
						detectedROI = RoiTools.intersection(detectedROI, roi);
					
					if (detectedROI.isEmpty()) {
						logger.debug("ROI detected, but empty");
						continue;
					}
					
					var newObject = PathObjects.createAnnotationObject(
							detectedROI,
							PathClass.fromString(item.getClassName())
							);
					try (var ml = newObject.getMeasurementList()) {
						ml.put("Class probability", item.getProbability());
					}
					nDetected++;
					parent.addChildObject(newObject);
				}
			}
			
		}
		imageData.getHierarchy().fireHierarchyChangedEvent(DjlZoo.class);
		return nDetected;
	}
	
	/**
	 * Apply an image-to-image model to an input image.
	 * @param model
	 * @param img
	 * @return
	 * @throws TranslateException
	 */
	public static BufferedImage segment(ZooModel<Image, Image> model, BufferedImage img) throws TranslateException {
		try (var manager = model.getNDManager()) {
			var predictor = model.newPredictor();
			var image = BufferedImageFactory.getInstance().fromImage(img);
			var output = predictor.predict(image);
			return (BufferedImage)output.getWrappedImage();
		}
	}
	
	/**
	 * Apply a detection model to an input image.
	 * @param model
	 * @param img
	 * @return
	 * @throws TranslateException
	 */
	public static DetectedObjects detect(ZooModel<Image, DetectedObjects> model, BufferedImage img) throws TranslateException {
		try (var manager = model.getNDManager()) {
			var predictor = model.newPredictor();
			var image = ImageFactory.getInstance().fromImage(img);
			return predictor.predict(image);
		}
	}
	
	/**
	 * Apply a classification model to an input image.
	 * @param model
	 * @param img
	 * @return
	 * @throws TranslateException
	 */
	public static Classifications classify(ZooModel<Image, Classifications> model, BufferedImage img) throws TranslateException {
		try (var manager = model.getNDManager()) {
			var predictor = model.newPredictor();
			var image = ImageFactory.getInstance().fromImage(img);
			return predictor.predict(image);
		}
	}

}
