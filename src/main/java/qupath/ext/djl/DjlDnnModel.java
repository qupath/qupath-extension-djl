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

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.lib.io.UriResource;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.dnn.DnnShape;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

class DjlDnnModel implements DnnModel, AutoCloseable, UriResource {

	private static final Logger logger = LoggerFactory.getLogger(DjlDnnModel.class);

	private List<URI> uris;
	private String engine;
	private String ndLayout;

	private Map<String, DnnShape> inputs;
	private Map<String, DnnShape> outputs;
	
	private boolean lazyInitialize;

	private transient boolean failed;
	private transient ZooModel<Mat[], Mat[]> model;
	private transient Predictor<Mat[], Mat[]> predictor;

	/**
	 * Default layout for an OpenCV Mat
	 */
	private static final String DEFAULT_MAT_LAYOUT = getLayout(LayoutType.HEIGHT, LayoutType.WIDTH, LayoutType.CHANNEL);

	DjlDnnModel(String engine, Collection<URI> uris, String ndLayout, Map<String, DnnShape> inputs, Map<String, DnnShape> outputs, boolean lazyInitialize) {
		this.engine = engine;
		this.uris = new ArrayList<>(uris);
		if (ndLayout == null) {
			logger.warn("ndLayout not specified - I'll need to try to guess");
		} else
			this.ndLayout = ndLayout.toUpperCase();
		this.inputs = inputs;
		this.outputs = outputs;
		this.lazyInitialize = lazyInitialize;
		if (!lazyInitialize)
			ensureInitialized();
	}

	private void ensureInitialized() {
		if (model != null)
			return;
		if (!failed && model == null) {
			synchronized (this) {
				if (!failed && model == null) {
					try {
						logger.debug("Initializing DjlDnnModel");
						model = DjlTools.loadModel(engine,
								Mat[].class, Mat[].class,
								new ModelMatTranslator(),
								uris.toArray(URI[]::new));
//						if (ndLayout != null && ndLayout.contains("N"))
						predictor = model.newPredictor();


						// TODO: Better handling of missing inputs/outputs - we may need to run a prediction for this to work
						if (this.inputs == null || this.inputs.isEmpty()) {
							var description = model.describeInput();
							if (description != null && !description.isEmpty())
								inputs = description.stream().collect(Collectors.toMap(p -> p.getKey(), p -> DjlTools.convertShape(p.getValue())));
							else
								inputs = Map.of(DnnModel.DEFAULT_INPUT_NAME, DnnShape.UNKNOWN_SHAPE);
						}

						if (this.outputs == null || this.outputs.isEmpty()) {
							try {
								var description = model.describeOutput();
								if (description != null && !description.isEmpty())
									outputs = description.stream().collect(Collectors.toMap(p -> p.getKey(), p -> DjlTools.convertShape(p.getValue())));
							} catch (Exception e) {
								logger.debug(e.getMessage(), e);
							}
							if (this.outputs == null || this.outputs.isEmpty())
								outputs = Map.of(DnnModel.DEFAULT_OUTPUT_NAME, DnnShape.UNKNOWN_SHAPE);
						}
					} catch (Exception e) {
						failed = true;
						logger.debug("Failed to create DjlDnnModel");
						throw new RuntimeException(e);
					}
				}
			}
		}
	}

	@Override
	public Map<String, Mat> predict(Map<String, Mat> blobs) {
		synchronized (predictor) {
			try {
				var result = predictor.predict(blobs.values().stream().toArray(Mat[]::new));
				if (result.length == 1)
					return Map.of(DnnModel.DEFAULT_OUTPUT_NAME, result[0]);
				else if (result.length == 0)
					return Map.of();
				else {
					// Try to handle multiple outputs, naming them sequentially
					Map<String, Mat> output = new LinkedHashMap<>();
					for (int i = 0; i < result.length; i++) {
						output.put(DEFAULT_OUTPUT_NAME + i, result[i]);
					}
					return output;
				}
			} catch (TranslateException e) {
				throw new RuntimeException(e);
			}
		}
	}

	@Override
	public Mat predict(Mat mat) {
		return DnnModel.super.predict(mat);
	}

	@Override
	public List<Mat> batchPredict(List<? extends Mat> mats) {
		return DnnModel.super.batchPredict(mats);
	}

	@Override
	public synchronized void close() throws Exception {
		if (model != null) {
			model.close();
			model = null;
			logger.debug("Closed DjlDnnModel");
		}
	}

	private static String getLayout(LayoutType... layouts) {
		return LayoutType.toString(layouts);
	}
	
	private static String estimateInputLayout(Mat mat) {
		if (mat.channels() >= 1)
			return DEFAULT_MAT_LAYOUT;
		var sizes = mat.createIndexer().sizes();
		switch (sizes.length) {
		case 1: return getLayout(LayoutType.HEIGHT);
		case 2: return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH);
		case 3: return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH, LayoutType.CHANNEL);
		default:
			throw new IllegalArgumentException("Unknown layout for input Mat " + mat);
		}
	}
	
	private static String estimateOutputLayout(NDArray array) {
		var shape = array.getShape();
		if (shape.isLayoutKnown())
			return shape.toLayoutString();
		switch (shape.dimension()) {
		case 1: return getLayout(LayoutType.HEIGHT);
		case 2: return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH);
		case 3: 
			// Assume channels-first or channels-last, and the channels dimensions is shorter
			if (shape.get(2) < shape.get(0))
				return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH, LayoutType.CHANNEL);
			else
				return getLayout(LayoutType.CHANNEL, LayoutType.HEIGHT, LayoutType.WIDTH);
		default:
			throw new IllegalArgumentException("Unknown layout for output shape " + shape);
		}
	}
	
	

	private class ModelMatTranslator implements NoBatchifyTranslator<Mat[], Mat[]> {

		@Override
		public Mat[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
			String layout;
			if ((ndLayout == null || ndLayout.length() != list.get(0).getShape().dimension()) && !list.isEmpty())
				layout = estimateOutputLayout(list.get(0));
			else
				layout = ndLayout;
			var output = list.stream().map(b -> DjlTools.ndArrayToMat(b, layout)).toArray(Mat[]::new);
			return output;
		}

		@Override
		public NDList processInput(TranslatorContext ctx, Mat... input) throws Exception {
			NDList list = new NDList();
			String layout = ndLayout;
			for (var mat : input) {
				// Try to figure out the layout
				if (layout == null) {
					layout = estimateInputLayout(mat);
				}
				list.add(DjlTools.matToNDArray(ctx.getNDManager(), mat, layout));
			}
			return list;
		}
	}

	@Override
	public Collection<URI> getURIs() throws IOException {
		return new ArrayList<>(uris);
	}

	@Override
	public boolean updateURIs(Map<URI, URI> replacements) throws IOException {
		var newUris = uris.stream().map(u -> replacements.getOrDefault(u, u)).collect(Collectors.toList());
		if (Objects.equals(newUris, uris))
			return false;
		this.uris = newUris;
		return true;
	}

}